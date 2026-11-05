#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
using json=nlohmann::json;
static void check(bool ok,const char *msg){if(!ok)throw std::runtime_error(msg);}
struct Trace {std::string dir;int count=0;};
static bool trace(ggml_tensor *t,bool ask,void *user){
 std::string name=ggml_get_name(t);
 bool wanted=t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t)<20000000 &&
  (name.find("ffn")!=std::string::npos || name.find("attn")!=std::string::npos || name.find("norm")!=std::string::npos || name.find("fc")!=std::string::npos || name.find("inp_embd")!=std::string::npos);
 if(ask)return wanted;if(!wanted)return true;
 auto *x=(Trace*)user;std::string file=std::to_string(x->count++)+"-"+name+".f32";
 std::vector<float> values(ggml_nbytes(t)/4);ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
 std::ofstream out(x->dir+"/"+file,std::ios::binary);out.write((char*)values.data(),values.size()*4);
 std::ofstream meta(x->dir+"/index.jsonl",std::ios::app);meta<<json({{"file",file},{"name",name},{"shape",{t->ne[3],t->ne[2],t->ne[1],t->ne[0]}}}).dump()<<'\n';return true;
}
int main(int argc,char **argv){
 try {
 check(argc==7,"target draft capture anchor slots output");
 ggml_backend_load_all();llama_backend_init();
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;
 auto *target=llama_model_load_from_file(argv[1],mp);check(target,"target load");
 auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.n_outputs_max=512;cp.n_outputs_max_per_seq=512;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
 auto *tc=llama_init_from_model(target,cp);check(tc,"target ctx");
 auto *draft=llama_model_load_from_file(argv[2],mp);check(draft,"draft load");cp.ctx_other=tc;
 Trace tr{std::string(argv[6])+".trace"};std::filesystem::create_directory(tr.dir);cp.cb_eval=trace;cp.cb_eval_user_data=&tr;
 auto *dc=llama_init_from_model(draft,cp);check(dc,"draft ctx");llama_set_causal_attn(dc,false);
 std::string dir=argv[3];std::ifstream meta(dir+"/capture.json");json j;meta>>j;
 int anchor=std::stoi(argv[4]),slots=std::stoi(argv[5]);check(anchor>0 && slots>1 && slots<=16,"geometry");
 auto tokens=j["input_tokens"].get<std::vector<int>>();
 std::vector<float> features(size_t(anchor)*20480),raw(size_t(anchor)*2560);
 int tapidx=0;for(int tap:{2,6,10,14,18,22,26,30}){
  std::ifstream f(dir+"/layer-"+std::to_string(tap)+".f32",std::ios::binary);f.read((char*)raw.data(),raw.size()*4);check(bool(f),"feature read");
  for(int r=0;r<anchor;++r)std::copy_n(raw.data()+r*2560,2560,features.data()+size_t(r)*20480+tapidx*2560);++tapidx;
 }
 auto *b=llama_batch_ext_init(dc);
 for(int start=0;start<anchor;start+=512){llama_batch_ext_clear(b);for(int r=start;r<std::min(anchor,start+512);++r){int i=llama_batch_ext_add_embd(b,0,{features.data()+size_t(r)*20480,1,20480});check(i>=0 && llama_batch_ext_set_pos(b,i,&r),"inject batch");}check(llama_process(dc,LLAMA_PROCESS_TYPE_DECODE,b)==0,"inject");}
 llama_batch_ext_clear(b);
 auto *vocab=llama_model_get_vocab(draft);int mask=llama_vocab_mask(vocab);
 for(int i=0;i<slots;++i){int pos=anchor+i;int idx=llama_batch_ext_add_token(b,0,i==0?tokens.at(anchor):mask);check(idx>=0 && llama_batch_ext_set_pos(b,idx,&pos) && llama_batch_ext_set_output_logits(b,idx,true),"noise batch");}
 check(llama_process(dc,LLAMA_PROCESS_TYPE_DECODE,b)==0,"noise decode");llama_synchronize(dc);
 std::ofstream out(argv[6],std::ios::binary);int nv=llama_vocab_n_tokens(vocab);
 for(int i=0;i<slots;++i){auto *p=llama_get_logits_ith(dc,i);check(p,"logits");out.write((char*)p,nv*4);}out.close();check(bool(out),"output write");
 llama_batch_ext_free(b);llama_free(dc);llama_model_free(draft);llama_free(tc);llama_model_free(target);llama_backend_free();
 } catch(const std::exception &e){fprintf(stderr,"REPLAY ERROR: %s\n",e.what());return 1;}return 0;
}
