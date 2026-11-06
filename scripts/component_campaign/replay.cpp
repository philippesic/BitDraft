// Model-specific fixed external work with full cache reconstruction each repetition.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
using json=nlohmann::json;
static void check(bool ok,const char*m){if(!ok)throw std::runtime_error(m);}
static uint64_t hash(const void*p,size_t n){auto*b=(const uint8_t*)p;uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ull;return h;}
struct Audit {json tensors=json::array();bool enabled=false;};
static bool trace(ggml_tensor*t,bool ask,void*u){
 auto&a=*(Audit*)u;std::string name=ggml_get_name(t);
 bool vocabulary=t->src[0] && t->src[0]->ne[1]==248320 && (t->op==GGML_OP_GET_ROWS || t->op==GGML_OP_MUL_MAT);
 bool want=a.enabled && ggml_nbytes(t)<32*1024*1024 && (vocabulary || t->op==GGML_OP_W1AX_PACK || t->op==GGML_OP_CPY||name.find("norm")!=std::string::npos||name.find("ffn")!=std::string::npos||name.find("l_out")!=std::string::npos||name=="result_norm"||name=="result_output");
 if(ask)return want;if(!want)return true;
 for(auto*w:t->src){if(!w || w->type!=GGML_TYPE_F16 || ggml_nbytes(w)>1024*1024)continue;std::string wn=ggml_get_name(w);if(wn.find("norm")==std::string::npos && wn.find(".w1a1_scale")==std::string::npos && wn.find(".w1ax_midpoint")==std::string::npos && wn.find(".w1ax_gain")==std::string::npos)continue;std::vector<uint8_t>raw(ggml_nbytes(w));ggml_backend_tensor_get(w,raw.data(),0,raw.size());a.tensors.push_back({{"kind","half_metadata_read"},{"source_name",wn},{"source_type",ggml_type_name(w->type)},{"consumer_op",ggml_op_name(t->op)},{"consumer_output_type",ggml_type_name(t->type)},{"source_bytes",raw.size()},{"source_hash_fnv1a64",hash(raw.data(),raw.size())}});}
 if(vocabulary){auto*w=t->src[0];a.tensors.push_back({{"kind","vocabulary_allocation"},{"op",ggml_op_name(t->op)},{"name",name},{"weight_tensor_address",uint64_t(uintptr_t(w))},{"weight_data_address",uint64_t(uintptr_t(w->data))},{"weight_bytes",ggml_nbytes(w)},{"weight_type",ggml_type_name(w->type)},{"weight_buffer",w->buffer?ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer)):"none"}});}
 if(t->src[0] && (vocabulary || t->op==GGML_OP_W1A1_MUL_MAT || (t->op==GGML_OP_MUL_MAT && ggml_is_quantized(t->src[0]->type)))){auto*w=t->src[0];if(w->buffer)a.tensors.push_back({{"kind","permanent_weight_buffer"},{"name",name},{"buffer_address",uint64_t(uintptr_t(w->buffer))},{"buffer_bytes",ggml_backend_buffer_get_size(w->buffer)},{"buffer_type",ggml_backend_buft_name(ggml_backend_buffer_get_type(w->buffer))},{"tensor_bytes",ggml_nbytes(w)}});}
 std::vector<uint8_t>bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
 a.tensors.push_back({{"name",name},{"op",ggml_op_name(t->op)},{"type",ggml_type_name(t->type)},{"bytes",bytes.size()},{"hash_fnv1a64",hash(bytes.data(),bytes.size())},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}});return true;
}
static double now(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){try{
 check(argc==8,"target draft capture anchor csv_reps output audit(0/1)");
 int anchor=std::stoi(argv[4]),reps=std::stoi(argv[5]);Audit audit;audit.enabled=std::stoi(argv[7]);
 ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=99;
 auto*target=llama_model_load_from_file(argv[1],mp);check(target,"target load");
 auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.n_outputs_max=512;cp.n_outputs_max_per_seq=512;cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
 auto*tc=llama_init_from_model(target,cp);check(tc,"target context");auto*draft=llama_model_load_from_file(argv[2],mp);check(draft,"draft load");cp.ctx_other=tc;
 auto cache_type=[](const char*value){std::string v=value?value:"f16";return v=="q8_0"?GGML_TYPE_Q8_0:v=="q4_0"?GGML_TYPE_Q4_0:v=="q5_0"?GGML_TYPE_Q5_0:GGML_TYPE_F16;};
 cp.type_k=cache_type(getenv("COMPONENT_K"));cp.type_v=cache_type(getenv("COMPONENT_V"));
 if(audit.enabled){cp.cb_eval=trace;cp.cb_eval_user_data=&audit;}
 auto*dc=llama_init_from_model(draft,cp);check(dc,"draft context");llama_set_causal_attn(dc,false);
 std::string dir=argv[3];json meta;std::ifstream(dir+"/capture.json")>>meta;auto tokens=meta["input_tokens"].get<std::vector<int>>();
 check(anchor+3<int(tokens.size()),"capture context range");
 int maxanchor=anchor+2;std::vector<float>features(size_t(maxanchor)*20480),raw(size_t(maxanchor)*2560);
 int ti=0;for(int tap:{2,6,10,14,18,22,26,30}){std::ifstream f(dir+"/layer-"+std::to_string(tap)+".f32",std::ios::binary);f.read((char*)raw.data(),raw.size()*4);check(bool(f),"features");for(int r=0;r<maxanchor;++r)std::copy_n(raw.data()+r*2560,2560,features.data()+size_t(r)*20480+ti*2560);++ti;}
 auto*b=llama_batch_ext_init(dc);auto*v=llama_model_get_vocab(draft);int mask=llama_vocab_mask(v),nv=llama_vocab_n_tokens(v);
 auto begin=(void(*)())dlsym(RTLD_DEFAULT,"exec_region_begin");auto end=(float(*)(int*))dlsym(RTLD_DEFAULT,"exec_region_end");
 json rows=json::array();
 for(int rep=-2;rep<reps;++rep)for(int offset:{0,1,2}){
  int a=anchor+offset;llama_memory_clear(llama_get_memory(dc),true);audit.tensors=json::array();double t=now();
  for(int start=0;start<a;start+=512){llama_batch_ext_clear(b);for(int r=start;r<std::min(a,start+512);++r){int i=llama_batch_ext_add_embd(b,0,{features.data()+size_t(r)*20480,1,20480});check(i>=0&&llama_batch_ext_set_pos(b,i,&r),"inject input");}check(llama_process(dc,LLAMA_PROCESS_TYPE_DECODE,b)==0,"inject process");}llama_synchronize(dc);double catchup=now()-t;
  llama_batch_ext_clear(b);for(int i=0;i<3;++i){int pos=a+i,idx=llama_batch_ext_add_token(b,0,i==0?tokens.at(a):mask);check(idx>=0&&llama_batch_ext_set_pos(b,idx,&pos)&&llama_batch_ext_set_output_logits(b,idx,i>0||!getenv("EXEC_REQUEST_MASKS")),"decode input");}
  t=now();if(begin)begin();check(llama_process(dc,LLAMA_PROCESS_TYPE_DECODE,b)==0,"decode process");llama_synchronize(dc);double forward=now()-t;int graphs=0;float gpu=end?end(&graphs):0;
  json output=json::array();for(int i=getenv("EXEC_REQUEST_MASKS")?1:0;i<3;++i){auto*p=llama_get_logits_ith(dc,i);check(p,"logits");output.push_back({{"hash_fnv1a64",hash(p,nv*4)},{"argmax",int(std::max_element(p,p+nv)-p)}});}
  size_t size=llama_state_seq_get_size(dc,0);std::vector<uint8_t>state(size);check(llama_state_seq_get_data(dc,state.data(),size,0)==size,"cache snapshot");
  rows.push_back({{"rep",rep},{"warmup",rep<0},{"anchor",a},{"N",3},{"catchup_host_ms",catchup},{"forward_host_ms",forward},{"graph_event_ms",gpu},{"graph_count",graphs},{"outputs",output},{"cache_bytes",size},{"cache_hash_fnv1a64",hash(state.data(),state.size())},{"audit",audit.tensors}});
  std::ofstream(argv[6])<<json({{"scope","fixed external native captured DEV features/history; model-specific full cache rebuild before each forward; changing contexts; no proposal mutation reuse"},{"audit_mode",audit.enabled},{"rows",rows}}).dump(2)<<'\n';
 }
 llama_batch_ext_free(b);llama_free(dc);llama_model_free(draft);llama_free(tc);llama_model_free(target);llama_backend_free();return 0;
}catch(const std::exception&e){fprintf(stderr,"REPLAY ERROR %s\n",e.what());return 1;}}
