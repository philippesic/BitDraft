// Fixed actual-parent source-indexed windows. No sampler, target forward,
// feature recapture, dense output files, training, or serving-speed claim.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

void llama_set_embeddings_nextn(llama_context *, bool, bool);
float * llama_get_embeddings_nextn(llama_context *);
float * llama_get_embeddings_nextn_ith(llama_context *, int32_t);
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
static constexpr int H = 2560, W = 7680, V = 248320, D = 50000;
static volatile std::sig_atomic_t stopped = 0;
static void stop_handler(int) { stopped = 1; }
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
static void guard(const fs::path & output) {
    require(!stopped && !fs::exists(output/"STOP"), "stop requested");
    std::ifstream control_file("/home/philip/.config/binary-eagle-decoding/gpu-control.json");
    json control; control_file >> control;
    require(bool(control_file) && !control.at("rtx5080").value("pause_requested",true)
            && !control.at("rtx5080").value("stop_all_requested",false), "GPU pause/stop requested");
    for (const fs::path & path : {output, fs::path("/home/philip/binary-eagle-decoding"), fs::path("/mnt/c")})
        require(fs::space(path).available >= uint64_t(2)*1024*1024*1024 + 64*1024*1024, "storage floor");
}
static std::string hex(const unsigned char * data, unsigned count) {
    std::ostringstream out;
    for (unsigned i=0;i<count;++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(data[i]);
    return out.str();
}
static std::string hash_bytes(const void * data, size_t count) {
    unsigned char hash[EVP_MAX_MD_SIZE]; unsigned length=0;
    require(EVP_Digest(data,count,hash,&length,EVP_sha256(),nullptr)==1, "SHA256 failure");
    return hex(hash,length);
}
static std::string hash_file(const fs::path & path) {
    std::ifstream in(path,std::ios::binary); require(bool(in),"cannot read pinned file");
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    require(bool(ctx) && EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA256 init failure");
    std::vector<char> bytes(1024*1024);
    while(in) { require(!stopped,"stop requested during authentication");in.read(bytes.data(),std::streamsize(bytes.size()));
        if(in.gcount()) require(EVP_DigestUpdate(ctx.get(),bytes.data(),size_t(in.gcount()))==1,"SHA256 update failure"); }
    require(in.eof(),"pinned file read failure");
    unsigned char hash[EVP_MAX_MD_SIZE]; unsigned length=0;
    require(EVP_DigestFinal_ex(ctx.get(),hash,&length)==1,"SHA256 finish failure"); return hex(hash,length);
}
static json file_pin(const fs::path & path) {
    return {{"path",fs::canonical(path).string()},{"sha256",hash_file(path)},{"bytes",fs::file_size(path)}};
}
static std::map<std::string,json> authenticated;
static fs::path checked(const json & ref) {
    fs::path path=fs::canonical(ref.at("path").get<std::string>());
    const std::string key=path.string();
    if(!authenticated.count(key)) authenticated[key]=file_pin(path);
    require(authenticated[key]["sha256"]==ref.at("sha256") && authenticated[key]["bytes"]==ref.at("bytes"),"pin changed: "+key);
    return path;
}
static json read_json(const fs::path & path) {
    std::ifstream in(path); json result; in>>result; require(bool(in),"JSON read failed"); return result;
}
static void write_json(const fs::path & path,const json & value) {
    require(!fs::exists(path),"metadata already exists"); const fs::path tmp=path.string()+".tmp";
    std::ofstream out(tmp); out<<value.dump(2)<<'\n';out.close();require(bool(out),"metadata write failed"); fs::rename(tmp,path);
}
static ggml_backend_reg_t registration(const fs::path & path,const char * symbol) {
    void * handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL); require(handle!=nullptr,"backend dlopen failed");
    void * address=dlsym(handle,symbol); Dl_info info={};
    require(address && dladdr(address,&info) && info.dli_fname && fs::canonical(info.dli_fname)==path,"backend resolved elsewhere");
    return reinterpret_cast<ggml_backend_reg_t(*)()>(address)();
}
static llama_context_params params() {
    auto p=llama_context_default_params();p.n_ctx=4096;p.n_batch=512;p.n_ubatch=512;p.n_seq_max=1;
    p.n_threads=4;p.n_threads_batch=4;p.type_k=GGML_TYPE_F16;p.type_v=GGML_TYPE_F16;
    p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;p.n_outputs_max=512;p.n_outputs_max_per_seq=512;
    p.pooling_type=LLAMA_POOLING_TYPE_NONE;return p;
}
static void pair(llama_batch_ext * b,int token,int pos,const float * state,bool output) {
    const int32_t i=llama_batch_ext_add_token(b,0,token);
    require(i>=0 && llama_batch_ext_set_pos(b,i,&pos) && llama_batch_ext_set_embd_token(b,i,{state,1,H})
            && llama_batch_ext_set_output_logits(b,i,output),"shifted pair construction failed");
}
static void process(llama_context * ctx,llama_batch_ext * b,bool encode,const fs::path & output) {
    guard(output); const int result=llama_process(ctx,encode?LLAMA_PROCESS_TYPE_ENCODE:LLAMA_PROCESS_TYPE_DECODE,b);
    require(result==0,"native process failed: "+std::to_string(result));llama_synchronize(ctx);
}
static json loaded_images(const json & packet,const fs::path & output,const std::string & tag) {
    std::map<std::string,json> expected;
    for(const auto & ref:packet.at("runtime_libraries")) expected[fs::canonical(ref.at("path").get<std::string>()).string()]=ref;
    expected[fs::canonical(packet.at("helper").at("path").get<std::string>()).string()]=packet.at("helper");
    std::ifstream maps("/proc/self/maps");std::ofstream copy(output/("loaded-library-maps-"+tag+".txt"));
    std::string line;std::set<std::string> paths;json refs=json::array();
    while(std::getline(maps,line)) { copy<<line<<'\n'; std::istringstream fields(line);
        std::string address,permissions,offset,dev,inode,path;
        fields>>address>>permissions>>offset>>dev>>inode;std::getline(fields,path);
        const auto start=path.find_first_not_of(' ');
        if(start==std::string::npos) { continue; }
        path=path.substr(start);
        if(path.empty() || path[0]!='/' || (permissions.find('x')==std::string::npos && path.find(".so")==std::string::npos))continue;
        require(path.find(" (deleted)")==std::string::npos,"deleted loaded image");paths.insert(fs::canonical(path).string()); }
    for(const auto & path:paths) { require(expected.count(path),"unfrozen loaded image: "+path);checked(expected.at(path));refs.push_back(expected.at(path)); }
    copy.close();require(bool(copy),"maps write failed");return refs;
}
int main(int argc,char ** argv) {
    fs::path output;
    const auto begin=std::chrono::steady_clock::now();
    try {
        if(argc==2 && std::string(argv[1])=="--help") {
            std::cout<<"--cases CASES.json --model parent|candidate --out NEWDIR --allow-cuda [--calibration PARENT/replay.json]\n";
            return 0; }
        if(argc==2 && std::string(argv[1])=="--abi-check") {
            require(llama_context_default_params().n_batch>0,"invalid linked ABI");
            std::cout<<"{\"linked_api_only\":true,\"model_opened\":false,\"backend_initialized\":false}\n";return 0; }
        std::map<std::string,std::string> args;bool allowed=false;
        for(int i=1;i<argc;++i) { const std::string key=argv[i];
            if(key=="--allow-cuda") {require(!allowed,"duplicate CUDA authorization");allowed=true;continue;}
            require((key=="--cases"||key=="--model"||key=="--out"||key=="--calibration") && !args.count(key) && i+1<argc,"invalid argument");args[key]=argv[++i]; }
        require(allowed && args.count("--cases") && args.count("--model") && args.count("--out"),"explicit arguments and CUDA authorization required");
        std::signal(SIGTERM,stop_handler);std::signal(SIGINT,stop_handler);
        const std::string role=args.at("--model");require(role=="parent"||role=="candidate","invalid model role");
        require((role=="candidate")==bool(args.count("--calibration")),"candidate requires parent calibration; parent cannot borrow one");
        require(!fs::exists(args.at("--out")),"output already exists");output=args.at("--out");fs::create_directory(output);guard(output);
        require(fs::file_size(args.at("--cases"))<=2*1024*1024,"packet exceeds2MiB");
        const json packet=read_json(args.at("--cases"));
        require(packet.at("schema")=="w1a8_actual_native_cases_v1" && packet.at("diagnostic_only")==true,"unsupported diagnostic packet");
        require(!packet.at("cases").empty() && packet.at("cases").size()<=128,"require1..128 windows");
        require(packet.at("target").at("sha256")=="3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
                && packet.at("models").at("parent").at("sha256")=="6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048","target/parent ancestry differs");
        for(const auto & setting:std::map<std::string,std::string>{{"GGML_CUDA_DECODE_CONSISTENT","32"},{"GGML_W1AX_ACT_BITS","8"},{"CUDA_VISIBLE_DEVICES","0"}}) {
            const char * value=std::getenv(setting.first.c_str());require(value && value==setting.second,"runtime environment differs: "+setting.first); }
        require(!std::getenv("LD_PRELOAD") || !*std::getenv("LD_PRELOAD"),"preload forbidden");
        const json cases_pin=file_pin(args.at("--cases"));
        const json build=read_json(checked(packet.at("build")));
        require(build.at("original_runtime_only")==true && build.at("executable")==packet.at("helper")
                && build.at("runtime_libraries")==packet.at("runtime_libraries"),"helper/build/runtime binding differs");
        require(checked(packet.at("helper"))==fs::canonical("/proc/self/exe"),"different helper executable");
        std::set<std::string> ids;std::map<int,json> raw_rows;
        for(const auto & row:packet.at("raw_rows")) {const int id=row.at(0);require(raw_rows.emplace(id,row).second,"duplicate raw source row");}
        for(const auto & source:packet.at("files")) {
            checked(source);require(source.at("shape").size()==2 && source.at("shape").at(1)==W
                && source.at("bytes").get<uint64_t>()==source.at("shape").at(0).get<uint64_t>()*W*4,"raw source shape differs"); }
        for(const auto & ref:packet.at("runtime_libraries"))checked(ref);
        std::ifstream mapping(checked(packet.at("draft_map")));std::vector<int> draft_map;int mapped;
        while(mapping>>mapped)draft_map.push_back(mapped);
        require(mapping.eof() && draft_map.size()==D && std::set<int>(draft_map.begin(),draft_map.end()).size()==D
                && *std::min_element(draft_map.begin(),draft_map.end())>=0 && *std::max_element(draft_map.begin(),draft_map.end())<V,"absolute map differs");
        std::set<int> supported(draft_map.begin(),draft_map.end());
        for(const auto & item:packet.at("cases")) {
            const std::string id=item.at("id");const int anchor=item.at("anchor"),depth=item.at("max_depth");
            require(ids.insert(id).second && (item.at("role")=="fit"||item.at("role")=="development") && anchor>=0 && anchor+depth<=4096
                    && depth>=1 && depth<=2 && item.at("prefix_tokens").size()==size_t(anchor)
                    && item.at("prefix_raw_rows").size()==size_t(anchor) && item.at("labels").size()==size_t(depth)
                    && item.at("native_success_mask").size()==size_t(depth) && item.at("native_supervision_reachable_mask").size()==size_t(depth),"invalid actual window"); }
        json calibration=json::object(),calibration_ref=nullptr;
        if(role=="candidate") {
            calibration_ref=file_pin(args.at("--calibration"));calibration=read_json(args.at("--calibration"));
            require(calibration.at("status")=="ACTUAL_NATIVE_REPLAY_COMPLETE" && calibration.at("model_role")=="parent"
                    && calibration.at("model")==packet.at("models").at("parent") && calibration.at("cases_pin")==cases_pin
                    && calibration.at("build")==packet.at("build") && calibration.at("windows").size()==ids.size(),"unchanged parent calibration differs"); }
        if(role=="candidate") for(const auto & item:packet.at("cases")) {
            const json & row=calibration.at("windows").at(item.at("id").get<std::string>());
            const int prefix=row.at("correct_prefix");
            require(row.at("case")==item && prefix>=0 && prefix<=item.at("max_depth").get<int>(),"parent calibration window differs"); }
        const fs::path cpu_path=checked(packet.at("cpu_library")),cuda_path=checked(packet.at("cuda_library"));
        auto * cpu=registration(cpu_path,"ggml_backend_cpu_reg");auto * cuda=registration(cuda_path,"ggml_backend_cuda_reg");
        require(ggml_backend_reg_by_name("CPU")==cpu && ggml_backend_reg_by_name("CUDA")==cuda,"backend registration differs");
        auto * gpu=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(gpu && ggml_backend_dev_backend_reg(gpu)==cuda,"matching CUDA backend required");
        const std::string hardware=ggml_backend_dev_description(gpu);
        require(hardware.find("RTX 5080")!=std::string::npos,"RTX5080 required");llama_backend_init();
        const auto target_path=checked(packet.at("target")),model_path=checked(packet.at("models").at(role));
        ggml_backend_dev_t no_devices[]={nullptr};auto tp=llama_model_default_params();tp.n_gpu_layers=0;tp.devices=no_devices;tp.load_mtp=false;tp.split_mode=LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> target(llama_model_load_from_file(target_path.c_str(),tp),llama_model_free);
        require(bool(target) && llama_model_n_embd(target.get())==H && llama_vocab_n_tokens(llama_model_get_vocab(target.get()))==V,"native embedding provider differs");
        auto tcp=params();tcp.n_outputs_max=1;tcp.n_outputs_max_per_seq=1;
        std::unique_ptr<llama_context,decltype(&llama_free)> target_ctx(llama_init_from_model(target.get(),tcp),llama_free);
        require(bool(target_ctx),"CPU embedding context failed");
        ggml_backend_dev_t devices[]={gpu,nullptr};auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.load_mtp=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(),mp),llama_model_free);
        require(bool(model) && llama_model_n_embd(model.get())==H && llama_model_n_layer(model.get())==1,"native draft geometry differs");
        auto cp=params();cp.ctx_other=target_ctx.get();
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);require(bool(ctx),"draft context failed");
        // No backend sampler is attached: eagle3.cpp expands the head into full V.
        llama_set_embeddings_nextn(ctx.get(),true,true);
        std::unique_ptr<llama_batch_ext,decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()),llama_batch_ext_free);require(bool(batch),"batch allocation failed");
        const json warm_images=loaded_images(packet,output,"loaded");
        std::vector<std::unique_ptr<std::ifstream>> streams;
        for(const auto & source:packet.at("files")) { streams.emplace_back(new std::ifstream(source.at("path").get<std::string>(),std::ios::binary));require(bool(*streams.back()),"raw file open failed"); }
        auto gather=[&](int id,float * dst) {
            require(raw_rows.count(id),"missing raw source row");const json & row=raw_rows.at(id);const int file=row.at(1),index=row.at(2);
            require(file>=0 && size_t(file)<streams.size() && index>=0 && index<packet["files"][file]["shape"][0].get<int>(),"raw source range differs");
            auto & input=*streams.at(file);input.clear();input.seekg(std::streamoff(index)*W*4);input.read(reinterpret_cast<char *>(dst),W*4);
            require(bool(input) && hash_bytes(dst,W*4)==row.at(3),"raw row hash differs");
            for(int j=0;j<W;++j)require(std::isfinite(dst[j]),"nonfinite raw row"); };
        json results=json::object();std::ofstream journal(output/"windows.jsonl");
        int native_reference=0,calibrated_reference=0,lost_reference=0;
        for(const auto & item:packet.at("cases")) {
            guard(output);const std::string id=item.at("id");const int anchor=item.at("anchor"),depth=item.at("max_depth");
            const auto tokens=item.at("prefix_tokens").get<std::vector<int>>();const auto labels=item.at("labels").get<std::vector<int>>();
            std::vector<int> source_ids=item.at("prefix_raw_rows").get<std::vector<int>>();source_ids.push_back(item.at("seed_raw_row"));
            std::vector<float> raw(64*W),encoded(size_t(anchor+1)*H);
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            for(int start=0;start<=anchor;start+=64) {
                const int count=std::min(64,anchor+1-start);llama_batch_ext_clear(batch.get());
                for(int j=0;j<count;++j) {gather(source_ids.at(start+j),raw.data()+size_t(j)*W);
                    const int32_t i=llama_batch_ext_add_embd(batch.get(),0,{raw.data()+size_t(j)*W,1,W});int pos=start+j;
                    require(i>=0 && llama_batch_ext_set_pos(batch.get(),i,&pos) && llama_batch_ext_set_output_logits(batch.get(),i,true),"FC batch failed"); }
                process(ctx.get(),batch.get(),true,output);const float * fc=llama_get_embeddings_nextn(ctx.get());require(fc,"FC output missing");
                for(size_t j=0;j<size_t(count)*H;++j)require(std::isfinite(fc[j]),"nonfinite native FC");
                std::copy_n(fc,size_t(count)*H,encoded.data()+size_t(start)*H); }
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            for(int start=0;start<anchor;start+=64) { llama_batch_ext_clear(batch.get());
                for(int j=start;j<std::min(anchor,start+64);++j)pair(batch.get(),tokens.at(j),j,encoded.data()+size_t(j)*H,false);
                process(ctx.get(),batch.get(),false,output); }
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==anchor-1,"prefix cache position differs");
            std::vector<float> state(encoded.begin()+size_t(anchor)*H,encoded.end());int token=item.at("seed_token"),prefix=0;json steps=json::array();
            for(int d=0;d<depth;++d) {
                llama_batch_ext_clear(batch.get());pair(batch.get(),token,anchor+d,state.data(),true);process(ctx.get(),batch.get(),false,output);
                const float * logits=llama_get_logits_ith(ctx.get(),0);require(logits,"full native logits missing");
                int proposal=-1;float best=-INFINITY,other=-INFINITY;
                for(int i=0;i<V;++i) { require(!std::isnan(logits[i]) && logits[i]!=INFINITY,"invalid full native logit");
                    if(logits[i]>best) {best=logits[i];proposal=i;}
                    if(i!=labels.at(d)) { other=std::max(other,logits[i]); } }
                require(proposal>=0 && std::isfinite(best) && supported.count(proposal),"invalid mapped native top1");
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==anchor+d,"step cache position differs");
                const bool support=supported.count(labels.at(d));json margin=nullptr,gold=nullptr;
                if(support) { require(std::isfinite(logits[labels.at(d)]) && std::isfinite(other),"supported native label not finite");gold=logits[labels.at(d)];margin=logits[labels.at(d)]-other; }
                const bool correct=proposal==labels.at(d);
                steps.push_back({{"depth",d},{"decoder_position",anchor+d},{"cache_position_max",anchor+d},{"input_token",token},
                    {"label",labels.at(d)},{"proposed",proposal},{"correct",correct},{"gold_supported",support},{"gold_logit",gold},{"gold_margin",margin},
                    {"native_success",item.at("native_success_mask").at(d)},{"native_supervision_reachable",item.at("native_supervision_reachable_mask").at(d)}});
                if(!correct) { break; }
                ++prefix;const float * pre=llama_get_embeddings_nextn_ith(ctx.get(),0);require(pre,"native pre-norm missing");
                state.assign(pre,pre+H);
                for(float value:state) { require(std::isfinite(value),"nonfinite pre-norm"); }
                token=labels.at(d); }
            const int original=item.at("native_prefix"),reference=role=="parent"?prefix:calibration.at("windows").at(id).at("correct_prefix").get<int>();
            const int protectable=std::min(original,reference);native_reference+=original;calibrated_reference+=protectable;lost_reference+=std::max(0,protectable-prefix);
            json row={{"case",item},{"correct_prefix",prefix},{"native_reference_prefix",original},{"parent_replay_prefix",reference},
                {"calibrated_reference_prefix",protectable},{"native_successes_outside_parent_calibration",original-protectable},
                {"lost_calibrated_reference_tokens",std::max(0,protectable-prefix)},{"steps",steps},
                {"stop_reason",prefix<depth?"first_gold_divergence":"depth_limit"}};
            results[id]=row;journal<<json{{"id",id},{"result",row}}.dump()<<'\n';journal.flush();require(bool(journal),"journal write failed");
            std::cout<<"window_complete "<<id<<" "<<prefix<<"/"<<depth<<std::endl; }
        const json final_images=loaded_images(packet,output,"final");
        write_json(output/"replay.json",{{"status","ACTUAL_NATIVE_REPLAY_COMPLETE"},{"model_role",role},{"model",packet.at("models").at(role)},
            {"target",packet.at("target")},{"cases_pin",cases_pin},{"build",packet.at("build")},{"calibration",calibration_ref},{"windows",results},
            {"native_reference_tokens",native_reference},{"calibrated_reference_tokens",calibrated_reference},{"lost_calibrated_reference_tokens",lost_reference},
            {"native_successes_outside_parent_calibration",native_reference-calibrated_reference},{"loaded_images",final_images},{"initial_loaded_images",warm_images},
            {"context",4096},{"batch",512},{"ubatch",512},{"prefix_chunk",64},{"kv","F16"},{"correctness_mode",32},{"activation_bits",8},
            {"host_logits","full target vocabulary; no backend sampler attached"},{"target_placement","CPU native embedding provider only; no target forward"},
            {"draft_placement","CUDA"},{"array_bytes_written",0},{"wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()},
            {"hardware",hardware},
            {"scope","bounded existing-window calibration; not actual serving verifier, speed, fresh quality or all549 protection"}});
        return 0;
    } catch(const std::exception & error) {
        if(!output.empty() && fs::exists(output) && !fs::exists(output/"failure.json")) {
            try {write_json(output/"failure.json",{{"error",error.what()},{"partial_windows","windows.jsonl"},
                {"wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}});}catch(...){} }
        std::cerr<<"actual_native_replay: "<<error.what()<<std::endl;return 1;
    }
}
