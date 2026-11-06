// One actual trained/native operator graph invocation before end-to-end timing.
// Build against the SAME GGML libraries and header as the private llama runtime.
#include "binary_reference.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char ** argv) { try {
    require(argc==6,"usage: native-binary-parity MODEL.gguf KIND INPUT.f32 OUTPUT.f32 ROWS");
    const std::string model_path=argv[1], mode=argv[2], input_path=argv[3], output_path=argv[4];
    const int64_t rows=std::stoll(argv[5]);require(rows>0 && rows<=16,"bounded parity rows");
    std::ifstream exists(output_path,std::ios::binary);require(!exists.good(),"exclusive parity output required");
    const std::string prefix="blk.4.ffn_up";
    ggml_context * metadata=nullptr;
    auto * file=gguf_init_from_file(model_path.c_str(),{true,&metadata});require(file&&metadata,"GGUF metadata");
    const int64_t kind_key=gguf_find_key(file,"dflash.binary.kind");
    require(kind_key>=0 && mode==gguf_get_val_str(file,kind_key),"actual GGUF mode differs");
    auto * gpu=ggml_backend_cuda_init(0);auto * cpu=ggml_backend_cpu_init();require(gpu&&cpu,"backend initialization");
    ggml_backend_t backends[]={gpu,cpu};
    ggml_context * weights=ggml_init({32*ggml_tensor_overhead(),nullptr,true});require(weights,"weight metadata context");
    binary_reference::tensors tensors;
    std::vector<std::pair<ggml_tensor*,int64_t>> load;
    for (int64_t i=0;i<gguf_get_n_tensors(file);++i) {
        const std::string name=gguf_get_tensor_name(file,i);
        if (name.rfind(prefix+".binary_",0)!=0) continue;
        auto * original=ggml_get_tensor(metadata,name.c_str());require(original&&original->type==GGML_TYPE_F32,"decoded F32 weight");
        auto * value=ggml_new_tensor_2d(weights,GGML_TYPE_F32,original->ne[0],original->ne[1]);
        ggml_set_name(value,name.c_str());tensors[name]=value;load.push_back({value,i});
    }
    require(!tensors.empty(),"decoded operator tensors missing");
    auto * weight_buffer=ggml_backend_alloc_ctx_tensors(weights,gpu);require(weight_buffer,"CUDA weights");
    ggml_backend_buffer_set_usage(weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::ifstream raw(model_path,std::ios::binary);require(bool(raw),"GGUF open");
    for (const auto & item:load) {
        std::vector<char> bytes(ggml_nbytes(item.first));
        raw.seekg(gguf_get_data_offset(file)+gguf_get_tensor_offset(file,item.second));
        raw.read(bytes.data(),bytes.size());require(bool(raw),"weight payload read");
        ggml_backend_tensor_set(item.first,bytes.data(),0,bytes.size());
    }
    const int64_t width=tensors.at(prefix+".binary_signs")->ne[0];
    std::vector<float> inputs(size_t(width*rows));
    std::ifstream in(input_path,std::ios::binary);in.read((char*)inputs.data(),inputs.size()*4);require(bool(in),"actual input read");
    require(in.peek()==std::char_traits<char>::eof(),"input size differs");
    std::vector<float> output, all_codes;
    const std::vector<int64_t> chunks=rows==9 ? std::vector<int64_t>{7,2} : std::vector<int64_t>{rows};
    int64_t offset=0;
    for (const auto chunk:chunks) {
        ggml_context * ctx=ggml_init({1024*ggml_tensor_overhead()+ggml_graph_overhead_custom(1024,false),nullptr,true});
        require(ctx,"graph context");
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,chunk);ggml_set_input(x);ggml_set_name(x,"binary_parity_input");
        auto * y=binary_reference::graph(ctx,tensors,prefix,x,binary_reference::parse_kind(mode));ggml_set_output(y);
        auto activation=binary_reference::a8(ctx,x);ggml_set_output(activation.codes);
        auto * graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,y);
        ggml_build_forward_expand(graph,activation.codes);
        auto * scheduler=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);require(scheduler,"scheduler");
        require(ggml_backend_sched_alloc_graph(scheduler,graph),"graph allocation");
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            require(ggml_backend_sched_get_tensor_backend(scheduler,ggml_graph_node(graph,i))==gpu,"operator fell back off CUDA");
        ggml_backend_tensor_set(x,inputs.data()+offset*width,0,size_t(chunk*width)*4);
        require(ggml_backend_sched_graph_compute(scheduler,graph)==GGML_STATUS_SUCCESS,"native graph compute");
        ggml_backend_sched_synchronize(scheduler);
        std::vector<float> values(ggml_nelements(y)), codes(size_t(chunk*width));
        ggml_backend_tensor_get(y,values.data(),0,values.size()*4);
        ggml_backend_tensor_get(activation.codes,codes.data(),0,codes.size()*4);
        output.insert(output.end(),values.begin(),values.end());all_codes.insert(all_codes.end(),codes.begin(),codes.end());
        offset+=chunk;ggml_backend_sched_free(scheduler);ggml_free(ctx);
    }
    std::ofstream out(output_path,std::ios::binary);out.write((const char*)output.data(),output.size()*4);require(bool(out),"parity output write");out.close();
    std::ofstream code_out(output_path+".codes.f32",std::ios::binary);code_out.write((const char*)all_codes.data(),all_codes.size()*4);require(bool(code_out),"A8 code output write");code_out.close();
    std::cout<<"NATIVE_BINARY_PARITY_GPU_COMPLETE mode="<<mode<<" rows="<<rows<<" width="<<width<<" actual_graph_rows=7 fixture_graph_rows=2\n";
    ggml_backend_buffer_free(weight_buffer);ggml_free(weights);
    ggml_free(metadata);gguf_free(file);ggml_backend_free(gpu);ggml_backend_free(cpu);return 0;
} catch(const std::exception & error) { std::cerr<<error.what()<<"\n";return 1; } }
