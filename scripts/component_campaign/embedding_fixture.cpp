#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=11)return 2;
 bool cuda=std::string(argv[1])=="cuda";auto type=ggml_type(std::stoi(argv[2]));int64_t offset=std::stoll(argv[4]),rows=std::stoll(argv[5]),width=std::stoll(argv[6]);int n=std::stoi(argv[8]);
 std::vector<int32_t> ids(n);std::ifstream(argv[7],std::ios::binary).read((char*)ids.data(),n*4);
 for(auto id:ids)if(id<0||id>=rows)return 3;
 auto backend=cuda?ggml_backend_cuda_init(0):ggml_backend_cpu_init();if(!backend)return 4;
 if(!cuda)ggml_backend_cpu_set_n_threads(backend,4);
 auto ctx=ggml_init({ggml_tensor_overhead()*12+ggml_graph_overhead_custom(12,false),nullptr,true});
 auto weight=ggml_new_tensor_2d(ctx,type,width,rows);auto indices=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,n);auto out=ggml_get_rows(ctx,weight,indices);
 auto graph=ggml_new_graph_custom(ctx,12,false);ggml_build_forward_expand(graph,out);auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 5;
 std::vector<char> raw(ggml_nbytes(weight));std::ifstream file(argv[3],std::ios::binary);file.seekg(offset);file.read(raw.data(),raw.size());if(!file)return 6;
 ggml_backend_tensor_set(weight,raw.data(),0,raw.size());ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*4);
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 7;
 std::vector<float> output(width*n);ggml_backend_tensor_get(out,output.data(),0,output.size()*4);std::ofstream(argv[9],std::ios::binary).write((char*)output.data(),output.size()*4);
 std::ofstream receipt(argv[10]);receipt<<"{\"backend\":\""<<(cuda?"cuda":"cpu")<<"\",\"type\":\""<<ggml_type_name(type)<<"\",\"matrix_bytes\":"<<ggml_nbytes(weight)<<",\"rows\":"<<rows<<",\"width\":"<<width<<",\"lookups\":"<<n<<",\"buffer\":\""<<ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer))<<"\"}\n";
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return 0;
}
