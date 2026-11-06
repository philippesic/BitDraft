#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <string>
#ifdef COMPONENT_CACHE_DIRECT_READ
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using cache_to_half = void (*)(const void *, half *, int64_t, cudaStream_t);
cache_to_half ggml_get_to_fp16_cuda(ggml_type);
#endif
int main(int argc,char**argv) {
 if(argc!=3 && argc!=4)return 2;
 auto backend=ggml_backend_cuda_init(0);if(!backend)return 3;
 ggml_type qt=std::string(argv[1])=="q8_0"?GGML_TYPE_Q8_0:std::string(argv[1])=="q4_0"?GGML_TYPE_Q4_0:GGML_TYPE_Q5_0;
 int k=128,n=19;std::vector<float> input(k*n);
 for(int i=0;i<k*n;i++)input[i]=std::sin(float(i)*.137f)*float(1+i%5);
 for(int i=0;i<32;i++)input[i]=0;
 // Explicit roundf ties and signed-extremum ties, with separate blocks.
 for(int i=32;i<64;i++)input[i]=float(i-48)+.5f;
 input[32]=127;input[64]=8;input[65]=-8;
 auto ctx=ggml_init({ggml_tensor_overhead()*12+ggml_graph_overhead_custom(12,false),nullptr,true});
 auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);auto q=ggml_new_tensor_2d(ctx,qt,k,n);
 auto cp=ggml_cpy(ctx,a,q);auto out=ggml_cast(ctx,cp,GGML_TYPE_F32);
 auto graph=ggml_new_graph_custom(ctx,12,false);ggml_build_forward_expand(graph,out);
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 4;
 ggml_backend_tensor_set(a,input.data(),0,input.size()*4);
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 5;
 std::vector<float> result(input.size());ggml_backend_tensor_get(out,result.data(),0,result.size()*4);
#ifdef COMPONENT_CACHE_DIRECT_READ
 if(argc==4) {
  // Flash attention uses convert.cu directly; generic GGML CPY does not
  // implement quantized-to-half and is not the native attention read path.
  half * device_half=nullptr;
  if(cudaMalloc(&device_half,input.size()*sizeof(half))!=cudaSuccess)return 6;
  ggml_get_to_fp16_cuda(qt)(q->data,device_half,input.size(),nullptr);
  std::vector<ggml_fp16_t> host_half(input.size());
  if(cudaMemcpy(host_half.data(),device_half,input.size()*sizeof(half),cudaMemcpyDeviceToHost)!=cudaSuccess)return 7;
  cudaFree(device_half);
  for(size_t i=0;i<input.size();i++)result[i]=ggml_fp16_to_fp32(host_half[i]);
 }
#endif
 std::ofstream(std::string(argv[2])+".input.f32",std::ios::binary).write((char*)input.data(),input.size()*4);
 std::ofstream(std::string(argv[2])+".output.f32",std::ios::binary).write((char*)result.data(),result.size()*4);
 std::vector<char> packed(ggml_nbytes(q));ggml_backend_tensor_get(q,packed.data(),0,packed.size());
 std::ofstream(std::string(argv[2])+".packed",std::ios::binary).write(packed.data(),packed.size());
 ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return 0;
}
