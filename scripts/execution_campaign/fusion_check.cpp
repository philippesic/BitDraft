#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("output");auto backend=ggml_backend_cuda_init(0);std::ofstream output(argv[1],std::ios::binary);
 for(int bits:{1,2,4,8})for(int n:{1,3,8}){
  int k=9216;ggml_init_params params={ggml_tensor_overhead()*16+ggml_graph_overhead_custom(32,false),nullptr,true};auto*ctx=ggml_init(params);
  auto*gate=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);auto*up=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);auto*gain=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);auto*mul=ggml_mul(ctx,up,ggml_silu(ctx,gate));auto*pack=ggml_w1ax_pack(ctx,mul,bits);pack->src[1]=gain;int mode=2;float clip=1.005f;memcpy((char*)pack->op_params+4,&mode,4);memcpy((char*)pack->op_params+8,&clip,4);
  auto*g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,pack);auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
  std::vector<float>a(k*n),b(k*n),c(k);for(int j=0;j<k;++j)c[j]=float(j%5+1)/2.f;
  for(int j=0;j<k*n;++j){a[j]=std::sin(j*.0137f)*12;b[j]=std::cos(j*.037f)*2;if(j%11==0){uint32_t v=0x80000001u+(j%127);memcpy(&a[j],&v,4);b[j]=1;}if(j%13==0)a[j]=float(j%511-255);}
  ggml_backend_tensor_set(gate,a.data(),0,a.size()*4);ggml_backend_tensor_set(up,b.data(),0,b.size()*4);ggml_backend_tensor_set(gain,c.data(),0,c.size()*4);if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");ggml_backend_synchronize(backend);
  std::vector<char>bytes(ggml_nbytes(pack));ggml_backend_tensor_get(pack,bytes.data(),0,bytes.size());output.write((char*)&bits,4);output.write((char*)&n,4);output.write(bytes.data(),bytes.size());ggml_backend_buffer_free(buffer);ggml_free(ctx);
 }
 ggml_backend_free(backend);return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
