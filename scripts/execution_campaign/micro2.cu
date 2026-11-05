#include EXEC_IMPL
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "/home/philip/binary-eagle-decoding/runs/w1a8-algorithms-20261020/residual30-native03/ggml/src/ggml-cuda/w1a8_algorithms.cuh"
static __global__ void exec_expand(const uint32_t*w,int8_t*dst,int m,int k){int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<int64_t(m)*k)dst[j]=(w[j/32]>>(j%32))&1?1:-1;}
static __global__ void exec_half(const uint32_t*w,const int8_t*q,__half*hw,__half*hq,int m,int k,int n){int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<int64_t(m)*k)hw[j]=__float2half((w[j/32]>>(j%32))&1?1.f:-1.f);if(j<int64_t(n)*k)hq[j]=__float2half(float(q[j]));}
static __global__ void exec_half_codes(const int8_t*q,__half*hq,int count){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<count)hq[j]=__float2half(float(q[j]));}
static __global__ void exec_epilogue(float*y,const float*a,const float*b,const float*mid,const float*s,int m,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<m*n){int r=j%m,t=j/m;y[j]=exec_affine(int(y[j]),a[r],b[t],mid[r],s[t]);}}
static __global__ void exec_expanded_mma(const int8_t*w,const float*a,const int8_t*q,const float*b,int m,int k,float*y,const float*mid,const float*s,int n){
 int lane=threadIdx.x,g=lane/4,t=lane%4,base=(blockIdx.x*4+threadIdx.y)*16,c[4]={};
 for(int j=0;j<k;j+=32){uint32_t ar[4];for(int z=0;z<4;++z){int r=base+g+(z%2)*8;ar[z]=r<m?*(const uint32_t*)(w+r*k+j+t*4+(z/2)*16):0;}uint32_t b0=g<n?*(const uint32_t*)(q+g*k+j+t*4):0,b1=g<n?*(const uint32_t*)(q+g*k+j+t*4+16):0;asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]):"r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(b0),"r"(b1));}
 for(int z=0;z<4;++z){int r=base+g+(z/2)*8,col=t*2+z%2;if(r<m&&col<n)y[col*m+r]=exec_affine(c[z],a[r],b[col],mid[r],s[col]);}
}
static __global__ void exec_midonly(float*y,const float*mid,const float*s,int m,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<m*n)y[j]=__fadd_rn(y[j],__fmul_rn(mid[j%m],s[j/m]));}
extern "C" int exec_launch(int prep,int dot,int phase,const float*x,const float*g,const uint32_t*w,const float*a,const float*mid,int m,int k,int n,int bits,int mode,float clip,int8_t*q,float*b,float*s,float*y,int8_t*expanded,__half*hw,__half*hq,void*stream){
 auto st=(cudaStream_t)stream;
 if(phase==3){exec_expand<<<(m*k+255)/256,256,0,st>>>(w,expanded,m,k);exec_half<<<(m*k+255)/256,256,0,st>>>(w,q,hw,hq,m,k,n);return cudaGetLastError();}
 if(phase!=2){if(prep==0||mode==0)w1ax_quantize<<<n,256,0,st>>>(x,k,q,b,s,bits,mode,clip,g);else if(prep==1)exec_known_bound<256><<<n,256,0,st>>>(x,k,q,b,s,bits,clip,g);else if(prep==2)exec_known_bound<32><<<n,32,0,st>>>(x,k,q,b,s,bits,clip,g);else exec_known_bound<128><<<n,128,0,st>>>(x,k,q,b,s,bits,clip,g);}
 if(phase!=2 && (dot==6||dot==7))w1a8_make_planes<<<dim3((k/32+7)/8,n),256,0,st>>>(q,k,k/32,(uint32_t*)expanded,(int32_t*)hw);
 if(phase!=2 && dot==8)w1a8_make_lut4<<<dim3((k/4*16+255)/256,n),256,0,st>>>(q,k,k/4,(int16_t*)expanded);
 if(phase!=1){
  if(dot==0)w1ax_integer_dot<<<dim3((m+3)/4,n),dim3(32,4),0,st>>>(w,a,q,b,m,k,k/32,y,mid,s);
  else if(dot==1)exec_cross_dot<4><<<(m+3)/4,dim3(32,4),0,st>>>(w,a,q,b,m,k,k/32,y,mid,s,n);
  else if(dot==2)exec_mma_dot<<<(m+63)/64,dim3(32,4),0,st>>>(w,a,q,b,m,k,k/32,y,mid,s,n);
  else if(dot==3)exec_expanded_mma<<<(m+63)/64,dim3(32,4),0,st>>>(expanded,a,q,b,m,k,y,mid,s,n);
  else if(dot==6||dot==7||dot==8){
   if(dot==6)w1a8_dot_planes<1><<<dim3((m+3)/4,n),dim3(32,4),0,st>>>(w,a,(const uint32_t*)expanded,(const int32_t*)hw,b,m,k,k/32,n,y,nullptr);
   if(dot==7)w1a8_dot_planes<3><<<dim3((m+3)/4,(n+2)/3),dim3(32,4),0,st>>>(w,a,(const uint32_t*)expanded,(const int32_t*)hw,b,m,k,k/32,n,y,nullptr);
   if(dot==8)w1a8_dot_lut4<3><<<dim3((m+3)/4,(n+2)/3),dim3(32,4),0,st>>>(w,a,(const int16_t*)expanded,b,m,k,k/32,n,y,nullptr);
   exec_midonly<<<(m*n+255)/256,256,0,st>>>(y,mid,s,m,n);
  }
  else if(dot==4||dot==5){
   static cublasHandle_t handle=nullptr;if(!handle)cublasCreate(&handle);cublasSetStream(handle,st);
   if(dot==5)exec_half<<<(m*k+255)/256,256,0,st>>>(w,q,hw,hq,m,k,n);
   else exec_half_codes<<<(k*n+255)/256,256,0,st>>>(q,hq,k*n);
   const float one=1,zero=0;auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&one,hw,CUDA_R_16F,k,hq,CUDA_R_16F,k,&zero,y,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);if(status!=CUBLAS_STATUS_SUCCESS)return 1000+status;
   exec_epilogue<<<(m*n+255)/256,256,0,st>>>(y,a,b,mid,s,m,n);
  }
 }
 return cudaGetLastError();
}
