// Screening prototypes only; not a ggml backend or deployed model.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

template<int B,bool BYTES> __global__ void pack(const float*x,int k,int8_t*codes,uint32_t*planes,float*beta,int32_t*code_sum){
 __shared__ uint32_t mx[256];__shared__ double av[256];__shared__ int sums[256];__shared__ float maximum,scale;
 int t=threadIdx.x,n=blockIdx.x;uint32_t m=0;double a=0;
 for(int i=t;i<k;i+=256){float v=x[n*k+i];uint32_t u=__float_as_uint(v)&0x7fffffff;if(u>=0x7f800000u)__trap();m=max(m,u);if(B==1)a+=(double)__uint_as_float(u);}
 mx[t]=m;if(B==1)av[t]=a;__syncthreads();
 for(int s=128;s;s>>=1){if(t<s){mx[t]=max(mx[t],mx[t+s]);if(B==1)av[t]+=av[t+s];}__syncthreads();}
 if(t==0){maximum=__uint_as_float(mx[0]);scale=B==1?(float)(av[0]/k):__fdiv_rn(maximum,(float)((1<<(B-1))-1));beta[n]=scale;}__syncthreads();
 int total=0;float inv=(B==1||maximum==0)?0:__fdiv_rn((float)((1<<(B-1))-1),maximum);
 for(int base=0;base<k;base+=256){int i=base+t,q=0;
  if(i<k){float v=x[n*k+i];if(B==1)q=v>=0?1:-1;else{float z=maximum==0?0:isfinite(inv)?__fmul_rn(v,inv):__double2float_rn((double)v/(double)maximum*((1<<(B-1))-1));q=max(-((1<<(B-1))-1),min((1<<(B-1))-1,__float2int_rn(z)));}total+=q;if(BYTES)codes[n*k+i]=(int8_t)q;}
  if(!BYTES){for(int b=0;b<B;b++){bool bit=B==1?q>0:((q>>b)&1);uint32_t bits=__ballot_sync(0xffffffffu,bit);if((t&31)==0&&i<k)planes[(n*B+b)*(k/32)+i/32]=bits;}}
 }
 sums[t]=total;__syncthreads();for(int s=128;s;s>>=1){if(t<s)sums[t]+=sums[t+s];__syncthreads();}if(t==0)code_sum[n]=sums[0];
}

template<int B> __global__ void binary_dot(const uint32_t*w,const float*alpha,const uint32_t*p,const float*beta,const int32_t*qs,int m,int k,float*y,int32_t*integer,bool check){
 int row=blockIdx.x*4+threadIdx.y,n=blockIdx.y,lane=threadIdx.x,words=k/32;int sum=0;
 if(row<m)for(int i=lane;i<words;i+=32){uint32_t weight=w[row*words+i];if(B==1)sum+=__popc(weight^p[n*words+i]);else for(int b=0;b<B;b++){int coefficient=b==B-1?-(1<<(B-1)):(1<<b);sum+=coefficient*__popc(weight&p[(n*B+b)*words+i]);}}
 for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,s);
 if(row<m&&lane==0){int z=B==1?k-2*sum:2*sum-qs[n];if(check)integer[n*m+row]=z;y[n*m+row]=__fmul_rn(__fmul_rn((float)z,alpha[row]),beta[n]);}
}
// Same packed-sign register expansion and four-row/warp geometry as retained
// native W1A8; output integer storage is disabled during timing.
__global__ void dp4a_dot(const uint32_t*w,const float*alpha,const int8_t*x,const float*beta,int m,int k,float*y,int32_t*integer,bool check){
 int row=blockIdx.x*4+threadIdx.y,n=blockIdx.y,lane=threadIdx.x,sum=0;
 if(row<m)for(int i=lane*4;i<k;i+=128){uint32_t signs=w[row*(k/32)+i/32]>>(i%32);uint32_t spread=((signs&15u)*0x00204081u)&0x01010101u;uint32_t s=((spread^0x01010101u)*0xfeu)|0x01010101u;sum=__dp4a(*(const int*)(x+n*k+i),(int)s,sum);}
 for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,s);
 if(row<m&&lane==0){if(check)integer[n*m+row]=sum;y[n*m+row]=__fmul_rn(__fmul_rn((float)sum,alpha[row]),beta[n]);}
}
extern "C" int screen_launch(int mode,int phase,const float*x,const uint32_t*w,const float*alpha,int m,int k,int n,int8_t*codes,uint32_t*planes,float*beta,int32_t*qs,float*y,int32_t*integer,int check,void*stream){
 if(k%32||m<1||n<1)return -1;cudaStream_t s=(cudaStream_t)stream;
 if(phase!=2){if(mode==0)pack<8,true><<<n,256,0,s>>>(x,k,codes,planes,beta,qs);else if(mode==1)pack<1,false><<<n,256,0,s>>>(x,k,codes,planes,beta,qs);else if(mode==4)pack<4,false><<<n,256,0,s>>>(x,k,codes,planes,beta,qs);else if(mode==8)pack<8,false><<<n,256,0,s>>>(x,k,codes,planes,beta,qs);else return -2;}
 if(phase!=1){dim3 grid((m+3)/4,n),block(32,4);if(mode==0)dp4a_dot<<<grid,block,0,s>>>(w,alpha,codes,beta,m,k,y,integer,check);else if(mode==1)binary_dot<1><<<grid,block,0,s>>>(w,alpha,planes,beta,qs,m,k,y,integer,check);else if(mode==4)binary_dot<4><<<grid,block,0,s>>>(w,alpha,planes,beta,qs,m,k,y,integer,check);else if(mode==8)binary_dot<8><<<grid,block,0,s>>>(w,alpha,planes,beta,qs,m,k,y,integer,check);else return -2;}
 return (int)cudaGetLastError();
}
