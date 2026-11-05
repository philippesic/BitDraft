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
 if(!BYTES&&B!=1){sums[t]=total;__syncthreads();for(int s=128;s;s>>=1){if(t<s)sums[t]+=sums[t+s];__syncthreads();}if(t==0)code_sum[n]=sums[0];}
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

#include "w1a8_algorithms.cuh"
#include "bmma.cuh"
#include "lutgemm_exact.cuh"
__global__ void unpack_a1(const uint32_t*p,int8_t*q,int k,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<k*n)q[i]=((p[i/32]>>(i%32))&1)?1:-1;}
static __global__ void w1a8_dot_token7(
        const uint32_t * weights, const float * weight_scales,
        const int8_t * codes, const float * act_scales,
        int64_t m, int64_t k, int64_t words, float * output, int32_t * raw_dots) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    const int64_t row = (int64_t) blockIdx.x*4 + threadIdx.y;
    const int lane = threadIdx.x;
    int sums_lo[7] = {}, sums_hi[7] = {};
    if (row < m) {
        for (int64_t i = (int64_t) lane*8; i < k; i += 256) {
            const uint32_t signs = weights[row*words + i/32] >> (i%32);
            const uint32_t spread_lo = ((signs & 15u)*0x00204081u) & 0x01010101u;
            const uint32_t spread_hi = (((signs >> 4) & 15u)*0x00204081u) & 0x01010101u;
            const int signs_lo = (int) (((spread_lo ^ 0x01010101u)*0xfeu) | 0x01010101u);
            const int signs_hi = (int) (((spread_hi ^ 0x01010101u)*0xfeu) | 0x01010101u);
#pragma unroll
            for (int token = 0; token < 7; ++token) {
                const int2 packed_codes = *(const int2 *) (codes + token*k + i);
                sums_lo[token] = __dp4a(packed_codes.x, signs_lo, sums_lo[token]);
                sums_hi[token] = __dp4a(packed_codes.y, signs_hi, sums_hi[token]);
            }
        }
    }
#pragma unroll
    for (int token = 0; token < 7; ++token) {
        int sum = sums_lo[token] + sums_hi[token];
        for (int stride = 16; stride > 0; stride /= 2) sum += __shfl_down_sync(0xffffffffu, sum, stride, 32);
        if (row < m && lane == 0) {
            if (raw_dots) raw_dots[token*m + row] = sum;
            const float weighted = (float) sum * weight_scales[row];
            output[token*m + row] = weighted * act_scales[token];
        }
    }
#endif
}

extern "C" int method_launch(int mode,int phase,const float*x,const uint32_t*w,const float*a,int m,int k,int n,int8_t*q,uint32_t*p,float*b,int32_t*s,int16_t*lut,float*y,void*stream){
 cudaStream_t st=(cudaStream_t)stream; bool a1=mode>=10;
 if(phase!=2){
  if(a1)pack<1,false><<<n,256,0,st>>>(x,k,q,p,b,s);else pack<8,true><<<n,256,0,st>>>(x,k,q,p,b,s);
  if(mode==0 && k>4096)w1a8_sum_codes<<<n,256,0,st>>>(q,k,s);
  if(mode==2 || mode==3)w1a8_make_planes<<<dim3((k/32+7)/8,n),256,0,st>>>(q,k,k/32,p,s);
  if(mode==4)w1a8_make_lut4<<<dim3((k/4*16+255)/256,n),256,0,st>>>(q,k,k/4,lut);
  if(mode==12||mode==13)unpack_a1<<<(n*k+255)/256,256,0,st>>>(p,q,k,n);
 }
 if(phase!=2 && (mode==5||mode==13))lut_transpose<<<(m*(k/32)+255)/256,256,0,st>>>(w,p+n*k,m,k/32);
 if(phase!=1){
  if(mode==5||mode==13){
   cudaMemsetAsync(y,0,n*m*sizeof(float),st);int kt=k<=4096?64:128,mt=k<=4096?2048:512;
   if(kt==64)kernel::_nqmv<64><<<dim3((m+mt-1)/mt,k/64,n),256,0,st>>>(p+n*k,nullptr,q,y,m,k,1,mt,k);
   else kernel::_nqmv<128><<<dim3((m+mt-1)/mt,k/128,n),256,0,st>>>(p+n*k,nullptr,q,y,m,k,1,mt,k);
   lut_scale<<<(n*m+255)/256,256,0,st>>>(y,a,b,m,n);
  }
  else if(mode==0||mode==12){if(n!=7||k%128)return -2;if(mode==0&&k>4096)imma_expanded<4,true><<<(m+15)/16,128,0,st>>>(w,a,q,s,b,m,k,k/32,y,nullptr);else imma_expanded<4,false><<<(m+15)/16,128,0,st>>>(w,a,q,nullptr,b,m,k,k/32,y,nullptr);}
  else if(mode==1 && n==7)w1a8_dot_token7<<<(m+3)/4,dim3(32,4),0,st>>>(w,a,q,b,m,k,k/32,y,nullptr);
  else if(mode==1)dp4a_dot<<<dim3((m+3)/4,n),dim3(32,4),0,st>>>(w,a,q,b,m,k,y,nullptr,false);
  else if(mode==2)w1a8_dot_planes<1><<<dim3((m+3)/4,n),dim3(32,4),0,st>>>(w,a,p,s,b,m,k,k/32,n,y,nullptr);
  else if(mode==3)w1a8_dot_planes<7><<<dim3((m+3)/4,(n+6)/7),dim3(32,4),0,st>>>(w,a,p,s,b,m,k,k/32,n,y,nullptr);
  else if(mode==4)w1a8_dot_lut4<7><<<dim3((m+3)/4,(n+6)/7),dim3(32,4),0,st>>>(w,a,lut,b,m,k,k/32,n,y,nullptr);
  else if(mode==10)binary_dot<1><<<dim3((m+3)/4,n),dim3(32,4),0,st>>>(w,a,p,b,s,m,k,y,nullptr,false);
  else if(mode==11)w1a1_binary_mma<<<dim3((m+7)/8,(n+7)/8),32,0,st>>>(w,a,p,b,m,n,k,k/32,y);
 }
 return cudaGetLastError();
}
