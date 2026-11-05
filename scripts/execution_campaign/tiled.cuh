// Multi-block fixed-bound preparation; workspace and completion are charged.
template<int B> static __global__ void exec_tiled_codes(const float*x,const float*g,int64_t k,int bits,float clip,int8_t*q,int*partial,int tiles){
 int token=blockIdx.y,j=blockIdx.x*256+threadIdx.x,qm=B?((B<=2)?1:(1<<(B-1))-1):((bits<=2)?1:(1<<(bits-1))-1),c=0;
 if(!isfinite(clip)||clip<=0)__trap();
 if(j<k){float v=gained_value(x+token*k,g,j);if(!isfinite(v))__trap();
  if((B?B:bits)==1)c=v>=0?1:-1;else{float inv=__fdiv_rn(float(qm),clip);float z=isfinite(inv)?__fmul_rn(v,inv):__double2float_rn(double(v)/double(clip)*qm);c=max(-qm,min(qm,__float2int_rn(fminf(float(qm),fmaxf(-float(qm),z)))));}q[token*k+j]=int8_t(c);
 }
 for(int d=16;d;d/=2)c+=__shfl_down_sync(0xffffffff,c,d);__shared__ int ws[8];if((threadIdx.x&31)==0)ws[threadIdx.x/32]=c;__syncthreads();if(threadIdx.x==0){int total=0;for(int i=0;i<8;++i)total+=ws[i];partial[token*tiles+blockIdx.x]=total;}
}
static __global__ void exec_tiled_finish(const int*p,int tiles,int bits,float clip,float*b,float*s){int token=blockIdx.x,sum=0;for(int i=threadIdx.x;i<tiles;i+=32)sum+=p[token*tiles+i];for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(threadIdx.x==0){int qm=bits<=2?1:(1<<(bits-1))-1;float beta=__fdiv_rn(clip,float(qm));b[token]=beta;s[token]=__fmul_rn(float(sum),beta);}}
