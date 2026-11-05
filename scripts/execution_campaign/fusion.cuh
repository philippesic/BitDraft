// Fixed-bound SiLU/up/gain/down-quant fusion. Whole-position code sum is retained.
template<int T> static __global__ void exec_silu_pack(const float*gate,const float*up,const float*g,int64_t k,int bits,float clip,int8_t*q,float*b,float*s){
 int token=blockIdx.x,qm=bits<=2?1:(1<<(bits-1))-1,total=0;if(!isfinite(clip)||clip<=0)__trap();
 float beta=__fdiv_rn(clip,float(qm)),inv=__fdiv_rn(float(qm),clip);
 for(int64_t j=threadIdx.x;j<k;j+=T){
  float v=gate[token*k+j];v=__fdividef(v,__fadd_rn(1.f,__expf(-v)));v=__fmul_rn(v,up[token*k+j]);
  if(g){if(!isfinite(g[j])||g[j]<=0)__trap();v=__fmul_rn(v,g[j]);}
  if(!isfinite(v))__trap();float z=isfinite(inv)?__fmul_rn(v,inv):__double2float_rn(double(v)/double(clip)*qm);
  int c=bits==1?(v>=0?1:-1):max(-qm,min(qm,__float2int_rn(fminf(float(qm),fmaxf(-float(qm),z)))));q[token*k+j]=int8_t(c);total+=c;
 }
 for(int d=16;d;d/=2)total+=__shfl_down_sync(0xffffffff,total,d);__shared__ int totals[T/32];if((threadIdx.x&31)==0)totals[threadIdx.x/32]=total;__syncthreads();
 if(threadIdx.x==0){int sum=0;for(int i=0;i<T/32;++i)sum+=totals[i];b[token]=beta;s[token]=__fmul_rn(float(sum),beta);}
}
