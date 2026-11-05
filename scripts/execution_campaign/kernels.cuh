// Execution-only alternatives; preserve explicit F32 multiply/add ordering.
template<int T> static __global__ void exec_known_bound(
 const float* x,int64_t k,int8_t* q,float* scales,float* sums,int bits,float clip,const float* gains) {
 const int token=blockIdx.x;
 if(!isfinite(clip)||clip<=0) __trap();
 const int qm=bits<=2?1:(1<<(bits-1))-1;
 const float beta=__fdiv_rn(clip,float(qm)),inv=__fdiv_rn(float(qm),clip);
 int total=0;
 for(int64_t j=threadIdx.x;j<k;j+=T){
  float v=gained_value(x+token*k,gains,j);
  if((__float_as_uint(v)&0x7fffffffu)>=0x7f800000u)__trap();
  int c;
  if(bits==1)c=v>=0?1:-1;
  else {
   float z=isfinite(inv)?__fmul_rn(v,inv):__double2float_rn(double(v)/double(clip)*double(qm));
   c=max(-qm,min(qm,__float2int_rn(fminf(float(qm),fmaxf(-float(qm),z)))));
  }
  q[token*k+j]=int8_t(c);total+=c;
 }
 for(int d=16;d;d/=2)total+=__shfl_down_sync(0xffffffff,total,d);
 __shared__ int warp_totals[T/32];
 if((threadIdx.x&31)==0)warp_totals[threadIdx.x/32]=total;
 if(T>32)__syncthreads();
 if(threadIdx.x==0){int sum=0;for(int i=0;i<T/32;++i)sum+=warp_totals[i];scales[token]=beta;sums[token]=__fmul_rn(float(sum),beta);}
}
static __device__ __forceinline__ float exec_affine(int v,float a,float b,float mid,float s){
 if((__float_as_uint(a)&0x80000000u)||!isfinite(a)||(__float_as_uint(b)&0x80000000u)||!isfinite(b))__trap();
 return __fadd_rn(__fmul_rn(__fmul_rn(float(v),a),b),__fmul_rn(mid,s));
}
static __device__ __forceinline__ uint32_t exec_sign4(uint32_t signs){
 const uint32_t spread=((signs&15u)*0x00204081u)&0x01010101u;
 return ((spread^0x01010101u)*0xfeu)|0x01010101u;
}
template<int C> static __global__ void exec_cross_dot(
 const uint32_t*w,const float*a,const int8_t*q,const float*b,int64_t m,int64_t k,int64_t words,float*y,const float*mid,const float*s,int64_t n){
 int64_t row=blockIdx.x*4+threadIdx.y;int lane=threadIdx.x;int acc[C]={};
 if(row<m)for(int64_t i=lane*4;i<k;i+=128){
  int sign=exec_sign4(w[row*words+i/32]>>(i%32));
  #pragma unroll
  for(int t=0;t<C;++t)if(t<n)acc[t]=__dp4a(*(const int*)(q+t*k+i),sign,acc[t]);
 }
 #pragma unroll
 for(int t=0;t<C;++t){
  for(int d=16;d;d/=2)acc[t]+=__shfl_down_sync(0xffffffff,acc[t],d);
  if(row<m && t<n && lane==0)y[t*m+row]=exec_affine(acc[t],a[row],b[t],mid?mid[row]:0,s[t]);
 }
}
static __global__ void exec_mma_dot(
 const uint32_t*w,const float*a,const int8_t*q,const float*b,int64_t m,int64_t k,int64_t words,float*y,const float*mid,const float*s,int64_t n){
 const int lane=threadIdx.x,g=lane/4,t=lane%4;
 const int rowbase=(blockIdx.x*4+threadIdx.y)*16;
 int c[4]={};
 for(int j=0;j<k;j+=32){
  uint32_t ar[4];
  #pragma unroll
  for(int z=0;z<4;++z){int row=rowbase+g+(z%2)*8;int shift=t*4+(z/2)*16;ar[z]=row<m?exec_sign4(w[row*words+j/32]>>shift):0;}
  uint32_t br0=g<n?*(const uint32_t*)(q+g*k+j+t*4):0;
  uint32_t br1=g<n?*(const uint32_t*)(q+g*k+j+t*4+16):0;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br0),"r"(br1));
 }
 #pragma unroll
 for(int z=0;z<4;++z){int row=rowbase+g+(z/2)*8,col=t*2+(z%2);if(row<m && col<n)y[col*m+row]=exec_affine(c[z],a[row],b[col],mid?mid[row]:0,s[col]);}
}
