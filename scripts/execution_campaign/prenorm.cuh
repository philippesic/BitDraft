// Approximate lane: native RMS reduction copied, but fused translation-unit FTZ differs.
static __global__ void exec_prenorm_kernel(const float*x,const float*norm,const float*g,int k,int bits,float eps,float clip,int8_t*q,float*b,float*s){
 int token=blockIdx.x,tid=threadIdx.x,i=tid*4;float4 v=make_float4(0,0,0,0);float tmp=0;
 if(i<k){v=((const float4*)(x+token*k))[tid];tmp+=v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;}
 __shared__ float shared[32];tmp=block_reduce<block_reduce_method::SUM,1024>(tmp,shared);float scale=rsqrtf(tmp/k+eps);int qm=bits<=2?1:(1<<(bits-1))-1,sum=0;
 if(!isfinite(clip)||clip<=0)__trap();float beta=__fdiv_rn(clip,float(qm)),inv=__fdiv_rn(float(qm),clip);
 if(i<k)for(int z=0;z<4;++z){float value=((&v.x)[z]*scale)*norm[i+z];if(g){if(!isfinite(g[i+z])||g[i+z]<=0)__trap();value=__fmul_rn(value,g[i+z]);}if(!isfinite(value))__trap();float normalized=isfinite(inv)?__fmul_rn(value,inv):__double2float_rn(double(value)/double(clip)*qm);int c=bits==1?(value>=0?1:-1):max(-qm,min(qm,__float2int_rn(fminf(float(qm),fmaxf(-float(qm),normalized)))));q[token*k+i+z]=int8_t(c);sum+=c;}
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);__shared__ int sums[32];if((tid&31)==0)sums[tid/32]=sum;__syncthreads();if(tid==0){int total=0;for(int j=0;j<32;++j)total+=sums[j];b[token]=beta;s[token]=__fmul_rn(float(total),beta);}
}
