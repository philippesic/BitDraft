#ifdef BINARY_RSR
#include "rsr-upstream.cuh"

struct RsrIndex {
    uint16_t * permutations;
    int4 * groups;
    int32_t * metadata;
    int block;
    int padded;
};

static RsrIndex rsr_index(Memory & memory,const std::string & prefix,int m,int k,int block) {
    const auto name=prefix+".rsr"+std::to_string(block);
    const auto count=read_values<uint32_t>(name+".sizes",1)[0];
    const int blocks=(m+block-1)/block;
    return {memory.upload(read_values<uint16_t>(name+".perms",size_t(blocks)*k)),memory.upload(read_values<int4>(name+".groups",count)),memory.upload(read_values<int32_t>(name+".meta",blocks*2)),block,blocks*block};
}

static __global__ void rsr_codes(const int8_t * q,float * values,int * totals,int k) {
    const int token=blockIdx.x;
    int sum=0;
    for(int j=threadIdx.x;j<k;j+=blockDim.x) { int value=q[token*k+j];values[token*k+j]=float(value);sum+=value; }
    for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
    __shared__ int partial[8];
    if((threadIdx.x&31)==0) partial[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x==0) {int total=0;for(int i=0;i<8;++i)total+=partial[i];totals[token]=total;}
}

static __global__ void rsr_finish(const float * binary,const int * totals,int * integers,float * output,const float * a,const float * b,const float * mid,const float * sums,int m,int padded,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n) {
        const int token=i/m,row=i%m;
        const int acc=2*__float2int_rn(binary[token*padded+row])-totals[token];
        integers[i]=acc;
        output[i]=exec_affine(acc,a[row],b[token],mid[row],sums[token]);
    }
}

static void launch_rsr(RsrIndex index,const float * values,float * binary,int * integers,const int * totals,Matrix matrix,const float * b,const float * sums,float * output,int m,int k,int n,cudaStream_t stream) {
    for(int token=0;token<n;++token) {
        const int blocks=index.padded/index.block;
        if(index.block<=4) rsr_upstream::rsr_gemv_v5_9_kernel<256><<<blocks,256,8*index.block*sizeof(float),stream>>>(index.permutations,index.groups,index.metadata,values+token*k,binary+token*index.padded,k,index.block);
        else rsr_upstream::rsr_gemv_v5_9_kernel<512><<<blocks,512,16*index.block*sizeof(float),stream>>>(index.permutations,index.groups,index.metadata,values+token*k,binary+token*index.padded,k,index.block);
    }
    rsr_finish<<<(m*n+255)/256,256,0,stream>>>(binary,totals,integers,output,matrix.scales,b,matrix.midpoint,sums,m,index.padded,n);
}
#endif
