#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstring>
#include <functional>
#include <fstream>
#include <stdexcept>
#include "runtime-kernels.cuh"
#include "kernel-variants.cuh"
#ifdef RUNTIME_RECOVERED_IMMA
#define spread_u8 harness_spread_u8
#define spread_s8 harness_spread_s8
#define recovered_imma harness_recovered_imma
#endif
#include "kernel-imma.cuh"

static void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> static std::vector<T> read_values(const std::string & path, size_t count) {
    std::vector<T> result(count);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(result.data()), count * sizeof(T)))
        throw std::runtime_error("truncated input: " + path);
    if (input.peek() != EOF) throw std::runtime_error("oversized input: " + path);
    return result;
}

struct Memory {
    std::vector<void *> allocations;
    template<class T> T * upload(const std::vector<T> & values) {
        T * pointer;
        checked(cudaMalloc(&pointer, values.size() * sizeof(T)));
        allocations.push_back(pointer);
        checked(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
        return pointer;
    }
    ~Memory() { for (void * pointer : allocations) cudaFree(pointer); }
};

static std::vector<float> download(const float * pointer, size_t count) {
    std::vector<float> result(count);
    checked(cudaMemcpy(result.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

static void measure(const std::function<void()> & launch, cudaStream_t stream, const float * output,
                    size_t count, const std::vector<float> & reference, int warmup, int repeats, int trials, const int * integers = nullptr, const std::vector<int> & integer_reference = {}) {
    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    bool identical = true;
    bool integer_identical = true;
    for (int j = 0; j < warmup; ++j) launch();
    checked(cudaStreamSynchronize(stream));
    printf("warmup,%d,per_trial,%d\n",warmup,std::min(warmup,200));
    for (int trial = -1; trial < trials; ++trial) {
        checked(cudaMemsetAsync(const_cast<float *>(output), 0xff, count * sizeof(float), stream));
        if(integers) checked(cudaMemsetAsync(const_cast<int *>(integers),0x80,count*sizeof(int),stream));
        for (int j = 0; j < std::min(warmup,200); ++j) launch();
        checked(cudaEventRecord(start, stream));
        for (int j = 0; j < repeats; ++j) launch();
        checked(cudaEventRecord(stop, stream));
        checked(cudaEventSynchronize(stop));
        checked(cudaGetLastError());
        float milliseconds;
        checked(cudaEventElapsedTime(&milliseconds, start, stop));
        const auto actual = download(output, count);
        identical = identical && memcmp(actual.data(), reference.data(), count * sizeof(float)) == 0;
        if (integers) {
            std::vector<int> actual_integers(count);
            checked(cudaMemcpy(actual_integers.data(),integers,count*sizeof(int),cudaMemcpyDeviceToHost));
            integer_identical = integer_identical && actual_integers == integer_reference;
        }
        if (trial >= 0) printf("sample,%d,%.12g,%d\n", trial, double(milliseconds) * 1000 / repeats, identical);
    }
    if(integers) printf("integers,%d\n",integer_identical);
    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
}

struct Matrix {
    uint32_t * weights;
    float * scales;
    float * midpoint;
};

static Matrix matrix(Memory & memory, const std::string & prefix, int m, int k) {
    return {
        memory.upload(read_values<uint32_t>(prefix + ".weights", size_t(m) * k / 32)),
        memory.upload(read_values<float>(prefix + ".scales", m)),
        memory.upload(read_values<float>(prefix + ".midpoint", m))
    };
}

#include "kernel-rsr.cuh"
#include "kernel-vocabulary.cuh"

static void launch_one(const std::string & variant, Matrix weight, const int8_t * q,
                       const float * b, const float * sums, float * output, int m, int k, int n,
                       cudaStream_t stream, const int32_t * code_sums) {
    if (variant.rfind("imma-",0)==0) {
        if(k%128) throw std::runtime_error("recovered IMMA requires columns divisible by 128");
        for(int offset=0;offset<n;offset+=8) {
            const int columns=std::min(8,n-offset);
            if(variant=="imma-expanded" && k>4096) recovered_imma<4,true><<<(m+15)/16,128,0,stream>>>(weight.weights,weight.scales,q+offset*k,code_sums+offset,b+offset,m,k,k/32,output+offset*m,nullptr,weight.midpoint,sums+offset,columns);
            else if(variant=="imma-split1") recovered_imma<1,false><<<(m+15)/16,32,0,stream>>>(weight.weights,weight.scales,q+offset*k,nullptr,b+offset,m,k,k/32,output+offset*m,nullptr,weight.midpoint,sums+offset,columns);
            else recovered_imma<4,false><<<(m+15)/16,128,0,stream>>>(weight.weights,weight.scales,q+offset*k,nullptr,b+offset,m,k,k/32,output+offset*m,nullptr,weight.midpoint,sums+offset,columns);
        }
    } else if (variant == "reference") {
        w1ax_integer_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums);
    } else if (variant == "cross") {
        if (n == 1) exec_cross_dot<1><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
        else if (n == 2) exec_cross_dot<2><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
        else if (n == 4) exec_cross_dot<4><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
        else if (n == 8) exec_cross_dot<8><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
        else exec_cross_dot<16><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
    } else if (variant == "mma" || variant == "mma-broadcast" || variant == "mma-warp") {
        for (int offset = 0; offset < n; offset += 8) {
            const int columns = std::min(8,n-offset);
            if (variant == "mma") exec_mma_dot<<<dim3((m+63)/64),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q+offset*k,b+offset,m,k,k/32,output+offset*m,weight.midpoint,sums+offset,columns);
            else if (variant == "mma-warp") mma_warp_dot<<<dim3(m/16),dim3(32),0,stream>>>(weight.weights,weight.scales,q+offset*k,b+offset,m,k,k/32,output+offset*m,weight.midpoint,sums+offset,columns);
            else mma_broadcast_dot<<<dim3((m+63)/64),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q+offset*k,b+offset,m,k,k/32,output+offset*m,weight.midpoint,sums+offset,columns);
        }
    } else throw std::runtime_error("unknown variant");
}

static void binary(const std::string & prefix, const std::string & variant, int m, int k, int n,
                   int bits, int mode, float clip, int warmup, int repeats, int trials, bool joint) {
    Memory memory;
    const auto first = matrix(memory,prefix,m,k);
    const auto second = joint ? matrix(memory,prefix+".up",m,k) : first;
    auto x = memory.upload(read_values<float>(prefix+".input",size_t(k)*n));
    auto gains = memory.upload(read_values<float>(prefix+".gain",k));
    auto q = memory.upload(std::vector<int8_t>(k*n));
    auto b = memory.upload(std::vector<float>(n));
    auto sums = memory.upload(std::vector<float>(n));
    auto output = memory.upload(std::vector<float>(size_t(m)*n*(joint?2:1)));
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    w1ax_quantize<<<n,256,0,stream>>>(x,k,q,b,sums,bits,mode,clip,gains);
    auto code_sums = memory.upload(std::vector<int32_t>(n));
    if(variant=="imma-expanded" && k>4096) w1a8_sum_codes<<<n,256,0,stream>>>(q,k,code_sums);
    int * integer_output = nullptr;
    std::vector<int> integer_reference;
#ifdef BINARY_RSR
    const bool is_rsr = variant.rfind("rsr-",0)==0;
    RsrIndex rsr_first{},rsr_second{};
    float * values = nullptr;
    float * binary_output = nullptr;
    int * totals = nullptr;
    if(is_rsr) {
        const int block=std::stoi(variant.substr(4));
        rsr_first=rsr_index(memory,prefix,m,k,block);
        rsr_second=joint ? rsr_index(memory,prefix+".up",m,k,block) : rsr_first;
        values=memory.upload(std::vector<float>(k*n));
        totals=memory.upload(std::vector<int>(n));
        binary_output=memory.upload(std::vector<float>(rsr_first.padded*n*(joint?2:1)));
        integer_output=memory.upload(std::vector<int>(m*n*(joint?2:1)));
        rsr_codes<<<n,256,0,stream>>>(q,values,totals,k);
        auto unit_rows=memory.upload(std::vector<float>(m,1.0f));
        auto unit_slots=memory.upload(std::vector<float>(n,1.0f));
        auto zero_slots=memory.upload(std::vector<float>(n));
        auto raw=memory.upload(std::vector<float>(m*n*(joint?2:1)));
        w1ax_integer_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(first.weights,unit_rows,q,unit_slots,m,k,k/32,raw,nullptr,zero_slots);
        if(joint) w1ax_integer_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(second.weights,unit_rows,q,unit_slots,m,k,k/32,raw+m*n,nullptr,zero_slots);
        checked(cudaStreamSynchronize(stream));
        for(float value:download(raw,size_t(m)*n*(joint?2:1))) integer_reference.push_back(int(value));
    }
#endif
    auto dispatch = [&](const std::string & name) {
#ifdef BINARY_RSR
        if(is_rsr && name==variant) {
            launch_rsr(rsr_first,values,binary_output,integer_output,totals,first,b,sums,output,m,k,n,stream);
            if(joint) launch_rsr(rsr_second,values,binary_output+rsr_first.padded*n,integer_output+m*n,totals,second,b,sums,output+m*n,m,k,n,stream);
            return;
        }
#endif
        if (joint && name == "reference") {
            exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(first.weights,first.scales,first.midpoint,output,second.weights,second.scales,second.midpoint,output+m*n,q,b,sums,m,k);
        } else {
            launch_one(name,first,q,b,sums,output,m,k,n,stream,code_sums);
            if(joint) launch_one(name,second,q,b,sums,output+m*n,m,k,n,stream,code_sums);
        }
    };
    dispatch("reference");
    checked(cudaStreamSynchronize(stream));
    const size_t count = size_t(m)*n*(joint?2:1);
    const auto reference = download(output,count);
    for(float value : reference) if(!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    dispatch(variant);
    checked(cudaStreamEndCapture(stream,&graph));
    checked(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    measure([&] { checked(cudaGraphLaunch(executable,stream)); },stream,output,count,reference,warmup,repeats,trials,integer_output,integer_reference);
    checked(cudaGraphExecDestroy(executable));
    checked(cudaGraphDestroy(graph));
    checked(cudaStreamDestroy(stream));
}

static void baseline(const std::string & prefix, int m, int k, int n, int kind, int warmup, int repeats, int trials) {
    if(kind != GGML_TYPE_Q4_K && kind != GGML_TYPE_Q6_K) throw std::runtime_error("unsupported ordinary weight type");
    ggml_init_params parameters = {1024*1024,nullptr,true};
    ggml_context * context = ggml_init(parameters);
    if(!context) throw std::runtime_error("ggml init failed");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if(!backend) throw std::runtime_error("CUDA backend init failed");
    ggml_tensor * weights = ggml_new_tensor_2d(context,static_cast<ggml_type>(kind),k,m);
    ggml_tensor * input = ggml_new_tensor_2d(context,GGML_TYPE_F32,k,n);
    ggml_tensor * output = ggml_mul_mat(context,weights,input);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph,output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context,backend);
    if(!buffer) throw std::runtime_error("ggml allocation failed");
    const auto raw = read_values<uint8_t>(prefix+".quantized",ggml_nbytes(weights));
    const auto x = read_values<float>(prefix+".input",size_t(k)*n);
    ggml_backend_tensor_set(weights,raw.data(),0,raw.size());
    ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
    for(int i=0;i<4;++i) {
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("ggml execution failed");
    }
    auto * cuda = static_cast<ggml_backend_cuda_context *>(backend->context);
    std::vector<cudaGraphExec_t> executions;
    for(const auto & item : cuda->cuda_graphs)
        for(auto instance : item.second->instances) executions.push_back(instance);
    if(executions.size()!=1) throw std::runtime_error("expected one captured ordinary runtime graph");
    const auto reference = download(static_cast<float *>(output->data),size_t(m)*n);
    for(float value : reference) if(!std::isfinite(value)) throw std::runtime_error("nonfinite Baseline reference");
    measure([&] { checked(cudaGraphLaunch(executions[0],cuda->stream())); },cuda->stream(),static_cast<float *>(output->data),size_t(m)*n,reference,warmup,repeats,trials);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(context);
}

int main(int argc,char ** argv) {
    try {
        if(argc>1 && std::string(argv[1])=="vocabulary") return vocabulary_main(argc,argv);
        if(argc!=14) throw std::runtime_error("prefix variant m k slots bits mode clip warmup repeats trials joint required");
        const std::string prefix=argv[1],variant=argv[2];
        const int m=std::stoi(argv[3]),k=std::stoi(argv[4]),n=std::stoi(argv[5]);
        const int bits=std::stoi(argv[6]),mode=std::stoi(argv[7]);
        const float clip=std::stof(argv[8]);
        const int warmup=std::stoi(argv[9]),repeats=std::stoi(argv[10]),trials=std::stoi(argv[11]);
        const bool joint=std::stoi(argv[12])!=0;
        if(std::string(argv[13])!="v1" || m<=0 || k<=0 || k%32 || m%16 || (n!=1&&n!=2&&n!=4&&n!=8&&n!=16) || warmup<1 || repeats<1 || trials<1)
            throw std::runtime_error("invalid benchmark bounds");
        cudaDeviceProp properties;
        checked(cudaGetDeviceProperties(&properties,0));
        printf("device,%s,SMs=%d,L2_bytes=%d\n",properties.name,properties.multiProcessorCount,properties.l2CacheSize);
        if(variant=="baseline") baseline(prefix,m,k,n,bits,warmup,repeats,trials);
        else binary(prefix,variant,m,k,n,bits,mode,clip,warmup,repeats,trials,joint);
        return 0;
    } catch(const std::exception & error) {
        fprintf(stderr,"%s\n",error.what());
        return 2;
    }
}
