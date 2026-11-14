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
                    size_t count, const std::vector<float> & reference, int warmup, int repeats, int trials) {
    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    bool identical = true;
    for (int j = 0; j < warmup; ++j) launch();
    checked(cudaStreamSynchronize(stream));
    printf("warmup,%d,per_trial,%d\n",warmup,std::min(warmup,200));
    for (int trial = -1; trial < trials; ++trial) {
        checked(cudaMemsetAsync(const_cast<float *>(output), 0xff, count * sizeof(float), stream));
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
        if (trial >= 0) printf("sample,%d,%.12g,%d\n", trial, double(milliseconds) * 1000 / repeats, identical);
    }
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

static void launch_one(const std::string & variant, Matrix weight, const int8_t * q,
                       const float * b, const float * sums, float * output, int m, int k, int n,
                       cudaStream_t stream) {
    if (variant == "reference") {
        w1ax_integer_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums);
    } else if (variant == "cross") {
        if (n == 2) exec_cross_dot<2><<<dim3((m+3)/4),dim3(32,4),0,stream>>>(weight.weights,weight.scales,q,b,m,k,k/32,output,weight.midpoint,sums,n);
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
    auto dispatch = [&](const std::string & name) {
        if (joint && name == "reference") {
            exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,stream>>>(first.weights,first.scales,first.midpoint,output,second.weights,second.scales,second.midpoint,output+m*n,q,b,sums,m,k);
        } else {
            launch_one(name,first,q,b,sums,output,m,k,n,stream);
            if(joint) launch_one(name,second,q,b,sums,output+m*n,m,k,n,stream);
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
    measure([&] { checked(cudaGraphLaunch(executable,stream)); },stream,output,count,reference,warmup,repeats,trials);
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
        if(argc!=14) throw std::runtime_error("prefix variant m k slots bits mode clip warmup repeats trials joint required");
        const std::string prefix=argv[1],variant=argv[2];
        const int m=std::stoi(argv[3]),k=std::stoi(argv[4]),n=std::stoi(argv[5]);
        const int bits=std::stoi(argv[6]),mode=std::stoi(argv[7]);
        const float clip=std::stof(argv[8]);
        const int warmup=std::stoi(argv[9]),repeats=std::stoi(argv[10]),trials=std::stoi(argv[11]);
        const bool joint=std::stoi(argv[12])!=0;
        if(std::string(argv[13])!="v1" || m<=0 || k<=0 || k%32 || m%16 || (n!=2&&n!=4&&n!=8&&n!=16) || warmup<1 || repeats<1 || trials<1)
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
