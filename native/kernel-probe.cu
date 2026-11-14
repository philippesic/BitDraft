#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "runtime-kernels.cuh"

static void checked(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(status));
        exit(1);
    }
}

template<class T> static T * uploaded(const std::vector<T> & values) {
    T * pointer;
    checked(cudaMalloc(&pointer, values.size() * sizeof(T)));
    checked(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    return pointer;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const int m = 9216, k = 2560, n = 4;
    std::vector<uint32_t> weights(m * k / 32);
    FILE * input = fopen(argv[1], "rb");
    if (!input) return 2;
    if (fread(weights.data(), sizeof(uint32_t), weights.size(), input) != weights.size()) return 2;
    fclose(input);
    auto w = uploaded(weights);
    auto a = uploaded(std::vector<float>(m, 0.01f));
    auto b = uploaded(std::vector<float>(n, 0.1f));
    auto s = uploaded(std::vector<float>(n, 0.0f));
    auto y = uploaded(std::vector<float>(m * n));
    std::vector<int8_t> codes(k * n);
    for (int j = 0; j < k * n; ++j) codes[j] = (j * 17 % 15) - 7;
    auto q = uploaded(codes);
    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    for (int j = 0; j < 100; ++j)
        w1ax_integer_dot<<<dim3(m / 4, n), dim3(32, 4)>>>(w,a,q,b,m,k,k/32,y,nullptr,s);
    checked(cudaEventRecord(start));
    for (int j = 0; j < 1000; ++j)
        w1ax_integer_dot<<<dim3(m / 4, n), dim3(32, 4)>>>(w,a,q,b,m,k,k/32,y,nullptr,s);
    checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop));
    checked(cudaGetLastError());
    float milliseconds;
    checked(cudaEventElapsedTime(&milliseconds,start,stop));
    printf("gate,4,%.9f ms\n", milliseconds / 1000);
    for (void * pointer : {static_cast<void *>(w),static_cast<void *>(a),static_cast<void *>(b),static_cast<void *>(s),static_cast<void *>(y),static_cast<void *>(q)}) checked(cudaFree(pointer));
    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
}
