#include <cuda_runtime.h>
#include <stdint.h>
#include "bmma.cuh"
extern "C" void probe(const uint32_t*w,const float*a,const uint32_t*x,const float*b,float*y){w1a1_binary_mma<<<1,32>>>(w,a,x,b,8,8,128,4,y);}
