__device__ __forceinline__ uint32_t spread_u8(uint32_t nibble) {
  return ((nibble & 15u) * 0x00204081u) & 0x01010101u;
}

__device__ __forceinline__ uint32_t spread_s8(uint32_t nibble) {
  return ((spread_u8(nibble) ^ 0x01010101u) * 0xfeu) | 0x01010101u;
}


template <int SPLIT, bool UNSIGNED>
__global__ void __launch_bounds__(SPLIT * 32)
    recovered_imma(const uint32_t *weights, const float *weight_scales,
                  const int8_t *codes, const int32_t *code_sums,
                  const float *act_scales, int64_t m, int64_t k, int64_t words,
                  float *output, int32_t *raw_dots, const float *midpoints, const float *sums, int n) {
  __shared__ int32_t partial[SPLIT][4][32];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = lane >> 2, in_group = lane & 3;
  const int64_t row0 = (int64_t)blockIdx.x * 16 + group, row1 = row0 + 8;
  const bool token_ok = group < n;
  const int8_t *token_codes = codes + (int64_t)(token_ok ? group : 0) * k;
  int32_t acc[4] = {};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  for (int64_t base = (int64_t)warp * 4; base < words; base += SPLIT * 4) {
    const uint4 w0 = row0 < m ? *(const uint4 *)(weights + row0 * words + base)
                              : make_uint4(0, 0, 0, 0);
    const uint4 w1 = row1 < m ? *(const uint4 *)(weights + row1 * words + base)
                              : make_uint4(0, 0, 0, 0);
    const uint32_t r0[4] = {w0.x, w0.y, w0.z, w0.w};
    const uint32_t r1[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t kk = (base + j) * 32;
      const int lo = in_group * 4, hi = 16 + in_group * 4;
      const uint32_t a0 = UNSIGNED ? spread_u8(r0[j] >> lo) : spread_s8(r0[j] >> lo);
      const uint32_t a1 = UNSIGNED ? spread_u8(r1[j] >> lo) : spread_s8(r1[j] >> lo);
      const uint32_t a2 = UNSIGNED ? spread_u8(r0[j] >> hi) : spread_s8(r0[j] >> hi);
      const uint32_t a3 = UNSIGNED ? spread_u8(r1[j] >> hi) : spread_s8(r1[j] >> hi);
      const uint32_t b0 = token_ok ? *(const uint32_t *)(token_codes + kk + lo) : 0u;
      const uint32_t b1 = token_ok ? *(const uint32_t *)(token_codes + kk + hi) : 0u;
      if constexpr (UNSIGNED) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
#endif
#pragma unroll
  for (int i = 0; i < 4; ++i)
    partial[warp][i][lane] = acc[i];
  __syncthreads();
  if (warp != 0)
    return;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int32_t s = 0;
#pragma unroll
    for (int w = 0; w < SPLIT; ++w)
      s += partial[w][i][lane];
    const int64_t row = i < 2 ? row0 : row1;
    const int token = in_group * 2 + (i & 1);
    if (row < m && token < n) {
      const int32_t dot = UNSIGNED ? 2 * s - code_sums[token] : s;
      if (raw_dots)
        raw_dots[token * m + row] = dot;
      output[token * m + row] = exec_affine(dot,weight_scales[row],act_scales[token],midpoints ? midpoints[row] : 0.0f,sums[token]);
    }
  }
}


static __global__ void w1a8_sum_codes(const int8_t * codes, int64_t k, int32_t * sums) {
    __shared__ int partial[256];
    const int token=blockIdx.x;
    int total=0;
    for(int64_t i=threadIdx.x;i<k;i+=256) total+=(int)codes[token*k+i];
    partial[threadIdx.x]=total;__syncthreads();
    for(int stride=128;stride>0;stride/=2){if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)sums[token]=partial[0];
}
