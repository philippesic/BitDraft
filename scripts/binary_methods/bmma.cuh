static __global__ void w1a1_binary_mma(
        const uint32_t * weights, const float * weight_scales,
        const uint32_t * activations, const float * activation_scales,
        int64_t m, int64_t n, int64_t k, int64_t words, float * output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int in_group = lane & 3;
    const int64_t row = (int64_t) blockIdx.x*8 + group;
    const int64_t token_tiles = (n - 1)/8 + 1;
    const int64_t k_tiles = (words - 1)/4 + 1;
    const uint32_t tail_mask = (k & 31) == 0 ? UINT32_MAX : (uint32_t(1) << (k & 31)) - 1;

    for (int64_t token_tile = blockIdx.y; token_tile < token_tiles; token_tile += gridDim.y) {
        const int64_t token_for_b = token_tile*8 + group;
        int mismatch0 = 0;
        int mismatch1 = 0;
        for (int64_t tile = 0; tile < k_tiles; ++tile) {
            const int64_t word = tile*4 + in_group;
            uint32_t a_word = 0;
            uint32_t b_word = 0;
            if (word < words) {
                const uint32_t mask = word == words - 1 ? tail_mask : UINT32_MAX;
                if (row < m) a_word = weights[row*words + word] & mask;
                if (token_for_b < n) b_word = activations[token_for_b*words + word] & mask;
            }
            asm volatile(
                "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
                "{%0, %1}, {%2}, {%3}, {%0, %1};"
                : "+r"(mismatch0), "+r"(mismatch1)
                : "r"(a_word), "r"(b_word));
        }

        const int64_t token0 = token_tile*8 + 2*in_group;
        if (row < m && token0 < n) {
            const float dot = (float) (k - 2*(int64_t) mismatch0);
            const float weighted = dot*weight_scales[row];
            output[token0*m + row] = weighted*activation_scales[token0];
        }
        if (row < m && token0 + 1 < n) {
            const float dot = (float) (k - 2*(int64_t) mismatch1);
            const float weighted = dot*weight_scales[row];
            output[(token0 + 1)*m + row] = weighted*activation_scales[token0 + 1];
        }
    }
#else
    asm volatile("trap;");
#endif
}