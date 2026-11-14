static __global__ void mma_broadcast_dot(
 const uint32_t*w,const float*a,const int8_t*q,const float*b,int64_t m,int64_t k,int64_t words,float*y,const float*mid,const float*s,int64_t n) {
    const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
    const int rowbase = (blockIdx.x * 4 + threadIdx.y) * 16;
    int c[4] = {};
    for (int j = 0; j < k; j += 32) {
        uint32_t low = 0, high = 0;
        if (t == 0 && rowbase + g < m) low = w[(rowbase + g) * words + j / 32];
        if (t == 0 && rowbase + g + 8 < m) high = w[(rowbase + g + 8) * words + j / 32];
        low = __shfl_sync(0xffffffff, low, g * 4);
        high = __shfl_sync(0xffffffff, high, g * 4);
        const uint32_t a0 = exec_sign4(low >> (t * 4));
        const uint32_t a1 = exec_sign4(high >> (t * 4));
        const uint32_t a2 = exec_sign4(low >> (t * 4 + 16));
        const uint32_t a3 = exec_sign4(high >> (t * 4 + 16));
        const uint32_t b0 = g < n ? *(const uint32_t *)(q + g * k + j + t * 4) : 0;
        const uint32_t b1 = g < n ? *(const uint32_t *)(q + g * k + j + t * 4 + 16) : 0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    for (int z = 0; z < 4; ++z) {
        const int row = rowbase + g + (z / 2) * 8, col = t * 2 + z % 2;
        if (row < m && col < n) y[col * m + row] = exec_affine(c[z], a[row], b[col], mid[row], s[col]);
    }
}

static __global__ void mma_warp_dot(
 const uint32_t*w,const float*a,const int8_t*q,const float*b,int64_t m,int64_t k,int64_t words,float*y,const float*mid,const float*s,int64_t n) {
    const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
    const int rowbase = blockIdx.x * 16;
    int c[4] = {};
    for (int j = 0; j < k; j += 32) {
        const uint32_t low = w[(rowbase + g) * words + j / 32];
        const uint32_t high = w[(rowbase + g + 8) * words + j / 32];
        const uint32_t a0 = exec_sign4(low >> (t * 4));
        const uint32_t a1 = exec_sign4(high >> (t * 4));
        const uint32_t a2 = exec_sign4(low >> (t * 4 + 16));
        const uint32_t a3 = exec_sign4(high >> (t * 4 + 16));
        const uint32_t b0 = g < n ? *(const uint32_t *)(q + g * k + j + t * 4) : 0;
        const uint32_t b1 = g < n ? *(const uint32_t *)(q + g * k + j + t * 4 + 16) : 0;
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    for (int z = 0; z < 4; ++z) {
        const int row = rowbase + g + (z / 2) * 8, col = t * 2 + z % 2;
        if (col < n) y[col * m + row] = exec_affine(c[z], a[row], b[col], mid[row], s[col]);
    }
}
