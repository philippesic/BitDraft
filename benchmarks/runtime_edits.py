SOURCE_EDITS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "kernel_recovered_imma_v1": (
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r"""static __global__ void exec_joint_dot(""",
            r"""__device__ __forceinline__ uint32_t spread_u8(uint32_t nibble) {
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



static __global__ void exec_joint_dot(""",
        ),
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r"""    w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>(""",
            r"""    if ((n == 2 || n == 4 || n == 8) && m == 2560 && k == 9216 && dst->src[4]) {
        recovered_imma<4,false><<<(m+15)/16,128,0,ctx.stream()>>>((const uint32_t*)dst->src[0]->data,(const float*)dst->src[1]->data,(const int8_t*)packed,nullptr,(const float*)(packed+layout.scale_offset),m,k,k/32,(float*)dst->data,nullptr,(const float*)dst->src[4]->data,(const float*)(packed+layout.scale_offset+n),n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>(""",
        ),
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r""" exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,ctx.stream()>>>((const uint32_t*)a->src[0]->data,(const float*)a->src[1]->data,a->src[4]?(const float*)a->src[4]->data:nullptr,(float*)a->data,(const uint32_t*)b->src[0]->data,(const float*)b->src[1]->data,b->src[4]?(const float*)b->src[4]->data:nullptr,(float*)b->data,(const int8_t*)p,(const float*)(p+layout.scale_offset),(const float*)(p+layout.scale_offset+n),m,k);CUDA_CHECK(cudaGetLastError());""",
            r""" if ((n == 2 || n == 4 || n == 8) && m == 9216 && k == 2560 && a->src[4] && b->src[4]) {
  for (ggml_tensor * dst : {a,b}) {
   recovered_imma<4,false><<<(m+15)/16,128,0,ctx.stream()>>>((const uint32_t*)dst->src[0]->data,(const float*)dst->src[1]->data,(const int8_t*)p,nullptr,(const float*)(p+layout.scale_offset),m,k,k/32,(float*)dst->data,nullptr,(const float*)dst->src[4]->data,(const float*)(p+layout.scale_offset+n),n);
  }
  CUDA_CHECK(cudaGetLastError());
  return;
 }
 exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,ctx.stream()>>>((const uint32_t*)a->src[0]->data,(const float*)a->src[1]->data,a->src[4]?(const float*)a->src[4]->data:nullptr,(float*)a->data,(const uint32_t*)b->src[0]->data,(const float*)b->src[1]->data,b->src[4]?(const float*)b->src[4]->data:nullptr,(float*)b->data,(const int8_t*)p,(const float*)(p+layout.scale_offset),(const float*)(p+layout.scale_offset+n),m,k);CUDA_CHECK(cudaGetLastError());""",
        ),
    ),
    "kernel_gate_mma_warp_v1": (
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r"""static __global__ void exec_joint_dot(""",
            r"""static __global__ void mma_warp_dot(
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

static __global__ void exec_joint_dot(""",
        ),
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r""" exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,ctx.stream()>>>((const uint32_t*)a->src[0]->data,(const float*)a->src[1]->data,a->src[4]?(const float*)a->src[4]->data:nullptr,(float*)a->data,(const uint32_t*)b->src[0]->data,(const float*)b->src[1]->data,b->src[4]?(const float*)b->src[4]->data:nullptr,(float*)b->data,(const int8_t*)p,(const float*)(p+layout.scale_offset),(const float*)(p+layout.scale_offset+n),m,k);CUDA_CHECK(cudaGetLastError());""",
            r""" if ((n == 4 || n == 8 || n == 16) && m == 9216 && k == 2560 && a->src[4] && b->src[4]) {
  for (ggml_tensor * dst : {a,b}) {
   for (int offset = 0; offset < n; offset += 8) {
    mma_warp_dot<<<dim3(m/16),dim3(32),0,ctx.stream()>>>((const uint32_t*)dst->src[0]->data,(const float*)dst->src[1]->data,(const int8_t*)p+offset*k,(const float*)(p+layout.scale_offset)+offset,m,k,k/32,(float*)dst->data+offset*m,(const float*)dst->src[4]->data,(const float*)(p+layout.scale_offset+n)+offset,std::min(8,n-offset));
   }
  }
  CUDA_CHECK(cudaGetLastError());
  return;
 }
 exec_joint_dot<<<dim3((m+3)/4,n),dim3(32,4),0,ctx.stream()>>>((const uint32_t*)a->src[0]->data,(const float*)a->src[1]->data,a->src[4]?(const float*)a->src[4]->data:nullptr,(float*)a->data,(const uint32_t*)b->src[0]->data,(const float*)b->src[1]->data,b->src[4]?(const float*)b->src[4]->data:nullptr,(float*)b->data,(const int8_t*)p,(const float*)(p+layout.scale_offset),(const float*)(p+layout.scale_offset+n),m,k);CUDA_CHECK(cudaGetLastError());""",
        ),
    ),
    "checkpoint_warm_buffers_v1": (
        (
            "common/common.cpp",
            r"""#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
""",
            r"""#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
""",
        ),
        (
            "common/common.cpp",
            r"""#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
""",
            r"""#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
""",
        ),
        (
            "common/common.cpp",
            r"""
    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

""",
            r"""
    return true;
}

struct common_state_pool {
    std::mutex mtx;
    std::condition_variable cv;
    std::multimap<size_t, void *> idle;
    std::unordered_map<void *, size_t> caps;
    size_t idle_bytes = 0;
    size_t recent[4] = { 0, 0, 0, 0 };
    size_t n_recent = 0;
    bool   worker = false;
};

static common_state_pool & common_state_pool_get() {
    static common_state_pool * pool = new common_state_pool();
    return *pool;
}

static constexpr size_t COMMON_STATE_POOL_MIN  = 1u << 20;
static constexpr size_t COMMON_STATE_POOL_ALGN = 2u << 20;
static constexpr size_t COMMON_STATE_IDLE_MAX  = (size_t) 2 << 30;
static constexpr size_t COMMON_STATE_RESERVE   = 4;

static void * common_state_map(size_t cap) {
#if defined(__linux__)
    void * ptr = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    return ptr;
#else
    return ::operator new(cap);
#endif
}

static void common_state_unmap(void * ptr, size_t cap) {
#if defined(__linux__)
    munmap(ptr, cap);
#else
    GGML_UNUSED(cap);
    ::operator delete(ptr);
#endif
}

static size_t common_state_reserve_cap(const common_state_pool & pool) {
    size_t cap = 0;
    for (size_t i = 0; i < std::min<size_t>(pool.n_recent, 4); ++i) {
        cap = std::max(cap, pool.recent[i]);
    }
    return cap;
}

static size_t common_state_reserve_count(const common_state_pool & pool, size_t cap) {
    size_t count = 0;
    for (auto it = pool.idle.lower_bound(cap); it != pool.idle.end(); ++it) {
        count++;
    }
    return count;
}

static void common_state_reserve_loop() {
    auto & pool = common_state_pool_get();
    std::unique_lock<std::mutex> lock(pool.mtx);
    while (true) {
        pool.cv.wait(lock, [&] {
            const size_t cap = common_state_reserve_cap(pool);
            return cap > 0 && common_state_reserve_count(pool, cap) < COMMON_STATE_RESERVE &&
                   pool.idle_bytes + cap <= COMMON_STATE_IDLE_MAX;
        });
        const size_t cap = common_state_reserve_cap(pool);
        lock.unlock();
        void * ptr = nullptr;
        try {
            ptr = common_state_map(cap);
            std::memset(ptr, 0, cap);
        } catch (const std::bad_alloc &) {
            ptr = nullptr;
        }
        lock.lock();
        if (ptr == nullptr) {
            pool.n_recent = 0;
            continue;
        }
        pool.caps[ptr] = cap;
        pool.idle.emplace(cap, ptr);
        pool.idle_bytes += cap;
    }
}

void * common_state_alloc(size_t size, bool pooled) {
    if (!pooled || size < COMMON_STATE_POOL_MIN) {
        return ::operator new(size);
    }
    auto & pool = common_state_pool_get();
    std::lock_guard<std::mutex> lock(pool.mtx);
    const size_t cap = GGML_PAD(size + size/8, COMMON_STATE_POOL_ALGN);
    pool.recent[pool.n_recent++ % 4] = cap;
    if (!pool.worker) {
        pool.worker = true;
        std::thread(common_state_reserve_loop).detach();
    }
    void * ptr = nullptr;
    auto it = pool.idle.lower_bound(size);
    if (it != pool.idle.end() && it->first <= 2*cap) {
        ptr = it->second;
        pool.idle_bytes -= it->first;
        pool.idle.erase(it);
    } else {
        ptr = common_state_map(cap);
        pool.caps[ptr] = cap;
    }
    pool.cv.notify_one();
    return ptr;
}

void common_state_free(void * ptr, size_t size) {
    if (ptr == nullptr) {
        return;
    }
    if (size >= COMMON_STATE_POOL_MIN) {
        auto & pool = common_state_pool_get();
        std::lock_guard<std::mutex> lock(pool.mtx);
        auto it = pool.caps.find(ptr);
        if (it != pool.caps.end()) {
            const size_t cap = it->second;
            if (pool.idle_bytes + cap <= COMMON_STATE_IDLE_MAX) {
                pool.idle.emplace(cap, ptr);
                pool.idle_bytes += cap;
            } else {
                pool.caps.erase(it);
                common_state_unmap(ptr, cap);
            }
            return;
        }
    }
    ::operator delete(ptr);
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

""",
        ),
        (
            "common/common.h",
            r"""#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
""",
            r"""#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <new>
#include <type_traits>
#include <utility>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
""",
        ),
        (
            "common/common.h",
            r"""//
// prompt utils
//

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
""",
            r"""//
// prompt utils
//

void * common_state_alloc(size_t size, bool pooled);
void   common_state_free(void * ptr, size_t size);

template <typename T>
struct common_state_allocator {
    using value_type = T;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    bool pooled = true;

    common_state_allocator() = default;
    explicit common_state_allocator(bool pooled) : pooled(pooled) {}
    template <typename U>
    common_state_allocator(const common_state_allocator<U> & other) : pooled(other.pooled) {}

    T * allocate(size_t n) { return static_cast<T *>(common_state_alloc(n * sizeof(T), pooled)); }
    void deallocate(T * ptr, size_t n) { common_state_free(ptr, n * sizeof(T)); }

    template <typename U>
    void construct(U * ptr) noexcept { ::new (static_cast<void *>(ptr)) U; }
    template <typename U, typename... Args>
    void construct(U * ptr, Args &&... args) { ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...); }

    common_state_allocator select_on_container_copy_construction() const { return common_state_allocator(false); }

    template <typename U>
    bool operator==(const common_state_allocator<U> &) const { return true; }
    template <typename U>
    bool operator!=(const common_state_allocator<U> &) const { return false; }
};

using common_state_bytes = std::vector<uint8_t, common_state_allocator<uint8_t>>;

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
""",
        ),
        (
            "common/common.h",
            r"""
    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;
""",
            r"""
    llama_pos pos_min;
    llama_pos pos_max;

    common_state_bytes data_tgt;
    common_state_bytes data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
""",
            r"""        n_read += size;
        return true;
    }

    template <typename Bytes>
    static bool ckpt_read_buf(std::ifstream & ifs, Bytes & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
""",
            r"""        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Bytes>
    static void ckpt_write_buf(std::ofstream & ofs, const Bytes & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
""",
        ),
    ),
    "device_feature_handoff_v1": (
        (
            "common/speculative.cpp",
            r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
""",
            r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
""",
        ),
        (
            "common/speculative.cpp",
            r"""    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;
""",
            r"""    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    bool                features_dev = false;
    llama_batch_ext_ptr batch_dev;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;
""",
        ),
        (
            "common/speculative.cpp",
            r"""        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }
""",
            r"""        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        bool ids_ascending = true;
        for (uint32_t k = 1; k < target_layer_ids_n; ++k) {
            ids_ascending = ids_ascending && target_layer_ids[k] > target_layer_ids[k - 1];
        }
        features_dev = ids_ascending && !is_lilicorr && llama_model_dflash_device_features(model_dft) &&
                       std::getenv("LLAMA_SPEC_HOST_FEATURES") == nullptr;
        if (features_dev) {
            llama_set_embeddings_layer_inp_device(ctx_tgt, true);
            batch_dev.reset(llama_batch_ext_init(ctx_dft));
        }
        LOG_INF("%s: - target features handed over on the %s\n", __func__, features_dev ? "device" : "host");

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }
""",
        ),
        (
            "common/speculative.cpp",
            r"""
            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
""",
            r"""
            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                if (features_dev) {
                    const ggml_tensor * src = llama_get_embeddings_layer_inp_device(ctx_tgt);
                    if (!src) {
                        GGML_ABORT("DFlash: target layer inputs not extracted on the device.");
                    }
                    llama_batch_ext * b = batch_dev.get();
                    llama_batch_ext_clear(b);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                        const llama_pos pos_arr[4] = { p, p, p, 0 };
                        llama_pos pos_row[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < batch_inject.n_pos; ++j) {
                            pos_row[j] = pos_arr[j];
                        }
                        const int32_t idx = llama_batch_ext_add_embd_device(b, seq_id);
                        if (idx < 0 || !llama_batch_ext_set_pos(b, idx, pos_row)) {
                            GGML_ABORT("DFlash: failed to add device feature row %d", i);
                        }
                    }
                    llama_batch_ext_set_embd_device(b, src, i_batch_beg[seq_id] + offset);
                    const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, b);
                    if (rc != 0) {
                        LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                    llama_embeddings_layer_inp_consumed(ctx_tgt, ctx_dft);
                    continue;
                }

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    }

    const bool mixed = n_tok_only > 0 && n_embd_only > 0;

    if (mixed && !allow_mixed) {
        LLAMA_LOG_ERROR("%s: this model or context does not support batches mixing token and embedding entries\n", __func__);
        return false;
    }
""",
            r"""    }

    const bool mixed = n_tok_only > 0 && n_embd_only > 0;

    if (batch_inp.embd_device && (mixed || n_both > 0 || batch_inp.embd_dev == nullptr)) {
        LLAMA_LOG_ERROR("%s: device embedding rows need an embedding-only batch with a device source\n", __func__);
        return false;
    }

    if (mixed && !allow_mixed) {
        LLAMA_LOG_ERROR("%s: this model or context does not support batches mixing token and embedding entries\n", __func__);
        return false;
    }
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""                const float * src = batch_inp.embd.data() + batch_inp.tokens[i].embd_off;
                std::copy(src, src + n_embd, embd_vec.data() + (size_t) i*n_embd);
            }
        }
    } else if (has_embd) {
        embd_vec = batch_inp.embd;
    }

    if (has_state) {
        state_vec = batch_inp.state;
    }

""",
            r"""                const float * src = batch_inp.embd.data() + batch_inp.tokens[i].embd_off;
                std::copy(src, src + n_embd, embd_vec.data() + (size_t) i*n_embd);
            }
        }
    } else if (has_embd && !batch_inp.embd_device) {
        embd_vec = batch_inp.embd;
    }

    embd_dev      = batch_inp.embd_device ? batch_inp.embd_dev : nullptr;
    embd_dev_row0 = batch_inp.embd_device ? batch_inp.embd_dev_row0 : 0;

    if (has_state) {
        state_vec = batch_inp.state;
    }

""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    //

    batch.n_tokens = n_tok;
    batch.token    = has_token ? token_vec.data() : nullptr;
    batch.embd     = has_embd  ? embd_vec.data()  : nullptr;
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = output.data();
""",
            r"""    //

    batch.n_tokens = n_tok;
    batch.token    = has_token ? token_vec.data() : nullptr;
    batch.embd     = has_embd  ? (embd_dev ? llama_ubatch_embd_device_marker() : embd_vec.data()) : nullptr;
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = output.data();
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    batch = {};

    token_vec   .clear();
    embd_vec    .clear();
    is_embd_vec .clear();
    state_vec   .clear();
    seq_id_data .clear();
    pos         .clear();
""",
            r"""    batch = {};

    token_vec   .clear();
    embd_vec    .clear();
    embd_dev      = nullptr;
    embd_dev_row0 = 0;
    is_embd_vec .clear();
    state_vec   .clear();
    seq_id_data .clear();
    pos         .clear();
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    const bool use_token = batch.token && !(mixed_batch && n_embd_rows == n_tokens);
    const bool use_embd  = batch.embd  && !(mixed_batch && n_embd_rows == 0);
    const bool has_state = !state_vec.empty();

    const int64_t n_embd_all  = use_embd  ? (int64_t) n_tokens*n_embd       : 0;
    const int64_t n_state_all = has_state ? (int64_t) n_tokens*n_embd_state : 0;
    const int64_t n_pos_all   =             (int64_t) n_tokens*n_pos_per_embd;

    udata->token     .resize(n_tokens);
""",
            r"""    const bool use_token = batch.token && !(mixed_batch && n_embd_rows == n_tokens);
    const bool use_embd  = batch.embd  && !(mixed_batch && n_embd_rows == 0);
    const bool has_state = !state_vec.empty();

    const bool    on_device   = use_embd && embd_dev != nullptr;
    const int64_t n_embd_all  = use_embd && !on_device ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_state_all = has_state ? (int64_t) n_tokens*n_embd_state : 0;
    const int64_t n_pos_all   =             (int64_t) n_tokens*n_pos_per_embd;

    udata->token     .resize(n_tokens);
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        if (use_token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (use_embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        if (mixed) {
""",
            r"""        if (use_token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (on_device) {
            GGML_ASSERT(idxs[i] == idxs[0] + (int32_t) i && "device embedding rows must stay contiguous");
        } else if (use_embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        if (mixed) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ use_token ? udata->token.data() : nullptr,
        /*.embd         =*/ use_embd  ? udata->embd.data()  : nullptr,
        /*.embd_state   =*/ has_state ? udata->embd_state.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
""",
            r"""        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ use_token ? udata->token.data() : nullptr,
        /*.embd         =*/ use_embd  ? (on_device ? llama_ubatch_embd_device_marker() : udata->embd.data()) : nullptr,
        /*.embd_state   =*/ has_state ? udata->embd_state.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        /*.decision_order =*/ udata->decision_order.empty() ? nullptr : udata->decision_order.data(),
        /*.data         =*/ std::move(udata),
    };

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
""",
            r"""        /*.decision_order =*/ udata->decision_order.empty() ? nullptr : udata->decision_order.data(),
        /*.data         =*/ std::move(udata),
    };

    if (on_device) {
        res.embd_dev      = embd_dev;
        res.embd_dev_row0 = embd_dev_row0 + idxs[0];
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    tokens.clear();
    embd  .clear();
    state .clear();
    n_embd = 0;
}

int32_t llama_batch_ext::add_token(llama_seq_id seq_id) {
    if (tokens.size() >= n_tokens_max) {
""",
            r"""    tokens.clear();
    embd  .clear();
    state .clear();
    n_embd = 0;
    embd_device   = false;
    embd_dev      = nullptr;
    embd_dev_row0 = 0;
}

float * llama_ubatch_embd_device_marker() {
    static float marker = 0.0f;
    return &marker;
}

bool llama_batch_ext::set_token_embd_device(int32_t idx) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || n_embd_inp_enc == 0 || !embd.empty()) {
        return false;
    }
    if (n_embd == 0) {
        n_embd = n_embd_inp_enc;
    } else if (n_embd != n_embd_inp_enc) {
        return false;
    }

    token & t = tokens[idx];

    if (t.has_embd) {
        return false;
    }

    t.has_embd  = true;
    t.embd_off  = 0;
    embd_device = true;

    return true;
}

int32_t llama_batch_ext::add_token(llama_seq_id seq_id) {
    if (tokens.size() >= n_tokens_max) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    if (!embd_in.data) {
        return false;
    }

    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
""",
            r"""bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    if (!embd_in.data || embd_device) {
        return false;
    }

    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    }
    return idx;
}

bool llama_batch_ext_add_seq(llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id) {
    return batch->add_seq(idx, seq_id);
}

""",
            r"""    }
    return idx;
}

int32_t llama_batch_ext_add_embd_device(llama_batch_ext * batch, llama_seq_id seq_id) {
    int32_t idx = batch->add_token(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!batch->set_token_embd_device(idx)) {
        return -2;
    }
    return idx;
}

void llama_batch_ext_set_embd_device(llama_batch_ext * batch, const struct ggml_tensor * src, int64_t row0) {
    batch->embd_dev      = src;
    batch->embd_dev_row0 = row0;
}

bool llama_batch_ext_add_seq(llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id) {
    return batch->add_seq(idx, seq_id);
}

""",
        ),
        (
            "src/llama-batch.h",
            r"""    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    std::shared_ptr<data_t> data;
};

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) GGML_ASSERT(!(ubatch).is_mixed() && "mixed token/embd ubatch is not supported here")

struct llama_hparams;
""",
            r"""    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    std::shared_ptr<data_t> data;

    const struct ggml_tensor * embd_dev = nullptr;
    int64_t embd_dev_row0 = 0;
};

float * llama_ubatch_embd_device_marker();

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) GGML_ASSERT(!(ubatch).is_mixed() && "mixed token/embd ubatch is not supported here")

struct llama_hparams;
""",
        ),
        (
            "src/llama-batch.h",
            r"""    std::vector<token> tokens;
    std::vector<float> embd;
    std::vector<float> state;

    llama_batch_ext(llama_context * ctx);

    // build without a llama_context, used by tests
    llama_batch_ext(
""",
            r"""    std::vector<token> tokens;
    std::vector<float> embd;
    std::vector<float> state;

    bool embd_device = false;
    const struct ggml_tensor * embd_dev = nullptr;
    int64_t embd_dev_row0 = 0;

    llama_batch_ext(llama_context * ctx);

    // build without a llama_context, used by tests
    llama_batch_ext(
""",
        ),
        (
            "src/llama-batch.h",
            r"""
    bool add_seq(int32_t idx, llama_seq_id seq_id);
    bool set_token_id(int32_t idx, llama_token id);
    bool set_token_embd(int32_t idx, llama_embd embd_in);
    bool set_token_state(int32_t idx, llama_embd state_in);
    bool set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool set_output(int32_t idx, bool output_last);
    bool set_decision_order(int32_t idx, int32_t order);
""",
            r"""
    bool add_seq(int32_t idx, llama_seq_id seq_id);
    bool set_token_id(int32_t idx, llama_token id);
    bool set_token_embd(int32_t idx, llama_embd embd_in);
    bool set_token_embd_device(int32_t idx);
    bool set_token_state(int32_t idx, llama_embd state_in);
    bool set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool set_output(int32_t idx, bool output_last);
    bool set_decision_order(int32_t idx, int32_t order);
""",
        ),
        (
            "src/llama-batch.h",
            r"""    uint32_t n_outputs;

    std::vector<llama_token>    token_vec;    // owned token IDs built from llama_batch_ext
    std::vector<float>          embd_vec;     // owned embeddings built from llama_batch_ext
    std::vector<int8_t>         is_embd_vec;  // mixed batch only (= 1 if embd, 0 if text token)
    std::vector<float>          state_vec;    // owned state embeddings built from llama_batch_ext, llama_batch has no slot for them
    std::vector<llama_seq_id>   seq_id_data;  // flat storage for seq_id pointers below

""",
            r"""    uint32_t n_outputs;

    std::vector<llama_token>    token_vec;    // owned token IDs built from llama_batch_ext
    std::vector<float>          embd_vec;     // owned embeddings built from llama_batch_ext
    const struct ggml_tensor *  embd_dev = nullptr;
    int64_t                     embd_dev_row0 = 0;
    std::vector<int8_t>         is_embd_vec;  // mixed batch only (= 1 if embd, 0 if text token)
    std::vector<float>          state_vec;    // owned state embeddings built from llama_batch_ext, llama_batch has no slot for them
    std::vector<llama_seq_id>   seq_id_data;  // flat storage for seq_id pointers below

""",
        ),
        (
            "src/llama-context.cpp",
            r"""#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
""",
            r"""#include "llama-context.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
""",
        ),
        (
            "src/llama-context.cpp",
            r"""
    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
""",
            r"""
    return embd_layer_inp[lid].data;
}

const ggml_tensor * llama_context::get_embeddings_layer_inp_device() {
    return t_layer_inp_dev;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

""",
            r"""    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp_device(bool enable) {
    synchronize();

    layer_inp_device = enable;

    t_layer_inp_dev = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();
}

void llama_context::embeddings_layer_inp_consumed(llama_context * consumer) {
    if (t_layer_inp_dev == nullptr) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf_layer_inp_dev.get()));
    ggml_backend_t backend = nullptr;
    for (auto & cur : consumer->backends) {
        if (dev != nullptr && ggml_backend_get_device(cur.get()) == dev) {
            backend = cur.get();
            break;
        }
    }
    if (backend != nullptr && !layer_inp_event) {
        layer_inp_event.reset(ggml_backend_event_new(dev));
    }
    if (backend == nullptr || !layer_inp_event) {
        consumer->synchronize();
        return;
    }
    ggml_backend_event_record(layer_inp_event.get(), backend);
    layer_inp_event_pending = true;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    t_layer_inp_dev = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

""",
        ),
        (
            "src/llama-context.cpp",
            r"""                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
""",
            r"""                    }
            }
        }

        if (layer_inp_device) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                GGML_ASSERT(ubatch.data->batch_idxs[i] == (int32_t) (n_tokens_prev + i) && "device layer inputs need batch-ordered ubatches");
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
""",
            r"""    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    if (layer_inp_device) {
        std::vector<uint32_t> lids;
        for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
            if (cparams.embeddings_layer_inp[il]) {
                lids.push_back(il);
            }
        }
        if (lids.empty()) {
            return false;
        }

        ggml_tensor * first = res->get_layer_inp((int) lids[0]);
        GGML_ASSERT(first != nullptr);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), first);
        GGML_ASSERT(backend != nullptr);

        const int64_t n_embd_l = model.hparams.n_embd;
        const int64_t n_rows   = cparams.n_batch;

        if (t_layer_inp_dev == nullptr) {
            ggml_init_params params_dev = { ggml_tensor_overhead(), nullptr, true };
            ctx_layer_inp_dev.reset(ggml_init(params_dev));
            t_layer_inp_dev = ggml_new_tensor_3d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_embd_l, n_rows, (int64_t) lids.size());
            buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), ggml_backend_get_default_buffer_type(backend)));
            GGML_ASSERT(buf_layer_inp_dev != nullptr && "failed to allocate the device layer input buffer");

            ggml_init_params params_views = { ggml_tensor_overhead() * (LLAMA_MAX_LAYERS + 1), nullptr, true };
            ctx_layer_inp_views.reset(ggml_init(params_views));
        }

        GGML_ASSERT(t_layer_inp_dev->ne[2] == (int64_t) lids.size());
        GGML_ASSERT((int64_t) (token_offset + n_tokens) <= n_rows);

        ggml_reset(ctx_layer_inp_views.get());

        if (layer_inp_event_pending) {
            ggml_backend_event_wait(backend, layer_inp_event.get());
            layer_inp_event_pending = false;
        }

        for (size_t k = 0; k < lids.size(); ++k) {
            ggml_tensor * t = res->get_layer_inp((int) lids[k]);
            GGML_ASSERT(t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            GGML_ASSERT(t->ne[0] == n_embd_l && ggml_nelements(t) == n_embd_l * (int64_t) n_tokens);
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), t) == backend);

            const size_t offset = k * t_layer_inp_dev->nb[2] + token_offset * t_layer_inp_dev->nb[1];
            ggml_tensor * dst = ggml_view_4d(ctx_layer_inp_views.get(), t_layer_inp_dev,
                    t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3], offset);
            GGML_ASSERT(ggml_backend_view_init(dst) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_copy_async(backend, backend, t, dst);
        }

        return false;
    }

    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
""",
        ),
        (
            "src/llama-context.cpp",
            r"""
    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

""",
            r"""
    return ctx->get_embeddings_layer_inp(lid);
}

void llama_set_embeddings_layer_inp_device(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_device(value);
}

void llama_embeddings_layer_inp_consumed(llama_context * ctx, llama_context * consumer) {
    ctx->embeddings_layer_inp_consumed(consumer);
}

const ggml_tensor * llama_get_embeddings_layer_inp_device(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp_device();
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

""",
        ),
        (
            "src/llama-context.h",
            r"""    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

""",
            r"""    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);
    const ggml_tensor * get_embeddings_layer_inp_device();

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

""",
        ),
        (
            "src/llama-context.h",
            r"""
    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

""",
            r"""
    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_embeddings_layer_inp_device(bool enable);
    void embeddings_layer_inp_consumed(llama_context * consumer);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

""",
        ),
        (
            "src/llama-context.h",
            r"""    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

""",
            r"""    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    bool                    layer_inp_device = false;
    ggml_context_ptr        ctx_layer_inp_dev;
    ggml_context_ptr        ctx_layer_inp_views;
    ggml_backend_buffer_ptr buf_layer_inp_dev;
    ggml_tensor *           t_layer_inp_dev = nullptr;
    ggml_backend_event_ptr  layer_inp_event;
    bool                    layer_inp_event_pending = false;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

""",
        ),
        (
            "src/llama-ext.h",
            r"""// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
""",
            r"""// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API void llama_set_embeddings_layer_inp_device(struct llama_context * ctx, bool value);

LLAMA_API const struct ggml_tensor * llama_get_embeddings_layer_inp_device(struct llama_context * ctx);

LLAMA_API void llama_embeddings_layer_inp_consumed(struct llama_context * ctx, struct llama_context * consumer);

LLAMA_API int32_t llama_batch_ext_add_embd_device(struct llama_batch_ext * batch, llama_seq_id seq_id);

LLAMA_API void llama_batch_ext_set_embd_device(struct llama_batch_ext * batch, const struct ggml_tensor * src, int64_t row0);

LLAMA_API bool llama_model_dflash_device_features(const struct llama_model * model);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
""",
        ),
        (
            "src/llama-graph.cpp",
            r"""    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
""",
            r"""    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);
        GGML_ASSERT(ubatch->embd_dev == nullptr && "this graph does not read device embedding rows");

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
""",
        ),
        (
            "src/llama-graph.cpp",
            r"""    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
""",
            r"""    }
    return n;
}

bool llm_graph_input_embd_dev::can_reuse(const llm_graph_params & params) {
    return params.ubatch.embd_dev == src && params.ubatch.embd_dev_row0 == row0;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= params.ubatch.embd_dev == nullptr;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
""",
        ),
        (
            "src/llama-graph.h",
            r"""};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;
""",
            r"""};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd_dev : public llm_graph_input_i {
public:
    llm_graph_input_embd_dev(const ggml_tensor * src, int64_t row0) : src(src), row0(row0) {}
    virtual ~llm_graph_input_embd_dev() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->embd_dev == src && ubatch->embd_dev_row0 == row0);
    }

    bool can_reuse(const llm_graph_params & params) override;

    const ggml_tensor * src;
    const int64_t row0;
};

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;
""",
        ),
        (
            "src/llama-model.cpp",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

""",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

bool llama_model_dflash_device_features(const llama_model * model) {
    const auto & hparams = model->hparams;
    return model->arch == LLM_ARCH_DFLASH && hparams.n_lora_kv == 0 && hparams.dsv4_hc_mult == 0 && hparams.lilicorr_candidate_topk == 0;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

""",
        ),
        (
            "src/models/dflash.cpp",
            r"""
    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc->type==GGML_TYPE_F32?model.output_norm_enc:ggml_cast(ctx0,model.output_norm_enc,GGML_TYPE_F32), NULL, LLM_NORM_RMS, -1);
""",
            r"""
    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        ggml_tensor * inp_target = nullptr;

        if (ubatch.embd_dev) {
            ggml_tensor * src = const_cast<ggml_tensor *>(ubatch.embd_dev);
            GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] * src->ne[2] == n_embd_inp);
            GGML_ASSERT(ubatch.embd_dev_row0 >= 0 && ubatch.embd_dev_row0 + n_tokens <= src->ne[1]);

            ggml_tensor * rows = ggml_view_3d(ctx0, src, src->ne[0], n_tokens, src->ne[2], src->nb[1], src->nb[2], ubatch.embd_dev_row0 * src->nb[1]);
            inp_target = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, rows, 0, 2, 1, 3)), n_embd_inp, n_tokens);

            res->add_input(std::make_unique<llm_graph_input_embd_dev>(ubatch.embd_dev, ubatch.embd_dev_row0));
        } else {
            auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

            inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
            ggml_set_input(inp->embd);

            inp_target = inp->embd;

            res->add_input(std::move(inp));
        }
        cb(inp_target, "inp_target_features", -1);

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc->type==GGML_TYPE_F32?model.output_norm_enc:ggml_cast(ctx0,model.output_norm_enc,GGML_TYPE_F32), NULL, LLM_NORM_RMS, -1);
""",
        ),
    ),
    "stop_token_cache_v1": (
        (
            "tools/server/server-context.cpp",
            r"""            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();
""",
            r"""            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            const bool can_drop_stop_tail =
                ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                ctx_dft_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_FULL &&
                (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS || n_draft <= llama_n_rs_seq(ctx_tgt));

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    if (can_drop_stop_tail && i + 1 < ids.size()) {
                        slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - (int) (ids.size() - 1 - i));
                    }
                    slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            slot.print_timings_tg();
""",
        ),
    ),
    "round_profile_nvtx_v1": (
        (
            "src/CMakeLists.txt",
            r"""target_link_libraries(llama PUBLIC ggml)
""",
            r"""target_link_libraries(llama PUBLIC ggml)
find_package(CUDAToolkit REQUIRED)
target_include_directories(llama PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""#include "ggml-cuda.h"
#include "ggml-impl.h"
""",
            r"""#include "ggml-cuda.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

static bool bitdraft_nvtx_nodes() {
    static const bool enabled = getenv("BITDRAFT_NVTX_NODES") != nullptr;
    return enabled;
}

static void bitdraft_nvtx_push_node(const ggml_tensor * node, int index, const char * kind) {
    char label[4096];
    int used = snprintf(label, sizeof(label), "%s|%d|%s|%s|%lld,%lld,%lld,%lld", kind, index, ggml_op_desc(node), node->name,
        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
    for (int j = 0; j < GGML_MAX_SRC && used > 0 && used < (int) sizeof(label); ++j) {
        const ggml_tensor * src = node->src[j];
        if (src == nullptr) {
            continue;
        }
        const int weight = src->buffer != nullptr && ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        used += snprintf(label + used, sizeof(label) - used, "|%d:%s:%s:%lld,%lld,%lld,%lld:%zu:%d", j, src->name, ggml_type_name(src->type),
            (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3], ggml_nbytes(src), weight);
    }
    nvtxRangePushA(label);
}
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
""",
            r"""                if (bitdraft_nvtx_nodes()) {
                    bitdraft_nvtx_push_node(node, i, "node");
                }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                if (bitdraft_nvtx_nodes() && nodes_to_skip != 0) {
                    for (int fused = 1; fused <= nodes_to_skip; ++fused) {
                        bitdraft_nvtx_push_node(cgraph->nodes[i + fused], i + fused, "fused");
                        nvtxRangePop();
                    }
                    nvtxRangePop();
                }
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
""",
            r"""                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (bitdraft_nvtx_nodes()) {
                    nvtxRangePop();
                }
""",
        ),
        (
            "src/llama-context.cpp",
            r"""#include "llama-context.h"
""",
            r"""#include "llama-context.h"
#include <nvtx3/nvToolsExt.h>
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    const auto gparams = graph_params(res, ubatch, mctx, gtype);
""",
            r"""    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    nvtxRangePushA("llama.graph");
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    // set the input data for the input tensors
    {
""",
            r"""    nvtxRangePop();
    nvtxRangePushA("llama.inputs");
    {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
""",
            r"""    nvtxRangePop();
    nvtxRangePushA("llama.compute");
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    nvtxRangePop();
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens""",
            r"""    nvtxRangePushA("llama.sync");
    ggml_backend_sched_synchronize(sched.get());
    nvtxRangePop();

    // FIXME: if multiple single tokens""",
        ),
        (
            "common/speculative.cpp",
            r"""#include "speculative.h"
""",
            r"""#include "speculative.h"
#include <nvtx3/nvToolsExt.h>

namespace {
struct bitdraft_range {
    explicit bitdraft_range(const char * label) { nvtxRangePushA(label); }
    ~bitdraft_range() { nvtxRangePop(); }
    bitdraft_range(const bitdraft_range &) = delete;
    bitdraft_range & operator=(const bitdraft_range &) = delete;
};
}
""",
        ),
        (
            "common/speculative.cpp",
            r"""            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
""",
            r"""            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        bitdraft_range bitdraft_select("dft.select");
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""#include "server-context.h"
""",
            r"""#include "server-context.h"
#include <nvtx3/nvToolsExt.h>
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
""",
            r"""            nvtxRangePushA("srv.draft");
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
            nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
""",
            r"""        int ret = 0;
        nvtxRangePushA(("srv.target|" + std::to_string(batch.view.size())).c_str());
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });
""",
            r"""            nvtxRangePushA("srv.process");
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });
            nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // make checkpoints if needed
""",
            r"""        nvtxRangePushA("srv.checkpoint");
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // update the batch with the sampled/drafted tokens
""",
            r"""        nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // speculative decoding - main model sample and accept
""",
            r"""        nvtxRangePushA("srv.verify");
        struct bitdraft_verify_end {
            ~bitdraft_verify_end() { nvtxRangePop(); }
        } bitdraft_verify;
""",
        ),
    ),
    "slot_rounding_v1": (
        (
            "src/models/dflash.cpp",
            r"""            memcpy(packed.scales->op_params,&abits,4); memcpy((char*)packed.scales->op_params+4,&mode,4); memcpy((char*)packed.scales->op_params+8,&clip,4);
""",
            r"""            memcpy(packed.scales->op_params,&abits,4); memcpy((char*)packed.scales->op_params+4,&mode,4); memcpy((char*)packed.scales->op_params+8,&clip,4);
            float clip_anchor=0; const int anchor_key=gguf_find_key(ml->metadata,(base+".w1ax.clip_anchor").c_str());
            if(anchor_key>=0) {
                if(gguf_get_kv_type(ml->metadata,anchor_key)!=GGUF_TYPE_FLOAT32) throw std::runtime_error("invalid anchor quantizer metadata type");
                clip_anchor=gguf_get_val_f32(ml->metadata,anchor_key);
                if(!mode || !std::isfinite(clip_anchor) || clip_anchor<=0) throw std::runtime_error("invalid anchor quantizer metadata");
            }
            memcpy((char*)packed.scales->op_params+12,&clip_anchor,4);
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""                if(p.scales->op_params[0]) memcpy((char*)result->op_params+4,(char*)p.scales->op_params+4,8);
""",
            r"""                if(p.scales->op_params[0]) memcpy((char*)result->op_params+4,(char*)p.scales->op_params+4,8);
                if(p.scales->op_params[3]) {
                    const int64_t rows=input->ne[1], blocks=ubatch.n_seqs_unq;
                    int32_t period=0;
                    if(!exec_cropped) {
                        if(blocks<=0 || rows%blocks) throw std::runtime_error("per-slot rounding needs equal blocks of slots");
                        period=(int32_t)(rows/blocks);
                    } else if(rows!=n_tokens-blocks) throw std::runtime_error("per-slot rounding needs the anchors outside the cropped rows");
                    memcpy((char*)result->op_params+12,(char*)p.scales->op_params+12,4);
                    memcpy((char*)result->op_params+16,&period,4);
                }
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""            if(memcmp(layer.ffn_gate_w1a8.scales->op_params,layer.ffn_up_w1a8.scales->op_params,12)) throw std::runtime_error("gate/up quantizers must match");""",
            r"""            if(memcmp(layer.ffn_gate_w1a8.scales->op_params,layer.ffn_up_w1a8.scales->op_params,16)) throw std::runtime_error("gate/up quantizers must match");""",
        ),
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r"""void ggml_cuda_w1ax_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
""",
            r"""template<int B> static __global__ void exec_slot_codes(const float*x,const float*g,int64_t k,int bits,float clip,float anchor,int period,int8_t*q,int*partial,int tiles){
 int token=blockIdx.y,j=blockIdx.x*256+threadIdx.x,qm=B?((B<=2)?1:(1<<(B-1))-1):((bits<=2)?1:(1<<(bits-1))-1),c=0;
 const float bound=(period>0&&token%period==0)?anchor:clip;
 if(!isfinite(bound)||bound<=0)__trap();
 if(j<k){float v=gained_value(x+token*k,g,j);if(!isfinite(v))__trap();
  if((B?B:bits)==1)c=v>=0?1:-1;else{float inv=__fdiv_rn(float(qm),bound);float z=isfinite(inv)?__fmul_rn(v,inv):__double2float_rn(double(v)/double(bound)*qm);c=max(-qm,min(qm,__float2int_rn(fminf(float(qm),fmaxf(-float(qm),z)))));}q[token*k+j]=int8_t(c);
 }
 for(int d=16;d;d/=2)c+=__shfl_down_sync(0xffffffff,c,d);__shared__ int ws[8];if((threadIdx.x&31)==0)ws[threadIdx.x/32]=c;__syncthreads();if(threadIdx.x==0){int total=0;for(int i=0;i<8;++i)total+=ws[i];partial[token*tiles+blockIdx.x]=total;}
}
static __global__ void exec_slot_finish(const int*p,int tiles,int bits,float clip,float anchor,int period,float*b,float*s){int token=blockIdx.x,sum=0;const float bound=(period>0&&token%period==0)?anchor:clip;for(int i=threadIdx.x;i<tiles;i+=32)sum+=p[token*tiles+i];for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);if(threadIdx.x==0){int qm=bits<=2?1:(1<<(bits-1))-1;float beta=__fdiv_rn(bound,float(qm));b[token]=beta;s[token]=__fmul_rn(float(sum),beta);}}

void ggml_cuda_w1ax_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
""",
        ),
        (
            "ggml/src/ggml-cuda/w1a1.cu",
            r"""        #define TILED(B) exec_tiled_codes<B><<<dim3(tiles,n),256,0,ctx.stream()>>>((const float*)acts->data,g,k,bits,clip,codes,partial.get(),tiles)
        switch(bits){case 1:TILED(1);break;case 2:TILED(2);break;case 4:TILED(4);break;case 8:TILED(8);break;default:GGML_ABORT("unsupported width");}
        #undef TILED
        exec_tiled_finish<<<n,32,0,ctx.stream()>>>(partial.get(),tiles,bits,clip,scales,scales+n);
""",
            r"""        const float anchor=ggml_get_op_params_f32(dst,3);const int period=ggml_get_op_params_i32(dst,4);
        if(anchor>0) {
        #define SLOT(B) exec_slot_codes<B><<<dim3(tiles,n),256,0,ctx.stream()>>>((const float*)acts->data,g,k,bits,clip,anchor,period,codes,partial.get(),tiles)
        switch(bits){case 1:SLOT(1);break;case 2:SLOT(2);break;case 4:SLOT(4);break;case 8:SLOT(8);break;default:GGML_ABORT("unsupported width");}
        #undef SLOT
        exec_slot_finish<<<n,32,0,ctx.stream()>>>(partial.get(),tiles,bits,clip,anchor,period,scales,scales+n);
        } else {
        #define TILED(B) exec_tiled_codes<B><<<dim3(tiles,n),256,0,ctx.stream()>>>((const float*)acts->data,g,k,bits,clip,codes,partial.get(),tiles)
        switch(bits){case 1:TILED(1);break;case 2:TILED(2);break;case 4:TILED(4);break;case 8:TILED(8);break;default:GGML_ABORT("unsupported width");}
        #undef TILED
        exec_tiled_finish<<<n,32,0,ctx.stream()>>>(partial.get(),tiles,bits,clip,scales,scales+n);
        }
""",
        ),
        (
            "ggml/src/ggml-impl.h",
            r"""        for (int i = 2; i < GGML_MAX_SRC; ++i) {
            if (op->src[i] != NULL) return false;
        }
        for (uint32_t i = 3; i < GGML_MAX_OP_PARAMS/sizeof(int32_t); ++i) {
            if (ggml_get_op_params_i32(op, i) != 0) return false;
        }
""",
            r"""        for (int i = 2; i < GGML_MAX_SRC; ++i) {
            if (op->src[i] != NULL) return false;
        }
        float anchor;
        memcpy(&anchor, (const char *) op->op_params + 12, sizeof(anchor));
        const int32_t period = ggml_get_op_params_i32(op, 4);
        if (ggml_get_op_params_i32(op, 3) != 0 && (ggml_get_op_params_i32(op, 1) == 0 || !isfinite(anchor) || !(anchor > 0.0f))) return false;
        if (period != 0 && (ggml_get_op_params_i32(op, 3) == 0 || period < 0 || a->ne[1] % period != 0)) return false;
        for (uint32_t i = 5; i < GGML_MAX_OP_PARAMS/sizeof(int32_t); ++i) {
            if (ggml_get_op_params_i32(op, i) != 0) return false;
        }
""",
        ),
        (
            "ggml/src/ggml-cpu/ggml-cpu.c",
            r"""    GGML_ASSERT(ggml_w1ax_is_supported(dst));
    GGML_ASSERT(fegetround() == FE_TONEAREST);
""",
            r"""    GGML_ASSERT(ggml_w1ax_is_supported(dst));
    GGML_ASSERT(ggml_get_op_params_i32(dst, 3) == 0);
    GGML_ASSERT(fegetround() == FE_TONEAREST);
""",
        ),
    ),
    "vocabulary_shortlist_v1": (
        (
            "src/llama-cparams.h",
            r"""    llama_context * ctx_other;
};
""",
            r"""    llama_context * ctx_other;

    const struct llama_vocab_shortlist * vocab_shortlist = nullptr;
};

struct llama_vocab_shortlist {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    const ggml_tensor * source = nullptr;
    ggml_tensor * rows = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * request_rows = nullptr;
    ggml_tensor * request_ids = nullptr;
    int64_t n_vocab = 0;
    size_t row_bytes = 0;
    uint32_t rows_per_seq = 0;
    std::vector<uint8_t> fixed;
    std::vector<int32_t> slot_of;
    std::vector<int32_t> slot_token;
    std::vector<uint64_t> slot_seqs;
    std::vector<uint32_t> seq_rows;
    std::vector<uint8_t> row;

    llama_vocab_shortlist() = default;
    llama_vocab_shortlist(const llama_vocab_shortlist &) = delete;
    llama_vocab_shortlist & operator=(const llama_vocab_shortlist &) = delete;
    ~llama_vocab_shortlist();
};
""",
        ),
        (
            "src/llama-graph.h",
            r"""//
// llm_graph_result
//
""",
            r"""class llm_graph_input_vocab_shortlist : public llm_graph_input_i {
public:
    llm_graph_input_vocab_shortlist(const llama_vocab_shortlist * shortlist) : shortlist(shortlist) {}
    virtual ~llm_graph_input_vocab_shortlist() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    const llama_vocab_shortlist * shortlist;
    ggml_tensor * mask = nullptr;
};

//
// llm_graph_result
//
""",
        ),
        (
            "src/llama-graph.cpp",
            r"""llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
""",
            r"""void llm_graph_input_vocab_shortlist::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(mask && shortlist);
    const int64_t n_rows = mask->ne[0];
    const int64_t n_columns = mask->ne[1];
    std::vector<float> values((size_t) (n_rows * n_columns), -INFINITY);
    int64_t column = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens && column < n_columns; ++i) {
        if (ubatch->output && !ubatch->output[i]) {
            continue;
        }
        const llama_seq_id seq_id = ubatch->seq_id[i][0];
        if (seq_id >= 0 && seq_id < 64) {
            const uint64_t bit = uint64_t(1) << seq_id;
            for (int64_t r = 0; r < n_rows; ++r) {
                if (shortlist->slot_seqs[(size_t) r] & bit) {
                    values[(size_t) (column * n_rows + r)] = 0.0f;
                }
            }
        }
        ++column;
    }
    ggml_backend_tensor_set(mask, values.data(), 0, ggml_nbytes(mask));
}

bool llm_graph_input_vocab_shortlist::can_reuse(const llm_graph_params & params) {
    return mask && params.cparams.vocab_shortlist == shortlist && mask->ne[1] == (int64_t) params.n_outputs;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
""",
        ),
        (
            "src/llama-model.h",
            r"""    struct ggml_tensor * d2t = nullptr;  // draft to target vocabulary mapping
""",
            r"""    struct ggml_tensor * d2t = nullptr;  // draft to target vocabulary mapping
    std::vector<int32_t> dflash_vocab_shortlist;
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);
""",
            r"""    ml.get_key(LLM_KV_DFLASH_SELECTOR_TOP_K,   hparams.dflash_selector_top_k,   false);
    {
        const int64_t shortlist_key = gguf_find_key(ml.metadata, "dflash.vocab_shortlist");
        if (shortlist_key >= 0) {
            if (gguf_get_kv_type(ml.metadata, shortlist_key) != GGUF_TYPE_ARRAY ||
                    gguf_get_arr_type(ml.metadata, shortlist_key) != GGUF_TYPE_INT32) {
                throw std::runtime_error("dflash.vocab_shortlist must be an int32 array");
            }
            const int32_t * shortlist_data = (const int32_t *) gguf_get_arr_data(ml.metadata, shortlist_key);
            dflash_vocab_shortlist.assign(shortlist_data, shortlist_data + gguf_get_arr_n(ml.metadata, shortlist_key));
        }
    }
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""    if (!exec_cropped && n_outputs<n_tokens && !model.dspark_markov_w1 && !model.dflash_selector_hidden && !model.lilicorr_pass_hidden_proj) cur=ggml_get_rows(ctx0,cur,build_inp_out_ids());
    cur = build_lora_mm(output, cur, output_s);
""",
            r"""    if (!exec_cropped && n_outputs<n_tokens && !model.dspark_markov_w1 && !model.dflash_selector_hidden && !model.lilicorr_pass_hidden_proj) cur=ggml_get_rows(ctx0,cur,build_inp_out_ids());
    const llama_vocab_shortlist * shortlist = cparams.vocab_shortlist;
    if (shortlist && shortlist->source == output && !output_s && !model.d2t && !model.dspark_markov_w1 && !model.dflash_selector_hidden && !model.lilicorr_pass_hidden_proj) {
        const int64_t n_columns = cur->ne[1];
        ggml_tensor * scored = ggml_mul_mat(ctx0, shortlist->rows, cur);
        if (prec_policy) {
            prec_policy->apply(scored);
        }
        cb(scored, "result_output_shortlist", -1);
        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, shortlist->n_vocab, n_columns), -INFINITY);
        logits = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, scored, 1, shortlist->rows->ne[1], n_columns),
                ggml_reshape_3d(ctx0, shortlist->ids, shortlist->ids->ne[0], 1, 1));
        if (shortlist->request_rows) {
            auto inp_shortlist = std::make_unique<llm_graph_input_vocab_shortlist>(shortlist);
            inp_shortlist->mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, shortlist->request_rows->ne[1], n_columns);
            ggml_set_input(inp_shortlist->mask);
            ggml_tensor * requested = ggml_mul_mat(ctx0, shortlist->request_rows, cur);
            if (prec_policy) {
                prec_policy->apply(requested);
            }
            cb(requested, "result_output_request", -1);
            requested = ggml_add(ctx0, requested, inp_shortlist->mask);
            res->add_input(std::move(inp_shortlist));
            logits = ggml_set_rows(ctx0, logits,
                    ggml_reshape_3d(ctx0, requested, 1, shortlist->request_rows->ne[1], n_columns),
                    ggml_reshape_3d(ctx0, shortlist->request_ids, shortlist->request_ids->ne[0], 1, 1));
        }
        cur = ggml_reshape_2d(ctx0, logits, shortlist->n_vocab, n_columns);
    } else {
        cur = build_lora_mm(output, cur, output_s);
    }
""",
        ),
        (
            "src/llama-model.cpp",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}
""",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

const int32_t * llama_model_dflash_vocab_shortlist(const llama_model * model, size_t * n) {
    *n = model->dflash_vocab_shortlist.size();
    return model->dflash_vocab_shortlist.empty() ? nullptr : model->dflash_vocab_shortlist.data();
}
""",
        ),
        (
            "src/llama-ext.h",
            r"""LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);
""",
            r"""LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

LLAMA_API const int32_t * llama_model_dflash_vocab_shortlist(const struct llama_model * model, size_t * n);

LLAMA_API bool llama_vocab_shortlist_init(struct llama_context * ctx, const int32_t * ids, size_t n, uint32_t rows_per_seq);

LLAMA_API void llama_vocab_shortlist_reset(struct llama_context * ctx, llama_seq_id seq_id);

LLAMA_API uint32_t llama_vocab_shortlist_add(struct llama_context * ctx, llama_seq_id seq_id, const llama_token * tokens, size_t n);
""",
        ),
        (
            "src/llama-context.h",
            r"""    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);
""",
            r"""    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    bool vocab_shortlist_init(const int32_t * ids, size_t n, uint32_t rows_per_seq);
    void vocab_shortlist_reset(llama_seq_id seq_id);
    uint32_t vocab_shortlist_add(llama_seq_id seq_id, const llama_token * tokens, size_t n);
""",
        ),
        (
            "src/llama-context.h",
            r"""    bool graph_reuse_disable = false;
""",
            r"""    bool graph_reuse_disable = false;

    std::unique_ptr<llama_vocab_shortlist> vocab_shortlist;
""",
        ),
        (
            "src/llama-context.cpp",
            r"""bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
""",
            r"""llama_vocab_shortlist::~llama_vocab_shortlist() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

bool llama_context::vocab_shortlist_init(const int32_t * ids, size_t n, uint32_t rows_per_seq) {
    if (vocab_shortlist) {
        LLAMA_LOG_ERROR("%s: a vocabulary shortlist is already attached\n", __func__);
        return false;
    }
    if (model.arch != LLM_ARCH_DFLASH || model.d2t || model.dflash_selector_hidden || model.dspark_markov_w1 ||
            model.lilicorr_pass_hidden_proj || model.hparams.n_lora_kv > 0 || model.hparams.dsv4_hc_mult > 0 ||
            !model.hdmd.rot.empty()) {
        LLAMA_LOG_ERROR("%s: a vocabulary shortlist needs a plain DFlash drafter\n", __func__);
        return false;
    }
    const ggml_tensor * source = model.output;
    const ggml_tensor * source_scale = model.output_s;
    if (source == nullptr) {
        if (cparams.ctx_other == nullptr) {
            LLAMA_LOG_ERROR("%s: the drafter has no output table and no target context\n", __func__);
            return false;
        }
        const llama_model * other = llama_get_model(cparams.ctx_other);
        source = other->output;
        source_scale = other->output_s;
    }
    if (source == nullptr || source_scale != nullptr || source->buffer == nullptr || source->type == GGML_TYPE_NVFP4 ||
            !ggml_is_contiguous(source) || source->ne[2] != 1 || source->ne[3] != 1 ||
            source->ne[1] != (int64_t) model.vocab.n_tokens() ||
            model.prec_policy.prec_acc.count(source) > 0 || model.prec_policy.prec_src1.count(source) > 0) {
        LLAMA_LOG_ERROR("%s: the output table cannot be shortlisted\n", __func__);
        return false;
    }
    const int64_t n_vocab = source->ne[1];
    const int64_t n_fixed = (int64_t) n;
    const int64_t capacity = (int64_t) rows_per_seq * (int64_t) cparams.n_seq_max;
    if (n_fixed < 1 || n_fixed > n_vocab || cparams.n_seq_max > 64 || capacity > n_vocab - n_fixed) {
        LLAMA_LOG_ERROR("%s: invalid vocabulary shortlist size %" PRId64 " or request rows %" PRId64 "\n", __func__, n_fixed, capacity);
        return false;
    }
    auto shortlist = std::make_unique<llama_vocab_shortlist>();
    shortlist->fixed.assign((size_t) n_vocab, 0);
    for (int64_t i = 0; i < n_fixed; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= n_vocab || shortlist->fixed[(size_t) id]) {
            LLAMA_LOG_ERROR("%s: vocabulary shortlist token %d is out of range or repeated\n", __func__, id);
            return false;
        }
        shortlist->fixed[(size_t) id] = 1;
    }
    const size_t row_bytes = ggml_row_size(source->type, source->ne[0]);
    if (source->nb[1] != row_bytes) {
        LLAMA_LOG_ERROR("%s: the output table rows are not contiguous\n", __func__);
        return false;
    }
    ggml_init_params init_params = { 4 * ggml_tensor_overhead(), nullptr, true };
    shortlist->ctx = ggml_init(init_params);
    if (shortlist->ctx == nullptr) {
        return false;
    }
    shortlist->rows = ggml_new_tensor_2d(shortlist->ctx, source->type, source->ne[0], n_fixed);
    ggml_set_name(shortlist->rows, "output_shortlist.weight");
    shortlist->ids = ggml_new_tensor_1d(shortlist->ctx, GGML_TYPE_I64, n_fixed);
    ggml_set_name(shortlist->ids, "output_shortlist.ids");
    if (capacity > 0) {
        shortlist->request_rows = ggml_new_tensor_2d(shortlist->ctx, source->type, source->ne[0], capacity);
        ggml_set_name(shortlist->request_rows, "output_request.weight");
        shortlist->request_ids = ggml_new_tensor_1d(shortlist->ctx, GGML_TYPE_I64, capacity);
        ggml_set_name(shortlist->request_ids, "output_request.ids");
    }
    shortlist->buffer = ggml_backend_alloc_ctx_tensors_from_buft(shortlist->ctx, ggml_backend_buffer_get_type(source->buffer));
    if (shortlist->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: cannot allocate the vocabulary shortlist\n", __func__);
        return false;
    }
    ggml_backend_buffer_set_usage(shortlist->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    {
        std::vector<uint8_t> table(ggml_nbytes(source));
        ggml_backend_tensor_get(source, table.data(), 0, table.size());
        std::vector<uint8_t> gathered((size_t) n_fixed * row_bytes);
        std::vector<int64_t> fixed_ids((size_t) n_fixed);
        for (int64_t i = 0; i < n_fixed; ++i) {
            std::memcpy(gathered.data() + (size_t) i * row_bytes, table.data() + (size_t) ids[i] * row_bytes, row_bytes);
            fixed_ids[(size_t) i] = ids[i];
        }
        ggml_backend_tensor_set(shortlist->rows, gathered.data(), 0, gathered.size());
        ggml_backend_tensor_set(shortlist->ids, fixed_ids.data(), 0, fixed_ids.size() * sizeof(int64_t));
        if (capacity > 0) {
            shortlist->slot_of.assign((size_t) n_vocab, -1);
            shortlist->slot_token.assign((size_t) capacity, -1);
            shortlist->slot_seqs.assign((size_t) capacity, 0);
            std::vector<uint8_t> padding((size_t) capacity * row_bytes);
            std::vector<int64_t> padding_ids((size_t) capacity);
            int64_t slot = 0;
            for (int64_t token = 0; token < n_vocab && slot < capacity; ++token) {
                if (shortlist->fixed[(size_t) token]) {
                    continue;
                }
                std::memcpy(padding.data() + (size_t) slot * row_bytes, table.data() + (size_t) token * row_bytes, row_bytes);
                padding_ids[(size_t) slot] = token;
                shortlist->slot_token[(size_t) slot] = (int32_t) token;
                shortlist->slot_of[(size_t) token] = (int32_t) slot;
                ++slot;
            }
            ggml_backend_tensor_set(shortlist->request_rows, padding.data(), 0, padding.size());
            ggml_backend_tensor_set(shortlist->request_ids, padding_ids.data(), 0, padding_ids.size() * sizeof(int64_t));
        }
    }
    shortlist->source = source;
    shortlist->n_vocab = n_vocab;
    shortlist->row_bytes = row_bytes;
    shortlist->rows_per_seq = rows_per_seq;
    shortlist->seq_rows.assign(cparams.n_seq_max, 0);
    shortlist->row.resize(row_bytes);
    vocab_shortlist = std::move(shortlist);
    cparams.vocab_shortlist = vocab_shortlist.get();
    sched_need_reserve = true;
    LLAMA_LOG_INFO("%s: vocabulary shortlist of %" PRId64 " %s rows, %" PRId64 " request rows\n", __func__, n_fixed, ggml_type_name(source->type), capacity);
    return true;
}

void llama_context::vocab_shortlist_reset(llama_seq_id seq_id) {
    if (!vocab_shortlist || seq_id < 0 || (size_t) seq_id >= vocab_shortlist->seq_rows.size()) {
        return;
    }
    const uint64_t bit = uint64_t(1) << seq_id;
    for (auto & seqs : vocab_shortlist->slot_seqs) {
        seqs &= ~bit;
    }
    vocab_shortlist->seq_rows[(size_t) seq_id] = 0;
}

uint32_t llama_context::vocab_shortlist_add(llama_seq_id seq_id, const llama_token * tokens, size_t n) {
    if (!vocab_shortlist || seq_id < 0 || (size_t) seq_id >= vocab_shortlist->seq_rows.size()) {
        return 0;
    }
    auto & shortlist = *vocab_shortlist;
    const uint64_t bit = uint64_t(1) << seq_id;
    uint32_t refused = 0;
    for (size_t i = 0; i < n; ++i) {
        const llama_token token = tokens[i];
        if (token < 0 || token >= shortlist.n_vocab || shortlist.fixed[(size_t) token]) {
            continue;
        }
        if (shortlist.request_rows == nullptr) {
            ++refused;
            continue;
        }
        int32_t slot = shortlist.slot_of[(size_t) token];
        if (slot >= 0 && (shortlist.slot_seqs[(size_t) slot] & bit)) {
            continue;
        }
        if (shortlist.seq_rows[(size_t) seq_id] >= shortlist.rows_per_seq) {
            ++refused;
            continue;
        }
        if (slot < 0) {
            size_t free_slot = 0;
            while (free_slot < shortlist.slot_seqs.size() && shortlist.slot_seqs[free_slot] != 0) {
                ++free_slot;
            }
            GGML_ASSERT(free_slot < shortlist.slot_seqs.size());
            shortlist.slot_of[(size_t) shortlist.slot_token[free_slot]] = -1;
            ggml_backend_tensor_get(shortlist.source, shortlist.row.data(), (size_t) token * shortlist.row_bytes, shortlist.row_bytes);
            ggml_backend_tensor_set(shortlist.request_rows, shortlist.row.data(), free_slot * shortlist.row_bytes, shortlist.row_bytes);
            const int64_t id = token;
            ggml_backend_tensor_set(shortlist.request_ids, &id, free_slot * sizeof(int64_t), sizeof(int64_t));
            shortlist.slot_token[free_slot] = token;
            shortlist.slot_of[(size_t) token] = (int32_t) free_slot;
            slot = (int32_t) free_slot;
        }
        shortlist.slot_seqs[(size_t) slot] |= bit;
        shortlist.seq_rows[(size_t) seq_id] += 1;
    }
    return refused;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
""",
            r"""bool llama_vocab_shortlist_init(llama_context * ctx, const int32_t * ids, size_t n, uint32_t rows_per_seq) {
    return ctx->vocab_shortlist_init(ids, n, rows_per_seq);
}

void llama_vocab_shortlist_reset(llama_context * ctx, llama_seq_id seq_id) {
    ctx->vocab_shortlist_reset(seq_id);
}

uint32_t llama_vocab_shortlist_add(llama_context * ctx, llama_seq_id seq_id, const llama_token * tokens, size_t n) {
    return ctx->vocab_shortlist_add(seq_id, tokens, n);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
""",
        ),
        (
            "common/common.h",
            r"""    bool probabilistic = false; // sample the draft and verify by rejection, instead of argmax and match
""",
            r"""    bool probabilistic = false; // sample the draft and verify by rejection, instead of argmax and match

    std::string shortlist;
    int32_t shortlist_request_rows = 512;
""",
        ),
        (
            "common/arg.cpp",
            r"""    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
""",
            r"""    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
    add_opt(common_arg(
        {"--spec-draft-shortlist"}, "FNAME",
        "score DFlash drafts over this vocabulary shortlist (little-endian int32 token ids) plus the request's tokens",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.shortlist = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-shortlist-request-rows"}, "N",
        string_format("rows per sequence for request tokens outside the vocabulary shortlist (default: %d)", params.speculative.draft.shortlist_request_rows),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("shortlist request rows must not be negative");
            }
            params.speculative.draft.shortlist_request_rows = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
""",
        ),
        (
            "common/speculative.cpp",
            r"""    // block-internal attention
    bool causal_attn = false;
""",
            r"""    // block-internal attention
    bool causal_attn = false;

    bool shortlist_active = false;
    std::vector<size_t> shortlist_seen;
    std::vector<uint8_t> shortlist_warned;
""",
        ),
        (
            "common/speculative.cpp",
            r"""        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }
""",
            r"""        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise

        size_t n_embedded = 0;
        const int32_t * embedded = llama_model_dflash_vocab_shortlist(model_dft, &n_embedded);
        std::vector<int32_t> sidecar;
        if (!this->params.shortlist.empty()) {
            if (n_embedded > 0) {
                throw std::runtime_error("the drafter already carries a vocabulary shortlist; refusing --spec-draft-shortlist");
            }
            std::ifstream input(this->params.shortlist, std::ios::binary | std::ios::ate);
            if (!input) {
                throw std::runtime_error("cannot read the vocabulary shortlist " + this->params.shortlist);
            }
            const std::streamsize size = input.tellg();
            if (size <= 0 || size % (std::streamsize) sizeof(int32_t) != 0) {
                throw std::runtime_error("the vocabulary shortlist must hold little-endian int32 token ids");
            }
            sidecar.resize((size_t) size / sizeof(int32_t));
            input.seekg(0);
            input.read((char *) sidecar.data(), size);
            if (!input) {
                throw std::runtime_error("cannot read the vocabulary shortlist " + this->params.shortlist);
            }
        }
        const int32_t * shortlist_ids = n_embedded > 0 ? embedded : sidecar.data();
        const size_t n_shortlist = n_embedded > 0 ? n_embedded : sidecar.size();
        if (n_shortlist > 0) {
            if (is_dflash2 || is_lilicorr || is_dspark) {
                throw std::runtime_error("a vocabulary shortlist supports plain DFlash drafting only");
            }
            if (!llama_vocab_shortlist_init(ctx_dft, shortlist_ids, n_shortlist, (uint32_t) std::max(0, this->params.shortlist_request_rows))) {
                throw std::runtime_error("the vocabulary shortlist could not be attached to the draft context");
            }
            shortlist_active = true;
            shortlist_seen.assign(n_seq, 0);
            shortlist_warned.assign(n_seq, 0);
            LOG_INF("%s: - vocabulary shortlist: %zu tokens from %s, %d request rows per sequence\n", __func__,
                    n_shortlist, n_embedded > 0 ? "the drafter" : this->params.shortlist.c_str(), this->params.shortlist_request_rows);
        }
    }
""",
        ),
        (
            "common/speculative.cpp",
            r"""        // reseed per request rather than per round, or two identical requests differ
        dflash2_seed[seq_id] = -1;
""",
            r"""        // reseed per request rather than per round, or two identical requests differ
        dflash2_seed[seq_id] = -1;

        if (shortlist_active) {
            llama_vocab_shortlist_reset(params.ctx_dft, seq_id);
            shortlist_seen[seq_id] = 0;
            shortlist_warned[seq_id] = 0;
        }
""",
        ),
        (
            "common/speculative.cpp",
            r"""            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;
            const int32_t n_block_tokens = is_lilicorr ? block_size""",
            r"""            if (shortlist_active) {
                uint32_t refused = 0;
                if (dp.prompt) {
                    const llama_tokens & request = *dp.prompt;
                    if (request.size() < shortlist_seen[seq_id]) {
                        llama_vocab_shortlist_reset(ctx_dft, seq_id);
                        shortlist_seen[seq_id] = 0;
                    }
                    refused += llama_vocab_shortlist_add(ctx_dft, seq_id, request.data() + shortlist_seen[seq_id], request.size() - shortlist_seen[seq_id]);
                    shortlist_seen[seq_id] = request.size();
                }
                refused += llama_vocab_shortlist_add(ctx_dft, seq_id, &dp.id_last, 1);
                if (refused > 0 && params.shortlist_request_rows > 0 && !shortlist_warned[seq_id]) {
                    shortlist_warned[seq_id] = 1;
                    LOG_WRN("%s: seq_id %d has more than %d request tokens outside the vocabulary shortlist; later ones are not scored\n",
                            __func__, (int) seq_id, params.shortlist_request_rows);
                }
            }

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;
            const int32_t n_block_tokens = is_lilicorr ? block_size""",
        ),
    ),
    "vocabulary_shortlist_only_v1": (
        (
            "src/llama-cparams.h",
            r"""    std::vector<uint8_t> row;

    llama_vocab_shortlist() = default;
""",
            r"""    std::vector<uint8_t> row;
    ggml_tensor * table = nullptr;
    ggml_tensor * input_map = nullptr;
    const ggml_tensor * convert_source = nullptr;
    int64_t n_fixed = 0;
    int64_t n_request = 0;
    int64_t n_input = 0;
    size_t input_next = 0;
    std::vector<int32_t> row_of;
    std::vector<int32_t> input_token;
    std::vector<uint8_t> input_pinned;
    std::vector<uint8_t> source_row;
    std::vector<float> values;

    llama_vocab_shortlist() = default;
""",
        ),
        (
            "src/llama-model.h",
            r"""    std::vector<int32_t> dflash_vocab_shortlist;
""",
            r"""    std::vector<int32_t> dflash_vocab_shortlist;
    struct ggml_tensor * dflash_shortlist_rows = nullptr;
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""    const bool component_tied = component_tied_it != gguf_kv.end() && component_tied_it->second == "true";
    if (component_tied) {
""",
            r"""    const bool component_tied = component_tied_it != gguf_kv.end() && component_tied_it->second == "true";
    const auto shortlist_rows_it = gguf_kv.find("dflash.vocab_shortlist.rows_only");
    if (shortlist_rows_it != gguf_kv.end()) {
        if (shortlist_rows_it->second != "true" || !component_tied || output || tok_embd || n_vocab_draft != n_vocab ||
                dflash_vocab_shortlist.empty()) {
            throw std::runtime_error("invalid shortlist-only DFlash vocabulary table");
        }
        dflash_shortlist_rows = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "shortlist"), { n_embd, (int64_t) dflash_vocab_shortlist.size() }, 0);
        if (!ggml_is_quantized(dflash_shortlist_rows->type) || ggml_quantize_requires_imatrix(dflash_shortlist_rows->type)) {
            throw std::runtime_error("shortlist-only DFlash vocabulary rows need a quantized type without an importance matrix");
        }
        LLAMA_LOG_INFO("DFlash shortlist-only vocabulary: %zu %s rows, no full table\n", dflash_vocab_shortlist.size(), ggml_type_name(dflash_shortlist_rows->type));
    } else if (component_tied) {
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
""",
            r"""    const llama_vocab_shortlist * input_rows = cparams.vocab_shortlist;
    ggml_tensor * inpL = input_rows && input_rows->input_map
            ? ggml_get_rows(ctx0, input_rows->table, ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, input_rows->input_map, inp_tokens), n_tokens))
            : ggml_get_rows(ctx0, tok_embd, inp_tokens);
""",
        ),
        (
            "src/models/dflash.cpp",
            r"""    if (shortlist && shortlist->source == output && !output_s && !model.d2t""",
            r"""    if (shortlist && (shortlist->input_map || (shortlist->source == output && !output_s)) && !model.d2t""",
        ),
        (
            "src/llama-context.h",
            r"""    uint32_t vocab_shortlist_add(llama_seq_id seq_id, const llama_token * tokens, size_t n);
""",
            r"""    uint32_t vocab_shortlist_add(llama_seq_id seq_id, const llama_token * tokens, size_t n);
    bool vocab_shortlist_init_rows(const int32_t * ids, size_t n, uint32_t rows_per_seq);
    void vocab_shortlist_convert(llama_token token, int64_t row);
    void vocab_shortlist_map(llama_token token, int32_t row);
    bool vocab_shortlist_inputs(const llama_batch_ext & batch);
""",
        ),
        (
            "src/llama-context.cpp",
            r"""        LLAMA_LOG_ERROR("%s: a vocabulary shortlist needs a plain DFlash drafter\n", __func__);
        return false;
    }
    const ggml_tensor * source = model.output;
""",
            r"""        LLAMA_LOG_ERROR("%s: a vocabulary shortlist needs a plain DFlash drafter\n", __func__);
        return false;
    }
    if (model.dflash_shortlist_rows) {
        return vocab_shortlist_init_rows(ids, n, rows_per_seq);
    }
    const ggml_tensor * source = model.output;
""",
        ),
        (
            "src/llama-context.cpp",
            r"""void llama_context::vocab_shortlist_reset(llama_seq_id seq_id) {
""",
            r"""static void vocab_shortlist_convert_row(const ggml_tensor * source, ggml_type type, llama_token token,
        std::vector<uint8_t> & encoded, std::vector<float> & values, uint8_t * out) {
    const int64_t columns = source->ne[0];
    const size_t width = source->nb[1];
    encoded.resize(width);
    values.resize((size_t) columns);
    ggml_backend_tensor_get(source, encoded.data(), (size_t) token * width, width);
    if (!ggml_validate_row_data(source->type, encoded.data(), width)) {
        GGML_ABORT("vocabulary row %d of the conversion source is invalid", (int) token);
    }
    ggml_get_type_traits(source->type)->to_float(encoded.data(), values.data(), columns);
    const size_t written = ggml_quantize_chunk(type, values.data(), out, 0, 1, columns, nullptr);
    if (written != ggml_row_size(type, columns)) {
        GGML_ABORT("vocabulary row %d did not convert to one %s row", (int) token, ggml_type_name(type));
    }
}

void llama_context::vocab_shortlist_convert(llama_token token, int64_t row) {
    auto & shortlist = *vocab_shortlist;
    vocab_shortlist_convert_row(shortlist.convert_source, shortlist.table->type, token, shortlist.source_row, shortlist.values, shortlist.row.data());
    ggml_backend_tensor_set(shortlist.table, shortlist.row.data(), (size_t) row * shortlist.row_bytes, shortlist.row_bytes);
}

void llama_context::vocab_shortlist_map(llama_token token, int32_t row) {
    auto & shortlist = *vocab_shortlist;
    shortlist.row_of[(size_t) token] = row;
    const int32_t value = row < 0 ? 0 : row;
    ggml_backend_tensor_set(shortlist.input_map, &value, (size_t) token * sizeof(int32_t), sizeof(int32_t));
}

bool llama_context::vocab_shortlist_init_rows(const int32_t * ids, size_t n, uint32_t rows_per_seq) {
    const ggml_tensor * rows_source = model.dflash_shortlist_rows;
    const llama_model * other = cparams.ctx_other ? llama_get_model(cparams.ctx_other) : nullptr;
    const ggml_tensor * convert_source = other ? other->tok_embd : nullptr;
    const int64_t n_vocab = (int64_t) model.vocab.n_tokens();
    const int64_t columns = rows_source->ne[0];
    if (convert_source == nullptr || convert_source->buffer == nullptr || !ggml_is_contiguous(convert_source) ||
            convert_source->ne[0] != columns || convert_source->ne[1] != n_vocab || convert_source->ne[2] != 1 || convert_source->ne[3] != 1 ||
            ggml_get_type_traits(convert_source->type)->to_float == nullptr ||
            convert_source->nb[1] != ggml_row_size(convert_source->type, columns)) {
        LLAMA_LOG_ERROR("%s: a shortlist-only drafter needs the target's token embedding table to convert rows\n", __func__);
        return false;
    }
    if (n != model.dflash_vocab_shortlist.size() || rows_source->ne[1] != (int64_t) n ||
            !std::equal(ids, ids + n, model.dflash_vocab_shortlist.begin())) {
        LLAMA_LOG_ERROR("%s: the shortlist differs from the drafter's shortlist rows\n", __func__);
        return false;
    }
    const int64_t n_fixed = (int64_t) n;
    const int64_t capacity = (int64_t) rows_per_seq * (int64_t) cparams.n_seq_max;
    const int64_t n_input = std::max<int64_t>((int64_t) cparams.n_batch, 1);
    if (n_fixed < 1 || n_fixed > n_vocab || cparams.n_seq_max > 64 || capacity > n_vocab - n_fixed) {
        LLAMA_LOG_ERROR("%s: invalid vocabulary shortlist size %" PRId64 " or request rows %" PRId64 "\n", __func__, n_fixed, capacity);
        return false;
    }
    const ggml_type type = rows_source->type;
    const size_t row_bytes = ggml_row_size(type, columns);
    if (!ggml_is_contiguous(rows_source) || rows_source->nb[1] != row_bytes || rows_source->ne[2] != 1 || rows_source->ne[3] != 1) {
        LLAMA_LOG_ERROR("%s: the shortlist rows are not contiguous\n", __func__);
        return false;
    }
    if (model.output_norm == nullptr || model.output_norm->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: the drafter has no placed output layer\n", __func__);
        return false;
    }
    auto shortlist = std::make_unique<llama_vocab_shortlist>();
    shortlist->fixed.assign((size_t) n_vocab, 0);
    shortlist->row_of.assign((size_t) n_vocab, -1);
    for (int64_t i = 0; i < n_fixed; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= n_vocab || shortlist->fixed[(size_t) id]) {
            LLAMA_LOG_ERROR("%s: vocabulary shortlist token %d is out of range or repeated\n", __func__, id);
            return false;
        }
        shortlist->fixed[(size_t) id] = 1;
        shortlist->row_of[(size_t) id] = (int32_t) i;
    }
    ggml_init_params init_params = { 8 * ggml_tensor_overhead(), nullptr, true };
    shortlist->ctx = ggml_init(init_params);
    if (shortlist->ctx == nullptr) {
        return false;
    }
    shortlist->table = ggml_new_tensor_2d(shortlist->ctx, type, columns, n_fixed + capacity + n_input);
    ggml_set_name(shortlist->table, "output_shortlist.table");
    shortlist->rows = ggml_view_2d(shortlist->ctx, shortlist->table, columns, n_fixed, shortlist->table->nb[1], 0);
    ggml_set_name(shortlist->rows, "output_shortlist.weight");
    shortlist->ids = ggml_new_tensor_1d(shortlist->ctx, GGML_TYPE_I64, n_fixed);
    ggml_set_name(shortlist->ids, "output_shortlist.ids");
    if (capacity > 0) {
        shortlist->request_rows = ggml_view_2d(shortlist->ctx, shortlist->table, columns, capacity, shortlist->table->nb[1], (size_t) n_fixed * shortlist->table->nb[1]);
        ggml_set_name(shortlist->request_rows, "output_request.weight");
        shortlist->request_ids = ggml_new_tensor_1d(shortlist->ctx, GGML_TYPE_I64, capacity);
        ggml_set_name(shortlist->request_ids, "output_request.ids");
    }
    shortlist->input_map = ggml_new_tensor_2d(shortlist->ctx, GGML_TYPE_I32, 1, n_vocab);
    ggml_set_name(shortlist->input_map, "output_shortlist.input_map");
    shortlist->buffer = ggml_backend_alloc_ctx_tensors_from_buft(shortlist->ctx, ggml_backend_buffer_get_type(model.output_norm->buffer));
    if (shortlist->buffer == nullptr) {
        LLAMA_LOG_ERROR("%s: cannot allocate the vocabulary shortlist\n", __func__);
        return false;
    }
    ggml_backend_buffer_set_usage(shortlist->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(shortlist->buffer, 0);
    ggml_backend_tensor_copy(rows_source, shortlist->rows);
    {
        std::vector<int64_t> fixed_ids(ids, ids + n_fixed);
        ggml_backend_tensor_set(shortlist->ids, fixed_ids.data(), 0, fixed_ids.size() * sizeof(int64_t));
    }
    if (capacity > 0) {
        shortlist->slot_of.assign((size_t) n_vocab, -1);
        shortlist->slot_token.assign((size_t) capacity, -1);
        shortlist->slot_seqs.assign((size_t) capacity, 0);
        std::vector<uint8_t> padding((size_t) capacity * row_bytes);
        std::vector<int64_t> padding_ids((size_t) capacity);
        int64_t slot = 0;
        for (int64_t token = 0; token < n_vocab && slot < capacity; ++token) {
            if (shortlist->fixed[(size_t) token]) {
                continue;
            }
            vocab_shortlist_convert_row(convert_source, type, (llama_token) token, shortlist->source_row, shortlist->values, padding.data() + (size_t) slot * row_bytes);
            padding_ids[(size_t) slot] = token;
            shortlist->slot_token[(size_t) slot] = (int32_t) token;
            shortlist->slot_of[(size_t) token] = (int32_t) slot;
            shortlist->row_of[(size_t) token] = (int32_t) (n_fixed + slot);
            ++slot;
        }
        ggml_backend_tensor_set(shortlist->request_rows, padding.data(), 0, padding.size());
        ggml_backend_tensor_set(shortlist->request_ids, padding_ids.data(), 0, padding_ids.size() * sizeof(int64_t));
    }
    {
        std::vector<int32_t> map((size_t) n_vocab);
        for (int64_t token = 0; token < n_vocab; ++token) {
            map[(size_t) token] = std::max(shortlist->row_of[(size_t) token], 0);
        }
        ggml_backend_tensor_set(shortlist->input_map, map.data(), 0, map.size() * sizeof(int32_t));
    }
    shortlist->convert_source = convert_source;
    shortlist->n_vocab = n_vocab;
    shortlist->row_bytes = row_bytes;
    shortlist->rows_per_seq = rows_per_seq;
    shortlist->seq_rows.assign(cparams.n_seq_max, 0);
    shortlist->row.resize(row_bytes);
    shortlist->n_fixed = n_fixed;
    shortlist->n_request = capacity;
    shortlist->n_input = n_input;
    shortlist->input_token.assign((size_t) n_input, -1);
    shortlist->input_pinned.assign((size_t) n_input, 0);
    vocab_shortlist = std::move(shortlist);
    cparams.vocab_shortlist = vocab_shortlist.get();
    sched_need_reserve = true;
    LLAMA_LOG_INFO("%s: vocabulary shortlist of %" PRId64 " %s rows from the shortlist-only table, %" PRId64 " request rows and %" PRId64 " input rows converted from the target's %s %s\n",
            __func__, n_fixed, ggml_type_name(type), capacity, n_input, ggml_type_name(convert_source->type), convert_source->name);
    return true;
}

bool llama_context::vocab_shortlist_inputs(const llama_batch_ext & batch) {
    if (model.dflash_shortlist_rows == nullptr) {
        return true;
    }
    if (!vocab_shortlist || !vocab_shortlist->input_map) {
        LLAMA_LOG_ERROR("%s: a shortlist-only drafter decodes only with its vocabulary shortlist attached\n", __func__);
        return false;
    }
    auto & shortlist = *vocab_shortlist;
    const int64_t first_input = shortlist.n_fixed + shortlist.n_request;
    std::fill(shortlist.input_pinned.begin(), shortlist.input_pinned.end(), 0);
    std::vector<llama_token> missing;
    for (const auto & tok : batch.tokens) {
        const llama_token token = tok.id;
        if (token < 0 || token >= shortlist.n_vocab) {
            continue;
        }
        const int32_t row = shortlist.row_of[(size_t) token];
        if (row >= first_input) {
            shortlist.input_pinned[(size_t) (row - first_input)] = 1;
        } else if (row < 0) {
            bool seen = false;
            for (const llama_token other : missing) {
                seen = seen || other == token;
            }
            if (!seen) {
                missing.push_back(token);
            }
        }
    }
    if (missing.empty()) {
        return true;
    }
    if ((int64_t) missing.size() > shortlist.n_input) {
        LLAMA_LOG_ERROR("%s: %zu input tokens outside the vocabulary rows exceed %" PRId64 " input rows\n", __func__, missing.size(), shortlist.n_input);
        return false;
    }
    ggml_backend_sched_synchronize(sched.get());
    for (const llama_token token : missing) {
        size_t slot = shortlist.input_next;
        while (shortlist.input_pinned[slot]) {
            slot = (slot + 1) % (size_t) shortlist.n_input;
        }
        shortlist.input_next = (slot + 1) % (size_t) shortlist.n_input;
        shortlist.input_pinned[slot] = 1;
        const int32_t row = (int32_t) (first_input + (int64_t) slot);
        const llama_token evicted = shortlist.input_token[slot];
        if (evicted >= 0 && shortlist.row_of[(size_t) evicted] == row) {
            vocab_shortlist_map(evicted, -1);
        }
        vocab_shortlist_convert(token, row);
        vocab_shortlist_map(token, row);
        shortlist.input_token[slot] = token;
    }
    return true;
}

void llama_context::vocab_shortlist_reset(llama_seq_id seq_id) {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""            ggml_backend_tensor_get(shortlist.source, shortlist.row.data(), (size_t) token * shortlist.row_bytes, shortlist.row_bytes);
            ggml_backend_tensor_set(shortlist.request_rows, shortlist.row.data(), free_slot * shortlist.row_bytes, shortlist.row_bytes);
""",
            r"""            if (shortlist.input_map) {
                const llama_token evicted = shortlist.slot_token[free_slot];
                const int32_t slot_row = (int32_t) (shortlist.n_fixed + (int64_t) free_slot);
                if (evicted >= 0 && shortlist.row_of[(size_t) evicted] == slot_row) {
                    vocab_shortlist_map(evicted, -1);
                }
                vocab_shortlist_convert(token, slot_row);
                vocab_shortlist_map(token, slot_row);
            } else {
                ggml_backend_tensor_get(shortlist.source, shortlist.row.data(), (size_t) token * shortlist.row_bytes, shortlist.row_bytes);
                ggml_backend_tensor_set(shortlist.request_rows, shortlist.row.data(), free_slot * shortlist.row_bytes, shortlist.row_bytes);
            }
""",
        ),
        (
            "src/llama-context.cpp",
            r"""        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }
""",
            r"""        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    if (!vocab_shortlist_inputs(batch_inp)) {
        return -1;
    }
""",
        ),
    ),
}
