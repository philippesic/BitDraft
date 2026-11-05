// Independent fixed-A8 pack/integer-dot checks; CPU reference and actual CUDA.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <xmmintrin.h>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Case { int k, m, n; };

static void run_case(ggml_backend_t backend, const Case shape, const char * name, int bits, int mode, bool fused_gain) {
    const int qmax=bits<=2?1:(1<<(bits-1))-1; const float clip=1.25f;
    const int k = shape.k, m = shape.m, n = shape.n;
    std::vector<float> acts(size_t(k)*n), scales(m),midpoints(m),gains(k);
    for(int j=0;j<k;++j)gains[j]=float(1+j%5)/2.f;
    std::vector<uint32_t> signs(size_t(k/32)*m);
    for (int row = 0; row < m; ++row) {
        scales[row] = float(1 + row%13)/16.0f; midpoints[row]=float(row%7-3)/32.f;
        for (int word = 0; word < k/32; ++word) {
            signs[size_t(row)*(k/32) + word] = 0xa5f04b39u ^ (uint32_t(row)*0x9e3779b9u + uint32_t(word)*0x45d9f3bu);
        }
    }
    for (int token = 0; token < n; ++token) {
        for (int j = 0; j < k; ++j) {
            float value;
            switch (token%4) {
                case 0: value = token == 0 && n > 1 ? 0.0f : std::sin(float(j)*.137f); break;
                case 1: value = j == 0 ? 127.0f : float(j%15 - 7) + .5f; break; // exact RNE ties
                case 2: {
                    // Construct actual subnormal bits without host FTZ loss.
                    const uint32_t bits = uint32_t(1 + j%101) | (j%2 ? 0x80000000u : 0u);
                    std::memcpy(&value, &bits, sizeof(value));
                } break;
                default: value = std::cos(float(j)*.019f)*float(1 + token); break;
            }
            acts[size_t(token)*k + j] = value;
        }
    }
    const auto raw=acts;
    for(int token=0;token<n;++token)for(int j=0;j<k;++j)acts[size_t(token)*k+j]*=gains[j];
    std::vector<int8_t> expected_codes(size_t(k)*n);
    std::vector<float> beta(n);
    for (int token = 0; token < n; ++token) {
        float maximum = 0;
        for (int j = 0; j < k; ++j) maximum = std::max(maximum, std::abs(acts[size_t(token)*k + j]));
        if(mode) maximum=clip;
        beta[token] = maximum/float(qmax);
        const float inverse = maximum == 0 ? 0 : float(qmax)/maximum;
        for (int j = 0; j < k; ++j) {
            const float value = acts[size_t(token)*k + j];
            const float normalized = maximum == 0 ? 0 : std::isfinite(inverse) ? value*inverse :
                float(double(value)/double(maximum)*double(qmax));
            expected_codes[size_t(token)*k + j] = int8_t(bits==1 ? (value>=0?1:-1) : std::clamp(int(std::nearbyint(normalized)), -qmax, qmax));
        }
    }
    std::vector<float> expected(size_t(m)*n);
    for (int token = 0; token < n; ++token) {
        for (int row = 0; row < m; ++row) {
            int64_t sum = 0;
            for (int j = 0; j < k; ++j) {
                const int sign = (signs[size_t(row)*(k/32) + j/32] >> (j%32)) & 1u ? 1 : -1;
                sum += sign*int(expected_codes[size_t(token)*k + j]);
            }
            const float weighted = float(sum)*scales[row];
            int qsum=0; for(int j=0;j<k;++j)qsum+=expected_codes[size_t(token)*k+j];
            const float middle=float(qsum)*beta[token];
            expected[size_t(token)*m + row] = weighted*beta[token]+midpoints[row]*middle;
        }
    }
    ggml_init_params params = {ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation failed");
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k/32, m);
    ggml_tensor * alpha = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * mid = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,m);
    ggml_tensor * gain=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);
    ggml_tensor * pack = ggml_w1ax_pack(ctx, fused_gain?a:ggml_mul(ctx,a,gain), bits);
    if(fused_gain)pack->src[1]=gain;
    memcpy((char*)pack->op_params+4,&mode,4);memcpy((char*)pack->op_params+8,&clip,4);
    ggml_tensor * out = ggml_w1ax_mul_mat_shared(ctx, w, alpha, pack, k, bits);
    out->src[4]=mid;
    require(ggml_backend_supports_op(backend, pack) && ggml_backend_supports_op(backend, out), "backend rejected valid fixed W1A8");
    // Admission negatives are metadata-only: no invalid GPU operation executes.
    ggml_tensor bad_pack = *pack;
    int bad_bits = 3;
    std::memcpy(bad_pack.op_params, &bad_bits, sizeof(bad_bits));
    require(!ggml_backend_supports_op(backend, &bad_pack), "A3 pack was accepted");
    ggml_tensor bad_scale = *alpha, bad_out = *out;
    bad_scale.type = GGML_TYPE_F16;
    bad_out.src[1] = &bad_scale;
    require(!ggml_backend_supports_op(backend, &bad_out), "F16 scales were accepted");
    ggml_tensor bad_acts = *a;
    bad_acts.ne[0] = k - 1;
    bad_pack = *pack; bad_pack.src[0] = &bad_acts;
    require(!ggml_backend_supports_op(backend, &bad_pack), "unaligned K was accepted");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend tensor allocation failed");
    ggml_backend_tensor_set(w, signs.data(), 0, signs.size()*sizeof(uint32_t));
    ggml_backend_tensor_set(alpha, scales.data(), 0, scales.size()*sizeof(float));
    ggml_backend_tensor_set(mid,midpoints.data(),0,midpoints.size()*4);
    ggml_backend_tensor_set(gain,gains.data(),0,gains.size()*4);
    ggml_backend_tensor_set(a, raw.data(), 0, raw.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend graph failed");
    ggml_backend_synchronize(backend);
    std::vector<uint32_t> packed_words(size_t(k)*n/4 + n);
    std::vector<float> observed(size_t(m)*n);
    ggml_backend_tensor_get(pack, packed_words.data(), 0, packed_words.size()*sizeof(uint32_t));
    ggml_backend_tensor_get(out, observed.data(), 0, observed.size()*sizeof(float));
    require(std::memcmp(packed_words.data(), expected_codes.data(), expected_codes.size()) == 0, "native A8 codes differ from independent reference");
    require(std::memcmp(packed_words.data() + size_t(k)*n/4, beta.data(), n*sizeof(float)) == 0, "native A8 scales differ from independent reference");
    require(std::memcmp(observed.data(), expected.data(), expected.size()*sizeof(float)) == 0, "native integer-dot products differ from independent reference");
    std::cout << "{\"backend\":\"" << name << "\",\"K\":" << k << ",\"M\":" << m
              << ",\"bits\":" << bits << ",\"mode\":" << mode << ",\"N\":" << n << ",\"gained_codes_scales_products_byteexact\":true,\"admission_negatives_pass\":true}" << std::endl;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: op_check OUTPUT_LIBRARY_MAPS generic|fused");
        const bool fused_gain=std::string(argv[2])=="fused";
        std::fesetround(FE_TONEAREST);
        _mm_setcsr(_mm_getcsr() & ~(unsigned(1 << 15) | unsigned(1 << 6)));
        ggml_backend_t cuda = ggml_backend_cuda_init(0);
        require(cuda,"real CUDA required");
        for(int bits:{1,2,4,8}) for(int mode:{0,1,2})
        for(const Case shape:{Case{32,7,4},Case{2560,9216,4},Case{9216,2560,4}}) run_case(cuda,shape,"CUDA",bits,mode,fused_gain);
        std::ifstream maps("/proc/self/maps");
        std::ofstream receipt(argv[1]); receipt << maps.rdbuf();
        require(bool(receipt), "library maps write failed");
        ggml_backend_free(cuda);
        std::cout << "{\"status\":\"CUDA_AFFINE_PRECISION_OP_PASS\",\"model_qualified\":false}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "op_check: " << error.what() << std::endl;
        return 1;
    }
}
