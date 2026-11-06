#pragma once
#include "ggml.h"
#include <map>
#include <string>
#include <stdexcept>
#include <limits>

// Decoded-F32 reference graph. No claim of packed-W1 storage or bitwise
// equivalence of ordinary GGML/CUBLAS reductions to the training actor.
namespace binary_reference {
using tensors = std::map<std::string, ggml_tensor *>;
enum kind { parent = 1, mos = 2, mos_static = 3, factors = 4, ternary = 5 };
inline int parse_kind(const std::string & name) {
    if (name == "decoded-parent") return parent;
    if (name == "mos") return mos;
    if (name == "mos-static") return mos_static;
    if (name == "double-binary") return factors;
    if (name == "ternary-a8") return ternary;
    throw std::runtime_error("unknown binary reference kind");
}
inline ggml_tensor * weight(const tensors & ts, const std::string & p, const char * suffix) {
    const auto it = ts.find(p + ".binary_" + suffix);
    if (it == ts.end()) throw std::runtime_error("missing binary reference tensor: " + p + suffix);
    return it->second;
}
inline ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    auto * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}
inline ggml_tensor * factor_out_f32_columns(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    // N>=2 may select native MMF/TF32 despite F32 accumulation. N=1 selects
    // the existing F32 MMVF vector path. Keep every token separate, then
    // concatenate in original token order; charge all launches/copies.
    GGML_ASSERT(w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[1] > 0 && x->ne[2] == 1 && x->ne[3] == 1);
    ggml_tensor * result = nullptr;
    for (int64_t column = 0; column < x->ne[1]; ++column) {
        auto * token = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], size_t(column)*x->nb[1]);
        auto * output = mm(ctx, w, token);
        result = result ? ggml_concat(ctx, result, output, 1) : output;
    }
    return result;
}
struct activation { ggml_tensor * codes; ggml_tensor * beta; };
inline activation a8(ggml_context * ctx, ggml_tensor * x) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[2] == 1 && x->ne[3] == 1);
    x = ggml_cont(ctx, x);
    auto * maximum = ggml_pool_1d(ctx, ggml_abs(ctx, x), GGML_OP_POOL_MAX,
                                 int(x->ne[0]), int(x->ne[0]), 0);
    // Exact zero rows remain zero. Subnormal rows use the explicit reference
    // FLT_MIN clamp, not the native packer's F64 overflow fallback.
    auto * zero = ggml_scale_bias(ctx, ggml_step(ctx, maximum), -1.0f, 1.0f);
    auto * safe = ggml_clamp(ctx, ggml_add(ctx, maximum, zero),
                            std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    auto * inverse = ggml_div(ctx, ggml_scale_bias(ctx, safe, 0.0f, 127.0f), safe);
    auto * normalized = ggml_mul(ctx, x, inverse);
    auto * lower = ggml_floor(ctx, normalized);
    auto * fraction = ggml_sub(ctx, normalized, lower);
    auto * tie = ggml_scale_bias(ctx, ggml_abs(ctx, ggml_sgn(ctx,
                    ggml_scale_bias(ctx, fraction, 1.0f, -0.5f))), -1.0f, 1.0f);
    auto * odd = ggml_sub(ctx, lower, ggml_scale(ctx, ggml_floor(ctx, ggml_scale(ctx, lower, 0.5f)), 2.0f));
    // Fraction-based RNE avoids double rounding z+.5 (the F32 predecessor
    // of .5 would otherwise round upward to1 before floor).
    auto * rounded = ggml_add(ctx, ggml_add(ctx, lower,
            ggml_step(ctx, ggml_scale_bias(ctx, fraction, 1.0f, -0.5f))), ggml_mul(ctx, tie, odd));
    auto * codes = ggml_clamp(ctx, rounded, -127.0f, 127.0f);
    auto * beta = ggml_div(ctx, maximum, ggml_scale_bias(ctx, maximum, 0.0f, 127.0f));
    return {codes, beta};
}
inline ggml_tensor * graph(ggml_context * ctx, const tensors & ts, const std::string & p,
                           ggml_tensor * input, int mode) {
    auto * x = input;
    auto * output_scale = weight(ts, p, "scales");
    if (mode == mos || mode == mos_static) {
        auto * router = weight(ts, p, "router");
        ggml_tensor * logits;
        if (mode == mos_static) {
            logits = ggml_reshape_2d(ctx, ggml_mean(ctx, router), 2, 1);
            logits = ggml_repeat_4d(ctx, logits, 2, x->ne[1], 1, 1);
        } else {
            logits = mm(ctx, router, x);
        }
        auto * gates = ggml_soft_max(ctx, logits);
        auto * inscale = mm(ctx, weight(ts, p, "input_experts"), gates);
        auto * offset = mm(ctx, weight(ts, p, "output_offsets"), gates);
        output_scale = ggml_mul(ctx, ggml_scale_bias(ctx, offset, 1.0f, 1.0f), output_scale);
        x = ggml_mul(ctx, x, inscale);
    } else if (mode == factors) {
        x = ggml_mul(ctx, x, weight(ts, p, "input_scale"));
    }
    const auto packed = a8(ctx, x);
    auto * dot = mm(ctx, weight(ts, p, "signs"), packed.codes);
    if (mode == factors) {
        auto * intermediate = ggml_mul(ctx, dot, packed.beta);
        dot = factor_out_f32_columns(ctx, weight(ts, p, "factor_out"), intermediate);
        return ggml_mul(ctx, dot, output_scale);
    }
    // Preserve the trained scale order: integer dot * token/row output scale
    // THEN activation beta; input scaling always precedes A8 quantization.
    return ggml_mul(ctx, ggml_mul(ctx, dot, output_scale), packed.beta);
}
} // namespace binary_reference
