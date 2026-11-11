#pragma once

#include "llama.h"
#include "json.h"
#include "log.h"
#include <cmath>
#include <stdexcept>
#include <vector>

inline void binary_verifier_trace(
    llama_context * ctx,
    const std::vector<int> & indices,
    const std::vector<llama_token> & proposed,
    const std::vector<llama_token> & accepted,
    bool replay,
    bool rejection,
    bool committed) {
    const auto * vocab = llama_model_get_vocab(llama_get_model(ctx));
    const int count = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> targets;
    std::vector<llama_token> eog;
    for (int token = 0; token < count; ++token) {
        if (llama_vocab_is_eog(vocab, token)) eog.push_back(token);
    }
    for (int index : indices) {
        const float * logits = llama_get_logits_ith(ctx, index);
        if (!logits) throw std::runtime_error("verifier logits unavailable");
        int best = 0;
        for (int token = 0; token < count; ++token) {
            if (!std::isfinite(logits[token])) throw std::runtime_error("nonfinite verifier logits");
            if (logits[token] > logits[best]) best = token;
        }
        targets.push_back(best);
    }
    common_json result = {
        {"proposed", proposed}, {"accepted_tokens", accepted}, {"target_argmax", targets},
        {"eog_ids", eog}, {"replay", replay}, {"rejection", rejection}, {"committed", committed},
    };
    LOG_INF("BINARY_VERIFIER_TRACE %s\n", result.dump().c_str());
}
