#include "llama.h"
#include "sampling.h"
#include "json.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("usage: binary-verifier-test MODEL TRIALS");
        const int trials = std::stoi(argv[2]);
        if (trials < 10000 || trials > 100000) throw std::runtime_error("bounded trials required");
        llama_backend_init();
        auto model_params = llama_model_default_params();
        const std::string device = argc == 4 ? argv[3] : "cuda";
        if (device != "cpu" && device != "cuda") throw std::runtime_error("device must be cpu or cuda");
        model_params.n_gpu_layers = device == "cuda" ? 99 : 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("target model load failed");
        auto context_params = llama_context_default_params();
        context_params.n_ctx = 512;
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_threads = 2;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), context_params), llama_free);
        if (!ctx) throw std::runtime_error("target context failed");
        llama_token input = 1;
        if (llama_decode(ctx.get(), llama_batch_get_one(&input, 1)) != 0) throw std::runtime_error("target prefill failed");
        auto * logits = llama_get_logits_ith(ctx.get(), 0);
        const auto * vocab = llama_model_get_vocab(model.get());
        const int size = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> ids;
        for (int id = 0; id < size && ids.size() < 3; ++id) {
            if (!llama_vocab_is_eog(vocab, id)) ids.push_back(id);
        }
        common_json rows = common_json::array();
        const std::vector<std::pair<std::array<float, 3>, std::array<float, 3>>> cases = {
            {{0.55f, 0.35f, 0.10f}, {0.10f, 0.70f, 0.20f}},
            {{0.55f, 0.35f, 0.10f}, {1.00f, 0.00f, 0.00f}},
            {{0.55f, 0.35f, 0.10f}, {0.55f, 0.35f, 0.10f}},
            {{0.00f, 0.60f, 0.40f}, {0.50f, 0.50f, 0.00f}},
        };
        bool passed = true;
        for (const auto & [p, q] : cases) {
          for (bool rejection : {false, true}) {
            std::fill(logits, logits + size, -std::numeric_limits<float>::infinity());
            for (int i = 0; i < 3; ++i) if (p[i] > 0) logits[ids[i]] = std::log(p[i]);
            common_params_sampling params;
            params.seed = 8101;
            params.temp = 1.0f;
            params.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
            std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), params), common_sampler_free);
            if (!sampler) throw std::runtime_error("sampler initialization failed");
            std::mt19937 generator(9101);
            std::discrete_distribution<int> propose(q.begin(), q.end());
            std::vector<llama_token_data> distribution;
            for (int i = 0; i < 3; ++i) if (q[i] > 0) distribution.push_back({ids[i], std::log(q[i]), q[i]});
            std::array<int, 3> observed = {0, 0, 0};
            int accepted = 0;
            for (int trial = 0; trial < trials; ++trial) {
                const auto draft = ids[propose(generator)];
                const auto result = rejection ? common_sampler_sample_and_accept_n_rejection(sampler.get(), ctx.get(), {0, 0}, {draft}, {distribution}) : common_sampler_sample_and_accept_n(sampler.get(), ctx.get(), std::vector<int>{0, 0}, std::vector<llama_token>{draft}, false);
                if (result.empty()) throw std::runtime_error("rejection sampler made no progress");
                auto found = std::find(ids.begin(), ids.end(), result.front());
                if (found == ids.end()) throw std::runtime_error("sample outside target support");
                observed[found - ids.begin()]++;
                accepted += result.size() == 2;
            }
            float acceptance = 0;
            for (int i = 0; i < 3; ++i) {
                acceptance += rejection ? std::min(p[i], q[i]) : p[i] * q[i];
                passed = passed && std::abs(float(observed[i]) / trials - p[i]) <= 0.035f;
            }
            passed = passed && std::abs(float(accepted) / trials - acceptance) <= 0.035f;
            rows.push_back({{"algorithm", rejection ? "rejection" : "sample-and-match"}, {"p", common_json::array({p[0],p[1],p[2]})}, {"q", common_json::array({q[0],q[1],q[2]})}, {"observed", common_json::array({observed[0],observed[1],observed[2]})}, {"accepted", accepted}, {"trials", trials}, {"expected_acceptance", acceptance}});
        }
          }
        llama_token eos = -1;
        for (int id = 0; id < size; ++id) if (llama_vocab_is_eog(vocab, id)) {eos = id; break;}
        if (eos < 0) throw std::runtime_error("target has no EOS");
        std::fill(logits, logits + size, -std::numeric_limits<float>::infinity());
        logits[eos] = 0;
        common_params_sampling params;
        params.seed = 8101;
        params.temp = 1.0f;
        params.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler(common_sampler_init(model.get(), params), common_sampler_free);
        const std::vector<llama_token_data> distribution = {{eos, 0, 1}};
        const auto ending = common_sampler_sample_and_accept_n_rejection(sampler.get(), ctx.get(), {0, 0, 0}, {eos, eos}, {distribution, distribution});
        const auto matching = common_sampler_sample_and_accept_n(sampler.get(), ctx.get(), std::vector<int>{0, 0, 0}, std::vector<llama_token>{eos, eos}, false);
        const bool stopped = ending.size() == 1 && ending.front() == eos && matching.size() == 1 && matching.front() == eos;
        passed = passed && stopped;
        std::cout << common_json({{"status", passed ? "passed" : "failed"}, {"distribution_cases", rows}, {"eos_stopped", stopped}, {"eos_returned_tokens", ending}, {"match_eos_returned_tokens", matching}, {"device", device}, {"scope", "actual common rejection and sample-and-match samplers with controlled target logits; neural drafter behavior requires separate native requests"}}).dump() << std::endl;
        return passed ? 0 : 2;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
