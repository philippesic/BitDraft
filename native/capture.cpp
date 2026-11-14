#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

void llama_set_embeddings_layer_inp(llama_context *, uint32_t, bool);
float * llama_get_embeddings_layer_inp(llama_context *, uint32_t);
namespace fs = std::filesystem;
using json = nlohmann::json;
static void require(bool okay, const std::string & text) {
    if (!okay) throw std::runtime_error(text);
}
static ggml_backend_reg_t backend(const fs::path & path, const char * symbol) {
    auto expected = fs::canonical(path);
    void * handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_LOCAL);
    require(handle != nullptr, "cannot open required backend");
    void * address = dlsym(handle, symbol);
    Dl_info info = {};
    require(address && dladdr(address, &info) && info.dli_fname && fs::canonical(info.dli_fname) == expected,
            "backend resolved to another module");
    return reinterpret_cast<ggml_backend_reg_t (*)()>(address)();
}
static void write_features(std::ofstream & file, const float * values, size_t size, bool half) {
    require(values != nullptr, "missing feature tap");
    for (size_t i = 0; i < size; ++i) require(std::isfinite(values[i]), "nonfinite feature");
    if (half) {
        std::vector<ggml_fp16_t> rounded(size);
        ggml_fp32_to_fp16_row(values, rounded.data(), size);
        for (auto value : rounded) require(std::isfinite(ggml_fp16_to_fp32(value)), "FP16 feature overflow");
        file.write(reinterpret_cast<const char *>(rounded.data()), size * 2);
    } else file.write(reinterpret_cast<const char *>(values), size * 4);
    require(bool(file), "feature write failed");
}
int main(int argc, char ** argv) {
    try {
        require(argc == 5, "capture TARGET CPU_LIBRARY CUDA_LIBRARY CONTEXT; requests on stdin");
        const char * consistent = std::getenv("GGML_CUDA_DECODE_CONSISTENT");
        require(consistent && std::string(consistent) == "32", "decode consistency must be 32");
        auto cpu = backend(argv[2], "ggml_backend_cpu_reg");
        auto cuda = backend(argv[3], "ggml_backend_cuda_reg");
        require(ggml_backend_reg_by_name("CPU") == cpu && ggml_backend_reg_by_name("CUDA") == cuda,
                "backend registry identity mismatch");
        auto gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(gpu && ggml_backend_dev_backend_reg(gpu) == cuda, "GPU backend required; fallback prohibited");
        llama_backend_init();
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices; mp.n_gpu_layers = -1; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        char architecture[128] = {};
        llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
        require(std::string(architecture) == "qwen35" && llama_model_n_embd(model.get()) == 2560 &&
                llama_model_n_layer(model.get()) == 32, "only qualified Qwen3.5-4B geometry supported");
        const int context = std::stoi(argv[4]);
        require(context >= 1024 && context <= 32768, "context outside qualified bounds");
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.n_outputs_max = 1; cp.n_outputs_max_per_seq = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE; cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        cp.embeddings = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx) && int(llama_n_ctx(ctx.get())) == context, "context creation mismatch");
        const std::vector<int> taps = {2,6,10,14,18,22,26,30};
        for (int tap : taps) llama_set_embeddings_layer_inp(ctx.get(), tap, true);
        auto vocab = llama_model_get_vocab(model.get());
        const int vocabulary = llama_vocab_n_tokens(vocab);
        std::vector<int> eog;
        for (int token = 0; token < vocabulary; ++token) if (llama_vocab_is_eog(vocab, token)) eog.push_back(token);
        std::cout << json({{"ready",true},{"gpu",ggml_backend_dev_description(gpu)}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            require(line.size() <= 2 * 1024 * 1024, "capture request too large");
            const auto request = json::parse(line);
            std::vector<llama_token> inputs = request.at("tokens").get<std::vector<llama_token>>();
            const int prompt = inputs.size(), cap = request.at("output_cap").get<int>();
            const fs::path output = request.at("output").get<std::string>();
            const std::string dtype = request.at("feature_dtype").get<std::string>();
            const bool half = dtype == "float16";
            require(half || dtype == "float32", "unsupported feature dtype");
            std::vector<std::pair<int,int>> spans;
            if (request.contains("spans")) spans = request.at("spans").get<std::vector<std::pair<int,int>>>();
            int previous = -1;
            for (auto [first, last] : spans) {
                require(previous < first && first <= last && last + 1 < prompt, "invalid supervised span");
                previous = last;
            }
            require(prompt > 0 && (spans.empty() ? cap > 0 : cap == 0) && cap <= 2048 && prompt + cap <= context,
                    "capture request exceeds context");
            for (int token : inputs) require(token >= 0 && token < vocabulary, "invalid input token");
            require(!fs::exists(output) && fs::is_directory(output.parent_path()), "new output directory required");
            const uint64_t bound = uint64_t(prompt + cap) * 2560 * taps.size() * (half ? 2 : 4);
            require(fs::space(output.parent_path()).available > bound + 1024 * 1024 * 1024ULL, "insufficient capture space");
            require(fs::create_directory(output), "cannot create capture output");
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            std::vector<std::ofstream> files;
            for (int tap : taps) {
                files.emplace_back(output / ("layer-" + std::to_string(tap) + (half ? ".f16" : ".f32")), std::ios::binary);
                require(bool(files.back()), "cannot create feature file");
            }
            std::vector<int> positions, labels, generated;
            int next = -1;
            auto consume = [&](int start, int count, bool prediction) {
                auto batch = llama_batch_init(count, 0, 1);
                require(batch.token && batch.pos && batch.logits && batch.seq_id && batch.n_seq_id, "batch allocation failed");
                batch.n_tokens = count;
                for (int i = 0; i < count; ++i) {
                    batch.token[i] = inputs.at(start+i); batch.pos[i] = start+i;
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                    batch.logits[i] = prediction && i == count-1;
                }
                const int status = llama_decode(ctx.get(), batch);
                llama_batch_free(batch);
                require(status == 0, "native decode failed");
                llama_synchronize(ctx.get());
                for (size_t t = 0; t < taps.size(); ++t)
                    write_features(files[t], llama_get_embeddings_layer_inp(ctx.get(), taps[t]), size_t(count)*2560, half);
                if (prediction) {
                    const float * logits = llama_get_logits_ith(ctx.get(), count-1);
                    require(logits != nullptr, "missing prediction logits");
                    for (int i = 0; i < vocabulary; ++i) require(std::isfinite(logits[i]), "nonfinite target logits");
                    next = std::max_element(logits, logits+vocabulary)-logits;
                    positions.push_back(start+count-1); labels.push_back(next);
                }
            };
            auto prefill = [&](int from, int to, bool prediction) {
                for (int start = from; start < to; start += 512)
                    consume(start, std::min(512, to-start), prediction && start+512 >= to);
            };
            if (spans.empty()) prefill(0, prompt, true);
            int position = 0;
            for (auto [first, last] : spans) {
                prefill(position, first+1, true);
                for (int token = first+1; token <= last; ++token) consume(token, 1, true);
                position = last+1;
            }
            if (!spans.empty()) prefill(position, prompt, false);
            while (spans.empty() && int(generated.size()) < cap && !llama_vocab_is_eog(vocab, next)) {
                generated.push_back(next); inputs.push_back(next);
                consume(inputs.size()-1, 1, true);
            }
            for (auto & file : files) { file.close(); require(bool(file), "feature close failed"); }
            const bool stopped = llama_vocab_is_eog(vocab, next);
            json metadata = {{"schema",2},{"status","native_capture_complete"},{"feature_dtype",dtype},
                {"feature_semantics","zero_based_pre_block_input_all_tokens"},{"feature_shape",{inputs.size(),2560}},
                {"taps",taps},{"input_tokens",inputs},{"prompt_tokens",spans.empty() ? prompt : spans[0].first+1},
                {"greedy_ids",labels},{"logit_positions",positions},{"eog_ids",eog},{"generated_tokens",generated},
                {"stop_type",!spans.empty() ? "spans" : stopped ? "eos" : "limit"},{"output_cap",cap},
                {"settings",{{"context",context},{"batch",512},{"ubatch",512},{"kv","F16"},
                    {"flash_attention","on"},{"consistent",32},{"sampling","raw_argmax_lowest_id_tie"}}}};
            if (!spans.empty()) metadata["supervised_spans"] = spans;
            std::ofstream meta(output/"capture.json"); meta << metadata.dump(2) << '\n';
            meta.close(); require(bool(meta), "capture metadata write failed");
            std::ifstream maps("/proc/self/maps"); std::ofstream mapped(output/"loaded-library-maps.txt");
            mapped << maps.rdbuf(); mapped.close(); require(bool(mapped), "library maps write failed");
            std::cout << json({{"status","complete"},{"output",output.string()},
                    {"input_tokens",prompt},{"generated_tokens",generated.size()},{"stop_type",metadata["stop_type"]}}).dump() << std::endl;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "capture refused: " << error.what() << std::endl;
        return 1;
    }
}
