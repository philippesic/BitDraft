// Instrumented native EAGLE replay against captured native target features.
#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Existing native C++ exports, absent from public llama.h.
void llama_set_embeddings_nextn(llama_context *, bool, bool);
float * llama_get_embeddings_nextn(llama_context *);
float * llama_get_embeddings_nextn_ith(llama_context *, int32_t);

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
static constexpr int H = 2560, V = 248320, D = 50000;
static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
static void write_json(const fs::path & path, const json & value) {
    std::ofstream out(path);
    require(bool(out), "cannot create metadata");
    out << value.dump(2) << '\n';
    out.close();
    require(bool(out), "metadata write failed");
}
static std::vector<float> read_floats(const fs::path & path, size_t count) {
    require(fs::is_regular_file(path) && fs::file_size(path) == count*4, "input F32 size differs");
    std::ifstream input(path, std::ios::binary);
    std::vector<float> data(count);
    input.read(reinterpret_cast<char *>(data.data()), std::streamsize(count*4));
    require(bool(input), "F32 input read failed");
    for (float value : data) require(std::isfinite(value), "nonfinite captured feature");
    return data;
}
static void append_floats(std::ofstream & output, const float * values, size_t count) {
    require(values != nullptr, "null native F32 output");
    for (size_t i = 0; i < count; ++i) require(std::isfinite(values[i]), "nonfinite native F32 output");
    output.write(reinterpret_cast<const char *>(values), std::streamsize(count*4));
    require(bool(output), "native F32 write failed");
}
struct Capture {
    bool active = false;
    int hits = 0;
    std::vector<float> embedding;
};
static bool capture_embedding(ggml_tensor * tensor, bool ask, void * user) {
    auto & state = *static_cast<Capture *>(user);
    if (!state.active || std::string(ggml_get_name(tensor)) != "inp_embd") return false;
    if (ask) return true;
    require(tensor->type == GGML_TYPE_F32 && tensor->ne[0] == H &&
            tensor->ne[1] == 1 && ggml_is_contiguous(tensor), "native embedding trace shape differs");
    state.embedding.resize(H);
    ggml_backend_tensor_get(tensor, state.embedding.data(), 0, H*4);
    ++state.hits;
    return true;
}
static ggml_backend_reg_t registration(const fs::path & path, const char * symbol) {
    const auto expected = fs::canonical(path);
    void * handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_LOCAL);
    require(handle != nullptr, "cannot load required backend");
    void * address = dlsym(handle, symbol);
    Dl_info info = {};
    require(address && dladdr(address, &info) && info.dli_fname &&
            fs::canonical(info.dli_fname) == expected, "backend registration resolved elsewhere");
    return reinterpret_cast<ggml_backend_reg_t (*)()>(address)();
}
static llama_context_params context_params() {
    auto params = llama_context_default_params();
    params.n_ctx = 4096; params.n_batch = 512; params.n_ubatch = 512; params.n_seq_max = 1;
    params.n_threads = 2; params.n_threads_batch = 2;
    params.type_k = GGML_TYPE_F16; params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.n_outputs_max = 512; params.n_outputs_max_per_seq = 512;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    return params;
}
static void add_pair(llama_batch_ext * batch, int token, int position, const float * state, bool output) {
    int32_t index = llama_batch_ext_add_token(batch, 0, token);
    require(index >= 0 && llama_batch_ext_set_pos(batch, index, &position) &&
            llama_batch_ext_set_embd_token(batch, index, {state, 1, H}) &&
            llama_batch_ext_set_output_logits(batch, index, output), "cannot construct shifted native pair");
}
static void decode(llama_context * ctx, llama_batch_ext * batch) {
    const int result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    require(result == 0, "native decoder failed: " + std::to_string(result));
    llama_synchronize(ctx);
}
int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "modern_replay --target TARGET --draft PACKED --cases CASES.json --out NEW_DIR\n"
                         "  --cpu-library LIB --cuda-library LIB\n"
                         "Bounded GPU-drafter replay; target is CPU-mapped for native embeddings only.\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--abi-check") {
            require(llama_context_default_params().n_batch > 0, "linked context defaults invalid");
            std::cout << "{\"status\":\"linked_api_only\",\"model_opened\":false,\"backend_initialized\":false}" << std::endl;
            return 0;
        }
        std::map<std::string, std::string> args;
        const std::set<std::string> keys = {"--target", "--draft", "--cases", "--out", "--cpu-library", "--cuda-library"};
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            require(keys.count(key) && !args.count(key) && i + 1 < argc, "unknown/duplicate/incomplete argument");
            args[key] = argv[++i];
        }
        require(args.size() == keys.size(), "all replay arguments are required");
        require(!fs::exists(args["--out"]), "replay output already exists");
        require(fs::file_size(args["--cases"]) <= 2*1024*1024, "case packet exceeds 2 MiB");
        const char * consistent = std::getenv("GGML_CUDA_DECODE_CONSISTENT");
        const char * bits = std::getenv("GGML_W1AX_ACT_BITS");
        require(consistent && std::string(consistent) == "32" && bits && std::string(bits) == "8",
                "explicit correctness32 and A8 environment required");
        json packet;
        std::ifstream(args["--cases"]) >> packet;
        require(packet.at("schema") == "w1a8_modern_native_replay_v1", "unsupported replay packet");
        require(packet.at("cases").is_array() && packet["cases"].size() > 0 && packet["cases"].size() <= 8,
                "require 1..8 cases");
        std::ifstream map_file(packet.at("draft_map").get<std::string>());
        std::vector<int> draft_map;
        int mapped;
        while (map_file >> mapped) draft_map.push_back(mapped);
        require(map_file.eof() && draft_map.size() == D && std::set<int>(draft_map.begin(), draft_map.end()).size() == D &&
                *std::min_element(draft_map.begin(), draft_map.end()) >= 0 && *std::max_element(draft_map.begin(), draft_map.end()) < V,
                "invalid absolute draft map");
        uint64_t budget = 0;
        std::set<std::string> ids;
        for (const auto & item : packet["cases"]) {
            const std::string id = item.at("id");
            require(!id.empty() && id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos && ids.insert(id).second,
                    "unique safe case IDs required");
            const auto tokens = item.at("tokens").get<std::vector<int>>();
            const int prompt = item.at("prompt_tokens"), depth = item.at("depth");
            const auto labels = item.at("labels").get<std::vector<int>>();
            const auto anchors = item.at("anchors").get<std::vector<int>>();
            require(tokens.size() <= 4096 && prompt >= 2 && size_t(prompt) < tokens.size() && depth >= 1 && depth <= 5 &&
                    labels.size() == tokens.size() - prompt + 1 && anchors.size() > 0 && anchors.size() <= 32 &&
                    item.at("features").size() == 3, "invalid bounded case geometry");
            for (int token : tokens) require(token >= 0 && token < V, "input token outside vocabulary");
            for (int token : labels) require(token >= 0 && token < V, "label outside vocabulary");
            require(std::equal(tokens.begin() + prompt, tokens.end(), labels.begin()), "forced serving continuation differs");
            for (int anchor : anchors) require(anchor >= prompt - 1 && size_t(anchor + 1) < tokens.size(), "anchor lacks shifted input");
            budget += uint64_t(tokens.size())*H*4 + uint64_t(anchors.size())*(depth + 1)*(D + 2*H)*4;
        }
        require(budget < 256*1024*1024 && fs::space(fs::path(args["--out"]).parent_path()).available > budget + 32*1024*1024,
                "insufficient bounded replay output space");

        auto * cpu = registration(args["--cpu-library"], "ggml_backend_cpu_reg");
        auto * cuda = registration(args["--cuda-library"], "ggml_backend_cuda_reg");
        require(ggml_backend_reg_by_name("CPU") == cpu && ggml_backend_reg_by_name("CUDA") == cuda,
                "registered backends differ from required images");
        auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(gpu && ggml_backend_dev_backend_reg(gpu) == cuda, "real matching GPU backend required");
        llama_backend_init();
        ggml_backend_dev_t no_devices[] = {nullptr};
        auto target_params = llama_model_default_params();
        target_params.n_gpu_layers = 0; target_params.devices = no_devices; target_params.load_mtp = false;
        target_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> target(
            llama_model_load_from_file(args["--target"].c_str(), target_params), llama_model_free);
        require(bool(target) && llama_model_n_embd(target.get()) == H && llama_vocab_n_tokens(llama_model_get_vocab(target.get())) == V,
                "target native embedding model load failed or geometry differs");
        auto tp = context_params(); tp.n_outputs_max = 1; tp.n_outputs_max_per_seq = 1;
        std::unique_ptr<llama_context, decltype(&llama_free)> target_ctx(llama_init_from_model(target.get(), tp), llama_free);
        require(bool(target_ctx), "target embedding-provider context load failed");
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto dp = llama_model_default_params();
        dp.devices = devices; dp.n_gpu_layers = -1; dp.split_mode = LLAMA_SPLIT_MODE_NONE; dp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(args["--draft"].c_str(), dp), llama_model_free);
        require(bool(model) && llama_model_n_embd(model.get()) == H && llama_model_n_layer(model.get()) == 1,
                "packed native EAGLE model load failed");
        Capture trace;
        auto cp = context_params(); cp.ctx_other = target_ctx.get(); cp.cb_eval = capture_embedding; cp.cb_eval_user_data = &trace;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "native draft context load failed");
        llama_set_embeddings_nextn(ctx.get(), true, true);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        require(bool(batch), "native extended batch allocation failed");
        fs::create_directory(args["--out"]);
        const fs::path output = args["--out"];
        json results = json::object();
        for (const auto & item : packet["cases"]) {
            const std::string id = item["id"];
            const auto tokens = item["tokens"].get<std::vector<int>>();
            const auto labels = item["labels"].get<std::vector<int>>();
            const auto anchors = item["anchors"].get<std::vector<int>>();
            const int prompt = item["prompt_tokens"], depth_max = item["depth"];
            std::vector<float> features(tokens.size()*3*H), encoded(tokens.size()*H);
            for (int tap = 0; tap < 3; ++tap) {
                auto values = read_floats(item["features"][tap].get<std::string>(), tokens.size()*H);
                for (size_t row = 0; row < tokens.size(); ++row) {
                    std::copy_n(values.data() + row*H, H, features.data() + row*3*H + tap*H);
                }
            }
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            for (size_t start = 0; start < tokens.size(); start += 512) {
                llama_batch_ext_clear(batch.get());
                const size_t count = std::min(size_t(512), tokens.size() - start);
                for (size_t j = 0; j < count; ++j) {
                    int32_t index = llama_batch_ext_add_embd(batch.get(), 0, {features.data() + (start+j)*3*H, 1, 3*H});
                    llama_pos position = int(start+j);
                    require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &position) &&
                            llama_batch_ext_set_output_logits(batch.get(), index, true), "cannot construct encoder batch");
                }
                require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_ENCODE, batch.get()) == 0, "native feature encoder failed");
                const float * values = llama_get_embeddings_nextn(ctx.get());
                require(values != nullptr, "encoder missing pre-norm output");
                std::copy_n(values, count*H, encoded.data() + start*H);
            }
            std::ofstream encoded_file(output/(id + "-encoded.f32"), std::ios::binary);
            append_floats(encoded_file, encoded.data(), encoded.size());
            auto prefix = [&](int anchor) {
                trace.active = false;
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                for (int start = 0; start < anchor; start += 512) {
                    llama_batch_ext_clear(batch.get());
                    for (int j = start; j < std::min(anchor, start + 512); ++j) {
                        add_pair(batch.get(), tokens.at(j+1), j, encoded.data() + size_t(j)*H, false);
                    }
                    decode(ctx.get(), batch.get());
                }
                require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == anchor - 1,
                        "native context position differs before shifted seed");
            };
            auto mode = [&](bool recurrent) {
                const std::string tag = recurrent ? "recurrent" : "teacher";
                std::ofstream logits_file(output/(id + "-" + tag + "-logits.f32"), std::ios::binary);
                std::ofstream state_file(output/(id + "-" + tag + "-prenorm.f32"), std::ios::binary);
                std::ofstream embedding_file(output/(id + "-" + tag + "-embeddings.f32"), std::ios::binary);
                json records = json::array(), windows = json::array();
                if (!recurrent) prefix(anchors.front());
                size_t rows = 0;
                for (size_t ai = 0; ai < anchors.size(); ++ai) {
                    const int anchor = anchors[ai];
                    if (recurrent) prefix(anchor);
                    else require(ai == 0 || anchor == anchors[ai-1]+1, "teacher anchors must be contiguous");
                    int token = tokens.at(anchor+1);
                    std::vector<float> state(encoded.begin() + size_t(anchor)*H, encoded.begin() + size_t(anchor+1)*H);
                    int correct_prefix = 0;
                    std::string stop = recurrent ? "depth_limit" : "teacher_forced_restart";
                    for (int depth = 0; depth < (recurrent ? depth_max : 1); ++depth) {
                        const int label_index = anchor - prompt + 2 + depth;
                        if (label_index >= int(labels.size())) { stop = "no_admitted_next_teacher_row"; break; }
                        const int position = anchor + depth;
                        trace.active = true; trace.hits = 0; trace.embedding.clear();
                        llama_batch_ext_clear(batch.get());
                        add_pair(batch.get(), token, position, state.data(), true);
                        decode(ctx.get(), batch.get());
                        trace.active = false;
                        require(trace.hits == 1, "expected exactly one native input embedding observation");
                        const float * full = llama_get_logits_ith(ctx.get(), 0);
                        require(full != nullptr, "native draft logits missing");
                        int proposed = int(std::max_element(full, full+V) - full);
                        std::vector<float> subset(D);
                        for (int i = 0; i < D; ++i) subset[i] = full[draft_map[i]];
                        const float * prenorm = llama_get_embeddings_nextn_ith(ctx.get(), 0);
                        append_floats(logits_file, subset.data(), subset.size());
                        append_floats(state_file, prenorm, H);
                        append_floats(embedding_file, trace.embedding.data(), H);
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == position,
                                "native cache position differs after step");
                        records.push_back({{"row", rows++}, {"anchor", anchor}, {"depth", depth},
                            {"decoder_position", position}, {"input_token", token}, {"label_index", label_index},
                            {"label", labels.at(label_index)}, {"proposed", proposed},
                            {"correct", proposed == labels.at(label_index)}, {"cache_position_max", position}});
                        state.assign(prenorm, prenorm+H);
                        if (proposed != labels.at(label_index)) { stop = "first_gold_divergence"; break; }
                        ++correct_prefix;
                        token = proposed;
                    }
                    windows.push_back({{"anchor", anchor}, {"correct_prefix", correct_prefix}, {"stop_reason", stop}});
                }
                logits_file.close(); state_file.close(); embedding_file.close();
                require(bool(logits_file) && bool(state_file) && bool(embedding_file), "native replay stream close failed");
                return json{{"records", records}, {"windows", windows}, {"rows", rows}};
            };
            results[id] = {{"tokens", tokens.size()}, {"prompt_tokens", prompt},
                           {"teacher", mode(false)}, {"recurrent", mode(true)}};
            std::cout << "replay_complete " << id << std::endl;
        }
        {
            std::ifstream maps("/proc/self/maps");
            std::ofstream out(output/"loaded-library-maps.txt"); out << maps.rdbuf();
            require(bool(out), "library maps write failed");
        }
        write_json(output/"replay.json", {{"status", "NATIVE_REPLAY_COMPLETE"}, {"cases", results},
            {"context", 4096}, {"batch", 512}, {"ubatch", 512}, {"kv", "F16"},
            {"target_placement", "CPU_mapped_native_embedding_provider_only_no_target_forward"},
            {"draft_placement", "CUDA"}, {"embedding_trace", true}, {"correctness_mode", 32},
            {"training_updates", 0}, {"scope", "Instrumented captured-history replay; not performance or serving verifier qualification"}});
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "modern_replay: " << error.what() << std::endl;
        return 1;
    }
}
