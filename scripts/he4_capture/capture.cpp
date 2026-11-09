// Native Qwen3.5 input taps and vocabulary-only tokenization; no runtime patch.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// These existing C++ exports are absent from the retained public llama.h.
// Exact definitions: retained src/llama-context.cpp, not extern "C".
void llama_set_embeddings_layer_inp(llama_context *, uint32_t, bool);
float * llama_get_embeddings_layer_inp(llama_context *, uint32_t);

namespace fs = std::filesystem;

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static std::string quote(const std::string & s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) {
            const char * hex = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << c;
    }
    return out.str() + '"';
}

static int integer(const std::string & s) {
    size_t end = 0;
    long long v = std::stoll(s, &end);
    require(end == s.size() && v >= 0 && v <= INT32_MAX, "invalid nonnegative integer: " + s);
    return static_cast<int>(v);
}

static std::vector<int> numbers(std::string s) {
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream in(s);
    std::vector<int> result;
    std::string word;
    while (in >> word) {
        require(result.size() < 4096, "at most 4096 integers permitted");
        result.push_back(integer(word));
    }
    return result;
}

static std::vector<int> token_file(const std::string & path) {
    require(fs::file_size(path) <= 65536, "token file exceeds 64 KiB");
    std::ifstream in(path);
    require(bool(in), "cannot read token file: " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return numbers(text.str());
}

template<typename T> static void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
}

static void floats(std::ofstream & out, const float * data, size_t count) {
    require(data != nullptr, "native output pointer is null");
    for (size_t i = 0; i < count; ++i) require(std::isfinite(data[i]), "nonfinite native output");
    out.write(reinterpret_cast<const char *>(data), count * sizeof(float));
    require(bool(out), "capture write failed");
}

static int tokenize_text(int argc, char ** argv) {
    std::map<std::string, std::string> args;
    const std::set<std::string> keys = {"--model", "--text-file", "--out"};
    bool mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--tokenize") { require(!mode, "duplicate --tokenize"); mode = true; continue; }
        require(keys.count(key) && i + 1 < argc && !args.count(key), "unknown, duplicate, or incomplete tokenization argument: " + key);
        args[key] = argv[++i];
    }
    for (const char * key : {"--model", "--text-file", "--out"})
        require(args.count(key), std::string("required for --tokenize: ") + key);
    const fs::path output = args.at("--out");
    require(!fs::exists(output), "output directory already exists");
    require(fs::is_regular_file(args.at("--text-file")), "text input must be a regular file");
    std::ifstream input(args.at("--text-file"), std::ios::binary);
    require(bool(input), "cannot open text file");
    std::string text(65537, '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    require(!input.bad(), "text read failed");
    text.resize(static_cast<size_t>(input.gcount()));
    require(text.size() <= 65536, "text file exceeds 64 KiB");

    // Empty devices skips registry/device enumeration in llama_prepare_model_devices.
    // vocab_only returns before load_tensors; no backend or context init is needed.
    ggml_backend_dev_t devices[] = {nullptr};
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.devices = devices;
    params.n_gpu_layers = 0;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(args.at("--model").c_str(), params), llama_model_free);
    require(bool(model), "target vocabulary load failed");
    std::vector<llama_token> tokens(4096);
    const int32_t count = llama_tokenize(llama_model_get_vocab(model.get()), text.data(),
        static_cast<int32_t>(text.size()), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    require(count >= 0 && count <= 4096, "native tokenization failed or exceeds 4096 IDs");
    tokens.resize(count);
    require(fs::create_directory(output), "cannot create unique output directory");
    std::ofstream ids(output / "tokens.ids");
    for (llama_token token : tokens) ids << token << '\n';
    ids.close(); require(bool(ids), "token IDs write failed");
    std::ofstream metadata(output / "tokenization.json");
    metadata << "{\n\"status\":\"native_tokenization_complete\",\n\"model\":"
        << quote(fs::absolute(args.at("--model")).string())
        << ",\n\"model_bytes\":" << fs::file_size(args.at("--model"))
        << ",\n\"text_file\":" << quote(fs::absolute(args.at("--text-file")).string())
        << ",\n\"text_bytes\":" << text.size()
        << ",\n\"add_special\":true,\n\"parse_special\":true,\n\"chat_template_applied\":false,"
        << "\n\"vocab_only\":true,\n\"backend_initialized\":false,\n\"weights_loaded\":false,"
        << "\n\"token_ids_file\":\"tokens.ids\",\n\"token_count\":" << count << "\n}\n";
    metadata.close(); require(bool(metadata), "tokenization metadata write failed");
    std::cout << "tokenization_complete " << (output / "tokenization.json").string() << '\n';
    return 0;
}

static void help() {
    std::cout << "modern_capture --model TARGET.gguf --tokens PROMPT.ids --out NEW_DIRECTORY\n"
        "  --cpu-library LIB --cuda-library IMMUTABLE_LIB\n"
        "  [--decode-tokens CONTINUATION.ids | --greedy-steps N] [--decode-chunk 1]\n"
        "  [--taps 2,16,29] [--logit-positions P0,P1,...] [--prefill-chunk 512]\n"
        "  [--threads 8] [--output-cap 512] [--max-output-mib 256] [--no-features]\n"
        "Token files contain decimal token IDs separated by whitespace or commas.\n"
        "Positions are zero based over prompt then continuation. Default logits:\n"
        "last prompt row plus every continuation row. Greedy mode requires these rows.\n"
        "Fixed continuation can use decode chunks 1..32; greedy mode requires 1.\n"
        "Features: layer-N.f32, shape [all input rows,2560], little endian F32.\n"
        "Logits: logits.f32, shape [selected rows,vocab]; capture.json maps rows.\n"
        "--no-features provides a target-only trajectory control.\n"
        "modern_capture --tokenize --model TARGET.gguf --text-file RAW.txt --out NEW_DIRECTORY\n"
        "  Vocabulary-only native tokenization: add_special=true, parse_special=true.\n"
        "  Raw text <=64 KiB; output <=4096 IDs in tokens.ids and tokenization.json.\n"
        "  No chat template, backend initialization, context, or weight loading.\n"
        "--abi-check checks linked defaults only, without backend or model loading.\n";
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { help(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--abi-check") {
            const auto p = llama_context_default_params();
            require(p.n_batch > 0, "invalid linked context defaults");
            std::cout << "{\"status\":\"linked_api_only\",\"model_loaded\":false,\"backend_initialized\":false}\n";
            return 0;
        }
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--tokenize") return tokenize_text(argc, argv);
        std::map<std::string, std::string> args;
        const std::set<std::string> keys = {"--model", "--tokens", "--out", "--cpu-library", "--cuda-library",
            "--decode-tokens", "--greedy-steps", "--decode-chunk", "--taps", "--logit-positions", "--prefill-chunk", "--threads", "--output-cap", "--max-output-mib"};
        bool features = true;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--no-features") { require(features, "duplicate --no-features"); features = false; continue; }
            require(keys.count(key) && i + 1 < argc && !args.count(key), "unknown, duplicate, or incomplete argument: " + key);
            args[key] = argv[++i];
        }
        for (const char * key : {"--model", "--tokens", "--out", "--cpu-library", "--cuda-library"})
            require(args.count(key), std::string("required: ") + key);
        auto get = [&](const char * key, const char * fallback) { return args.count(key) ? args.at(key) : std::string(fallback); };
        require(!(args.count("--decode-tokens") && args.count("--greedy-steps")), "choose fixed continuation or greedy steps");
        const auto prompt = token_file(args.at("--tokens"));
        const auto fixed = args.count("--decode-tokens") ? token_file(args.at("--decode-tokens")) : std::vector<int>{};
        const int steps = args.count("--greedy-steps") ? integer(args.at("--greedy-steps")) : static_cast<int>(fixed.size());
        require(!prompt.empty() && prompt.size() + steps <= 4096, "require 1..4096 total input tokens");
        const int chunk = integer(get("--prefill-chunk", "512"));
        const int decode_chunk = integer(get("--decode-chunk", "1"));
        require(decode_chunk >= 1 && decode_chunk <= 32, "decode chunk must be in [1,32]");
        require(!args.count("--greedy-steps") || decode_chunk == 1,
                "greedy continuation requires decode chunk 1");
        const int threads = integer(get("--threads", "8"));
        const int output_cap = integer(get("--output-cap", "512"));
        require(output_cap >= 1 && output_cap <= 512, "output cap must be in [1,512]");
        const int max_mib = integer(get("--max-output-mib", "256"));
        require(chunk >= 1 && chunk <= 512 && threads >= 1 && threads <= 64 && max_mib >= 1 && max_mib <= 1024, "invalid capture limits");
        auto taps = features ? numbers(get("--taps", "2,16,29")) : std::vector<int>{};
        require(!features || (!taps.empty() && taps.size() <= 8), "require 1..8 taps");
        require(std::set<int>(taps.begin(), taps.end()).size() == taps.size(), "duplicate taps");
        for (int tap : taps) require(tap < 32, "Qwen3.5 4B input tap must be in [0,31]");
        const int total = static_cast<int>(prompt.size()) + steps;
        std::vector<int> selected;
        if (args.count("--logit-positions")) selected = numbers(args.at("--logit-positions"));
        else for (int p = static_cast<int>(prompt.size()) - 1; p < total; ++p) selected.push_back(p);
        require(!selected.empty() && std::is_sorted(selected.begin(), selected.end()), "logit positions must be nonempty and sorted");
        const std::set<int> selected_set(selected.begin(), selected.end());
        require(selected_set.size() == selected.size() && selected.back() < total, "invalid or duplicate logit positions");
        if (args.count("--greedy-steps")) for (int p = static_cast<int>(prompt.size()) - 1; p < total; ++p)
            require(selected_set.count(p), "greedy mode requires logits at last prompt and every continuation row");
        const uint32_t endian = 1;
        require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4, "requires little endian F32 host");
        const char * consistent = std::getenv("GGML_CUDA_DECODE_CONSISTENT");
        require(consistent && std::string(consistent) == "32", "GGML_CUDA_DECODE_CONSISTENT must equal 32");
        const fs::path output = args.at("--out");
        require(!fs::exists(output), "output directory already exists");

        // These retained shared backends export *_reg, not the optional plugin
        // entrypoint ggml_backend_init. Bind the exact modules and verify the
        // registry actually selected them; do not treat them as DL plugins.
        auto registration = [&](const char * option, const char * symbol) {
            const auto expected = fs::canonical(args.at(option));
            void * handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_LOCAL);
            require(handle != nullptr, "required backend module could not be opened");
            void * address = dlsym(handle, symbol);
            Dl_info info = {};
            require(address && dladdr(address, &info) && info.dli_fname &&
                    fs::canonical(info.dli_fname) == expected, "backend registration resolved to another module");
            auto function = reinterpret_cast<ggml_backend_reg_t (*)()>(address);
            return function();
        };
        auto * required_cpu = registration("--cpu-library", "ggml_backend_cpu_reg");
        auto * required_cuda = registration("--cuda-library", "ggml_backend_cuda_reg");
        require(ggml_backend_reg_by_name("CPU") == required_cpu &&
                ggml_backend_reg_by_name("CUDA") == required_cuda, "active backend registry differs from required modules");
        auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(gpu != nullptr && ggml_backend_dev_backend_reg(gpu) == required_cuda,
                "no matching GPU device; CPU fallback prohibited");
        llama_backend_init();
        ggml_backend_dev_t devices[] = {gpu, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(args.at("--model").c_str(), mp), llama_model_free);
        require(bool(model), "target model load failed");
        char architecture[128] = {};
        char file_type[128] = {};
        char model_description[256] = {};
        llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture));
        llama_model_meta_val_str(model.get(), "general.file_type", file_type, sizeof(file_type));
        llama_model_desc(model.get(), model_description, sizeof(model_description));
        const int width = llama_model_n_embd(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(std::string(architecture) == "qwen35" && width == 2560 && llama_model_n_layer(model.get()) == 32,
                "capture is restricted to Qwen3.5 4B (32 layers, width 2560)");
        for (int token : prompt) require(token < vocab, "prompt token exceeds vocabulary");
        for (int token : fixed) require(token < vocab, "continuation token exceeds vocabulary");
        const uint64_t expected_bytes = (uint64_t(total) * width * taps.size() + uint64_t(selected.size()) * vocab) * sizeof(float);
        require(expected_bytes <= uint64_t(max_mib) * 1024 * 1024, "requested arrays exceed --max-output-mib");
        require(fs::space(output.has_parent_path() ? output.parent_path() : fs::path(".")).available > expected_bytes + 16 * 1024 * 1024,
                "insufficient disk space for bounded capture");
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.n_outputs_max = static_cast<uint32_t>(output_cap);
        cp.n_outputs_max_per_seq = static_cast<uint32_t>(output_cap);
        cp.n_threads = threads; cp.n_threads_batch = threads;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
        cp.embeddings = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "target context creation failed");
        require(llama_n_ctx(ctx.get()) == 4096 && llama_n_batch(ctx.get()) == 512 && llama_n_ubatch(ctx.get()) == 512 && llama_n_seq_max(ctx.get()) == 1,
                "effective context/batch settings differ from frozen runtime");
        for (int tap : taps) llama_set_embeddings_layer_inp(ctx.get(), static_cast<uint32_t>(tap), true);
        require(fs::create_directory(output), "cannot create unique output directory");
        std::vector<std::ofstream> feature_files;
        for (int tap : taps) {
            feature_files.emplace_back(output / ("layer-" + std::to_string(tap) + ".f32"), std::ios::binary);
            require(bool(feature_files.back()), "cannot create feature output");
        }
        std::ofstream logits_file(output / "logits.f32", std::ios::binary);
        require(bool(logits_file), "cannot create logits output");
        std::vector<int> inputs = prompt;
        std::vector<int> greedy_ids;
        int next = -1;
        auto consume = [&](int start, int count) {
            int output_count = 0;
            for (int i = 0; i < count; ++i) output_count += selected_set.count(start + i);
            require(output_count <= output_cap, "selected output rows exceed context output cap");
            auto batch = llama_batch_init(count, 0, 1);
            require(batch.token && batch.pos && batch.logits && batch.seq_id && batch.n_seq_id, "batch allocation failed");
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = inputs.at(start + i); batch.pos[i] = start + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = selected_set.count(start + i) ? 1 : 0;
            }
            const int result = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(result == 0, "llama_decode failed: " + std::to_string(result));
            llama_synchronize(ctx.get());
            for (size_t t = 0; t < taps.size(); ++t)
                floats(feature_files[t], llama_get_embeddings_layer_inp(ctx.get(), taps[t]), size_t(count) * width);
            for (int i = 0; i < count; ++i) if (selected_set.count(start + i)) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                floats(logits_file, logits, vocab);
                next = static_cast<int>(std::max_element(logits, logits + vocab) - logits);
                greedy_ids.push_back(next);
            }
        };
        for (int start = 0; start < static_cast<int>(prompt.size()); start += chunk)
            consume(start, std::min(chunk, static_cast<int>(prompt.size()) - start));
        for (int i = 0; i < (args.count("--greedy-steps") ? steps - 1 : steps); i += decode_chunk) {
            if (args.count("--greedy-steps") && llama_vocab_is_eog(llama_model_get_vocab(model.get()), next)) break;
            if (args.count("--greedy-steps")) require(next >= 0, "missing greedy predecessor logits");
            const int count = std::min(decode_chunk, steps - i);
            const int start = static_cast<int>(inputs.size());
            for (int j = 0; j < count; ++j)
                inputs.push_back(args.count("--greedy-steps") ? next : fixed.at(i + j));
            consume(start, count);
        }
        selected.resize(greedy_ids.size());
        std::vector<int> eog_ids;
        for (int t = 0; t < vocab; ++t) if (llama_vocab_is_eog(llama_model_get_vocab(model.get()), t)) eog_ids.push_back(t);
        logits_file.close(); require(bool(logits_file), "logits close failed");
        for (auto & file : feature_files) { file.close(); require(bool(file), "features close failed"); }
        {
            std::ifstream mapped("/proc/self/maps");
            std::ofstream receipt(output / "loaded-library-maps.txt");
            require(bool(mapped) && bool(receipt), "cannot record loaded native libraries");
            receipt << mapped.rdbuf();
            receipt.close(); require(bool(receipt), "loaded-library receipt write failed");
        }
        std::ofstream metadata(output / "capture.json");
        metadata << "{\n\"status\":\"native_capture_complete\",\n\"model\":" << quote(fs::absolute(args.at("--model")).string())
            << ",\n\"model_bytes\":" << fs::file_size(args.at("--model"))
            << ",\n\"model_description\":" << quote(model_description)
            << ",\n\"gguf_file_type\":" << quote(file_type)
            << ",\n\"cpu_library\":" << quote(fs::canonical(args.at("--cpu-library")).string())
            << ",\n\"cuda_library\":" << quote(fs::canonical(args.at("--cuda-library")).string())
            << ",\n\"gpu_name\":" << quote(ggml_backend_dev_name(gpu))
            << ",\n\"gpu_description\":" << quote(ggml_backend_dev_description(gpu))
            << ",\n\"dtype\":\"little_endian_float32\",\n\"feature_semantics\":\"zero_based_pre_block_input_all_tokens\",\n\"feature_shape\":[" << inputs.size() << ',' << width
            << "],\n\"logits_shape\":[" << selected.size() << ',' << vocab << "],\n\"taps\":";
        array(metadata, taps); metadata << ",\n\"eog_ids\":"; array(metadata, eog_ids); metadata << ",\n\"logit_positions\":"; array(metadata, selected);
        metadata << ",\n\"greedy_ids\":"; array(metadata, greedy_ids);
        metadata << ",\n\"input_tokens\":"; array(metadata, inputs);
        metadata << ",\n\"prompt_tokens\":" << prompt.size() << ",\n\"continuation_mode\":" << quote(args.count("--greedy-steps") ? "native_greedy" : "fixed_token_ids")
            << ",\n\"prefill_chunk\":" << chunk
            << ",\n\"decode_chunk\":" << decode_chunk
            << ",\n\"output_cap\":" << output_cap
            << ",\n\"threads\":" << threads
            << ",\n\"settings\":{\"context\":4096,\"batch\":512,\"ubatch\":512,\"seq_max\":1,\"kv\":\"F16\",\"flash_attention\":\"on\",\"consistent\":32,\"sampling\":\"raw_argmax_lowest_id_tie\",\"load_mtp\":false},\n\"array_bytes\":" << expected_bytes << "\n}\n";
        metadata.close(); require(bool(metadata), "metadata close failed");
        std::cout << "capture_complete " << (output / "capture.json").string() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "modern_capture: " << error.what() << '\n';
        return 1;
    }
}
