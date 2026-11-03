// Diagnostic interposition only: forwards the exact native call once, unchanged.
#include "llama.h"
#include "llama-batch.h"
#include "llama-ext.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
constexpr size_t CAP = 64 * 1024 * 1024, RESERVE = 4096;
constexpr int H = 2560, V = 248320, D = 50000;
using process_fn = int32_t (*)(llama_context *, llama_process_type, llama_batch_ext *);
std::atomic<bool> owns_directory{false};

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}
void write_all(int fd, const void * ptr, size_t bytes) {
    const auto * p = static_cast<const char *>(ptr);
    while (bytes) {
        const ssize_t result = ::write(fd, p, bytes);
        if (result < 0 && errno == EINTR) continue;
        require(result > 0, "trace write failed");
        p += result; bytes -= size_t(result);
    }
}
process_fn next_process() {
    static process_fn value = [] {
        dlerror();
        void * address = dlsym(RTLD_NEXT, "llama_process");
        require(address && dlerror() == nullptr, "RTLD_NEXT llama_process unavailable");
        return reinterpret_cast<process_fn>(address);
    }();
    return value;
}

struct Trace {
    bool enabled = false;
    fs::path root;
    std::mutex mutex;
    size_t bytes = 0;
    uint64_t serial = 0, events = 0;
    int rows_fd = -1;
    std::vector<int> map;

    Trace() {
        const char * dir = std::getenv("W1A8_TRACE_DIR");
        const char * library = std::getenv("W1A8_TRACE_LLAMA");
        const char * mapping = std::getenv("W1A8_TRACE_MAP");
        if (!dir && !library && !mapping) return;
        require(dir && *dir && library && *library && mapping && *mapping,
                "all three W1A8_TRACE_DIR/LLAMA/MAP variables are required");
        require(fs::path(dir).is_absolute() && fs::path(library).is_absolute() && fs::path(mapping).is_absolute(),
                "trace paths must be absolute");
        Dl_info info{};
        require(dladdr(reinterpret_cast<void *>(next_process()), &info) && info.dli_fname &&
                fs::canonical(info.dli_fname) == fs::canonical(library),
                "RTLD_NEXT llama_process library differs from W1A8_TRACE_LLAMA");
        std::ifstream input(mapping);
        int token;
        while (input >> token) map.push_back(token);
        require(input.eof() && map.size() == D && std::set<int>(map.begin(), map.end()).size() == D &&
                *std::min_element(map.begin(), map.end()) >= 0 && *std::max_element(map.begin(), map.end()) < V,
                "invalid absolute50000 trace map");
        root = fs::canonical(dir);
        require(fs::is_directory(root) && fs::is_empty(root), "trace directory must exist and be empty");
        require(fs::space(root).available > CAP + RESERVE, "less than64MiB trace headroom");
        enabled = true;
        owns_directory.store(true);
        json manifest = {{"schema", "w1a8_actual_native_actor_trace_v2"}, {"pid", getpid()},
            {"required_llama", fs::canonical(library).string()}, {"draft_map", fs::canonical(mapping).string()},
            {"cap_bytes", CAP}, {"batch_ext_sizeof", sizeof(llama_batch_ext)},
            {"token_sizeof", sizeof(llama_batch_ext::token)}, {"float_bytes", sizeof(float)},
            {"scope", "Diagnostic EAGLE input/output trace; synchronization and file IO invalidate performance timings"},
            {"operation_configuration_changed", false}, {"model_opened_by_interposer", false},
            {"row_log", "calls.jsonl"}};
        exclusive_json("manifest.json", manifest);
        rows_fd = ::open((root / "calls.jsonl").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        require(rows_fd >= 0, "exclusive row log creation failed");
    }
    ~Trace() { if (rows_fd >= 0) ::close(rows_fd); }
    void budget(size_t count) {
        require(count <= CAP - RESERVE && bytes <= CAP - RESERVE - count,
                "trace64MiB cap exceeded; process aborted with incomplete trace");
        bytes += count;
    }
    void exclusive_file(const std::string & name, const void * data, size_t count) {
        budget(count);
        const fs::path temporary = root / ("." + name + ".tmp");
        int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        require(fd >= 0, "exclusive trace temporary creation failed");
        try {
            write_all(fd, data, count);
            require(::fsync(fd) == 0, "trace fsync failed");
            ::close(fd); fd = -1;
            require(::link(temporary.c_str(), (root / name).c_str()) == 0, "exclusive trace publication failed");
            require(::unlink(temporary.c_str()) == 0, "trace temporary unlink failed");
        } catch (...) { if (fd >= 0) ::close(fd); ::unlink(temporary.c_str()); throw; }
    }
    void exclusive_json(const std::string & name, const json & value) {
        const std::string data = value.dump(2) + "\n";
        exclusive_file(name, data.data(), data.size());
    }
    void append(json value) {
        value["event"] = ++events;
        value["trace_bytes_before_event"] = bytes;
        const std::string data = value.dump() + "\n";
        budget(data.size()); write_all(rows_fd, data.data(), data.size());
        require(::fsync(rows_fd) == 0, "trace row log fsync failed");
    }
    json floats(const std::string & name, const float * data, size_t rows, size_t width) {
        require(data && rows > 0 && width > 0 && rows <= CAP / 4 / width,
                "invalid bounded F32 trace geometry");
        const size_t count = rows * width;
        for (size_t i = 0; i < count; ++i) require(std::isfinite(data[i]), "nonfinite F32 trace output");
        exclusive_file(name, data, count * sizeof(float));
        return {{"file", name}, {"shape", {rows, width}}, {"dtype", "little_endian_float32"}};
    }
};
Trace & trace() { static Trace value; return value; }
thread_local std::vector<uint64_t> target_stack;
thread_local uint64_t last_target = 0;
thread_local std::vector<uint64_t> active_calls;

[[noreturn]] void fatal(const char * message) {
    std::fprintf(stderr, "W1A8_TRACE_FATAL: %s\n", message);
    // Exit without a core dump. The missing after row/failure file marks incompleteness.
    const char * dir = std::getenv("W1A8_TRACE_DIR");
    if (owns_directory.load() && dir && fs::path(dir).is_absolute()) {
        const std::string text = json({{"status", "TRACE_ABORTED"}, {"reason", std::string(message).substr(0, 1024)}}).dump() + "\n";
        const int fd = ::open((fs::path(dir) / "failure.json").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) { const ssize_t written = ::write(fd, text.data(), text.size()); if (written != ssize_t(text.size())) std::fprintf(stderr, "W1A8_TRACE_FATAL: failure marker write incomplete\n"); ::close(fd); }
    }
    std::_Exit(86);
}
std::string architecture(llama_context * ctx) {
    require(ctx != nullptr, "null native context");
    char name[128]{};
    const int count = llama_model_meta_val_str(llama_get_model(ctx), "general.architecture", name, sizeof(name));
    require(count > 0 && count < int(sizeof(name)), "model architecture unavailable");
    return name;
}
std::string stem(uint64_t id) {
    char buffer[40]; std::snprintf(buffer, sizeof(buffer), "call-%06llu", static_cast<unsigned long long>(id));
    return buffer;
}
json batch_snapshot(Trace & tr, uint64_t id, const llama_batch_ext & batch, bool eagle) {
    require(batch.tokens.size() <= 4096 && batch.n_pos_per_embd <= 4 && batch.n_pos_per_embd > 0,
            "batch exceeds bounded diagnostic geometry");
    json rows = json::array();
    std::vector<float> embeddings, states;
    for (size_t i = 0; i < batch.tokens.size(); ++i) {
        const auto & token = batch.tokens[i];
        auto seq = std::vector<llama_seq_id>(token.seq_ids.begin(), token.seq_ids.end());
        std::sort(seq.begin(), seq.end());
        json row = {{"batch_row", i}, {"token", token.id}, {"seq_ids", seq},
                    {"positions", token.pos}, {"output", token.output},
                    {"decision_order", token.decision_order}, {"has_embd", token.has_embd},
                    {"has_state", token.has_state}, {"original_embd_offset", token.embd_off},
                    {"original_state_offset", token.state_off}};
        if (eagle && token.has_embd) {
            require(batch.n_embd > 0 && batch.n_embd <= 3*H && token.embd_off <= batch.embd.size() &&
                    batch.n_embd <= batch.embd.size() - token.embd_off, "input embedding range invalid");
            row["saved_embedding_row"] = embeddings.size() / batch.n_embd;
            embeddings.insert(embeddings.end(), batch.embd.begin() + token.embd_off,
                              batch.embd.begin() + token.embd_off + batch.n_embd);
        }
        if (eagle && token.has_state) {
            require(batch.n_embd_state > 0 && batch.n_embd_state <= 3*H && token.state_off <= batch.state.size() &&
                    batch.n_embd_state <= batch.state.size() - token.state_off, "input state range invalid");
            row["saved_state_row"] = states.size() / batch.n_embd_state;
            states.insert(states.end(), batch.state.begin() + token.state_off,
                          batch.state.begin() + token.state_off + batch.n_embd_state);
        }
        rows.push_back(row);
    }
    json result = {{"rows", rows}, {"n_tokens", batch.tokens.size()}, {"n_embd", batch.n_embd},
        {"n_embd_inp", batch.n_embd_inp}, {"n_embd_inp_enc", batch.n_embd_inp_enc},
        {"n_embd_state", batch.n_embd_state}, {"n_pos_per_embd", batch.n_pos_per_embd}};
    if (!embeddings.empty()) result["embeddings"] = tr.floats(stem(id) + "-input-embd.f32", embeddings.data(), embeddings.size()/batch.n_embd, batch.n_embd);
    if (!states.empty()) result["states"] = tr.floats(stem(id) + "-input-state.f32", states.data(), states.size()/batch.n_embd_state, batch.n_embd_state);
    return result;
}
} // namespace

extern "C" LLAMA_API const char * w1a8_trace_abi_check() {
    try {
        (void) next_process();
        static std::string value;
        value = json({{"status", "ABI_ONLY_NO_MODEL_NO_BACKEND"}, {"enabled", trace().enabled},
                      {"batch_ext_sizeof", sizeof(llama_batch_ext)},
                      {"token_sizeof", sizeof(llama_batch_ext::token)}}).dump();
        return value.c_str();
    } catch (const std::exception & error) { fatal(error.what()); }
}

extern "C" LLAMA_API int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    try {
        auto actual = next_process();
        auto & tr = trace();
        if (!tr.enabled) return actual(ctx, type, batch);
        const std::string arch = architecture(ctx);
        const bool eagle = arch == "eagle3";
        const bool target = arch.rfind("qwen35", 0) == 0;
        if (!eagle && !target) return actual(ctx, type, batch);
        require(batch != nullptr, "null native batch");
        if (eagle) require(llama_model_n_embd(llama_get_model(ctx)) == H &&
            llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx))) == V,
            "trace requires exact modern4B draft geometry");
        uint64_t id;
        json before;
        {
            std::lock_guard<std::mutex> lock(tr.mutex);
            id = ++tr.serial;
            before = {{"phase", "before"}, {"call", id}, {"architecture", arch},
                {"process_type", int(type)}, {"process_type_name", type == LLAMA_PROCESS_TYPE_ENCODE ? "encode" : "decode"},
                {"context", reinterpret_cast<uintptr_t>(ctx)}, {"thread", std::hash<std::thread::id>{}(std::this_thread::get_id())},
                {"parent_call", active_calls.empty() ? 0 : active_calls.back()},
                {"active_target_call", target_stack.empty() ? 0 : target_stack.back()},
                {"last_completed_target_call", last_target},
                {"batch", batch_snapshot(tr, id, *batch, eagle)}};
            tr.append(before);
        }
        active_calls.push_back(id);
        if (target) target_stack.push_back(id);
        // No trace mutex spans the real call: nested EAGLE invocations remain safe.
        const int32_t result = actual(ctx, type, batch);
        if (target) { target_stack.pop_back(); last_target = id; }
        active_calls.pop_back();
        // Read only already-requested outputs. Never enable embeddings or change context.
        if (eagle && result == 0) llama_synchronize(ctx);
        {
            std::lock_guard<std::mutex> lock(tr.mutex);
            json after = {{"phase", "after"}, {"call", id}, {"architecture", arch},
                          {"result", result}, {"eagle_outputs_requested_only", true}};
            if (eagle && result == 0) {
                json selected = json::array();
                std::vector<float> states;
                for (size_t i = 0; i < batch->tokens.size(); ++i) {
                    if (!batch->tokens[i].output) continue;
                    const float * state = llama_get_embeddings_nextn_ith(ctx, int32_t(i));
                    require(state, "selected native nextn output unavailable");
                    states.insert(states.end(), state, state + H);
                    json row = {{"batch_row", i}, {"saved_row", selected.size()}};
                    if (type == LLAMA_PROCESS_TYPE_DECODE) {
                        // llama_get_logits_ith prefers backend sampled logits, whose
                        // rows may be compact and require their explicit candidate IDs.
                        const float * sampled = llama_get_sampled_logits_ith(ctx, int32_t(i));
                        const uint32_t count = llama_get_sampled_logits_count_ith(ctx, int32_t(i));
                        const llama_token * ids = llama_get_sampled_candidates_ith(ctx, int32_t(i));
                        const uint32_t ids_count = llama_get_sampled_candidates_count_ith(ctx, int32_t(i));
                        if (sampled) {
                            require(count > 0 && count <= V && ids && ids_count == count,
                                    "backend sampled logits lack aligned explicit candidate IDs");
                            std::vector<int> candidates(ids, ids + count);
                            require(std::set<int>(candidates.begin(), candidates.end()).size() == count,
                                    "duplicate backend sampled candidate IDs");
                            int best = 0;
                            for (uint32_t j = 0; j < count; ++j) {
                                require(ids[j] >= 0 && ids[j] < V && std::isfinite(sampled[j]),
                                        "invalid backend candidate ID/logit");
                                if (sampled[j] > sampled[best]) best = int(j);
                            }
                            row["logits_mode"] = "backend_explicit_candidate_pairs";
                            row["candidate_ids"] = candidates;
                            row["candidate_count"] = count;
                            row["candidate_logits"] = tr.floats(stem(id) + "-row-" + std::to_string(i) +
                                "-candidate-logits.f32", sampled, 1, count);
                            row["argmax_target_token"] = ids[best];
                            row["backend_sampled_token"] = llama_get_sampled_token_ith(ctx, int32_t(i));
                        } else {
                            require(count == 0 && ids_count == 0,
                                    "inconsistent absent backend sampled logits");
                            const float * full = llama_get_logits_ith(ctx, int32_t(i));
                            require(full, "selected native dense decoder logits unavailable");
                            int proposal = 0;
                            for (int j = 0; j < V; ++j) {
                                require(!std::isnan(full[j]) && full[j] != INFINITY,
                                        "invalid dense logits; unsupported negative infinity is permitted");
                                if (full[j] > full[proposal]) proposal = j;
                            }
                            std::vector<float> mapped;
                            for (int token_id : tr.map) mapped.push_back(full[token_id]);
                            row["logits_mode"] = "dense_target_vocab_mapped_subset";
                            row["argmax_target_token"] = proposal;
                            row["mapped_logits"] = tr.floats(stem(id) + "-row-" + std::to_string(i) +
                                "-mapped-logits.f32", mapped.data(), 1, D);
                        }
                    }
                    selected.push_back(row);
                }
                after["selected"] = selected;
                if (!states.empty()) after[type == LLAMA_PROCESS_TYPE_ENCODE ? "encoded_fc" : "prenorm"] =
                    tr.floats(stem(id) + "-output-prenorm.f32", states.data(), selected.size(), H);
                if (type == LLAMA_PROCESS_TYPE_DECODE) after["logits_schema"] = "per_selected_row_explicit_mode_v2";
            }
            tr.append(after);
        }
        return result;
    } catch (const std::exception & error) { fatal(error.what()); }
    catch (...) { fatal("unknown trace exception"); }
}
