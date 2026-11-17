#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct VocabularyCase {
    std::string directory, variant, output;
    int type;
    int64_t rows, width;
    int slots, sets, keep, shortlist, rank, warmup, repeats, trials;
};

struct VocabularyGraph {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * scores = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * candidates = nullptr;
    ggml_tensor * rescored = nullptr;
    std::vector<int32_t> identifiers;
    cudaGraphExec_t replay = nullptr;
    ~VocabularyGraph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (context) ggml_free(context);
    }
};

static std::string vocabulary_file(const VocabularyCase & c, const std::string & name) {
    return c.directory + "/" + name;
}

template<class T> static void vocabulary_upload(ggml_tensor * tensor, const std::string & path) {
    const auto values = read_values<T>(path, ggml_nbytes(tensor) / sizeof(T));
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(T));
}

template<class T> static std::vector<T> vocabulary_read(ggml_tensor * tensor) {
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(T));
    return values;
}

static void vocabulary_build(VocabularyGraph & g, const VocabularyCase & c) {
    ggml_init_params parameters = {1024 * 1024, nullptr, true};
    g.context = ggml_init(parameters);
    if (!g.context) throw std::runtime_error("ggml init failed");
    g.backend = ggml_backend_cuda_init(0);
    if (!g.backend) throw std::runtime_error("CUDA backend init failed");
    ggml_context * ctx = g.context;
    const auto kind = static_cast<ggml_type>(c.type);
    g.hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.width, c.slots);
    ggml_tensor * table = nullptr;
    ggml_tensor * subtable = nullptr;
    ggml_tensor * signs = nullptr;
    ggml_tensor * scales = nullptr;
    ggml_tensor * left = nullptr;
    ggml_tensor * right = nullptr;
    if (c.variant == "topk" || c.variant == "argmax") {
        table = ggml_new_tensor_2d(ctx, kind, c.width, c.rows);
        g.scores = ggml_mul_mat(ctx, table, g.hidden);
        g.selected = c.variant == "topk" ? ggml_top_k(ctx, g.scores, 10) : ggml_argmax(ctx, g.scores);
    } else if (c.variant == "shortlist") {
        subtable = ggml_new_tensor_2d(ctx, kind, c.width, c.shortlist);
        g.scores = ggml_mul_mat(ctx, subtable, g.hidden);
        g.selected = ggml_argmax(ctx, g.scores);
    } else if (c.variant == "one-bit" || c.variant == "low-rank") {
        table = ggml_new_tensor_2d(ctx, kind, c.width, c.rows);
        if (c.variant == "one-bit") {
            signs = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.width / 32, c.rows);
            scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.rows);
            g.scores = ggml_w1ax_mul_mat(ctx, signs, scales, g.hidden, c.width, 8);
        } else {
            right = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.width, c.rank);
            left = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.rank, c.rows);
            g.scores = ggml_mul_mat(ctx, left, ggml_mul_mat(ctx, right, g.hidden));
        }
        g.candidates = ggml_top_k(ctx, g.scores, c.keep);
        ggml_tensor * rows = ggml_get_rows(ctx, table, ggml_reshape_1d(ctx, g.candidates, int64_t(c.keep) * c.slots));
        ggml_tensor * gathered = ggml_reshape_3d(ctx, rows, c.width, c.keep, c.slots);
        ggml_tensor * columns = ggml_reshape_3d(ctx, g.hidden, c.width, 1, c.slots);
        g.rescored = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, gathered, columns), c.keep, c.slots);
        g.selected = ggml_argmax(ctx, g.rescored);
    } else {
        throw std::runtime_error("unknown vocabulary variant");
    }
    g.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(g.graph, g.selected);
    g.buffer = ggml_backend_alloc_ctx_tensors(ctx, g.backend);
    if (!g.buffer) throw std::runtime_error("ggml allocation failed");
    if (table) vocabulary_upload<uint8_t>(table, vocabulary_file(c, "table.bin"));
    if (subtable) {
        vocabulary_upload<uint8_t>(subtable, vocabulary_file(c, "shortlist-" + std::to_string(c.shortlist) + ".bin"));
        g.identifiers = read_values<int32_t>(vocabulary_file(c, "shortlist-" + std::to_string(c.shortlist) + ".i32"), c.shortlist);
    }
    if (signs) {
        vocabulary_upload<int32_t>(signs, vocabulary_file(c, "signs.u32"));
        vocabulary_upload<float>(scales, vocabulary_file(c, "scales.f32"));
    }
    if (left) {
        vocabulary_upload<uint16_t>(left, vocabulary_file(c, "low-rank-left.f16"));
        vocabulary_upload<uint16_t>(right, vocabulary_file(c, "low-rank-right.f16"));
    }
}

static void vocabulary_set(VocabularyGraph & g, const std::vector<float> & hidden, const VocabularyCase & c, int set) {
    const size_t count = size_t(c.slots) * c.width;
    ggml_backend_tensor_set(g.hidden, hidden.data() + size_t(set) * count, 0, count * sizeof(float));
}

static void vocabulary_run(VocabularyGraph & g) {
    if (g.replay) {
        auto * cuda = static_cast<ggml_backend_cuda_context *>(g.backend->context);
        checked(cudaGraphLaunch(g.replay, cuda->stream()));
    } else if (ggml_backend_graph_compute(g.backend, g.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml execution failed");
    }
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static std::vector<int32_t> vocabulary_choices(VocabularyGraph & g, const VocabularyCase & c, std::vector<float> & values, std::vector<int32_t> & candidates) {
    auto * cuda = static_cast<ggml_backend_cuda_context *>(g.backend->context);
    checked(cudaStreamSynchronize(cuda->stream()));
    const auto selected = vocabulary_read<int32_t>(g.selected);
    std::vector<int32_t> tokens(c.slots);
    values.assign(c.slots, 0.0f);
    if (c.variant == "topk" || c.variant == "argmax" || c.variant == "shortlist") {
        const auto scores = vocabulary_read<float>(g.scores);
        const int64_t columns = g.scores->ne[0];
        for (int slot = 0; slot < c.slots; ++slot) {
            int64_t best = -1;
            if (c.variant == "topk") {
                for (int k = 0; k < 10; ++k) {
                    const int64_t candidate = selected[size_t(slot) * 10 + k];
                    if (candidate < 0 || candidate >= columns) throw std::runtime_error("top-k index out of range");
                    const float value = scores[size_t(slot) * columns + candidate];
                    const float current = best < 0 ? 0.0f : scores[size_t(slot) * columns + best];
                    if (best < 0 || value > current || (value == current && candidate < best)) best = candidate;
                }
            } else {
                best = selected[slot];
                if (best < 0 || best >= columns) throw std::runtime_error("argmax index out of range");
            }
            values[slot] = scores[size_t(slot) * columns + best];
            tokens[slot] = c.variant == "shortlist" ? g.identifiers[best] : int32_t(best);
        }
    } else {
        const auto rescored = vocabulary_read<float>(g.rescored);
        candidates = vocabulary_read<int32_t>(g.candidates);
        for (int slot = 0; slot < c.slots; ++slot) {
            const int32_t best = selected[slot];
            if (best < 0 || best >= c.keep) throw std::runtime_error("rescored index out of range");
            values[slot] = rescored[size_t(slot) * c.keep + best];
            tokens[slot] = candidates[size_t(slot) * c.keep + best];
        }
    }
    return tokens;
}

static void vocabulary(const VocabularyCase & c) {
    VocabularyGraph g;
    vocabulary_build(g, c);
    const auto hidden = read_values<float>(vocabulary_file(c, "hidden.f32"), size_t(c.sets) * c.slots * c.width);
    for (float value : hidden) if (!std::isfinite(value)) throw std::runtime_error("nonfinite hidden state");
    vocabulary_set(g, hidden, c, 0);
    for (int i = 0; i < 4; ++i) {
        if (ggml_backend_graph_compute(g.backend, g.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml execution failed");
    }
    auto * cuda = static_cast<ggml_backend_cuda_context *>(g.backend->context);
    std::vector<cudaGraphExec_t> executions;
    for (const auto & item : cuda->cuda_graphs)
        for (auto instance : item.second->instances) executions.push_back(instance);
    if (executions.size() == 1) g.replay = executions[0];
    printf("mode,%s\n", g.replay ? "graph" : "compute");
    std::vector<float> values;
    std::vector<int32_t> candidates;
    std::vector<int32_t> all_candidates;
    std::vector<int32_t> reference;
    for (int set = 0; set < c.sets; ++set) {
        vocabulary_set(g, hidden, c, set);
        vocabulary_run(g);
        const auto tokens = vocabulary_choices(g, c, values, candidates);
        if (set == 0) reference = tokens;
        for (int slot = 0; slot < c.slots; ++slot) printf("choice,%d,%d,%d,%08x\n", set, slot, tokens[slot], float_bits(values[slot]));
        all_candidates.insert(all_candidates.end(), candidates.begin(), candidates.end());
    }
    if (!all_candidates.empty()) {
        std::ofstream output(c.output + ".candidates", std::ios::binary);
        output.write(reinterpret_cast<const char *>(all_candidates.data()), all_candidates.size() * sizeof(int32_t));
        if (!output) throw std::runtime_error("candidate output failed");
    }
    vocabulary_set(g, hidden, c, 0);
    cudaEvent_t start, stop;
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    for (int j = 0; j < c.warmup; ++j) vocabulary_run(g);
    checked(cudaStreamSynchronize(cuda->stream()));
    printf("warmup,%d,per_trial,%d\n", c.warmup, std::min(c.warmup, 200));
    bool stable = true;
    for (int trial = -1; trial < c.trials; ++trial) {
        for (int j = 0; j < std::min(c.warmup, 200); ++j) vocabulary_run(g);
        checked(cudaEventRecord(start, cuda->stream()));
        for (int j = 0; j < c.repeats; ++j) vocabulary_run(g);
        checked(cudaEventRecord(stop, cuda->stream()));
        checked(cudaEventSynchronize(stop));
        checked(cudaGetLastError());
        float milliseconds;
        checked(cudaEventElapsedTime(&milliseconds, start, stop));
        stable = stable && vocabulary_choices(g, c, values, candidates) == reference;
        if (trial >= 0) printf("sample,%d,%.12g,%d\n", trial, double(milliseconds) * 1000 / c.repeats, stable);
    }
    checked(cudaEventDestroy(start));
    checked(cudaEventDestroy(stop));
}

static int vocabulary_main(int argc, char ** argv) {
    if (argc != 17 || std::string(argv[16]) != "v1") throw std::runtime_error("vocabulary directory variant output type rows width slots sets keep shortlist rank warmup repeats trials v1 required");
    VocabularyCase c;
    c.directory = argv[2];
    c.variant = argv[3];
    c.output = argv[4];
    c.type = std::stoi(argv[5]);
    c.rows = std::stoll(argv[6]);
    c.width = std::stoll(argv[7]);
    c.slots = std::stoi(argv[8]);
    c.sets = std::stoi(argv[9]);
    c.keep = std::stoi(argv[10]);
    c.shortlist = std::stoi(argv[11]);
    c.rank = std::stoi(argv[12]);
    c.warmup = std::stoi(argv[13]);
    c.repeats = std::stoi(argv[14]);
    c.trials = std::stoi(argv[15]);
    if ((c.type != GGML_TYPE_Q4_K && c.type != GGML_TYPE_Q6_K) || c.rows <= 0 || c.width <= 0 || c.width % 256 || c.slots < 1 || c.slots > 16 || c.sets < 1 || c.keep < 0 || c.keep > c.rows || c.shortlist < 0 || c.shortlist > c.rows || c.rank < 0 || c.rank > c.width || c.warmup < 1 || c.repeats < 1 || c.trials < 1)
        throw std::runtime_error("invalid vocabulary benchmark bounds");
    cudaDeviceProp properties;
    checked(cudaGetDeviceProperties(&properties, 0));
    printf("device,%s,SMs=%d,L2_bytes=%d\n", properties.name, properties.multiProcessorCount, properties.l2CacheSize);
    vocabulary(c);
    return 0;
}
