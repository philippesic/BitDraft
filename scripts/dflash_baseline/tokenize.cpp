// Vocabulary-only admission helper. No context, inference or generated labels.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("model path required");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        auto * model = llama_model_load_from_file(argv[1], params);
        if (!model) throw std::runtime_error("vocabulary load failed");
        auto * vocab = llama_model_get_vocab(model);
        std::cout << "{\"ready\":true,\"vocab_only\":true}" << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            const auto request = nlohmann::json::parse(line);
            const auto text = request.at("text").get<std::string>();
            if (text.size() >= size_t(std::numeric_limits<int32_t>::max())) throw std::runtime_error("input overflow");
            int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
            if (n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("token overflow");
            std::vector<llama_token> ids(n < 0 ? -n : n);
            n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), true, true);
            if (n < 0) throw std::runtime_error("token capacity mismatch");
            ids.resize(n);
            std::cout << nlohmann::json({{"tokens",ids},{"n_tokens",n}}).dump() << std::endl;
        }
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "tokenizer: " << e.what() << std::endl;
        return 1;
    }
}
