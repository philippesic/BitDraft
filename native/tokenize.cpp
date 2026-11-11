#include "llama.h"
#include "chat.h"
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
        auto templates = common_chat_templates_init(model, "");
        auto caps = common_chat_templates_get_caps(templates.get());
        std::cout << nlohmann::json({{"ready",true},{"vocab_only",true},{"native_chat_template",true},{"template",common_chat_templates_source(templates.get())}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
            if (line.size() > 2 * 1024 * 1024) throw std::runtime_error("prompt request exceeds 2 MiB");
            const auto request = nlohmann::json::parse(line);
            std::string text;
            if (request.contains("messages")) {
                common_chat_templates_inputs inputs;
                inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(request.at("messages").dump()));
                inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(request.value("tools",nlohmann::json::array()).dump()));
                inputs.use_jinja = true;
                inputs.enable_thinking = false;
                inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                inputs.parallel_tool_calls = caps.at("supports_parallel_tool_calls").get<bool>();
                inputs.add_generation_prompt = true;
                text = common_chat_templates_apply(templates.get(), inputs).prompt.str();
            } else text = request.at("text").get<std::string>();
            if (text.size() >= size_t(std::numeric_limits<int32_t>::max())) throw std::runtime_error("input overflow");
            int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
            if (n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("token overflow");
            std::vector<llama_token> ids(n < 0 ? -n : n);
            n = llama_tokenize(vocab, text.data(), text.size(), ids.data(), ids.size(), true, true);
            if (n < 0) throw std::runtime_error("token capacity mismatch");
            ids.resize(n);
            std::cout << nlohmann::json({{"prompt",text},{"tokens",ids},{"n_tokens",n}}).dump() << std::endl;
            } catch (const std::exception & error) {
                std::cout << nlohmann::json({{"error",error.what()}}).dump() << std::endl;
            }
        }
        templates.reset();
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "tokenizer: " << e.what() << std::endl;
        return 1;
    }
}
