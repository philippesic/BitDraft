#include "common.cuh"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

void ggml_cuda_w1ax_pack(ggml_backend_cuda_context & context, ggml_tensor * destination) {
    using Function = void (*)(ggml_backend_cuda_context &, ggml_tensor *);
    static Function original = reinterpret_cast<Function>(dlsym(RTLD_NEXT,"_Z19ggml_cuda_w1ax_packR25ggml_backend_cuda_contextP11ggml_tensor"));
    if(!original) throw std::runtime_error("cannot resolve original runtime quantizer");
    static std::mutex mutex;
    static int captured = 0;
    const char * directory = std::getenv("BITDRAFT_KERNEL_CAPTURE");
    const char * count_text = std::getenv("BITDRAFT_KERNEL_SLOTS");
    if(directory && count_text) {
        std::lock_guard<std::mutex> guard(mutex);
        const auto root = std::filesystem::path(directory);
        const int slots = std::stoi(count_text);
        const ggml_tensor * input = destination->src[0];
        if(captured < 12 && input->ne[1] == slots && std::filesystem::exists(root / "armed")) {
            const int layer = captured / 2;
            const bool down = captured % 2;
            const int expected_width = down ? 9216 : 2560;
            if(input->type != GGML_TYPE_F32 || input->ne[0] != expected_width || !ggml_is_contiguous(input))
                throw std::runtime_error("unexpected drafter capture geometry");
            std::vector<float> values(size_t(expected_width)*slots);
            CUDA_CHECK(cudaStreamSynchronize(context.stream()));
            CUDA_CHECK(cudaMemcpy(values.data(),input->data,values.size()*sizeof(float),cudaMemcpyDeviceToHost));
            const auto name = "blk."+std::to_string(layer)+".ffn_"+(down?"down":"gate")+"."+std::to_string(slots);
            std::ofstream output(root / (name+".f32"),std::ios::binary);
            output.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
            if(!output) throw std::runtime_error("activation capture write failed");
            std::ofstream log(root / "capture.tsv",std::ios::app);
            log << name << '\t' << input->name << '\t' << destination->op_params[0] << '\n';
            ++captured;
        }
    }
    original(context,destination);
}
