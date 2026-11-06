#include "llama.h"
#include <cstdio>
int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "input output\n"); return 2; }
    auto params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    params.nthread = 2;
    return llama_model_quantize(argv[1], argv[2], &params);
}
