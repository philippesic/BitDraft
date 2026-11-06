// Diagnostic only: observe the actual selector lattice read by speculative.cpp.
#include "llama.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
int32_t llama_model_dflash_selector_top_k(const llama_model *);
float * llama_get_embeddings_nextn(llama_context *);
float * llama_get_embeddings_nextn(llama_context *ctx) {
    static auto real = reinterpret_cast<decltype(&llama_get_embeddings_nextn)>(dlsym(RTLD_NEXT,"_Z26llama_get_embeddings_nextnP13llama_context"));
    float *out = real(ctx);
    auto *model = llama_get_model(ctx);
    int k = llama_model_dflash_selector_top_k(model), width = llama_model_n_embd(model);
    if (k > 0) {
        if (!out || width < k*(k+1)) std::abort();
        // Row1 is always a requested draft position at the frozen depths2/3.
        const float *row = out + width;
        int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        for (int i=0;i<k;++i) if (!std::isfinite(row[i]) || row[i]!=int(row[i]) || row[i]<0 || row[i]>=nv) std::abort();
        for (int i=k;i<k+k*k;++i) if (!std::isfinite(row[i])) std::abort();
        static FILE *f = [](){auto *p=getenv("DFLASH2_SELECTOR_AUDIT");if(!p)std::abort();auto *f=std::fopen(p,"wx");if(!f)std::abort();return f;}();
        std::fprintf(f,"{\"selector_top_k\":%d,\"width\":%d,\"finite_scores\":%d,\"valid_candidate_ids\":%d,\"first_candidate\":%d}\n",k,width,k*k,k,int(row[0]));
        std::fflush(f);
    }
    return out;
}
