// Non-mutating failure probe, separate from clean latency and the strict audit.
#include "llama.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
int32_t llama_model_dflash_selector_top_k(const llama_model *);
float * llama_get_embeddings_nextn(llama_context *);
float * llama_get_embeddings_nextn(llama_context *ctx) {
    static auto real=reinterpret_cast<decltype(&llama_get_embeddings_nextn)>(dlsym(RTLD_NEXT,"_Z26llama_get_embeddings_nextnP13llama_context"));
    float *out=real(ctx);auto *model=llama_get_model(ctx);
    int k=llama_model_dflash_selector_top_k(model),width=llama_model_n_embd(model);
    if(k>0){
        static FILE *f=[](){const char *p=getenv("DFLASH2_SELECTOR_AUDIT");return p?std::fopen(p,"wx"):stderr;}();
        if(!f)std::abort();
        if(!out){std::fprintf(f,"{\"null\":true}\n");std::fflush(f);return out;}
        const float *row=out+width;int nv=llama_vocab_n_tokens(llama_model_get_vocab(model)),valid=0,finite=0;
        for(int i=0;i<k;++i)valid+=std::isfinite(row[i])&&row[i]>=0&&row[i]<nv&&row[i]==std::floor(row[i]);
        for(int i=k;i<k+k*k;++i)finite+=std::isfinite(row[i]);
        unsigned bits;std::memcpy(&bits,row,4);
        std::fprintf(f,"{\"selector_top_k\":%d,\"width\":%d,\"finite_scores\":%d,\"valid_candidate_ids\":%d,\"first_candidate_text\":\"%a\",\"first_candidate_bits\":\"%08x\",\"first_score_text\":\"%a\"}\n",k,width,finite,valid,double(row[0]),bits,double(row[k]));std::fflush(f);
    }
    return out;
}
