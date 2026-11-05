// Diagnostic host spans only. No synchronization, model, batch or sampler changes.
#include "llama.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <mutex>
#include <nvtx3/nvToolsExt.h>
struct common_speculative;
struct common_batch;
static thread_local int nesting=0;
static thread_local const char * phase="none";
static std::mutex output_lock;
static uint64_t nanos(){ timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec; }
static FILE * output(){static FILE * f=[](){const char *p=getenv("DECODE_COST_TRACE");if(!p) std::abort();FILE *r=fopen(p,"wx");if(!r) std::abort();setvbuf(r,nullptr,_IOFBF,256*1024);return r;}();return f;}
template<typename F> static F next(const char * name){void * f=dlsym(RTLD_NEXT,name);if(!f){fprintf(stderr,"cost trace missing %s\n",name);std::abort();}return reinterpret_cast<F>(f);}
struct span {
 const char * name; const char * prior; uint64_t start;int depth;void * ctx;
 span(const char*n,void*c=nullptr,bool setphase=false):name(n),prior(phase),depth(nesting++),ctx(c){if(setphase)phase=n;nvtxRangePushA(n);start=nanos();}
 ~span(){auto end=nanos();nvtxRangePop();--nesting;phase=prior;std::lock_guard<std::mutex> guard(output_lock);fprintf(output(),"{\"name\":\"%s\",\"phase\":\"%s\",\"start_ns\":%llu,\"end_ns\":%llu,\"depth\":%d,\"context\":\"%p\"}\n",name,prior,(unsigned long long)start,(unsigned long long)end,depth,ctx);}
};
extern "C" int32_t llama_process(llama_context * c,llama_process_type t,llama_batch_ext * b){static auto f=next<decltype(&llama_process)>("llama_process");span s(nesting==0?"target_process":(t==LLAMA_PROCESS_TYPE_ENCODE?"draft_encode":"draft_decode"),c);return f(c,t,b);}
extern "C" void llama_synchronize(llama_context*c){static auto f=next<decltype(&llama_synchronize)>("llama_synchronize");span s(nesting==0?"target_sync":"nested_sync",c);f(c);}
extern "C" bool trace_process(common_speculative *,const common_batch&) asm("_Z26common_speculative_processP18common_speculativeRK12common_batch");
extern "C" bool trace_process(common_speculative*s,const common_batch&b){static auto f=next<decltype(&trace_process)>("_Z26common_speculative_processP18common_speculativeRK12common_batch");span timer("draft_catchup",s,true);return f(s,b);}
extern "C" void trace_draft(common_speculative *) asm("_Z24common_speculative_draftP18common_speculative");
extern "C" void trace_draft(common_speculative*s){static auto f=next<decltype(&trace_draft)>("_Z24common_speculative_draftP18common_speculative");span timer("draft_generate",s,true);f(s);}
extern "C" void trace_accept(common_speculative *,int,uint16_t) asm("_Z25common_speculative_acceptP18common_speculativeit");
extern "C" void trace_accept(common_speculative*s,int seq,uint16_t n){static auto f=next<decltype(&trace_accept)>("_Z25common_speculative_acceptP18common_speculativeit");span timer("draft_accept",s,true);f(s,seq,n);}
// std::vector is passed by reference; retain its ABI opaquely without inspecting it.
extern "C" void trace_begin(common_speculative *,int,const void *) asm("_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");
extern "C" void trace_begin(common_speculative*s,int seq,const void*p){static auto f=next<decltype(&trace_begin)>("_Z24common_speculative_beginP18common_speculativeiRKSt6vectorIiSaIiEE");span timer("generation_begin",s,true);f(s,seq,p);}
#include <vector>
#include <string>
struct common_sampler;
extern "C" llama_token trace_sample(common_sampler*,llama_context*,int,bool) asm("_Z21common_sampler_sampleP14common_samplerP13llama_contextib");
extern "C" llama_token trace_sample(common_sampler*s,llama_context*c,int i,bool g){static auto f=next<decltype(&trace_sample)>("_Z21common_sampler_sampleP14common_samplerP13llama_contextib");span timer(nesting==0?"target_sample":"nested_sample",c);auto token=f(s,c,i,g);if(phase==std::string("draft_generate")){static FILE*pf=[](){const char*p=getenv("EXEC_PROPOSAL_TRACE");if(!p)std::abort();FILE*f=fopen(p,"wx");if(!f)std::abort();return f;}();std::lock_guard<std::mutex>guard(output_lock);fprintf(pf,"{\"time_ns\":%llu,\"index\":%d,\"token\":%d}\n",(unsigned long long)nanos(),i,token);}return token;}
extern "C" std::vector<int> trace_verify(common_sampler*,llama_context*,const std::vector<int>&,const std::vector<int>&,bool) asm("_Z34common_sampler_sample_and_accept_nP14common_samplerP13llama_contextRKSt6vectorIiSaIiEES7_b");
extern "C" std::vector<int> trace_verify(common_sampler*s,llama_context*c,const std::vector<int>&i,const std::vector<int>&d,bool g){static auto f=next<decltype(&trace_verify)>("_Z34common_sampler_sample_and_accept_nP14common_samplerP13llama_contextRKSt6vectorIiSaIiEES7_b");span timer("verifier_sample",c,true);return f(s,c,i,d,g);}
