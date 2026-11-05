// CUDA-event graph regions only. No per-kernel synchronization.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <vector>
struct Pair {cudaEvent_t a,b;};
static thread_local bool active=false;
static thread_local std::vector<Pair> pairs;
extern "C" void exec_region_begin(){active=true;pairs.clear();}
extern "C" float exec_region_end(int*count){active=false;float total=0;for(auto p:pairs){float ms=0;cudaEventSynchronize(p.b);cudaEventElapsedTime(&ms,p.a,p.b);total+=ms;cudaEventDestroy(p.a);cudaEventDestroy(p.b);}*count=int(pairs.size());pairs.clear();return total;}
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t g,cudaStream_t s){
 static auto real=(cudaError_t(*)(cudaGraphExec_t,cudaStream_t))dlsym(RTLD_NEXT,"cudaGraphLaunch");
 if(!active)return real(g,s);
 Pair p;cudaEventCreate(&p.a);cudaEventCreate(&p.b);cudaEventRecord(p.a,s);auto result=real(g,s);cudaEventRecord(p.b,s);pairs.push_back(p);return result;
}
