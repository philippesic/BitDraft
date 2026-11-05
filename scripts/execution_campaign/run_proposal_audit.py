from pathlib import Path
import sys,subprocess,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
R=P/'runs/execution-campaign-20261105-01';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';s=(ROOT/'scripts/data_cost_transfer/cost_trace.cpp').read_text();s=s.replace('#include <mutex>','#include <mutex>\n#include <cstring>\nstatic thread_local int proposal_index=-1;');s=s.replace('return f(s,c,i,g);','if(strcmp(phase,"draft_generate")==0)proposal_index=i;return f(s,c,i,g);')
s+='''\nextern "C" void trace_sampler_accept(common_sampler*,llama_token,bool) asm("_Z21common_sampler_acceptP14common_samplerib");
extern "C" void trace_sampler_accept(common_sampler*s,llama_token token,bool generated){
 static auto f=next<decltype(&trace_sampler_accept)>("_Z21common_sampler_acceptP14common_samplerib");
 if(strcmp(phase,"draft_generate")==0){static FILE*pf=[](){const char*p=getenv("EXEC_PROPOSAL_TRACE");if(!p)std::abort();FILE*f=fopen(p,"wx");if(!f)std::abort();return f;}();std::lock_guard<std::mutex>guard(output_lock);fprintf(pf,"{\\"time_ns\\":%llu,\\"index\\":%d,\\"token\\":%d}\\n",(unsigned long long)nanos(),proposal_index,token);}
 f(s,token,generated);
}
'''
source=R/'proposal-accept-trace.cpp';source.write_text(s);lib=R/'libproposal-accept-trace.so';cmd=['c++','-O2','-shared','-fPIC','-std=c++17','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-I/usr/local/cuda/include',str(source),'-ldl','-pthread','-o',str(lib)];subprocess.run(cmd,check=True);(R/'proposal-accept-build.json').write_text(json.dumps(dict(command=cmd,source=pin(source),library=pin(lib),reason='DFlash ignores common_sampler_sample return; actual selected cur_p[0].id is passed into common_sampler_accept before result.push_back. Prior return-value traces preserved as superseded diagnostic evidence.'),indent=2)+'\n')
env={**os.environ,'EXEC_TRACE_LIBRARY':str(lib)};argv=[sys.executable,str(ROOT/'scripts/execution_campaign/native.py'),'--arms',str(R/'final-arms.json'),'--runtime',str(F),'--runtime-map',str(R/'final-runtime-map.json'),'--out',str(R/'native-proposal-audit'),'--cohort','natural12','--warmups','2','--reps','1','--trace'];subprocess.run(argv,env=env,check=True)
