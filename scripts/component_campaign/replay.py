"""Clean CUDA-event full forwards with model-specific cache history and declared formats."""
import argparse,json,os,subprocess
from .common import *
from .archive import hydrate,release,reference

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',type=Path);ap.add_argument('--cache-map',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reps',type=int,default=5);ap.add_argument('--runtime',type=Path,default=R/'runtime03');ap.add_argument('--audit',action='store_true');a=ap.parse_args()
 a.out.mkdir(exist_ok=False);source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';runtime=a.runtime;exe=R/'replay-binaries'/pin(ROOT/'scripts/component_campaign/replay.cpp')['sha256'][:12];exe.parent.mkdir(exist_ok=True)
 cmd=['c++','-O2','-std=c++17','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor/nlohmann'),str(ROOT/'scripts/component_campaign/replay.cpp'),'-L'+str(runtime/'bin'),'-lllama','-lggml-base','-lggml','-ldl','-Wl,-rpath,'+str(runtime/'bin'),'-o',str(exe)]
 if not exe.exists():subprocess.run(cmd,check=True)
 arms=json.load(open(a.arms));cache=json.load(open(a.cache_map)) if a.cache_map else {};records=[]
 for rep in range(a.reps):
  order=list(arms) if rep%2==0 else list(reversed(arms))
  for name in order:
   pause();model=arms[name];created=hydrate(model)
   try:
    for domain in ('code','prose','reasoning'):
     capture=P/f'runs/dflash-modern-capture-20261104-04/data/development-{domain}-00/capture';meta=json.load(open(capture/'capture.json'));anchor=meta['prompt_tokens']+5
     dest=a.out/f'{rep:02d}-{name}-{domain}.json'
     env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_'))}
     env.update(LD_LIBRARY_PATH=str(runtime/'bin'),LD_PRELOAD=str(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',EXEC_REQUEST_MASKS='1',COMPONENT_K=cache.get(name,['f16','f16'])[0],COMPONENT_V=cache.get(name,['f16','f16'])[1])
     argv=[str(exe),str(TARGET),str(model),str(capture),str(anchor),'1',str(dest),str(int(a.audit))]
     with dest.with_suffix('.log').open('w') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
     row=dict(rep=rep,arm=name,domain=domain,model=pin(model),cache=cache.get(name,['f16','f16']),result=json.load(open(dest)))
     if not a.audit and any(x['graph_count']!=1 or x['graph_event_ms']<=0 for x in row['result']['rows'] if not x['warmup']):raise ValueError('clean warmed CUDA graph region missing')
     records.append(row);save(a.out/'progress.json',records)
   finally:release(model,created)
   print('REPLAY_COMPLETE',rep,name,flush=True)
 save(a.out/'result.json',dict(status='COMPLETE',records=records,source=pin(__file__),cpp=pin(ROOT/'scripts/component_campaign/replay.cpp'),runtime=pin(runtime/'build.json'),audit=a.audit,scope=f'{a.reps} alternating process-cell repetitions; two same-process warmups per cell; three consumed DEV domains and3changing contexts each. '+('Diagnostic callback allocation/logit audit; timing excluded.' if a.audit else 'CUDA event graph region, no backend sampler. Host spans and catchup separate.')+' Cache formats explicit.'))

if __name__=='__main__':main()
