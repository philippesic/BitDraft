from pathlib import Path
import json,os,sys,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
from scripts.dflash_baseline.native_archive import hydrate,release
R=P/'runs/execution-campaign-20261105-01';r=R/'replay-final';r.mkdir(exist_ok=False);arms=json.load(open(R/'arms.json'));created=hydrate(arms['w1']);rows=[]
try:
 for rep in range(5):
  order=['w1_original','w1_optimized','q4','q4_optimized'];order=order if rep%2==0 else order[::-1]
  for domain in ['code','reasoning','prose']:
   capture=P/f'runs/dflash-modern-capture-20261104-04/data/development-{domain}-00/capture';meta=json.load(open(capture/'capture.json'));anchor=meta['prompt_tokens']+5
   for arm in order:
    if json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']:raise RuntimeError('user pause')
    opt=arm.endswith('optimized');runtime=R/'runtime-combined-last02' if opt else F;model=arms['q4'] if arm.startswith('q4') else arms['w1'];out=r/f'{rep:02d}-{domain}-{arm}.json'
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_','EXEC_'))};env.update(LD_LIBRARY_PATH=str(runtime/'bin'),LD_PRELOAD=str(R/'libgraph-timer.so'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
    if opt:env['EXEC_REQUEST_MASKS']='1'
    argv=[str(R/'replay-v6'),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),model,str(capture),str(anchor),'1',str(out),'0']
    with out.with_suffix('.log').open('w') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=json.load(open(out));rows.extend(dict(x,arm=arm,rep=rep,domain=domain) for x in data['rows']);(r/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('FINAL_REPLAY',rep,domain,arm,flush=True)
 # Full output/cache byte identity per history/context; compare required mask outputs.
 checks=[]
 for domain in ['code','reasoning','prose']:
  for rep in range(5):
   for base,opt in [('w1_original','w1_optimized'),('q4','q4_optimized')]:
    a=[x for x in rows if x['domain']==domain and x['rep']==rep and x['arm']==base];b=[x for x in rows if x['domain']==domain and x['rep']==rep and x['arm']==opt]
    ok=all(x['outputs'][1:]==y['outputs'] and x['cache_hash_fnv1a64']==y['cache_hash_fnv1a64'] and x['cache_bytes']==y['cache_bytes'] for x,y in zip(a,b));checks.append(dict(domain=domain,rep=rep,base=base,optimized=opt,required_logits_cache_exact=ok));assert ok,checks[-1]
 measured=[x for x in rows if not x['warmup']];assert all(x['graph_count']==1 for x in measured),'cold or non-graph final timing'
 (r/'result.json').write_text(json.dumps(dict(rows=rows,checks=checks,scope='3consumed captureDEV domains; identical external features/history; own cache rebuilt every iteration;3changing contexts;2warmup+5alternating process-cell reps;graph event region and synchronized host forward; backend sampler absent; required mask logits/cache exact',sources=dict(replay=pin(R/'replay-v6'),timer=pin(R/'libgraph-timer.so'),script=pin(Path(__file__)))),indent=2)+'\n')
finally:release(arms['w1'],created)
