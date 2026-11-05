from pathlib import Path
import sys,json,subprocess,os,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,pin
from scripts.dflash_baseline.native_archive import hydrate,release
R=P/'runs/execution-campaign-20261105-01';r=R/'replay-methods';r.mkdir(exist_ok=False);arms=json.load(open(R/'arms.json'));created=hydrate(arms['w1']);rows=[]
try:
 for rep in range(5):
  for domain in ['code','reasoning','prose']:
   for arm in (['joint','planes4'] if rep%2==0 else ['planes4','joint']):
    runtime=R/('runtime-combined-last02' if arm=='joint' else 'runtime-planes4');capture=P/f'runs/dflash-modern-capture-20261104-04/data/development-{domain}-00/capture';anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5;env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_','EXEC_'))};env.update(LD_LIBRARY_PATH=str(runtime/'bin'),LD_PRELOAD=str(R/'libgraph-timer.so'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',EXEC_REQUEST_MASKS='1');out=r/f'{rep:02d}-{domain}-{arm}.json'
    with out.with_suffix('.log').open('w') as log:subprocess.run([str(R/'replay-v6'),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),arms['w1'],str(capture),str(anchor),'1',str(out),'0'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rs=json.load(open(out))['rows'];rows.extend(dict(x,rep=rep,domain=domain,arm=arm) for x in rs);(r/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print('METHOD',rep,domain,arm,flush=True)
 checks=[]
 for d in ['code','reasoning','prose']:
  for rep in range(5):
   a=[x for x in rows if x['domain']==d and x['rep']==rep and x['arm']=='joint'];b=[x for x in rows if x['domain']==d and x['rep']==rep and x['arm']=='planes4'];ok=all(x['outputs']==y['outputs'] and x['cache_hash_fnv1a64']==y['cache_hash_fnv1a64'] for x,y in zip(a,b));assert ok;checks.append(dict(domain=d,rep=rep,required_logits_cache_exact=ok))
 measured=[x for x in rows if not x['warmup']];assert all(x['graph_count']==1 for x in measured);rng=np.random.default_rng(818);grid=lambda arm:np.array([[statistics.mean(x['graph_event_ms'] for x in measured if x['arm']==arm and x['domain']==d and x['rep']==rep) for rep in range(5)] for d in ['code','reasoning','prose']]);a=grid('joint');b=grid('planes4');samples=[]
 for _ in range(4000):
  w=np.bincount(rng.integers(3,size=3),minlength=3)[:,None]*np.bincount(rng.integers(5,size=5),minlength=5)[None,:];samples.append(float((b*w).sum()/(a*w).sum()))
 interval=np.quantile(samples,[.025,.975]).tolist();result=dict(rows=rows,checks=checks,gpu_ms=dict(joint=float(a.mean()),planes4=float(b.mean())),planes4_over_joint=dict(value=float(b.sum()/a.sum()),ci95=interval),promote_planes4=interval[1]<1,scope='5alternating pairs across3captureDEV domains,3changing contexts each;2warmups per process;CUDA graph events;exact required logits/cache');(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('rows','checks')},flush=True)
finally:release(arms['w1'],created)
