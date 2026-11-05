"""Measured same-runtime selection/final panels; no heldout payload selection here."""
import argparse,json,subprocess,sys
from .common import *
from .archive import available

def completed():
 rows={}
 for out in sorted((R/'arms').iterdir()):
  if out.name=='H0-pilot' or not (out/'result.json').exists():continue
  result=json.load(open(out/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']<4096:continue
  model=out/f"model-{result['progress']['updates']}.gguf"
  if not available(model):raise ValueError('completed native model not recoverable '+str(model))
  rows[out.name]=dict(model=str(model),spec=result['spec'],result=pin(out/'result.json'))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--arms',nargs='*');ap.add_argument('--reps',type=int,default=5);ap.add_argument('--trace',action='store_true');ap.add_argument('--cohort',choices=['capture3','natural12'],default='natural12');ap.add_argument('--runtime',type=Path);a=ap.parse_args()
 registry=completed();names=a.arms or list(registry);models=dict(target=None,q4=str(Q4),B0=str(B/'model-final.gguf'));cache={}
 for name in names:
  if name not in registry:raise ValueError('unknown complete arm '+name)
  models[name]=registry[name]['model'];cache[name]=registry[name]['spec'].get('kv',['f16','f16'])
 policy=json.load(open(R/'runtime-policy.json')) if (R/'runtime-policy.json').exists() else {}
 runtime=a.runtime or Path(policy.get('runtime',str(R/'runtime03')))
 out=R/'measurements'/a.tag;out.parent.mkdir(exist_ok=True)
 config=R/'panels'/(a.tag+'-arms.json');save(config,models);cachefile=R/'panels'/(a.tag+'-cache.json');save(cachefile,cache)
 save(R/'panels'/(a.tag+'-plan.json'),dict(arms=registry,names=names,cohort=a.cohort,repetitions=a.reps,warmups=2,runtime=pin(runtime/'build.json'),scope='Formal comparison' if a.reps>=5 else 'Consumed DEV selection smoke; no final timing claim',trace=a.trace))
 argv=[sys.executable,'-m','scripts.component_campaign.native','--arms',config,'--draft-cache-map',cachefile,'--runtime',runtime,'--out',out,'--cohort',a.cohort,'--cohort-label',a.tag,'--reps',str(a.reps),'--warmups','2']
 if a.trace:argv+=['--trace']
 subprocess.run(list(map(str,argv)),cwd=ROOT,check=True)
 print('MEASUREMENT_COMPLETE',a.tag,flush=True)

if __name__=='__main__':main()
