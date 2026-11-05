"""Qualify physical half clip metadata against the identical rounded F32 reference."""
import json,subprocess,sys
from .common import *
from .archive import hydrate,release,archive,available,reference
from .export import rewrite

def main():
 out=R/'qualification/clip-storage';out.mkdir(exist_ok=True)
 if (out/'result.json').exists():return
 models={'target':None,'q4':str(Q4),'B0-r3':str(B/'model-final.gguf'),'B0-r4':str(B/'model-final.gguf')};runtime={'B0-r3':str(R/'runtime03')}
 variants={}
 for name in ('Sclip','Sall'):
  original=R/'arms'/name/'model-4096.gguf';result=json.load(open(original.parent/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']!=4096:raise ValueError('scalar recovery must finish first')
  spec=dict(result['spec'],clip_bits_storage=True);dest=out/(name+'-half-clip.gguf')
  if not available(dest):
   created=hydrate(original)
   try:rewrite(original,dest,spec)
   finally:release(original,created)
   archive(dest,R/'bases/shared.gguf')
  variants[name]=dict(model=reference(dest),parent=reference(original),spec=spec,trained_result=pin(original.parent/'result.json'),storage_only_same_half_values=True)
  models[name+'-r3']=str(original);models[name+'-r4']=str(dest);runtime[name+'-r3']=str(R/'runtime03')
 save(out/'arms.json',models);save(out/'runtime-map.json',runtime)
 if not (out/'native/result.json').exists():
  subprocess.run([sys.executable,'-m','scripts.component_campaign.native','--arms',str(out/'arms.json'),'--runtime-map',str(out/'runtime-map.json'),'--runtime',str(R/'runtime04'),'--out',str(out/'native'),'--cohort','capture3','--reps','1','--output-cap','64'],cwd=ROOT,check=True)
 rows=json.load(open(out/'native/result.json'))['rows']
 for base in ('B0','Sclip','Sall'):
  for before in [x for x in rows if x['arm']==base+'-r3']:
   after=next(x for x in rows if x['arm']==base+'-r4' and x['prompt']==before['prompt'])
   if any(before['native'].get(k)!=after['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')):raise ValueError('clip storage changes proposals')
 # Required logits and serialized cache equivalence use the same inputs/masks.
 for version in ('r3','r4'):
  armfile=out/('replay-'+version+'.json');save(armfile,{name:models[name+'-'+version] for name in ('B0','Sclip','Sall')})
  runtime_path=R/('runtime03' if version=='r3' else 'runtime04');dest=out/('replay-'+version)
  if not (dest/'result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.replay',str(armfile),'--out',str(dest),'--runtime',str(runtime_path),'--reps','1'],cwd=ROOT,check=True)
 before=json.load(open(out/'replay-r3/result.json'))['records'];after=json.load(open(out/'replay-r4/result.json'))['records'];checks=0
 for b in before:
  a=next(x for x in after if x['arm']==b['arm'] and x['rep']==b['rep'] and x['domain']==b['domain'])
  for x,y in zip(b['result']['rows'],a['result']['rows'],strict=True):
   if any(x[k]!=y[k] for k in ('outputs','cache_bytes','cache_hash_fnv1a64')):raise ValueError('clip storage changes required logits/cache')
   checks+=1
 save(out/'result.json',dict(status='NATIVE_HALF_CLIP_STORAGE_PASS',variants=variants,replay_row_checks=checks,native=pin(out/'native/result.json'),source=pin(__file__),runtime=pin(R/'runtime04/build.json'),saving='18 binary16 metadata values save36 payload bytes on disk; file alignment/header effects counted separately. Loader converts each once into existing F32 clip op_params; no runtime clip-memory saving claimed.'))
 save(R/'deployment-variants.json',variants)
 save(R/'runtime-policy.json',dict(runtime=str(R/'runtime04'),qualification=pin(out/'result.json'),scope='Half scalar tensor reads and binary16 clip fields supported; optimized CUDA unchanged. All final controls/finalists use this runtime.'))
 print('CLIP_STORAGE_QUALIFIED',flush=True)

if __name__=='__main__':main()
