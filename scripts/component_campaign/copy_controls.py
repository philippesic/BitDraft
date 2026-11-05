"""Exact private Q6 controls reuse a qualified identical training forward/history."""
import copy,json
from .common import *
from .archive import hydrate,release,archive
from .export import rewrite

def main():
 source=R/'arms/H0';result=json.load(open(source/'result.json'))
 if result['progress']['updates']!=4096 or result['status']!='COMPLETE':raise ValueError('full matched continuation required')
 records=json.load(open(R/'arms/H0-pilot/readouts.json'))[:1]+json.load(open(source/'readouts.json'))
 for name,key in [('H6','head'),('E6','embedding')]:
  out=R/'arms'/name
  if (out/'result.json').exists():continue
  out.mkdir(exist_ok=False);spec=dict(name=name,**{key:'Q6_K'});save(out/'spec.json',spec);readouts=[]
  for record in records:
   model=Path(record['model']['path']);created=hydrate(model)
   try:
    destination=out/f"model-{record['update']}.gguf";rewrite(model,destination,spec)
    changed=copy.deepcopy(record);changed['model']=pin(destination);changed['reused_identical_training_forward']=True
    archive(destination);readouts.append(changed)
   finally:release(model,created)
  save(out/'readouts.json',readouts)
  save(out/'result.json',dict(result,spec=spec,readouts=readouts,physical_updates=0,optimizer_compute_seconds=0,reused_matched_control=pin(source/'result.json'),reuse_qualification='H6/E6 exactly copy targetQ6_K packed codes; Model training already uses this same dequantized frozen head/embedding. Identical trainable parameter groups, common B0, optimizer reset, example order/exposure and full forward values. Native private allocation/execution evaluated separately. No extra training exposure claimed.'))
 print('EXACT_PRIVATE_CONTROLS_DERIVED',flush=True)

if __name__=='__main__':main()
