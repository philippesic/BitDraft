"""Qualify one exact private Q6 table shared by draft head and embedding."""
import json,subprocess,sys
import numpy as np
from .common import *
from .archive import hydrate,release,archive,available,reference
from .export import rewrite
from .quant import target_tensor
from scripts.dflash_baseline.model import GGUFReader

def main():
 out=R/'qualification/tied-q6';out.mkdir(exist_ok=True)
 if (out/'verified.json').exists():return
 spec=dict(name='HE6-exact-admission',head='Q6_K',embedding='Q6_K',tied=True)
 original=B/'model-final.gguf';model=out/'model.gguf'
 if not available(model):
  created=hydrate(original)
  try:rewrite(original,model,spec)
  finally:release(original,created)
  archive(model,R/'bases/Q6_K.gguf')
 created=hydrate(model)
 try:
  reader=GGUFReader(model);head=next(t for t in reader.tensors if t.name=='output.weight')
  if head.tensor_type.name!='Q6_K' or not np.array_equal(head.data,target_tensor(TARGET).data) or any(t.name=='token_embd.weight' for t in reader.tensors):raise ValueError('tied Q6 must use one exact original packed matrix')
 finally:release(model,created)
 arms=dict(target=None,q4=str(Q4),B0=str(original),HE6=str(model));save(out/'arms.json',arms)
 if not (out/'native/result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.native','--arms',str(out/'arms.json'),'--runtime',str(R/'runtime04'),'--out',str(out/'native'),'--cohort','capture3','--output-cap','64','--reps','1'],cwd=ROOT,check=True)
 rows=json.load(open(out/'native/result.json'))['rows']
 for before in [x for x in rows if x['arm']=='B0']:
  after=next(x for x in rows if x['arm']=='HE6' and x['prompt']==before['prompt'])
  if any(before['native'].get(k)!=after['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')):raise ValueError('exact tied Q6 changes proposals')
 save(out/'replay-arms.json',dict(B0=str(original),HE6=str(model)))
 if not (out/'replay/result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.replay',str(out/'replay-arms.json'),'--runtime',str(R/'runtime04'),'--out',str(out/'replay'),'--reps','1','--audit'],cwd=ROOT,check=True)
 records=json.load(open(out/'replay/result.json'))['records'];checks=0
 for baseline in [x for x in records if x['arm']=='B0']:
  private=next(x for x in records if x['arm']=='HE6' and x['domain']==baseline['domain'])
  for a,b in zip(baseline['result']['rows'],private['result']['rows'],strict=True):
   if any(a[k]!=b[k] for k in ('outputs','cache_bytes','cache_hash_fnv1a64')):raise ValueError('exact tied Q6 changes required logits/cache')
   vocab=[v for v in b['audit'] if v.get('kind')=='vocabulary_allocation'];head=[v for v in vocab if v['op']=='MUL_MAT'];embedding=[v for v in vocab if v['op']=='GET_ROWS']
   if not head or not embedding or any(v['weight_type']!='q6_K' and v['weight_type']!='Q6_K' for v in head+embedding):raise ValueError('actual tied Q6 dispatch missing')
   if any(not h['weight_data_address'] or h['weight_data_address']!=e['weight_data_address'] or h['weight_tensor_address']!=e['weight_tensor_address'] for h in head for e in embedding):raise ValueError('tied Q6 not physically shared')
   checks+=1
 save(out/'verified.json',dict(status='EXACT_NATIVE_TIED_Q6_CONTROL_PASS',model=reference(model),native=pin(out/'native/result.json'),replay=pin(out/'replay/result.json'),required_logit_cache_and_allocation_checks=checks,source=pin(__file__),policy='One exact original Q6_K packed table, one deployed tensor/device allocation, target immutable. Same capacity as tied Q4 R2 control. Diagnostic callback timing excluded.'))
 print('TIED_Q6_QUALIFIED',flush=True)

if __name__=='__main__':main()
