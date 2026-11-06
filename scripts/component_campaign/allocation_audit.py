"""Prove native physical vocabulary sharing and count unique permanent buffers."""
import argparse,json
from .common import *

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=R/'qualification/physical-tie/result.json');args=parser.parse_args()
 source=args.input;data=json.load(open(source));arms={}
 for rec in data['records']:
  name=rec['arm'];entry=arms.setdefault(name,dict(cells=[],physical_tied_all_cells=True,head_types=set(),embedding_types=set()))
  for row in rec['result']['rows']:
   vocabulary=[x for x in row['audit'] if x.get('kind')=='vocabulary_allocation']
   head=[x for x in vocabulary if x['op']=='MUL_MAT'];embedding=[x for x in vocabulary if x['op']=='GET_ROWS']
   if not head or not embedding:raise ValueError('missing actual vocabulary callbacks '+name)
   if any(x['weight_data_address']==0 or x['weight_tensor_address']==0 for x in head+embedding):raise ValueError('unallocated vocabulary pointer in actual graph '+name)
   for h in head:
    for e in embedding:
     tied=h['weight_data_address']==e['weight_data_address'] and h['weight_tensor_address']==e['weight_tensor_address']
     entry['physical_tied_all_cells'] &= tied
     entry['head_types'].add(h['weight_type']);entry['embedding_types'].add(e['weight_type'])
   buffers={x['buffer_address']:x for x in row['audit'] if x.get('kind')=='permanent_weight_buffer'}
   entry['cells'].append(dict(domain=rec['domain'],anchor=row['anchor'],warmup=row['warmup'],head=head,embedding=embedding,unique_observed_buffers=list(buffers.values()),unique_observed_buffer_bytes=sum(x['buffer_bytes'] for x in buffers.values())))
 from .measure import completed
 registry=completed()
 for name,entry in arms.items():
  entry['head_types']=sorted(entry['head_types']);entry['embedding_types']=sorted(entry['embedding_types'])
  if (name=='HE4' or registry.get(name,{}).get('spec',{}).get('tied')) and not entry['physical_tied_all_cells']:raise ValueError('tied vocabulary does not physically share tensor and device data '+name)
  if name in ('H4','E4','H6','E6') and entry['physical_tied_all_cells']:raise ValueError('private component unexpectedly aliases target counterpart')
 control_checks=0
 for baseline in [x for x in data['records'] if x['arm']=='H0']:
  for name in ('H6','E6'):
   if name not in arms:continue
   private=next(x for x in data['records'] if x['arm']==name and x['domain']==baseline['domain'] and x['rep']==baseline['rep'])
   for a,b in zip(baseline['result']['rows'],private['result']['rows'],strict=True):
    if any(a[k]!=b[k] for k in ('outputs','cache_bytes','cache_hash_fnv1a64')):raise ValueError('exact private Q6 copy control changes required logits/cache '+name)
    control_checks+=1
 save(source.parent/'verified.json',dict(status='NATIVE_PHYSICAL_ALLOCATION_PASS',source=pin(source),arms=arms,exact_Q6_control_logits_cache_checks=control_checks,scope='Actual graph vocabulary tensor/data addresses and buffer types, not GGUF tying flags. Unique observed buffers are a lower bound when a frozen weight is outside callback scope; model load buffer totals and device peak are independently reported. Target head remains immutable.'))
 print('PHYSICAL_TIE_VERIFIED',flush=True)

if __name__=='__main__':main()
