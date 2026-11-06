"""Qualify a real full CUDA-produced state before adopting tensor-delta retention."""
import gc,hashlib,json,struct
from pathlib import Path
import numpy as np
import torch
from .common import *
from .tensor_checkpoint import memory_load,save_state,load

def fingerprint(state):
 def normalized(v):
  if torch.is_tensor(v):return ['tensor',str(v.dtype),list(v.shape),hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy()).hexdigest()]
  if isinstance(v,np.ndarray):return ['array',v.dtype.str,list(v.shape),hashlib.sha256(v).hexdigest()]
  if isinstance(v,dict):return ['dict',[[repr(k),normalized(x)] for k,x in v.items()]]
  if isinstance(v,(tuple,list)):return [type(v).__name__,[normalized(x) for x in v]]
  if isinstance(v,float):return ['float',struct.pack('<d',v).hex()]
  return [type(v).__name__,repr(v)]
 return hashlib.sha256(json.dumps(normalized(state),separators=(',',':')).encode()).hexdigest()

def main():
 torch.set_num_threads(2);directory=Path('/mnt/c/binary-eagle-component-20261105-01');rolling=json.load(open(directory/'rolling.json'));original=rolling['receipt'];arm=rolling['arm'];out=R/'qualification/full-tensor-checkpoint';out.mkdir(exist_ok=False)
 if not (R/'arms/H4/result.json').exists():raise ValueError('finish the first primary recovery pair before storage conversion')
 saved=memory_load(original['archive']);expected=fingerprint(saved)
 receipt=save_state(directory/('delta-'+arm+'-'+str(rolling['update'])+'.pt'),saved)
 del saved;gc.collect();restored=load(receipt['archive'])
 if fingerprint(restored)!=expected:raise ValueError('full optimizer/RNG/parameter state changed')
 from scripts.dflash_baseline.export_precision import export
 from .export import rewrite
 base=out/'base.gguf';export(None,base,state=restored,checkpoint_pin=pin(receipt['archive']))
 native=out/'recovered.gguf';rewrite(base,native,restored['component_spec'],head_log_d=restored['parameters'].get('head_log_d'))
 expected_native=json.load(open(R/'arms'/arm/'readouts.json'))[-1]['model']
 if pin(native)['sha256']!=expected_native['sha256'] or pin(native)['bytes']!=expected_native['bytes']:raise ValueError('full delta-recovered native export differs')
 save(out/'result.json',dict(status='FULL_CUDA_PARAMETERS_ADAM_RNG_CURSOR_NATIVE_BYTE_ROUNDTRIP_PASS',original=original,replacement=receipt,state_fingerprint=expected,native=expected_native,source=pin(__file__),codec=pin(ROOT/'scripts/component_campaign/tensor_checkpoint.py')))
 save(R/'checkpoint-policy.json',dict(format='tensor-delta-v1',qualification=pin(out/'result.json'),observed_bytes=receipt['archive_bytes'],semantics='Lossless trained tensors/optimizer/RNG/data state against authenticated immutable B0; no frozen weights duplicated; no training recipe or precision change.'))
 save(R/'arms'/arm/'recovery.json',receipt);save(directory/'rolling.json',dict(arm=arm,update=rolling['update'],receipt=receipt))
 if pin(original['archive'])['sha256']!=original['archive_sha256']:raise ValueError('original recovery state changed before replacement')
 with (R/'rolling-retention.jsonl').open('a') as f:f.write(json.dumps(dict(retired=original,replaced_by=receipt,policy='Full-state tensor-delta/native re-export qualification; only NEW exact redundant container retired'))+'\n')
 Path(original['archive']).unlink()
 # Both byte streams already survive as the original trained native export.
 for path in (base,base.with_suffix('.json'),native):path.unlink()
 print('FULL_TENSOR_CHECKPOINT_PASS',receipt['archive_bytes'],flush=True)

if __name__=='__main__':main()
