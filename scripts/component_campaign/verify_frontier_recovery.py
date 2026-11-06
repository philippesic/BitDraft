"""Authenticate retained full optimizer/RNG states and recover exact native bytes."""
import gc,json,shutil
from .common import *
from .tensor_checkpoint import load
from .archive import reference
from .export import rewrite
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.prototype import digest

def main():
 import torch
 torch.set_num_threads(2)
 retained=json.load(open(R/'frontier-retention.json'));out=R/'qualification/frontier-recovery';out.mkdir(exist_ok=True);records=[]
 for path,receipt in retained['states'].items():
  name=receipt['original_arm'];dest=out/name;dest.mkdir(exist_ok=True)
  if (dest/'result.json').exists():records.append(json.load(open(dest/'result.json')));continue
  if pin(path)['sha256']!=receipt['archive_sha256']:raise ValueError('retained full frontier state changed')
  state=load(path)
  if not state['resume_capable'] or state['config_hash']!=digest(state['config']) or state['progress']['updates']!=4096 or len(state['rng'])!=4 or state['rng'][3] is None:raise ValueError('frontier is not a full4096resume state')
  steps={int(v['step']) for v in state['optimizer']['state'].values()}
  if steps!={4096}:raise ValueError('frontier optimizer history incomplete')
  if any(not torch.isfinite(p).all() for p in state['parameters'].values()):raise ValueError('nonfinite saved master')
  model=receipt['native']
  if shutil.disk_usage('/').free<int(model['bytes'])+183_500_000+(512<<20):raise RuntimeError('native recovery verification needs model+184MBbase+512MiB free WSL bytes')
  base=dest/'base.gguf';native=dest/'recovered.gguf'
  export(None,base,state=state,checkpoint_pin=receipt)
  rewrite(base,native,state['component_spec'],head_log_d=state['parameters'].get('head_log_d'),deployment_exps=state.get('deployment_exps'))
  actual=pin(native)
  if actual['sha256']!=model['sha256'] or actual['bytes']!=model['bytes']:raise ValueError('native byte stream from retained full state differs')
  result=dict(status='FULL_FRONTIER_MASTER_ADAM_RNG_NATIVE_BYTE_RECOVERY_PASS',original_arm=name,full_state=receipt,native=model,optimizer_steps=sorted(steps),master_count=len(state['parameters']),deployment_half_values=sorted(state.get('deployment_exps',{})),source=pin(__file__))
  save(dest/'result.json',result);records.append(result)
  # These two exact byte streams survive as authenticated native exports and
  # full recovery states. Only this owned verification scratch is retired.
  for p in (base,base.with_suffix('.json'),native,native.with_suffix('.component.json')):p.unlink()
  del state;gc.collect();print('FRONTIER_FULL_RECOVERY_VERIFIED',name,flush=True)
 save(out/'result.json',dict(status='ALL_RETAINED_FRONTIERS_FULLY_RECOVERABLE',records=records,source=pin(__file__),scope='Actual full CPU decode checks masters/Adam/RNG/cursor and byte-identical export, including saved GPU exp-then-half deployment values. No training/source changes. Actual optimizer resume also qualified by the earlier live resumed trainer(s).'))

if __name__=='__main__':main()
