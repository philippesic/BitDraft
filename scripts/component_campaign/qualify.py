"""Real cache CUDA fixture, storage gradients and full B0/pilot checkpoint recovery."""
import json,os,subprocess,gc
import numpy as np
import torch
from .common import *
from .simulation import cache_round,hadamard,StorageProjection
from scripts.dflash_baseline.checkpoint_io import load
from scripts.dflash_baseline.export_precision import export
from .archive import archive

def main():
 out=R/'qualification';out.mkdir(exist_ok=False)
 source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';runtime=R/'runtime';exe=out/'cache-fixture'
 cmd=['c++','-O2','-std=c++17','-I'+str(source/'ggml/include'),'-I'+str(source/'ggml/src'),str(ROOT/'scripts/component_campaign/cache_fixture.cpp'),'-L'+str(runtime/'bin'),'-lggml-cuda','-lggml-base','-lggml','-Wl,-rpath,'+str(runtime/'bin'),'-o',str(exe)]
 subprocess.run(cmd,check=True)
 env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32')
 records=[]
 for format in ('q8_0','q4_0','q5_0'):
  dest=out/format;subprocess.run([str(exe),format,str(dest)],env=env,check=True)
  x=torch.from_numpy(np.fromfile(str(dest)+'.input.f32',dtype=np.float32)).cuda().reshape(19,8,16).reshape(19,128).requires_grad_()
  expected=np.fromfile(str(dest)+'.output.f32',dtype=np.float32).reshape(19,128)
  observed=cache_round(x,format);observed.sum().backward()
  error=float(np.max(np.abs(observed.detach().cpu().numpy()-expected)))
  if error>2e-6 or not torch.equal(x.grad,torch.ones_like(x)):raise ValueError('native cache round/STE mismatch '+format+': '+str(error))
  records.append(dict(format=format,max_absolute_error=error,positions=19,values=x.numel(),finite_gradient=True,grouping=32,packed=pin(str(dest)+'.packed')))
 for size in (64,128):
  x=torch.randn(4,8,128,device='cuda',requires_grad=True);y=hadamard(hadamard(x,size),size)
  if (y-x).abs().max()>2e-6:raise ValueError('Hadamard involution failed')
  y.sum().backward()
 scalar=[]
 for group in ('scale','midpoint','gain','clip'):
  m=StorageProjection(torch.randn(4,128),4,quantizer=dict(mode='learned',clip=1.01)).cuda();m.storage={group:True}
  with torch.no_grad():m.midpoint.fill_(.00317);m.log_gain.fill_(.0021)
  x=torch.randn(3,128,device='cuda',requires_grad=True);m(x).square().sum().backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad):raise ValueError('missing storage gradient '+group)
  scalar.append(dict(group=group,finite_gradients=True,exported_exp_rounding=True))
 del m,x,y;gc.collect();torch.cuda.empty_cache()
 import tempfile;tempfile.tempdir='/dev/shm'
 recovery=json.load(open(R/'arms/H0-pilot/recovery.json'));state=load(recovery['archive'])
 if state['progress']['updates']!=32 or not state['resume_capable']:raise ValueError('pilot recovery incomplete')
 steps={int(v['step']) for v in state['optimizer']['state'].values()}
 if steps!={32} or len(state['rng'])!=4 or state['rng'][3] is None:raise ValueError('pilot optimizer/RNG not preserved')
 native=out/'recovered-base.gguf';export(None,native,state=state,checkpoint_pin=pin(recovery['archive']))
 # Rewrite the recovered full state, then compare to the resident pilot point.
 from .export import rewrite
 recovered=out/'recovered.gguf';rewrite(native,recovered,state['component_spec'])
 expected=json.load(open(R/'arms/H0-pilot/readouts.json'))[-1]['model']
 if pin(recovered)['sha256']!=expected['sha256']:raise ValueError('full recovery/native export changed bytes')
 native.unlink();native.with_suffix('.json').unlink();archive(recovered,R/'bases/shared.gguf')
 save(out/'result.json',dict(status='CUDA_CACHE_STORAGE_GRADIENT_FULL_CHECKPOINT_EXPORT_PASS',cache=records,scalar=scalar,recovered_checkpoint=pin(recovery['archive']),optimizer_steps=sorted(steps),recovered_native=expected,source=pin(__file__),limitation='Full model native trajectory/history and exact optimizer continuation remain separate gates.'))
 print('QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
