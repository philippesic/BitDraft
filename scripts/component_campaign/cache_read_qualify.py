"""Actual CUDA cache-to-F16 conversion, source dispatch and upstream gradient gate."""
import os,subprocess
import numpy as np
import torch
from .common import *
from .simulation import cache_read

def main():
 output=R/'qualification/cache-read'
 if (output/'result.json').exists():
  if __import__('json').load(open(output/'result.json'))['status']!='CUDA_NATIVE_MMA_CACHE_READ_PASS':raise ValueError('cache read qualification invalid')
  return
 output.mkdir(exist_ok=True)
 source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';runtime=R/'runtime03';exe=output/'cache-read-fixture'
 cmd=['c++','-O2','-std=c++17','-DCOMPONENT_CACHE_DIRECT_READ','-I/usr/local/cuda/include','-I'+str(source/'ggml/include'),'-I'+str(source/'ggml/src'),str(ROOT/'scripts/component_campaign/cache_fixture.cpp'),'-L'+str(runtime/'bin'),'-L/usr/local/cuda/lib64','-lcudart','-lggml-cuda','-lggml-base','-lggml','-Wl,-rpath,'+str(runtime/'bin'),'-o',str(exe)]
 subprocess.run(cmd,check=True);records=[]
 for format in ('q8_0','q4_0','q5_0'):
  dest=output/format;subprocess.run([str(exe),format,str(dest),'read-f16'],env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32'),check=True)
  x=torch.from_numpy(np.fromfile(str(dest)+'.input.f32',dtype=np.float32).reshape(19,128)).cuda().requires_grad_();expected=np.fromfile(str(dest)+'.output.f32',dtype=np.float32).reshape(19,128)
  observed=cache_read(x,format);observed.square().sum().backward();error=float(np.max(np.abs(observed.detach().cpu().numpy()-expected)))
  if error!=0 or not torch.isfinite(x.grad).all():raise ValueError('native cache-to-F16 read mismatch '+format)
  records.append(dict(format=format,max_absolute_error=error,finite_upstream_gradient=True,values=x.numel()))
 save(output/'result.json',dict(status='CUDA_NATIVE_MMA_CACHE_READ_PASS',records=records,source=pin(__file__),fixture=pin(ROOT/'scripts/component_campaign/cache_fixture.cpp'),simulation=pin(ROOT/'scripts/component_campaign/simulation.py'),native_dispatch=pin(source/'ggml/src/ggml-cuda/fattn.cu'),native_conversion=pin(source/'ggml/src/ggml-cuda/fattn-common.cuh'),scope='SM120 head128, N3 ordinary depth2 and N16 fixed training block select MMA_F16; quantized K/V converted to F16 before attention. Hadamard K128/V64 write/read history is simulated. Native N<=2 tail/vector query arithmetic remains a bounded diagnostic outside the fixed N16 training surrogate; full actual request measurements include these tails. Existing query/softmax F32 surrogate remains the same as matched control.'))
 print('CACHE_READ_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
