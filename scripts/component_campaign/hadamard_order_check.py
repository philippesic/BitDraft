"""Bound F32 normalization-order drift against the pinned native FWHT source."""
import numpy as np
import torch
from .common import *
from .simulation import hadamard,cache_read

def main():
 torch.set_num_threads(2);torch.manual_seed(1205);rows=[]
 for size in (64,128):
  x=torch.randn(128,128,requires_grad=True)
  # Native fwht.cu scales each input then performs ascending butterfly stages.
  y=x.reshape(-1,size)*np.float32(1/np.sqrt(np.float32(size)))
  stride=1
  while stride<size:
   z=y.reshape(-1,size//(2*stride),2,stride);a,b=z[:,:,0],z[:,:,1]
   y=torch.stack((a+b,a-b),2).reshape(y.shape);stride*=2
  native=y.reshape(x.shape);sim=hadamard(x,size)
  formats=[]
  for fmt in ('q8_0','q4_0','q5_0'):
   a=cache_read(native,fmt);b=cache_read(sim,fmt);delta=(a-b).detach()
   formats.append(dict(format=fmt,changed_read_values=int((delta!=0).sum()),values=delta.numel(),max_absolute_read_difference=float(delta.abs().max()),rms_read_difference=float(delta.square().mean().sqrt())))
  (native.sum()).backward(retain_graph=True);g=x.grad.clone();x.grad=None;sim.sum().backward()
  rows.append(dict(group=size,max_rotation_difference=float((native-sim).abs().max()),max_analytical_gradient_difference=float((g-x.grad).abs().max()),formats=formats))
 source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source/ggml/src/ggml-cuda/fwht.cu'
 save(R/'qualification/hadamard-normalization-order.json',dict(status='NUMERIC_BOUND_RECORDED',records=rows,source=pin(source),simulation=pin(ROOT/'scripts/component_campaign/simulation.py'),scope='CPU F32 source-order diagnostic, not a native CUDA claim. Same normalized H64/H128, actual grouping/write/read format. Native scales before butterflies; mathematical surrogate scales after. Native request correctness/acceptance is measured separately. This bound does not claim bit-exact full native attention or independent training-seed stability.'))
 print(rows)

if __name__=='__main__':main()
