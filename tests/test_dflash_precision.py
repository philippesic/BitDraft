import tempfile
from pathlib import Path
import torch
from scripts.dflash_baseline.prototype import Projection,resolve,native_admit
from scripts.dflash_baseline.components import decoded,install_components
from scripts.dflash_baseline.checkpoint_io import load,archive

def test_a1_is_distinct_and_gradients():
 for mode in ('dynamic','fixed','learned'):
  a=Projection(torch.tensor([[.2,-.3,.4]]),1,quantizer=dict(mode=mode,clip=1.))
  b=Projection(torch.tensor([[.2,-.3,.4]]),2,quantizer=dict(mode=mode,clip=1.))
  x=torch.tensor([[0.,.1,-1.],[0.,0.,0.]],requires_grad=True)
  codes,step=a.stream(x);other,_=b.stream(x)
  assert set(codes.detach().flatten().tolist())=={-1.,1.} and 0 in other.detach()
  if mode=='dynamic':assert torch.equal(a(x)[1],torch.zeros(1))
  a(x).square().sum().backward();assert torch.isfinite(x.grad).all()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())

def test_archive_exact_recovery():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'state.pt';expected={'parameters':torch.randn(128,32),'progress':{'cursor':134}}
  torch.save(expected,p);r=archive(p);actual=load(r['archive']);assert not p.exists()
  assert actual['progress']==expected['progress'] and torch.equal(actual['parameters'],expected['parameters'])

def test_components_change_actual_frozen_weights():
 from torch import nn
 from scripts.dflash_baseline.model import NativeBF16Linear
 for format in ('Q4_0','Q8_0'):
  m=nn.Module();m.fc=NativeBF16Linear(torch.randn(32,32));m.register_buffer('head',torch.randn(64,32))
  old=m.fc.weight.clone();head=m.head.clone();install_components(m,dict(components={'fusion':format,'head':format}))
  assert not torch.equal(old,m.fc.weight) and not torch.equal(head,m.head)
  assert not list(m.parameters())
  torch.testing.assert_close(m.fc.weight,decoded(old.numpy(),format),rtol=0,atol=0)
  x=torch.randn(2,32,requires_grad=True);m.fc(x).square().sum().backward();assert torch.isfinite(x.grad).all()
