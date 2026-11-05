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

def test_shared_gate_up_learned_bound():
 import runpy
 from torch import nn
 from scripts.dflash_baseline.prototype_model import install,optimizer
 from scripts.dflash_baseline.pipeline import parameters,load_parameters
 class Old(nn.Module):
  def __init__(self):
   super().__init__();self.latent_sign=nn.Parameter(torch.tensor([[.2,-.3,.4]]))
  def effective_scales(self):return self.latent_sign.detach().abs().mean(-1)
 class Model(nn.Module):
  def __init__(self):
   super().__init__();self.layers=nn.ModuleList()
   for _ in range(6):
    layer=nn.Module()
    for n in ('ffn_gate','ffn_up','ffn_down'):setattr(layer,n,Old())
    for n in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm'):setattr(layer,n,nn.Parameter(torch.ones(1)))
    self.layers.append(layer)
   self.output_norm=nn.Parameter(torch.ones(1));self.output_norm_enc=nn.Parameter(torch.ones(1))
  def binary_linears(self):return {f'blk.{i}.{n}':getattr(l,n) for i,l in enumerate(self.layers) for n in ('ffn_gate','ffn_up','ffn_down')}
 request=runpy.run_path('tests/test_dflash_prototype.py')['config']();request['quantizer']=dict(mode='learned',clip=1.)
 c=resolve(request);m=install(Model(),c,{});opt=optimizer(m)
 for l in m.layers:
  assert l.ffn_gate.log_clip is l.ffn_up.log_clip and l.ffn_gate.log_clip is not l.ffn_down.log_clip
  x=torch.tensor([[2.,-.3,.2]]);(l.ffn_gate(x).square().sum()+l.ffn_up(x).square().sum()).backward()
 opt.step();state=parameters(m);other=install(Model(),c,{});load_parameters(other,state)
 for l in other.layers:assert l.ffn_gate.log_clip is l.ffn_up.log_clip
 assert not any('ffn_up.log_clip' in n for n in state)

def test_calibrated_quantizer_contract():
 import runpy,copy
 request=runpy.run_path('tests/test_dflash_prototype.py')['config']()
 request['quantizer']=dict(mode='learned',clip=1.,initial_bounds={'blk.0.ffn_gate':.3,'blk.0.ffn_up':.3,'blk.0.ffn_down':.12})
 c=resolve(request);assert native_admit(c)['contract_version']==2
 for wrong in (dict(mode='dynamic',clip=1.,initial_bounds={'blk.0.ffn_down':.1}),dict(mode='learned',clip=1.,initial_bounds={'blk.0.ffn_gate':.3}),dict(mode='learned',clip=1.,initial_bounds={'blk.0.ffn_down':0.})):
  bad=copy.deepcopy(request);bad['quantizer']=wrong
  try:resolve(bad)
  except ValueError:pass
  else:raise AssertionError('invalid calibrated quantizer accepted')
