import tempfile
from pathlib import Path
import torch
from torch import nn
from scripts.dflash_baseline.prototype import Projection
from scripts.dflash_baseline.prototype_model import optimizer
from scripts.dflash_baseline.pipeline import update,checkpoint,restore,packed_export,packed_import,parameters

def test_runner_restart_packed():
 for bits in (2,4,8):
  for mode in ('dynamic','fixed','learned'):
   m=nn.Sequential(Projection(torch.tensor([[.2,-.3,.4]]),bits,quantizer=dict(mode=mode,clip=1.)))
   m.prototype_config=dict(bits=bits,mode=mode)
   opt=optimizer(m)
   def loss():return (m(torch.randn(2,3))-1).square().mean()
   update(m,opt,loss)
   with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'checkpoint.pt';packed=Path(d)/'packed.pt'
    progress=dict(updates=1,cursor=1,order=[0,1],supervised_tokens=15,update_seconds=.1)
    checkpoint(p,m,opt,progress,{'optimizer_policy':'new'})
    update(m,opt,loss);expected=parameters(m)
    assert restore(p,m,opt)[0]==progress
    update(m,opt,loss)
    for n,v in parameters(m).items():torch.testing.assert_close(v,expected[n],rtol=0,atol=0)
    x=torch.randn(4,3);y=m(x).detach();packed_export(packed,m)
    with torch.no_grad():m[0].scale.zero_();m[0].midpoint.fill_(10)
    packed_import(packed,m);torch.testing.assert_close(m(x),y,rtol=0,atol=0)

def test_protected_roundtrip():
 from scripts.dflash_baseline.model import NativeBF16Linear
 m=nn.Module();m.ffn_down=NativeBF16Linear(torch.tensor([[.5,-.25]]));m.prototype_config={'protected':True}
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'packed.pt';packed_export(path,m)
  with torch.no_grad():m.ffn_down.weight.zero_()
  packed_import(path,m)
  torch.testing.assert_close(m.ffn_down(torch.ones(1,2)),torch.tensor([[.25]]),rtol=0,atol=0)
