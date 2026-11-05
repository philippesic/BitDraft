"""A streamed full optimizer checkpoint must resume the actual update trajectory."""
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.dflash_baseline.checkpoint_io import save_streamed,load
from scripts.dflash_baseline.pipeline import checkpoint,restore,update,parameters
from scripts.dflash_baseline.prototype import Projection
from scripts.dflash_baseline.prototype_model import optimizer

class StreamedCheckpointTests(unittest.TestCase):
 def test_exact_optimizer_rng_and_parameter_resume(self):
  torch.manual_seed(17)
  m=torch.nn.Sequential(Projection(torch.tensor([[.2,-.3,.4]]),4,quantizer=dict(mode='fixed',clip=1.)))
  m.prototype_config={'fixture':'fixed-step'};opt=optimizer(m)
  def step():return (m(torch.randn(2,3))-1).square().mean()
  update(m,opt,step)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.pt';progress=dict(updates=1,cursor=7,order=[0,1],update_seconds=.1,supervised_tokens=15)
   receipt=checkpoint(p,m,opt,progress,{'optimizer_policy':'reset'},compressed=True)
   self.assertFalse(p.exists());self.assertTrue(receipt['lossless_verified']);self.assertFalse(receipt['raw_materialized'])
   update(m,opt,step);expected=parameters(m);expected_opt={k:{n:v.clone() if torch.is_tensor(v) else v for n,v in x.items()} for k,x in opt.state_dict()['state'].items()}
   actual,lineage=restore(p.with_suffix('.pt.zst'),m,opt);self.assertEqual(actual,progress);self.assertEqual(lineage['optimizer_policy'],'reset')
   update(m,opt,step)
   for name,value in parameters(m).items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
   for key,state in opt.state_dict()['state'].items():
    for name,value in state.items():torch.testing.assert_close(value,expected_opt[key][name],rtol=0,atol=0)
 def test_recovery_coverage_matches_actual_autograd_flags(self):
  from scripts.dflash_baseline.recover_clip_checkpoint import trainable_names
  m=torch.nn.Sequential(Projection(torch.tensor([[.2,-.3,.4]]),4,quantizer=dict(mode='fixed',clip=1.)))
  c=dict(quantizer=dict(mode='fixed'),features=dict(sign=True,scale=True,midpoint=True,gain=False,norm=True))
  expected={name for name,p in m.named_parameters() if p.requires_grad}
  self.assertEqual(trainable_names(c,parameters(m)),expected)
  self.assertLess(len(expected),len(parameters(m)))
 def test_existing_archive_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.pt';save_streamed(p,{'x':torch.tensor([2.])});before=p.with_suffix('.pt.zst').read_bytes()
   with self.assertRaises(FileExistsError):save_streamed(p,{'x':torch.tensor([3.])})
   self.assertEqual(before,p.with_suffix('.pt.zst').read_bytes())
   self.assertEqual(load(p.with_suffix('.pt.zst'))['x'].item(),2.)
if __name__=='__main__':unittest.main()
