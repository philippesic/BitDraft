"""Real CUDA/CLI resume/adaptation/export gate, supervised as one GPU owner."""
import json,os,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.pipeline import sha,parameters
from scripts.dflash_baseline.prototype import Projection
from scripts.dflash_baseline.export_precision import export
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-01'
def main():
 assert not json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']
 env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'),LD_LIBRARY_PATH=str(runtime/'bin'))
 with (r/'ops.jsonl').open('x') as out:subprocess.run([str(runtime/'op-check'),str(r/'ops-library-maps.txt')],env=env,stdout=out,check=True)
 grad=[]
 for bits in (1,2,4,8):
  for mode in ('dynamic','fixed','learned'):
   m=Projection(torch.tensor([[.2,-.3,.4],[.1,.5,-.2]]),bits,quantizer={'mode':mode,'clip':.4}).cuda()
   with torch.no_grad():m.midpoint.fill_(.1);m.log_gain.fill_(.2)
   x=torch.tensor([[0.,0.,0.],[1e-40,-2e-40,0.],[2.,.23,-.42]],device='cuda',requires_grad=True)
   y=m(x);y.square().sum().backward()
   assert torch.isfinite(y).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()) and torch.isfinite(x.grad).all()
   if mode=='learned':assert m.log_clip.grad.abs()>0
   grad.append(dict(bits=bits,mode=mode,finite=True,clip_grad=float(m.log_clip.grad) if mode=='learned' else None))
 del m,x,y;torch.cuda.empty_cache();(r/'gradients.json').write_text(json.dumps(grad,indent=2)+'\n')
 cfg=r/'qualification-config.json'
 def run(out,updates,extra):
  subprocess.run([sys.executable,'-m','scripts.dflash_baseline.pipeline',str(cfg),'--out',str(r/out),'--updates',str(updates),'--max-windows','8','--device','cuda',*extra],env=env,check=True)
 run('qual-two',2,['--baseline-parent'])
 run('qual-resume',4,['--resume',str(r/'qual-two/checkpoint.pt')])
 run('qual-four',4,['--baseline-parent'])
 a=torch.load(r/'qual-resume/checkpoint.pt',mmap=True,map_location='cpu',weights_only=False);b=torch.load(r/'qual-four/checkpoint.pt',mmap=True,map_location='cpu',weights_only=False)
 assert a['progress']['cursor']==b['progress']['cursor']
 assert all(torch.equal(v,b['parameters'][k]) for k,v in a['parameters'].items()),'CUDA exact resume failed'
 del a,b
 export(r/'qual-four/checkpoint.pt',r/'qual-four/model.gguf')
 c=json.loads(cfg.read_text());c['parent']={'path':str(r/'qual-four/checkpoint.pt'),'sha256':sha(r/'qual-four/checkpoint.pt')};c['preset']='gate-up-a1';cfg=r/'qual-adapt-config.json';cfg.write_text(json.dumps(c,indent=2)+'\n')
 run('qual-adapt',2,['--adapt'])
 a=torch.load(r/'qual-adapt/checkpoint.pt',mmap=True,map_location='cpu',weights_only=False)
 assert a['lineage']['optimizer_policy']=='reset' and a['progress']['cursor']==14237 and a['progress']['updates']==2
 export(r/'qual-adapt/checkpoint.pt',r/'qual-adapt/model.gguf')
 (r/'qualification.json').write_text(json.dumps(dict(status='CUDA_CLI_RESUME_ADAPT_EXPORT_PASS',op_cases=36,grad_cases=len(grad),native_model_serving_pending=True),indent=2)+'\n')
if __name__=='__main__':main()
