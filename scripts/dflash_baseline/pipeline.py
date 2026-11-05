"""CPU-capable config runner and versioned packed reference interchange (not native GGUF)."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from .prototype import resolve, digest, Projection
from .prototype_model import install, optimizer, authenticate

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def pin(p):
 if sha(p['path'])!=p['sha256']:raise ValueError('artifact hash mismatch: '+p['path'])
def atomic(path,state):
 path=Path(path);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def parameters(m):return {n:p.detach().cpu().clone() for n,p in m.named_parameters()}
def load_parameters(m,state):
 expected=dict(m.named_parameters())
 if set(expected)!=set(state):raise ValueError('parameter coverage mismatch')
 for n,v in state.items():
  if n.endswith('ffn_gate.log_gain'):
   other=n.replace('ffn_gate.log_gain','ffn_up.log_gain')
   if other in state and not torch.equal(v,state[other]):raise ValueError('shared gain conflict')
 with torch.no_grad():
  for n,p in expected.items():p.copy_(state[n])
def rng():return (torch.get_rng_state(),random.getstate(),np.random.get_state())
def restore_rng(r):torch.set_rng_state(r[0]);random.setstate(r[1]);np.random.set_state(r[2])
def checkpoint(path,m,opt,progress,lineage):
 atomic(path,dict(version=2,config=m.prototype_config,config_hash=digest(m.prototype_config),parameters=parameters(m),optimizer=opt.state_dict(),progress=progress,rng=rng(),lineage=lineage))
def restore(path,m,opt):
 s=torch.load(path,weights_only=False,map_location='cpu')
 if s['version']!=2 or s['config_hash']!=digest(m.prototype_config) or digest(s['config'])!=s['config_hash']:raise ValueError('resume config mismatch')
 load_parameters(m,s['parameters']);opt.load_state_dict(s['optimizer']);restore_rng(s['rng'])
 return s['progress'],s['lineage']
def update(m,opt,loss):
 opt.zero_grad(set_to_none=True);value=loss();value.backward()
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
 for p in m.modules():
  if isinstance(p,Projection):p.project_scales_()
 return float(value.detach())
def packed_export(path,m):
 state=parameters(m);projections={}
 for name,p in m.named_modules():
  if isinstance(p,Projection):
   latent=state.pop(name+'.latent_sign');bits=np.packbits((latent.numpy()>=0).reshape(-1),bitorder='little')
   projections[name]=dict(shape=list(latent.shape),signs=torch.from_numpy(bits.copy()),bits=p.bits,quantizer=p.quantizer)
 # Protected/frozen tensors are reloaded from authenticated source, plus explicit protected projection weights.
 protected={n:p.weight.detach().cpu().clone() for n,p in m.named_modules() if n.endswith(('ffn_gate','ffn_up','ffn_down')) and not isinstance(p,Projection)}
 atomic(path,dict(format='dflash.packed.reference',version=1,config=m.prototype_config,config_hash=digest(m.prototype_config),parameters=state,projections=projections,protected=protected,native_supported=False,bit_order='little',sign_positive=1))
def packed_import(path,m):
 s=torch.load(path,weights_only=False,map_location='cpu')
 if s['format']!='dflash.packed.reference' or s['version']!=1 or s['config_hash']!=digest(m.prototype_config):raise ValueError('packed contract mismatch')
 state=dict(s['parameters']);modules=dict(m.named_modules())
 for n,v in s['projections'].items():
  p=modules[n]
  if p.bits!=v['bits'] or p.quantizer!=v['quantizer']:raise ValueError('quantizer mismatch')
  signs=np.unpackbits(v['signs'].numpy(),bitorder='little',count=int(np.prod(v['shape']))).reshape(v['shape'])
  state[n+'.latent_sign']=torch.from_numpy(signs.astype(np.float32)*2-1)
 load_parameters(m,state)
 with torch.no_grad():
  for n,w in s['protected'].items():modules[n].weight.copy_(w)
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--updates',type=int,required=True);ap.add_argument('--max-windows',type=int,required=True);ap.add_argument('--resume',type=Path);ap.add_argument('--adapt',action='store_true');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
 if a.updates<0 or a.max_windows<0 or (a.resume and a.adapt):raise ValueError('invalid budget/mode')
 c=resolve(json.loads(a.config.read_text()));authenticate(c)
 if not a.adapt and not a.resume and c['parent']!=c['source']:raise ValueError('from-source parent must equal source; use --adapt for checkpoint parent')
 manifest=json.loads(Path(c['data']['path']).read_text())
 if 'records' in manifest:
  records=[r for r in manifest['records'] if r['split']=='TRAIN']
  manifest=dict(target=c['evaluation']['target'],files=[p for r in records for p in [r['metadata'],*r['files']]],train_captures=[str(Path(r['metadata']['path']).parent) for r in records])
 pin(manifest['target'])
 for p in manifest['files']:pin(p)
 from .model import Model,Capture,GGUFReader,dequantize
 torch.set_num_threads(2);torch.manual_seed(8101);random.seed(8101);np.random.seed(8101)
 m=Model(c['source']['path'],manifest['target']['path']);weights={}
 reader=GGUFReader(c['source']['path'])
 for t in reader.tensors:
  key=t.name.removesuffix('.weight')
  if key in c['coverage']:weights[key]=torch.from_numpy(dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32)).bfloat16()
 lineage={'source':c['source'],'data':c['data'],'parent':c['parent'],'optimizer_policy':'new'}
 if a.adapt:
  parent=torch.load(c['parent']['path'],map_location='cpu',weights_only=False)
  if parent.get('version')!=2:raise ValueError('adaptation requires pipeline v2 checkpoint')
  if parent['config']['source']!=c['source']:raise ValueError('adapt source mismatch')
  install(m,parent['config'],weights);load_parameters(m,parent['parameters'])
  for layer in m.layers:
   gate,up=layer.ffn_gate,layer.ffn_up
   if isinstance(gate,Projection) and isinstance(up,Projection) and not torch.equal(gate.log_gain,up.log_gain):raise ValueError('shared gain conflict')
  lineage.update(parent_config_hash=parent['config_hash'],parent_progress=parent['progress'],optimizer_policy='reset')
 install(m,c,weights);opt=optimizer(m)
 train=[Capture(p) for p in manifest['train_captures']]
 # Every consumed capture file must be authenticated by the data manifest.
 authenticated={str(Path(p['path']).resolve()) for p in manifest['files']}
 for cap in train:
  for p in [cap.path/'capture.json',*[cap.path/f'layer-{t}.f32' for t in cap.meta['taps']]]:
   if str(p.resolve()) not in authenticated:raise ValueError('capture file missing from manifest')
 windows=[(i,x) for i,cap in enumerate(train) for x in cap.anchors]
 if not windows:raise ValueError('no training windows')
 progress=dict(updates=0,cursor=0,order=np.random.permutation(len(windows)).tolist(),update_seconds=0.,supervised_tokens=0)
 if a.resume:progress,lineage=restore(a.resume,m,opt)
 if sorted(progress['order'])!=list(range(len(windows))):raise ValueError('data order mismatch')
 a.out.mkdir(exist_ok=False)
 (a.out/'resolved.json').write_text(json.dumps(c,indent=2))
 if not a.dry_run:
  while progress['updates']<a.updates and progress['cursor']<a.max_windows and progress['update_seconds']<c['budget_seconds']:
   i,anchor=windows[progress['order'][progress['cursor']%len(windows)]];start=time.perf_counter()
   update(m,opt,lambda:train[i].loss(m,anchor)[0]);progress['update_seconds']+=time.perf_counter()-start
   progress['updates']+=1;progress['cursor']+=1;progress['supervised_tokens']+=15
   checkpoint(a.out/'checkpoint.pt',m,opt,progress,lineage)
  packed_export(a.out/'packed.pt',m)
 print(json.dumps(dict(progress=progress,layers=len(m.layers),native_supported=False,dry_run=a.dry_run)))
if __name__=='__main__':main()
