"""Bounded, resumable modern DFlash baseline training; update-time accounting."""
import argparse,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from model import Model,Capture,P

def atomic(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def cpu_tree(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
 return x
def digest(m):
 h=hashlib.sha256()
 for name,p in m.named_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def optimizer(m):
 return torch.optim.AdamW([{'params':[x.latent_sign for x in m.binary_linears().values()],'lr':1e-3},{'params':[p for n,p in m.named_parameters() if not n.endswith('latent_sign')],'lr':1e-5}],weight_decay=0,foreach=False)
def step(m,opt,c,anchor):
 torch.cuda.synchronize();start=time.perf_counter()
 loss,acc=c.loss(m,anchor);loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
 opt.step();opt.zero_grad(set_to_none=True)
 for x in m.binary_linears().values():x.project_scales_()
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 return dict(loss=float(loss.detach()),token_agreement=float(acc.detach()),grad_norm=float(norm),seconds=elapsed)
def snapshot(path,m,opt,updates,seconds,cursor,order):
 state=dict(parameters={n:p.detach().cpu() for n,p in m.named_parameters()},optimizer=cpu_tree(opt.state_dict()),updates=updates,update_seconds=seconds,cursor=cursor,order=order,rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),recipe=dict(sign_lr=1e-3,scale_norm_lr=1e-5,weight_decay=0,lr_schedule='constant',loss='native-label CE exp(-position/7)',seed=8101))
 tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
 return path

def restore(path,m):
 state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 with torch.no_grad():
  for n,p in m.named_parameters():p.copy_(state['parameters'][n])
 opt=optimizer(m);opt.load_state_dict(state['optimizer'])
 torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state(state['rng']['cuda']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
 return opt,{k:state[k] for k in ('updates','update_seconds','cursor','order')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--qualify',action='store_true');ap.add_argument('--seconds',type=float,default=1800);ap.add_argument('--resume',type=Path);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 stop=False
 def request_stop(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 torch.set_num_threads(4);torch.manual_seed(8101);np.random.seed(8101);random.seed(8101);torch.backends.cuda.matmul.allow_tf32=False
 m=Model(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-bf16.gguf',P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf').cuda();opt=optimizer(m)
 data=P/'runs/dflash-modern-capture-20261104-04/data'
 train=[Capture(p/'capture') for p in sorted(data.glob('fit-*'))];dev=[Capture(p/'capture') for p in sorted(data.glob('development-*'))]
 windows=[(i,anchor) for i,c in enumerate(train) for anchor in c.anchors]
 order=np.random.permutation(len(windows)).tolist();cursor=updates=0;seconds=0.
 if a.resume:
  del opt;torch.cuda.empty_cache();opt,st=restore(a.resume,m);updates=st['updates'];seconds=st['update_seconds'];cursor=st['cursor'];order=st['order']
 atomic(a.out/'recipe.json',dict(train_sequences=len(train),dev_sequences=len(dev),windows=len(windows),data=str(data),source=str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-bf16.gguf'),target=str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),allocation=a.seconds,notes='qualification steps excluded; frozen baseline starts from zero; exact fixed order cycles; 15 supervised mask positions per window'))
 def next_window():
  nonlocal cursor
  i,anchor=windows[order[cursor%len(order)]];cursor+=1;return train[i],anchor
 if a.qualify:
  c,anchor=next_window();one=step(m,opt,c,anchor)
  path=snapshot(a.out/'one.pt',m,opt,1,one['seconds'],cursor,order)
  c,anchor=next_window();two=step(m,opt,c,anchor);expected=digest(m)
  del opt;torch.cuda.empty_cache();opt,st=restore(path,m);cursor=st['cursor'];order=st['order']
  c,anchor=next_window();repeat=step(m,opt,c,anchor);observed=digest(m)
  atomic(a.out/'result.json',dict(first=one,second=two,repeated=repeat,expected_hash=expected,resumed_hash=observed,exact_resume=expected==observed,peak_bytes=torch.cuda.max_memory_allocated()))
  assert expected==observed,'resume parameters differ'
  snapshot(a.out/'qualified-two.pt',m,opt,2,one['seconds']+two['seconds'],cursor,order)
  print('Exact resume qualified',flush=True);return
 initial={n:p.detach().flatten()[::1024].cpu().clone() for n,p in m.named_parameters()}
 def evaluate():
  with torch.no_grad():return [dict(name=c.path.parent.name,loss=float(c.loss(m,c.anchors[0])[0])) for c in dev]
 def save(label):
  stats={}
  for n,p in m.named_parameters():
   x=p.detach().flatten()[::1024].cpu();stats[n]=dict(sample_count=x.numel(),max_abs_change=float((x-initial[n]).abs().max()),sample_sign_changes=int(((x<0)!=(initial[n]<0)).sum()))
  path=snapshot(a.out/f'checkpoint-{label}.pt',m,opt,updates,seconds,cursor,order)
  atomic(a.out/f'checkpoint-{label}.json',dict(path=str(path),updates=updates,update_seconds=seconds,windows_seen=cursor,supervised_tokens=cursor*15,effective_epochs=cursor/len(windows),movement=stats,validation=evaluate(),peak_bytes=torch.cuda.max_memory_allocated()))
  print('Saved',label,updates,seconds,flush=True)
 if not a.resume:save('zero')
 thresholds=[t for t in (300,900,a.seconds) if t>seconds];started=time.perf_counter()
 while seconds<a.seconds:
  if stop or json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']:
   save('interrupted');atomic(a.out/'result.json',dict(status='INTERRUPTED',updates=updates,update_seconds=seconds));return
  c,anchor=next_window();record=step(m,opt,c,anchor);updates+=1;seconds+=record['seconds'];record.update(update=updates,total_update_seconds=seconds,sequence=c.path.parent.name,anchor=anchor)
  with (a.out/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  if thresholds and seconds>=thresholds[0]:save(str(int(thresholds.pop(0))))
 atomic(a.out/'result.json',dict(status='TRAINING_ALLOCATION_COMPLETE',updates=updates,update_seconds=seconds,loop_wall_seconds=time.perf_counter()-started,windows_seen=cursor,supervised_tokens=cursor*15,peak_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
