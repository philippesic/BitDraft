"""Two diagnostic model updates; proves training/persistence plumbing, not generalization."""
from pathlib import Path
import copy,hashlib,json,os,sys,time,gc
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-modern-replay-20261103');R=P/'runs/w1a8-modern-gpu-training-qualification-20261103-01'
sys.path[:0]=[str(W/'scripts/w1a8_generalization'),str(W/'src'),str(P/'third_party/llama.cpp/gguf-py')]
import numpy as np
import torch
from modern_actor import build_modern_actor,collect_scale_state,load_scale_state,export_scale_only_gguf,file_pin
from modern_admission import build_actor_steps,require_actor_steps
from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import shared_round_hard_signs

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def guard():assert not read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'].get('pause_requested') and not (R/'STOP').exists()
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return x
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
started=time.monotonic();guard();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.manual_seed(37);torch.cuda.manual_seed_all(37)
assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5080';device=torch.device('cuda:0')
inputs={};rows={};pins=[]
for pid in [2,4]:
 a=P/f'runs/w1a8-modern-admission-20261103-01/prompt-{pid:02d}-admitted.json';receipt=read(a);pins.append(file_pin(a));e=receipt['evidence'];meta=read(e['capture']['path']);shape=tuple(meta['feature_shape']);features=torch.from_numpy(np.concatenate([np.fromfile(e['arrays'][f'layer-{tap}.f32']['path'],dtype='<f4').reshape(shape) for tap in [2,16,29]],axis=1)).to(device)
 emb=np.fromfile(e['arrays']['layer-0.f32']['path'],dtype='<f4').reshape(shape)
 for index,token in enumerate(meta['input_tokens']):
  row=torch.tensor(emb[index],device=device)
  if token in rows:assert torch.equal(rows[token],row)
  else:rows[token]=row
 steps=build_actor_steps(receipt);require_actor_steps(receipt,[s['actor_step_id'] for s in steps if s['supervision_eligible']]);inputs[pid]=(receipt,meta,features,steps)
lookup=ExternalNativeF32Embedding(lookup=rows.__getitem__,vocab_size=248320,hidden_size=2560,target_identity='admitted native F32 Qwen3.5-4B Q4_K_M',target_sha256=e['target']['sha256'],capture_sha256=hashlib.sha256(''.join(p['sha256'] for p in pins).encode()).hexdigest())
actor=build_modern_actor(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',embedding_lookup=lookup,device=device,scale_only=True)
absolute=actor.drafter.d2t+torch.arange(50000,device=device);reverse=torch.full((248320,),-1,device=device,dtype=torch.long);reverse[absolute]=torch.arange(50000,device=device)
optimizer=torch.optim.Adam(actor.scale_parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8)
write(R/'freeze.json',{'source':file_pin(__file__),'actor_source':file_pin(W/'scripts/w1a8_generalization/modern_actor.py'),'native_step':file_pin(W/'src/w1a1_eagle/native_step.py'),'recurrent_qat':file_pin(W/'src/w1a1_eagle/recurrent_qat.py'),'input_admissions':pins,'contract':actor.contract,'initializer':actor.initializer,'hardware':torch.cuda.get_device_name(0),'precision':'F32 Torch surrogate with existing hard A8, F16 KV; TF32 disabled','optimizer':{'name':'Adam','lr':1e-5,'betas':[.9,.999],'eps':1e-8},'objective':'Mean cross entropy on 7 eligible reused prompt02 teacher-forced rows; context detached/rebuilt each step','updates':2,'scope':'Training/backward/compact resume/export qualification only; not data/objective generalization or screening evidence'})
def logits(pid,grad=False):
 receipt,meta,features,steps=inputs[pid];adapter=actor.adapter;n=meta['prompt_tokens']-1;cache=adapter.build_context_cache(meta['input_tokens'][1:n+1],features[:n],chunk_size=32);values=[]
 for step in steps:
  p=step['decoder_position'];s=adapter.decode_step(step['input_token_id'],adapter.encode_feature(features[p]),p,cache);cache=s.cache;values.append(s.logits)
 return torch.stack(values)
def readout(tag):
 report={}
 with torch.no_grad(),shared_round_hard_signs(actor.linears):
  for pid in inputs:
   values=logits(pid);data=values.cpu().numpy();data.tofile(R/f'{tag}-prompt-{pid:02d}-teacher.f32');steps=inputs[pid][3];report[str(pid)]={'proposals':absolute[values.argmax(-1)].cpu().tolist(),'labels':[s['label_token_id'] for s in steps],'eligible':[s['supervision_eligible'] for s in steps]}
 return report
before=readout('before')
def update():
 guard();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
 with shared_round_hard_signs(actor.linears):
  values=logits(2,True);steps=inputs[2][3];selected=[s['actor_step_id'] for s in steps if s['supervision_eligible']];labels=torch.tensor([s['label_token_id'] for s in steps],device=device);mapped=reverse[labels];assert bool((mapped[selected]>=0).all());loss=torch.nn.functional.cross_entropy(values[selected],mapped[selected]);assert torch.isfinite(loss);loss.backward()
  grads={p:float(m.scale_offset.grad.norm()) for p,m in actor.linears.items()};assert all(m.scale_offset.grad is not None and bool(torch.isfinite(m.scale_offset.grad).all()) and m.latent_sign.grad is None for m in actor.linears.values());norm=torch.nn.utils.clip_grad_norm_(actor.scale_parameters(),1.);optimizer.step()
  for m in actor.linears.values():m.project_scales_()
 torch.cuda.synchronize();return {'loss':float(loss.detach()),'gradient_norm':float(norm),'scale_gradient_norms':grads,'wall_seconds':time.monotonic()-t}
first=update();saved={'scale_state':collect_scale_state(actor),'optimizer':cpu(optimizer.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'cursor':1,'contract':read(R/'freeze.json')}
with (R/'checkpoint-1.pt').open('xb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
second=update();expected=collect_scale_state(actor);expected_optimizer=cpu(optimizer.state_dict());restored=torch.load(R/'checkpoint-1.pt',weights_only=False,map_location='cpu');assert equal(saved,restored);load_scale_state(actor,restored['scale_state']);optimizer.load_state_dict(restored['optimizer']);torch.set_rng_state(restored['rng']);torch.cuda.set_rng_state(restored['cuda_rng']);repeated_second=update();assert equal(expected,collect_scale_state(actor)) and equal(expected_optimizer,cpu(optimizer.state_dict()));assert second['loss']==repeated_second['loss'];after=readout('after')
final={'scale_state':collect_scale_state(actor),'optimizer':cpu(optimizer.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'cursor':2,'contract':read(R/'freeze.json')}
with (R/'checkpoint-2.pt').open('xb') as f:torch.save(final,f);f.flush();os.fsync(f.fileno())
export=export_scale_only_gguf(actor,R/'model.gguf');write(R/'export.json',export)
write(R/'result.json',{'status':'GPU_BACKWARD_RESUME_EXPORT_PASS','before':before,'after':after,'updates':[first,second],'repeated_update2':repeated_second,'physical_updates':3,'unique_trajectory_updates':2,'checkpoint_roundtrip_and_resume_exact':True,'checkpoint':file_pin(R/'checkpoint-2.pt'),'export':file_pin(R/'model.gguf'),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'wall_seconds':time.monotonic()-started,'scope':read(R/'freeze.json')['scope'],'native_export_qualified':False})
print(json.dumps({'status':'GPU_BACKWARD_RESUME_EXPORT_PASS','updates':[first['loss'],second['loss']],'wall_seconds':time.monotonic()-started}),flush=True)
