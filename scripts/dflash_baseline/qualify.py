from pathlib import Path
import json,time,torch,numpy as np,subprocess,os
from model import Model,Capture,P
R=P/'runs/dflash-modern-qat-20261104-09';R.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(8101);torch.backends.cuda.matmul.allow_tf32=False
assert not json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']
cases=[]
for domain in ('code','prose','reasoning'):
 c=Capture(P/f'runs/dflash-modern-capture-20261104-04/data/fit-{domain}-00/capture')
 anchor=c.meta['prompt_tokens']+5;dest=P/f'runs/dflash-modern-qat-20261104-08/native-{domain}.f32'
 cmd=[str(P/'runs/dflash-modern-native-20261104-03/replay'),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),str(P/'runs/dflash-modern-20261104-02/zero.gguf'),str(c.path),str(anchor),'16',str(dest)]
 cases.append((c,anchor,dest))
t=time.perf_counter()
m=Model(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-bf16.gguf',P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf').cuda()
comparisons=[]
for c,anchor,dest in cases:
 with torch.no_grad():o=m(c.features,c.tokens,anchor).cpu().numpy()
 n=np.fromfile(dest,dtype=np.float32).reshape(o.shape);err=o-n
 result=dict(max_abs=float(np.abs(err).max()),rms=float(np.sqrt((err**2).mean())),relative_rms=float(np.linalg.norm(err)/np.linalg.norm(n)),cosine=float(np.sum(o*n)/(np.linalg.norm(o)*np.linalg.norm(n))),agreement=int((o[1:].argmax(-1)==n[1:].argmax(-1)).sum()),slots=15,native_top=n.argmax(-1).tolist(),qat_top=o.argmax(-1).tolist(),peak_bytes=torch.cuda.max_memory_allocated())
 comparisons.append(result);print(result,flush=True)
(R/'graph-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
assert all(r['agreement']>=14 and r['cosine']>0.999 and r['relative_rms']<0.05 for r in comparisons),'Graph alignment requires diagnosis'
c,anchor,_=cases[0]
opt=torch.optim.AdamW([{'params':[x.latent_sign for x in m.binary_linears().values()],'lr':1e-3},{'params':[p for name,p in m.named_parameters() if not name.endswith('latent_sign')],'lr':1e-5}],weight_decay=0,foreach=False)
before={name:(x.latent_sign.detach()<0).cpu() for name,x in m.binary_linears().items()}
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
loss,acc=c.loss(m,anchor);loss.backward()
assert torch.isfinite(loss)
grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
opt.step();opt.zero_grad(set_to_none=True)
for x in m.binary_linears().values():x.project_scales_()
torch.cuda.synchronize()
result.update(loss=float(loss),agreement_training=float(acc),grad_norm=float(grad),update_seconds=time.perf_counter()-t,peak_update_bytes=torch.cuda.max_memory_allocated(),flips={name:int(((x.latent_sign.detach()<0).cpu()!=before[name]).sum()) for name,x in m.binary_linears().items()})
(R/'update-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
