import json,torch,numpy as np
from model import Model,Capture,P,NativeBF16Linear,_FrozenLinear
from pathlib import Path
R=P/'runs/dflash-modern-native-20261104-03';T=R/'trace-zero.f32.trace'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
m=Model(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-bf16.gguf',P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf').cuda();c=Capture(P/'runs/dflash-modern-capture-20261104-04/data/fit-code-00/capture')
native={x['name']:np.fromfile(T/x['file'],np.float32).reshape(x['shape']).squeeze() for x in map(json.loads,(T/'index.jsonl').read_text().splitlines())}
def diff(a,b):return dict(rms=float(np.sqrt(((a-b)**2).mean())),relative=float(np.linalg.norm(a-b)/np.linalg.norm(b)),maximum=float(np.abs(a-b).max()))
reports={}
for mode in ['bf16','f32']:
 if mode=='f32':
  for x in m.modules():
   if isinstance(x,NativeBF16Linear):x.__class__=_FrozenLinear
 got={};hooks=[]
 for i,l in enumerate(m.layers):
  hooks.append(l.ffn_down.register_forward_hook(lambda mod,inp,out,i=i:got.update({f'ffn_out-{i}':out.detach().cpu().numpy()})))
  hooks.append(l.ffn_gate.register_forward_pre_hook(lambda mod,inp,i=i:got.update({f'ffn_norm-{i}':inp[0].detach().cpu().numpy()})))
  hooks.append(l.register_forward_hook(lambda mod,inp,out,i=i:got.update({f'ffn_inp-{i}':out[0].detach().cpu().numpy()-got[f'ffn_out-{i}']})))
 with torch.no_grad():o=m(c.features,c.tokens,196)
 for h in hooks:h.remove()
 reports[mode]={k:diff(v,native[k]) for k,v in got.items()}
 # Direct binary operation diagnostic on native normalized input at layer0.
 with torch.no_grad():
  l=m.layers[0];x=torch.from_numpy(native['ffn_norm-0']).cuda()
  direct=l.ffn_down(torch.nn.functional.silu(l.ffn_gate(x))*l.ffn_up(x)).cpu().numpy()
 reports[mode]['isolated_ffn0']=diff(direct,native['ffn_out-0'])
 print(mode,reports[mode],flush=True)
(R/'stage-diagnostic.json').write_text(json.dumps(reports,indent=2)+'\n')
