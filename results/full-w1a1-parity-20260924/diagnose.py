import json,sys,tomllib,hashlib,subprocess
from pathlib import Path
from datetime import UTC,datetime
import torch
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/"src")]
from scripts.evaluate_pytorch_w1a1 import encode_prompt,load_prompts,sha256_file
from w1a1_eagle.official_loader import load_official_eagle3
import angelslim.compressor.speculative.inference.models.eagle3.eagle3_model as em
from angelslim.compressor.speculative.utils.util import reset_tree_mode,evaluate_posterior as eval0
from angelslim.compressor.speculative.inference.models.eagle3.target.modeling_qwen3_kv import Qwen3Model
from angelslim.compressor.speculative.inference.models.eagle3.eagle3_model import GenerationConfig
R=Path(R);O=R/"results/full-w1a1-parity-20260924";O.mkdir(parents=True,exist_ok=True);c=tomllib.loads((R/"configs/pytorch_w1a1.toml").read_text());e=c["evaluation"];mc=c["models"];mp=R/"results/model-manifest-5080-20260924.json";pr=load_prompts(R/e["prompt_manifest"])[0]
torch.manual_seed(e["seed"]);m=load_official_eagle3(R/mc["target_dir"],R/mc["draft_dir"],angelslim_revision=mc["angelslim_revision"],total_token=e["total_token"],depth=e["depth"],top_k=e["top_k"],threshold=e["threshold"],target_load_kwargs={"dtype":torch.bfloat16,"device_map":"cuda:0"});m.eval();d=next(m.base_model.parameters()).device;ids=encode_prompt(m.tokenizer,pr["messages"],e["thinking_mode"],d)
def H(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
masks=[];rows=[];om=Qwen3Model._update_causal_mask
def mw(self,am,x,cp,pkv,oa=False):
 z=om(self,am,x,cp,pkv,oa)
 if len(masks)<6:
  q={"seq":x.shape[1],"past":int(pkv[0][0].current_length.item()) if pkv is not None else 0,"cache_position":cp.cpu().tolist(),"impl":self.config._attn_implementation,"tree_mask":list(self.tree_mask.shape) if getattr(self,"tree_mask",None) is not None else None}
  if z is None:q["mask_none"]=True
  else:
   f=z.float().cpu();q.update(shape=list(f.shape),sha=H(z),min=float(f.min()),max=float(f.max()),nonzero=int((f!=0).sum()),sample=f.flatten()[:20].tolist())
  masks.append(q)
 return z
Qwen3Model._update_causal_mask=mw;ot=em.tree_decoding
class Halt(Exception):pass
def snap(p):
 n=int(p[0][0].current_length.item());return n,[(k.data[:,:,:n,:].cpu().clone(),v.data[:,:,:n,:].cpu().clone()) for k,v in p]
def restore(p,s):
 n,z=s
 for (k,v),(a,b) in zip(p,z):k.data[:,:,:n,:].copy_(a.to(k.data.device));v.data[:,:,:n,:].copy_(b.to(v.data.device));k.current_length.fill_(n);v.current_length.fill_(n)
def td(self,t,p,tp,ctx,ri):
 s=snap(p);n,z=s;ret=ri.cpu();pos=tp.cpu().reshape(-1).tolist();abspos=tp+ctx.shape[1];abspos=abspos.unsqueeze(0) if abspos.dim()==1 else abspos
 a0=ot(self,t,p,tp,ctx,ri);lg,hs,_=a0;pad=self.generation_manager.get_padding_token(ctx.device);cand=torch.cat((t,pad),1)[0,ri];b,a,samp=eval0(lg,cand,None);bi=int(b);ai=int(a);path=ret[bi].tolist();slot=int(path[ai]);row=lg[bi,ai].float().cpu();used=set(path);off=[i for i in range(t.shape[1]) if i not in used];tm=t.clone()
 for i in off:tm[0,i]=(int(tm[0,i])+9973)%m.base_model.config.vocab_size
 restore(p,s);b0=ot(self,tm,p,tp,ctx,ri);rl,rhs,_=b0;rr=rl[bi,ai].float().cpu();ha=hs[0,slot].float().cpu();hb=rhs[0,slot].float().cpu();v,ix=torch.topk(row,10)
 rows.append({"context_ids":ctx.cpu().tolist(),"tree_ids":t.cpu().tolist(),"mutated_ids":tm.cpu().tolist(),"tree_positions":pos,"absolute_positions":abspos.cpu().tolist(),"retrieve":ret.tolist(),"prefix_cache_len":n,"prefix_cache_hashes":[[H(x),H(y)] for x,y in z],"accepted":ai,"best":bi,"chosen_path_slots":path,"selected_slot":slot,"mutated_siblings":off,"sample_id":int(samp.item()),"dtype":str(lg.dtype),"top10":[{"id":int(i),"logit":float(x)} for x,i in zip(v,ix)],"logit_ids_11_272_7578":[float(row[i]) for i in (11,272,7578)],"replay_ids_11_272_7578":[float(rr[i]) for i in (11,272,7578)],"row_equal":torch.equal(row,rr),"hidden_hashes":[H(ha),H(hb)],"hidden_equal":torch.equal(ha,hb)})
 restore(p,s);raise Halt()
em.tree_decoding=td
try:m.eagle_generate(ids,temperature=0.,max_new_tokens=3,max_length=e["max_length"],log=True)
except Halt:pass
em.tree_decoding=ot;Qwen3Model._update_causal_mask=om;reset_tree_mode(m)
st=m.generation_manager.prepare_generation(m,ids,GenerationConfig(temperature=0.,max_new_tokens=4,max_length=e["max_length"],log=False,is_llama3=False))
for k,v in st.past_key_values:k.current_length.zero_();v.current_length.zero_()
with torch.inference_mode():
 o=m.base_model(st.input_ids,past_key_values=st.past_key_values,use_cache=True)
 for t in (785,7406,41506):o=m.base_model(torch.tensor([[t]],device=d),past_key_values=st.past_key_values,use_cache=True)
 inc=o.logits[0,-1].float().cpu();prefix=torch.cat((ids,torch.tensor([[785,7406,41506]],device=d)),1);full=m.base_model(prefix,use_cache=False);fl=full.logits[0,-1].float().cpu()
torch.cuda.synchronize()
def S(x):
 v,i=torch.topk(x,10);return {"sha":H(x),"top10":[{"id":int(j),"logit":float(k)} for k,j in zip(v,i)],"id11_272_7578":[float(x[j]) for j in (11,272,7578)]}
tr={"utc":datetime.now(UTC).isoformat(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=R,text=True).strip(),"target":mc["target_revision"],"draft":mc["draft_revision"],"angelslim":mc["angelslim_revision"],"manifest_sha":sha256_file(mp),"config_sha":sha256_file(R/"configs/pytorch_w1a1.toml"),"prompt_sha":sha256_file(R/e["prompt_manifest"]),"prompt":pr["id"],"input_ids":ids.cpu().tolist(),"known_prefix":[785,7406,41506],"device":torch.cuda.get_device_name(0),"target_dtype":str(m.base_model.dtype),"draft_dtype":str(next(m.eagle_layer.parameters()).dtype),"tf32":torch.backends.cuda.matmul.allow_tf32,"tree":rows[0] if rows else None,"effective_masks":masks,"incremental_pos40":S(inc),"full_prefix_pos40":S(fl),"argmax_equal":int(inc.argmax())==int(fl.argmax()),"max_absdiff":float((inc-fl).abs().max())}
p=O/"trace.json";p.write_text(json.dumps(tr,indent=2)+"\n");print(json.dumps({"trace":str(p),"tree":tr["tree"],"masks":masks,"incremental":S(inc),"full":S(fl)},indent=2))

