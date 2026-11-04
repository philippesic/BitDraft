"""TRAIN-only zero-update precision perturbation and local representation screen."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time
import numpy as np
import torch
P=Path('/home/philip/binary-eagle-decoding');S=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103')
sys.path[:0]=[str(S/'scripts/w1a8_generalization'),str(S/'src'),str(P/'third_party/llama.cpp/gguf-py'),str(P/'scripts/contender_pipeline')]
from runner import read,pin,atomic,require
from modern_actor import build_modern_actor,load_scale_state
from modern_actual_data import ActualDataset
from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_qat import W1AxContract,shared_round_hard_signs,hard_activation
from sharded_rows import ShardRows
from safetensors import safe_open
PARENT=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt')
VARIANTS=['A8_control','A4_all','A1_all','A4_body_A8_head','A1_body_A8_head','A4_mlp_only']

def main(a):
 require(not a.out.exists(),'unique output required');a.out.mkdir();started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 inputs=read(a.inputs);require(pin(inputs['shards']['path'])==inputs['shards'],'shards changed');reader=ShardRows(inputs['shards']['path'],max_maps=2);datasets={};windows=[];embeddings={};embedding_hashes={}
 for item in inputs['selected']:
  ref=item['manifest'];require(pin(ref['path'])==ref,'selected metadata changed')
  if ref['path'] not in datasets:datasets[ref['path']]=ActualDataset(ref['path']);datasets[ref['path']]._rows=reader
  data=datasets[ref['path']];w=data._window_ids[item['window']];require(w['role']=='fit' and w['group_id']==item['group'] and w['anchor']==item['anchor'] and w['max_depth']==2,'TRAIN selection changed');windows.append((data,w,item))
  for token in set(w['prefix_tokens']+[w['seed_token']]+w['labels'][:-1]):
   h=data.manifest['embedding_rows'][str(token)]['sha256']
   if token in embeddings:require(embedding_hashes[token]==h,'embedding disagreement')
   else:embeddings[token]=data.embedding_lookup(token,'cuda');embedding_hashes[token]=h
 parent=torch.load(PARENT,map_location='cpu',weights_only=False);require(parent['cursor']==5760,'wrong parent');lookup=ExternalNativeF32Embedding(lookup=embeddings.__getitem__,vocab_size=248320,hidden_size=2560,target_identity='admitted native TRAIN features',target_sha256=next(iter(datasets.values())).manifest['target']['sha256'],capture_sha256=inputs['registry']['sha256'])
 actor=build_modern_actor(P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json',embedding_lookup=lookup,device='cuda',scale_only=True);require(actor.contract==parent['contract']['actor'],'actor source/arithmetic differs');load_scale_state(actor,parent['scales']);actor.drafter.eval();versions={n:p._version for n,p in actor.drafter.named_parameters()}
 absolute=actor.drafter.d2t+torch.arange(50000,device='cuda');reverse=torch.full((248320,),-1,device='cuda',dtype=torch.long);reverse[absolute]=torch.arange(50000,device='cuda');labels=torch.tensor([t for _,w,_ in windows for t in w['labels']],device='cuda');classes=reverse[labels];supported=classes>=0
 captures={name:[] for name in actor.linears};capture_meta={name:[] for name in actor.linears};capture_counts={};current={};handles=[]
 def hook(name):
  def collect(module,args):
   x=args[0].detach().reshape(-1,args[0].shape[-1]);remaining=8-capture_counts.get(name,0);count=min(2,remaining,len(x))
   if count:
    ix=torch.linspace(0,len(x)-1,count,device=x.device).long();captures[name].append(x[ix].cpu().clone());capture_meta[name].extend([dict(window=current['window'],domain=current['domain']) for _ in range(count)]);capture_counts[name]=capture_counts.get(name,0)+count
  return collect
 def forward():
  values=[]
  for data,w,item in windows:
   capture_counts.clear();current.clear();current.update(item);batch=data.materialize(w,'cuda');cache=actor.adapter.build_context_cache(batch.prefix_tokens,batch.prefix_rawfeatures,chunk_size=64);state=actor.adapter.encode_feature(batch.seed_rawfeature);token=w['seed_token']
   for d in range(2):
    step=actor.adapter.decode_step(token,state,w['anchor']+d,cache);cache=step.cache;values.append(step.logits);state=step.pre_norm;token=w['labels'][d]
  return torch.stack(values)
 metrics={};baseline=None
 with torch.no_grad(),shared_round_hard_signs(actor.linears):
  for variant in VARIANTS:
   for name,module in actor.linears.items():
    bits=8
    if variant.startswith('A4') and (variant=='A4_all' or variant=='A4_body_A8_head' and name!='lm_head' or variant=='A4_mlp_only' and '.mlp.' in name):bits=4
    if variant.startswith('A1') and (variant=='A1_all' or variant=='A1_body_A8_head' and name!='lm_head'):bits=1
    module.contract=W1AxContract(bits)
   if variant=='A8_control':handles=[module.register_forward_pre_hook(hook(name)) for name,module in actor.linears.items()]
   logits=forward();require(torch.isfinite(logits).all(),'nonfinite logits');predicted=absolute[logits.argmax(-1)];matches=predicted==labels
   if baseline is None:
    baseline=logits.clone()
    for h in handles:h.remove()
   path=a.out/(variant+'.npy');np.save(path,logits.cpu().numpy());m=dict(precision_by_projection={n:m.contract.activation_bits for n,m in actor.linears.items()},positions=len(labels),supported_labels=int(supported.sum()),gold_top1=int(matches.sum()),gold_first_position=int(matches[::2].sum()),gold_both_positions=int(matches.reshape(-1,2).all(-1).sum()),agreement_with_A8=int((logits.argmax(-1)==baseline.argmax(-1)).sum()),supported_mean_ce=float(torch.nn.functional.cross_entropy(logits[supported],classes[supported])),logits=pin(path));metrics[variant]=m;print(json.dumps(dict(variant=variant,**{k:v for k,v in m.items() if k not in ['precision_by_projection','logits']})),flush=True)
  for module in actor.linears.values():module.contract=W1AxContract(8)
  repeated=forward();require(torch.equal(repeated,baseline),'A8 repeat differs after perturbations');require(versions=={n:p._version for n,p in actor.drafter.named_parameters()},'model parameters mutated')
 del baseline,repeated,logits
 captured={name:torch.cat(values) for name,values in captures.items()};torch.save(captured,a.out/'activations.pt');atomic(a.out/'activation-rows.json',capture_meta)
 source=P/'runs/spec-shortlist-20261030/models/hf-eagle-4B/model.safetensors';original=read(P/'runs/w1a8-modern-packed-init-20261103-01/freeze.json')['source'];require(pin(source)==original,'dense original changed');mapping=read(P/'runs/w1a8-modern-packed-init-20261103-01/initialization-manifest.json')['projections'];local={}
 with safe_open(source,framework='pt',device='cpu') as store,torch.no_grad():
  for base,info in mapping.items():
   name=info['checkpoint_name'].removesuffix('.weight');W=store.get_tensor(info['checkpoint_name']).cuda().float();X=captured[name].cuda();require(W.shape[1]==X.shape[1],'projection shape differs');dense=torch.nn.functional.linear(X,W);den=dense.square().sum();sign=torch.where(W>=0,1.,-1.);alpha=W.abs().mean(-1,keepdim=True);row=sign*alpha;group=(sign.reshape(W.shape[0],-1,128)*W.reshape(W.shape[0],-1,128).abs().mean(-1,keepdim=True)).reshape_as(W);residual=W-row;two=row+torch.where(residual>=0,1.,-1.)*residual.abs().mean(-1,keepdim=True);results={}
   for rep,Q in [('row',row),('group128',group),('two_planes',two)]:
    for bits in ([1,4,8] if rep=='row' else [4,8]):
     xq=hard_activation(X,bits)[0];Y=torch.nn.functional.linear(xq,Q);results[f'{rep}_A{bits}']=dict(output_nmse=float((Y-dense).square().sum()/den),weight_nmse=float((Q-W).square().sum()/W.square().sum()),top1_agreement_with_dense=int((Y.argmax(-1)==dense.argmax(-1)).sum()) if base=='output' else None)
   local[base]=dict(rows=len(X),shape=list(W.shape),metrics=results);print(json.dumps(dict(projection=base,nmse={k:round(v['output_nmse'],4) for k,v in results.items()})),flush=True);del W,X,dense,sign,alpha,row,group,residual,two,Q,Y,xq
 atomic(a.out/'result.json',dict(status='TRAIN_ONLY_ZERO_UPDATE_SCREEN_PASS',training_updates=0,inputs=pin(a.inputs),parent=pin(PARENT),dense_source=original,source=pin(__file__),variants=metrics,A8_repeat_bit_exact=True,parameters_unchanged=True,activation_capture=pin(a.out/'activations.pt'),activation_rows=pin(a.out/'activation-rows.json'),local_projection_screen=local,wall_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),scope='12TRAIN groups/24teacher-forced positions, not serving acceptance or independent generalization. A8-trained signs/scales favor A8; precision perturbations are unadapted, not fair equal-QAT model rankings. Local group/two-plane approximations are separately initialized from original dense BF16 on common current-actor inputs; reconstruction/top1 proxy only, no native exporter/kernel deployment claim.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
