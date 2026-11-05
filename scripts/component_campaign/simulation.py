"""Declared storage semantics for frozen vocabulary, metadata and actual draft KV."""
import math
import torch
from torch.nn import functional as F
from scripts.dflash_baseline.model import Model
from scripts.dflash_baseline.prototype import Projection
from w1a1_eagle.block_qat import BlockLayer,_rms,_rope

def stored(x,enabled):
 if not enabled:return x
 rounded=x.half().float()
 if not torch.isfinite(rounded).all():raise ValueError('F16 storage overflow')
 return x+(rounded-x).detach()

def hadamard(x,size):
 # Same normalized Walsh-Hadamard matrix as llama-kv-cache.cpp; contiguous head groups.
 shape=x.shape;y=x.reshape(*shape[:-1],-1,size)
 stride=1
 while stride<size:
  z=y.reshape(*y.shape[:-1],-1,2,stride);a,b=z[...,0,:],z[...,1,:]
  y=torch.stack((a+b,a-b),-2).reshape(y.shape);stride*=2
 return (y/math.sqrt(size)).reshape(shape)

def cache_round(x,format):
 if format=='f16':return stored(x,True)
 shape=x.shape;v=x.reshape(-1,32)
 with torch.no_grad():
  if format=='q8_0':
   d=v.abs().amax(-1,keepdim=True)/127
   z=v*torch.where(d!=0,1/d,0)
   codes=torch.sign(z)*torch.floor(z.abs()+.5) # native CUDA roundf: ties away from zero
  elif format in ('q4_0','q5_0'):
   q=8 if format=='q4_0' else 16
   idx=v.abs().argmax(-1,keepdim=True);d=v.gather(-1,idx)/-q
   z=v*torch.where(d!=0,1/d,0)
   codes=(z+q+.5).to(torch.int32).clamp(max=2*q-1).float()-q
  else:raise ValueError('unqualified cache format '+format)
  rounded=(codes*d.half().float()).reshape(shape)
 return x+(rounded-x).detach()

class StorageProjection(Projection):
 def stream(self,x):
  # Round the EXPORTED exp values, with STE to the F32 latent masters.
  gain=stored(self.log_gain.exp(),self.storage.get('gain',False))
  x=x.float()*gain
  if not torch.isfinite(x).all():raise ValueError('nonfinite gained input')
  qmax=max(1,2**(self.bits-1)-1)
  bound=stored(self.log_clip.exp(),self.storage.get('clip',False))
  clipped=torch.maximum(torch.minimum(x,bound),-bound);z=clipped*(qmax/bound)
  hard=torch.where(x>=0,1.,-1.) if self.bits==1 else z.round().clamp(-qmax,qmax)
  return z+(hard-z).detach(),bound/qmax

 def forward(self,x):
  codes,step=self.stream(x)
  from w1a1_eagle.recurrent_binary import hard_sign_ste
  signs=hard_sign_ste(self.latent_sign)
  scale=stored(self.scale.clamp_min(0),self.storage.get('scale',False))
  midpoint=stored(self.midpoint,self.storage.get('midpoint',False))
  native=(F.linear(codes,signs)*scale)*step+(codes.sum(-1,keepdim=True)*step)*midpoint
  values=codes*step;surrogate=F.linear(values,signs)*scale+values.sum(-1,keepdim=True)*midpoint
  return native.detach()+(surrogate-surrogate.detach())

class ComponentLayer(BlockLayer):
 def norm(self,x,name):return _rms(x,stored(getattr(self,name),self.scalar.get('norm',False)),self.config.norm_eps)
 def project_context(self,encoded,positions):
  cfg=self.config
  k=self.attn_k(encoded).view(-1,cfg.num_kv_heads,cfg.head_dim)
  v=self.attn_v(encoded).view(-1,cfg.num_kv_heads,cfg.head_dim)
  k=_rope(self.norm(k,'attn_k_norm'),positions,cfg.rope_theta)
  if self.kformat!='f16':k=hadamard(k,128)
  if self.vformat!='f16':v=hadamard(v,64)
  return cache_round(k,self.kformat),cache_round(v,self.vformat)
 def forward(self,state,context_kv,positions,allowed):
  cfg=self.config;norm=self.norm(state,'attn_norm')
  q=self.attn_q(norm).view(-1,cfg.num_heads,cfg.head_dim)
  k=self.attn_k(norm).view(-1,cfg.num_kv_heads,cfg.head_dim)
  v=self.attn_v(norm).view(-1,cfg.num_kv_heads,cfg.head_dim)
  q=_rope(self.norm(q,'attn_q_norm'),positions,cfg.rope_theta)
  k=_rope(self.norm(k,'attn_k_norm'),positions,cfg.rope_theta)
  if self.kformat!='f16':q,k=hadamard(q,128),hadamard(k,128)
  if self.vformat!='f16':v=hadamard(v,64)
  noise=(k,v);ck,cv=context_kv
  k=torch.cat((ck,cache_round(k,self.kformat)),0);v=torch.cat((cv,cache_round(v,self.vformat)),0)
  groups=cfg.num_heads//cfg.num_kv_heads;q=q.view(cfg.block_size,cfg.num_kv_heads,groups,cfg.head_dim)
  score=torch.einsum('qhgd,khd->hgqk',q,k)/math.sqrt(cfg.head_dim)
  score=score.masked_fill(~allowed[None,None],-torch.inf)
  attended=torch.einsum('hgqk,khd->qhgd',F.softmax(score,-1),v)
  if self.vformat!='f16':attended=hadamard(attended,64)
  residual=state+self.attn_output(attended.reshape(cfg.block_size,-1))
  norm=self.norm(residual,'ffn_norm')
  return residual+self.ffn_down(F.silu(self.ffn_gate(norm))*self.ffn_up(norm)),noise

class ComponentModel(Model):
 def forward(self,features,tokens,anchor,slots=16):
  device=self.head.device;self.config.block_size=slots
  positions=torch.arange(anchor,anchor+slots,device=device);context_positions=torch.arange(anchor,device=device)
  inp=torch.tensor([tokens[anchor]]+[self.mask]*(slots-1))
  state=F.embedding(inp,self.embedding_cpu).to(device)
  encoded=_rms(self.fc(features[:anchor].to(device)),stored(self.output_norm_enc,self.scalar.get('norm',False)),1e-6)
  key_positions=torch.cat((context_positions,positions))
  for i,layer in enumerate(self.layers):
   context=layer.project_context(encoded,context_positions)
   allowed=torch.ones((slots,anchor+slots),dtype=torch.bool,device=device)
   if i<5:allowed&=(positions[:,None]-key_positions[None,:]).abs()<4096
   state,_=layer(state,context,positions,allowed)
  return F.linear(_rms(state,stored(self.output_norm,self.scalar.get('norm',False)),1e-6),self.head)

def install(model,spec):
 from .common import R
 import numpy as np,json
 from scripts.dflash_baseline.model import dequantize
 from gguf import GGMLQuantizationType as Type
 model.__class__=ComponentModel;model.scalar=spec.get('scalar',{})
 for layer in model.layers:
  layer.__class__=ComponentLayer;layer.scalar=model.scalar
  layer.kformat,layer.vformat=spec.get('kv',['f16','f16'])
  for name in ('ffn_gate','ffn_up','ffn_down'):
   p=getattr(layer,name)
   if not isinstance(p,Projection) or p.quantizer['mode']!='learned':raise ValueError('frozen B0 recipe required')
   p.__class__=StorageProjection;p.storage=model.scalar
 for key in ('head','embedding'):
  format=spec.get(key)
  if not format or format=='Q6_K':continue
  path=R/'matrices'/f'{format}.bin';meta=json.load(open(path.with_suffix('.json')))
  data=np.memmap(path,mode='r',dtype=np.uint8,shape=tuple(meta['packed_shape']))
  weight=model.head if key=='head' else model.embedding_cpu
  for start in range(0,len(data),256):
   value=dequantize(data[start:start+256],Type[format]).reshape(-1,2560).copy()
   weight[start:start+256].copy_(torch.from_numpy(value))
 return model
