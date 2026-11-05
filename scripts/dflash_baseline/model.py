"""Modern six-layer DFlash QAT, matching native input taps and mask slots."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(ROOT/'src'),str(P/'third_party/llama.cpp/gguf-py')]
from gguf import GGUFReader
from gguf.quants import dequantize
from w1a1_eagle.block_qat import BlockLayer,_FrozenLinear,_rms
TAPS=(2,6,10,14,18,22,26,30)
class NativeBF16Linear(_FrozenLinear):
 def forward(self,x):
  rounded=x.to(torch.bfloat16).float()
  x=x+(rounded-x).detach()
  return F.linear(x,self.weight)
class Model(nn.Module):
 def __init__(self,source,target):
  super().__init__()
  self.config=SimpleNamespace(hidden_size=2560,intermediate_size=9216,num_heads=32,num_kv_heads=8,head_dim=128,block_size=16,activation_bits=8,norm_eps=1e-6,rope_theta=1e7)
  reader=GGUFReader(source)
  assert reader.fields['dflash.target_layers'].contents()==list(TAPS)
  self.mask=int(reader.fields['tokenizer.ggml.mask_token_id'].contents())
  self.tensors={}
  for t in reader.tensors:
   self.tensors[t.name]=torch.from_numpy(dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32))
  self.fc=_FrozenLinear(self.tensors.pop('fc.weight'))
  self.output_norm_enc=nn.Parameter(self.tensors.pop('enc.output_norm.weight'))
  self.output_norm=nn.Parameter(self.tensors.pop('output_norm.weight'))
  self.layers=nn.ModuleList()
  for i in range(6):
   layer=BlockLayer(self.tensors,i,self.config,None)
   for name in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm'):
    v=getattr(layer,name);delattr(layer,name);setattr(layer,name,nn.Parameter(v))
   self.layers.append(layer)
   for name in list(self.tensors):
    if name.startswith(f'blk.{i}.'):del self.tensors[name]
  assert not self.tensors
  # Source target is frozen and quantized; use its actual dequantized tensors.
  target_reader=GGUFReader(target);lookup={t.name:t for t in target_reader.tensors}
  def decode(name):
   t=lookup[name];return torch.from_numpy(dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32))
  self.embedding_cpu=decode('token_embd.weight')
  self.register_buffer('head',decode('output.weight') if 'output.weight' in lookup else self.embedding_cpu.clone())
 def binary_linears(self):
  return {f'blk.{i}.{name}':getattr(l,name) for i,l in enumerate(self.layers) for name in ('ffn_gate','ffn_up','ffn_down')}
 def forward(self,features,tokens,anchor,slots=16):
  device=self.head.device;self.config.block_size=slots
  positions=torch.arange(anchor,anchor+slots,device=device)
  context_positions=torch.arange(anchor,device=device)
  inp=torch.tensor([tokens[anchor]]+[self.mask]*(slots-1))
  state=F.embedding(inp,self.embedding_cpu).to(device)
  features=features[:anchor].to(device)
  encoded=_rms(self.fc(features),self.output_norm_enc,1e-6)
  key_positions=torch.cat((context_positions,positions))
  for i,layer in enumerate(self.layers):
   context=layer.project_context(encoded,context_positions)
   allowed=torch.ones((slots,anchor+slots),dtype=torch.bool,device=device)
   if i<5:allowed &= (positions[:,None]-key_positions[None,:]).abs()<4096
   state,_=layer(state,context,positions,allowed)
  hidden=_rms(state,self.output_norm,1e-6)
  return F.linear(hidden,self.head)
class Capture:
 def __init__(self,path):
  self.path=Path(path);self.meta=json.load(open(self.path/'capture.json'))
  assert self.meta['taps']==list(TAPS) and self.meta['feature_semantics']=='zero_based_pre_block_input_all_tokens'
  self.tokens=self.meta['input_tokens'];n=len(self.tokens)
  self.features=torch.from_numpy(np.concatenate([np.memmap(self.path/f'layer-{t}.f32',dtype='<f4',mode='r',shape=(n,2560)) for t in TAPS],axis=1))
  self.labels=dict(zip(self.meta['logit_positions'],self.meta['greedy_ids']))
  # Native logit at p predicts token p+1; DFlash mask slot j predicts anchor+j.
  self.anchors=[a for a in range(self.meta['prompt_tokens'],n-15) if all(a+j-1 in self.labels for j in range(1,16))]
 def loss(self,model,anchor):
  logits=model(self.features,self.tokens,anchor)[1:]
  labels=torch.tensor([self.labels[anchor+j-1] for j in range(1,16)],device=logits.device)
  losses=F.cross_entropy(logits,labels,reduction='none')
  weights=torch.exp(-torch.arange(15,device=logits.device)/7)
  return (losses*weights).sum()/weights.sum(),(logits.argmax(-1)==labels).float().mean()
