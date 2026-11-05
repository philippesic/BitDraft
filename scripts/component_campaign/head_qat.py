"""Bounded R2: learn one existing native half block-scale per256 head weights.

Codes and subscales remain frozen. Q6/Q4/Q3 controls have identical trainable
capacity; no dense vocabulary gradient/master/Adam allocation is introduced.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .quant import matrix_data
from scripts.dflash_baseline.model import dequantize
from gguf import GGMLQuantizationType as Type,GGML_QUANT_SIZES
from .simulation import ComponentModel,stored

def scale_offset(format):
 return 0 if format=='Q4_K' else GGML_QUANT_SIZES[Type[format]][1]-2

def encoded_scales(data,format,log_d):
 result=np.array(data,copy=True);blocks=result.reshape(*log_d.shape,-1)
 offset=scale_offset(format)
 rounded=log_d.detach().cpu().float().exp().half().numpy()
 if not np.isfinite(rounded).all():raise ValueError('head block-scale overflow')
 blocks[...,offset:offset+2]=np.ascontiguousarray(rounded).view(np.uint8).reshape(*rounded.shape,2)
 return result

class Head(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,log_d,weight,coefficients):
  ctx.save_for_backward(x,log_d,weight,coefficients)
  return F.linear(x,weight)
 @staticmethod
 def backward(ctx,gradient):
  x,log_d,weight,coefficients=ctx.saved_tensors
  dx=gradient@weight if ctx.needs_input_grad[0] else None
  dd=torch.empty_like(log_d) if ctx.needs_input_grad[1] else None
  if dd is not None:
   for start in range(0,len(weight),1024):
    gw=gradient[:,start:start+1024].T@x
    dd[start:start+1024]=(gw*coefficients[start:start+1024].float()).reshape(-1,log_d.shape[1],256).sum(-1)*log_d[start:start+1024].exp()
  return dx,dd,None,None

class CalibratedModel(ComponentModel):
 @torch.no_grad()
 def refresh_head(self):
  d=self.head_log_d.exp().half().float()
  if not torch.isfinite(d).all():raise ValueError('head scale overflow')
  for start in range(0,len(self.head),1024):
   value=self.head_coefficients[start:start+1024].float().reshape(-1,d.shape[1],8,32)*d[start:start+1024,:,None,None]
   if self.head_offset is not None:value+=self.head_offset[start:start+1024,:,:,None]
   self.head[start:start+1024].copy_(value.reshape(-1,self.head.shape[1]))
 def project_logits(self,hidden):
  if not self.component_spec.get('head'):return F.linear(hidden,self.head)
  self.refresh_head()
  return Head.apply(hidden,self.head_log_d,self.head,self.head_coefficients)
 def token_lookup(self,ids,device):
  if not self.component_spec.get('embedding'):return super().token_lookup(ids,device)
  ids=ids.to(device);d=stored(self.head_log_d[ids].exp(),True)
  value=self.head_coefficients[ids].float().reshape(-1,d.shape[1],8,32)*d[:,:,None,None]
  if self.head_offset is not None:value=value+self.head_offset[ids,:,:,None]
  return value.reshape(len(ids),-1)

def install(model,spec):
 format=spec.get('head',spec.get('embedding'))
 if format not in ('Q6_K','Q4_K','Q3_K'):raise ValueError('R2 block-scale format unsupported')
 data=matrix_data(format);rows,width=model.head.shape;block,bytes_per_block=GGML_QUANT_SIZES[Type[format]]
 if block!=256:raise ValueError('R2 block geometry')
 coded=data.reshape(rows,width//256,bytes_per_block);offset=scale_offset(format)
 d=np.ascontiguousarray(coded[...,offset:offset+2]).view('<f2').reshape(rows,width//256).astype(np.float32)
 coefficients=np.empty((rows,width),dtype=np.int16)
 offsets=np.empty((rows,width//256,8),dtype=np.float32) if format=='Q4_K' else None
 for start in range(0,rows,256):
  raw=np.array(data[start:start+256],copy=True);blocks=raw.reshape(-1,width//256,bytes_per_block)
  blocks[...,offset:offset+2]=np.array([1.],dtype='<f2').view(np.uint8)
  if format=='Q4_K':blocks[...,2:4]=0
  value=dequantize(raw,Type[format]).reshape(-1,width)
  if not np.equal(value,value.astype(np.int16)).all():raise ValueError('head grid is not integral')
  coefficients[start:start+256]=value.astype(np.int16)
  if offsets is not None:
   original=np.array(data[start:start+256],copy=True).reshape(-1,width//256,bytes_per_block)
   original[...,0:2]=0
   value=dequantize(original.reshape(len(original),-1),Type[format]).reshape(-1,width//256,8,32)
   if not np.equal(value,value[...,:1]).all():raise ValueError('Q4 offset grouping differs')
   offsets[start:start+256]=value[...,0]
 model.__class__=CalibratedModel;model.component_spec=spec
 model.register_buffer('head_coefficients',torch.from_numpy(coefficients))
 model.register_buffer('head_offset',torch.from_numpy(offsets) if offsets is not None else None)
 model.head_log_d=nn.Parameter(torch.from_numpy(np.log(np.maximum(d,1e-12))))
 return dict(recipe='R2-native-block-scale',component='head+embedding' if spec.get('tied') else 'head' if spec.get('head') else 'embedding',format=format,trainable_scalars=d.size,grouping=256,initial_zero_blocks=int((d==0).sum()),frozen_codes=True,extra_vocabulary_F32_master=False,head_scale_lr=1e-4)

def optimizer(model):
 from scripts.dflash_baseline.prototype_model import optimizer as ordinary
 model.head_log_d.requires_grad_(False);opt=ordinary(model);model.head_log_d.requires_grad_(True)
 opt.add_param_group(dict(params=[model.head_log_d],lr=1e-4,name='head_native_block_scale'))
 return opt

def cpu_gradient_fixture():
 torch.manual_seed(987);x=torch.randn(3,256,requires_grad=True);d=torch.full((17,1),-4.1,requires_grad=True)
 a=torch.randint(-32,32,(17,256),dtype=torch.int16);weight=d.detach().exp().half().float()*a.float();g=torch.randn(3,17)
 y=Head.apply(x,d,weight,a);(y*g).sum().backward();dx=x.grad.clone();dd=d.grad.clone()
 x.grad=None;d.grad=None;scale=d.exp();rounded=scale+(scale.half().float()-scale).detach();reference=F.linear(x,rounded*a.float());(reference*g).sum().backward()
 if not torch.equal(dx,x.grad) or not torch.allclose(dd,d.grad,rtol=2e-6,atol=2e-6):raise ValueError('bounded head gradient fixture failed')
 return dict(status='CPU_ANALYTIC_GRADIENT_PASS',max_scale_gradient_error=float((dd-d.grad).abs().max()),native_full_head_cuda_and_export_resume='pending evidence-led R2 use')

if __name__=='__main__':print(cpu_gradient_fixture())
