"""Frozen, drafter-private weight quantization for the finite component queue."""
import numpy as np
import torch
from .model import GGUFReader,dequantize
from gguf import GGMLQuantizationType as Type
from gguf.quants import quantize
FORMATS={'Q8_0':Type.Q8_0,'Q4_0':Type.Q4_0}
def tensor_component(name):
 if name=='fc.weight':return 'fusion'
 if any(name.endswith('.'+p+'.weight') for p in ('attn_q','attn_k')):return 'attention_qk'
 if any(name.endswith('.'+p+'.weight') for p in ('attn_v','attn_output')):return 'attention_vo'
 if name=='output.weight':return 'head'
 return None

def encoded(weight,format):
 # Native GGUF weight block quantization; fixed weights, no new trainable masters.
 if format not in FORMATS:raise ValueError('unsupported component format')
 return quantize(np.ascontiguousarray(weight,dtype=np.float32),FORMATS[format])
def decoded(weight,format):
 # Bound conversion scratch for the large vocabulary head.
 result=np.empty(weight.shape,dtype=np.float32)
 for start in range(0,len(weight),256):
  result[start:start+256]=dequantize(encoded(weight[start:start+256],format),FORMATS[format]).reshape(weight[start:start+256].shape)
 return torch.from_numpy(result)
def install_components(model,config):
 specs=config.get('components',{})
 for name,module in model.named_modules():
  if not hasattr(module,'weight') or name.endswith(('ffn_gate','ffn_up','ffn_down')):continue
  canonical=name.replace('layers.','blk.')+'.weight'
  kind=tensor_component(canonical)
  if kind in specs:
   module.weight=decoded(module.weight.cpu().numpy(),specs[kind])
 if 'head' in specs:model.head=decoded(model.head.cpu().numpy(),specs['head'])
