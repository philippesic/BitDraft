"""Capture the actual GPU-rounded exported exp values, never round latent logs."""
import torch

def values(model,spec):
 result={};mismatches={};scalar=spec.get('scalar',{})
 with torch.no_grad():
  for base,module in model.binary_linears().items():
   for group,parameter,key in [('gain',module.log_gain,base+'.w1ax_gain'),('clip',module.log_clip,base+'.w1ax.clip')]:
    if not scalar.get(group) or (group=='gain' and base.endswith('_up')):continue
    gpu=parameter.exp().half().cpu();cpu=parameter.cpu().exp().half()
    result[key]=gpu;mismatches[key]=int((gpu!=cpu).sum())
  if spec.get('r2_vocab_scale'):
   gpu=model.head_log_d.exp().half().cpu();cpu=model.head_log_d.cpu().exp().half()
   result['vocabulary.block_scale']=gpu;mismatches['vocabulary.block_scale']=int((gpu!=cpu).sum())
 return result,mismatches
