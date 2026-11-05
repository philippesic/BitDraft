"""Actual full-head gradient/optimizer and native packed-grid qualification."""
import numpy as np
import torch
from .quant import matrix_data
from .head_qat import scale_offset
from scripts.dflash_baseline.model import GGUFReader,dequantize

def inspect(model,opt,spec,native):
 ids=np.array([0,1,123,100001,200001,248077,248319],dtype=np.int64)
 key='output.weight' if spec.get('head') else 'token_embd.weight'
 tensor=next(t for t in GGUFReader(native).tensors if t.name==key)
 native_values=dequantize(tensor.data[ids],tensor.tensor_type).reshape(len(ids),2560)
 with torch.no_grad():
  observed=model.head[ids].detach().cpu().numpy() if spec.get('head') else model.token_lookup(torch.tensor(ids),'cuda').detach().cpu().numpy()
 error=float(np.abs(native_values-observed).max())
 if error!=0:raise ValueError('R2 deployed packed weight differs from simulated full head/lookup '+str(error))
 parameter=model.head_log_d;gradient=parameter.grad;state=opt.state.get(parameter,{})
 if gradient is not None and (not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient)):raise ValueError('missing or nonfinite full-head scale gradient')
 for k in ('exp_avg','exp_avg_sq'):
  if state and (state[k].shape!=parameter.shape or not torch.isfinite(state[k]).all()):raise ValueError('R2 optimizer state is invalid')
 # Every code/subscale/offset byte must remain fixed; only native d is eligible.
 original=matrix_data(spec.get('head',spec.get('embedding')))
 offset=scale_offset(tensor.tensor_type.name);bytes_per_block=__import__('gguf').GGML_QUANT_SIZES[tensor.tensor_type][1]
 current=tensor.data[ids].reshape(len(ids),-1,bytes_per_block);base=original[ids].reshape(current.shape)
 keep=np.ones(bytes_per_block,dtype=bool);keep[offset:offset+2]=False
 if not np.array_equal(current[...,keep],base[...,keep]):raise ValueError('R2 unexpectedly changed frozen packed codes/subscales')
 return dict(status='CUDA_FULL_HEAD_NATIVE_GRID_PASS',sample_ids=ids.tolist(),sampled_native_dequant_max_error=error,frozen_code_and_subscale_bytes_preserved=True,scale_master_shape=list(parameter.shape),scale_master_dtype=str(parameter.dtype),gradient_finite=gradient is not None,nonzero_gradient_scalars=int(torch.count_nonzero(gradient)) if gradient is not None else None,gradient_norm=float(gradient.norm()) if gradient is not None else None,adam_step=float(state['step']) if state else 0.,adam_moments_valid=bool(state))
