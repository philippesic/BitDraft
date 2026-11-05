"""Component rewrite with bounded vocabulary scratch and explicit sharing metadata."""
import json
import numpy as np
from pathlib import Path
from .common import R,TARGET,pin,save
from .quant import target_tensor
from scripts.dflash_baseline.model import GGUFReader
from gguf import GGUFWriter,GGMLQuantizationType as Type,GGUFValueType

def rewrite(source,dest,spec,head_log_d=None,deployment_exps=None):
 dest=Path(dest);tmp=dest.with_suffix('.partial.gguf');reader=GGUFReader(source)
 writer=GGUFWriter(tmp,'dflash')
 scalar=spec.get('scalar',{});audit={};total=0
 deployment_exps=deployment_exps or {}
 for key,f in reader.fields.items():
  if key.startswith(('GGUF.','dflash.component.')) or key in ('general.architecture','dflash.w1ax.private_head'):continue
  value=f.contents()
  if key.endswith('.w1ax.clip'):
   if f.types[0]==GGUFValueType.UINT16:value=float(np.asarray(value,dtype=np.uint16).view(np.float16))
   if scalar.get('clip'):value=float(np.float16(value))
   if key in deployment_exps:value=float(deployment_exps[key])
   if scalar.get('clip') and spec.get('clip_bits_storage'):
    writer.add_uint16(key,int(np.asarray(value,dtype=np.float16).view(np.uint16)));continue
   writer.add_float32(key,value);continue
  writer.add_key_value(key,value,f.types[0],f.types[-1] if len(f.types)>1 else None)
 writer.add_bool('dflash.w1ax.private_head',bool(spec.get('head')))
 writer.add_bool('dflash.component.tied',bool(spec.get('tied')))
 writer.add_string('dflash.component.precision',json.dumps(spec,sort_keys=True))
 for t in reader.tensors:
  group='scale' if t.name.endswith('.w1a1_scale') else 'midpoint' if t.name.endswith('.w1ax_midpoint') else 'gain' if t.name.endswith('.w1ax_gain') else 'norm' if 'norm' in t.name and t.name.endswith('.weight') else None
  half=scalar.get(group,False);value=t.data.astype(np.float16) if half else t.data
  if t.name in deployment_exps:value=deployment_exps[t.name].cpu().numpy()
  if half and not np.isfinite(value).all():raise ValueError('scalar storage overflow '+t.name)
  writer.add_tensor(t.name,value,raw_dtype=Type.F16 if half else t.tensor_type)
  total+=value.nbytes
  if half:audit[t.name]=dict(group=group,type='F16',bytes=value.nbytes,zeros=int((value==0).sum()),original_zeros=int((t.data==0).sum()),max_absolute_rounding=float(np.max(np.abs(value.astype(np.float32)-t.data))))
 target=None
 for name,key in [('token_embd.weight','embedding'),('output.weight','head')]:
  format=spec.get(key)
  if not format or (key=='embedding' and spec.get('tied')):continue
  if format=='Q6_K':
   target=target or target_tensor(TARGET);value=target.data
  else:
   from .quant import matrix_data
   value=matrix_data(format)
  if spec.get('r2_vocab_scale') and (key=='head' or not spec.get('head')):
   if head_log_d is None:raise ValueError('R2 export requires trained native block-scales')
   from .head_qat import encoded_scales
   value=encoded_scales(value,format,head_log_d,deployment_exps.get('vocabulary.block_scale'))
  writer.add_tensor(name,value,raw_dtype=Type[format]);total+=value.nbytes
  audit[name]=dict(type=format,bytes=value.nbytes,exact_original_codes=format=='Q6_K')
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close();tmp.rename(dest)
 save(dest.with_suffix('.component.json'),dict(model=pin(dest),base=pin(source),spec=spec,tensors=audit,tensor_bytes=total,tied_physical_allocation=bool(spec.get('tied')),target_head_unchanged=True,metadata_conversion='F16 tensor values converted to F32 graph operands; conversion workspace charged. Clip uses binary16 bits in UINT16 GGUF scalar with runtime04 when requested, otherwise rounded F32 metadata; native clip op_params remain F32.',gpu_exported_exp_values=sorted(deployment_exps)))
 return dest
