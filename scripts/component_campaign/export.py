"""Component rewrite with bounded vocabulary scratch and explicit sharing metadata."""
import json
import numpy as np
from pathlib import Path
from .common import R,TARGET,pin,save
from .quant import target_tensor
from scripts.dflash_baseline.model import GGUFReader
from gguf import GGUFWriter,GGMLQuantizationType as Type

def rewrite(source,dest,spec):
 dest=Path(dest);tmp=dest.with_suffix('.partial.gguf');reader=GGUFReader(source)
 writer=GGUFWriter(tmp,'dflash')
 scalar=spec.get('scalar',{});audit={};total=0
 for key,f in reader.fields.items():
  if key.startswith('GGUF.') or key in ('general.architecture','dflash.w1ax.private_head'):continue
  value=f.contents()
  if key.endswith('.w1ax.clip') and scalar.get('clip'):value=float(np.float16(value))
  writer.add_key_value(key,value,f.types[0],f.types[-1] if len(f.types)>1 else None)
 writer.add_bool('dflash.w1ax.private_head',bool(spec.get('head')))
 writer.add_bool('dflash.component.tied',bool(spec.get('tied')))
 writer.add_string('dflash.component.precision',json.dumps(spec,sort_keys=True))
 for t in reader.tensors:
  group='scale' if t.name.endswith('.w1a1_scale') else 'midpoint' if t.name.endswith('.w1ax_midpoint') else 'gain' if t.name.endswith('.w1ax_gain') else 'norm' if 'norm' in t.name and t.name.endswith('.weight') else None
  half=scalar.get(group,False);value=t.data.astype(np.float16) if half else t.data
  if half and not np.isfinite(value).all():raise ValueError('scalar storage overflow '+t.name)
  writer.add_tensor(t.name,value,raw_dtype=Type.F16 if half else t.tensor_type)
  total+=value.nbytes
  if half:audit[t.name]=dict(group=group,type='F16',bytes=value.nbytes,zeros=int((value==0).sum()),original_zeros=int((t.data==0).sum()),max_absolute_rounding=float(np.max(np.abs(value.astype(np.float32)-t.data))))
 target=None
 for name,key in [('token_embd.weight','embedding'),('output.weight','head')]:
  format=spec.get(key)
  if not format or (key=='head' and spec.get('tied')):continue
  if format=='Q6_K':
   target=target or target_tensor(TARGET);value=target.data
  else:
   path=R/'matrices'/f'{format}.bin';meta=json.load(open(path.with_suffix('.json')))
   value=np.memmap(path,mode='r',dtype=np.uint8,shape=tuple(meta['packed_shape']))
  writer.add_tensor(name,value,raw_dtype=Type[format]);total+=value.nbytes
  audit[name]=dict(type=format,bytes=value.nbytes,exact_original_codes=format=='Q6_K')
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close();tmp.rename(dest)
 save(dest.with_suffix('.component.json'),dict(model=pin(dest),base=pin(source),spec=spec,tensors=audit,tensor_bytes=total,tied_physical_allocation=bool(spec.get('tied')),target_head_unchanged=True,metadata_conversion='F16 stored values converted to F32 graph operands; conversion workspace is charged; clip stored rounded F32 GGUF scalar, no clip-metadata byte saving'))
 return dest
