"""Export trained signs/scales/norms without modifying protected tensors."""
import sys,json
from pathlib import Path
import numpy as np
import torch
P=Path('/home/philip/binary-eagle-decoding');ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'third_party/llama.cpp/gguf-py'),str(ROOT/'scripts')]
from gguf import GGUFReader,GGUFWriter,GGMLQuantizationType as Type
from gguf.quants import dequantize
from export_recurrent_binary import pack,raw_hash,same_tensor,sha256

def export(checkpoint,destination):
 destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
 params=saved['parameters'];source=P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-bf16.gguf'
 reader=GGUFReader(source);selected={f'blk.{i}.ffn_{p}.weight' for i in range(6) for p in ('gate','up','down')}
 norms={'enc.output_norm.weight':'output_norm_enc','output_norm.weight':'output_norm'}
 norms.update({f'blk.{i}.{n}.weight':f'layers.{i}.{n}' for i in range(6) for n in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm')})
 tmp=destination.with_suffix('.partial.gguf');writer=GGUFWriter(tmp,'dflash')
 for key,f in reader.fields.items():
  if not key.startswith('GGUF.') and key!='general.architecture':writer.add_key_value(key,f.contents(),f.types[0],f.types[-1] if len(f.types)>1 else None)
 writer.add_uint32('dflash.w1ax.version',1);writer.add_uint32('dflash.w1ax.activation_bits',8)
 for key,value in {'profile':'ffn18','bit_order':'little','sign_rule':'nonnegative_is_one','scale_rule':'f32_learned_nonnegative'}.items():writer.add_string('dflash.w1ax.'+key,value)
 writer.add_array('dflash.w1ax.tensors',sorted(selected));audit={}
 for t in reader.tensors:
  if t.name in selected:
   base=t.name.removesuffix('.weight');key=base.replace('blk.','layers.')
   weight=dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32)
   signs=params[key+'.latent_sign'].numpy();scales=np.maximum(np.abs(weight).mean(-1)+params[key+'.scale_offset'].numpy(),0)
   assert np.isfinite(signs).all() and np.isfinite(scales).all()
   packed=pack(signs)
   writer.add_tensor(base+'.w1a1_packed',packed,raw_dtype=Type.I32);writer.add_tensor(base+'.w1a1_scale',scales,raw_dtype=Type.F32)
   audit[base]=dict(packed=raw_hash(packed),scales=raw_hash(scales))
  elif t.name in norms:writer.add_tensor(t.name,params[norms[t.name]].numpy(),raw_dtype=Type.F32)
  else:writer.add_tensor(t.name,t.data,raw_dtype=t.tensor_type)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 observed={t.name:t for t in GGUFReader(tmp).tensors}
 for t in reader.tensors:
  if t.name not in selected and t.name not in norms:assert same_tensor(t,observed[t.name]),t.name
 for base,a in audit.items():
  assert raw_hash(observed[base+'.w1a1_packed'].data)==a['packed']
  assert raw_hash(observed[base+'.w1a1_scale'].data)==a['scales']
 tmp.rename(destination)
 result=dict(path=str(destination),sha256=sha256(destination),checkpoint=str(checkpoint),updates=saved['updates'],update_seconds=saved['update_seconds'],protected_unchanged=True,trained_norm_tensors=list(norms))
 destination.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':print(export(sys.argv[1],sys.argv[2]))
