"""Version2 native GGUF export for the qualified affine FFN contract."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from .pipeline import sha
from .prototype import digest
from .checkpoint_io import load as load_checkpoint
from .components import tensor_component,encoded,FORMATS
P=Path('/home/philip/binary-eagle-decoding');ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'third_party/llama.cpp/gguf-py'),str(ROOT/'scripts')]
from gguf import GGUFReader,GGUFWriter,GGMLQuantizationType as Type
from export_recurrent_binary import pack,raw_hash,same_tensor
from gguf.quants import dequantize

def export(checkpoint,destination):
 destination=Path(destination)
 if destination.exists():raise FileExistsError(destination)
 saved=load_checkpoint(checkpoint)
 if saved['version']!=2 or saved['config_hash']!=digest(saved['config']):raise ValueError('checkpoint contract mismatch')
 c=saved['config'];params=saved['parameters'];source=Path(c['source']['path'])
 if sha(source)!=c['source']['sha256']:raise ValueError('source hash changed')
 if any(v['weight']!='W1' for v in c['coverage'].values()):raise ValueError('native v2 requires all18 binary FFN; protected FFN unqualified')
 reader=GGUFReader(source);selected={k+'.weight' for k in c['coverage']}
 norms={'enc.output_norm.weight':'output_norm_enc','output_norm.weight':'output_norm'}
 norms.update({f'blk.{i}.{n}.weight':f'layers.{i}.{n}' for i in range(6) for n in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm')})
 tmp=destination.with_suffix('.partial.gguf');writer=GGUFWriter(tmp,'dflash')
 for key,f in reader.fields.items():
  if not key.startswith('GGUF.') and key!='general.architecture':writer.add_key_value(key,f.contents(),f.types[0],f.types[-1] if len(f.types)>1 else None)
 writer.add_uint32('dflash.w1ax.version',2);writer.add_uint32('dflash.w1ax.activation_bits',8)
 for key,value in {'profile':'ffn18','bit_order':'little','sign_rule':'nonnegative_is_one','scale_rule':'f32_learned_nonnegative','arithmetic':'affine_same_quantized_stream_v1','a1':'signed_pm_bound_threshold0_nonnegative_positive_zero_row_scale0','execution':'packed_signs_int8_codes_dp4a','config_hash':saved['config_hash']}.items():writer.add_string('dflash.w1ax.'+key,value)
 writer.add_array('dflash.w1ax.tensors',sorted(selected));audit={}
 def tensor(name,v,type=Type.F32):
  v=np.asarray(v)
  if not np.isfinite(v).all():raise ValueError('nonfinite export '+name)
  writer.add_tensor(name,v,raw_dtype=type);audit[name]=raw_hash(v)
 for t in reader.tensors:
  if t.name in selected:
   base=t.name.removesuffix('.weight');key=base.replace('blk.','layers.');spec=c['coverage'][base]
   bits=spec['bits'];mode={'dynamic':0,'fixed':1,'learned':2}[c['quantizer']['mode']]
   clip=0. if mode==0 else float(params[key+'.log_clip'].exp()) if mode==2 else c['quantizer']['clip']
   writer.add_uint32(base+'.w1ax.bits',bits);writer.add_uint32(base+'.w1ax.mode',mode);writer.add_float32(base+'.w1ax.clip',clip)
   tensor(base+'.w1a1_packed',pack(params[key+'.latent_sign'].numpy()),Type.I32)
   scale=params[key+'.scale'].clamp_min(0).numpy();tensor(base+'.w1a1_scale',scale)
   tensor(base+'.w1ax_midpoint',params[key+'.midpoint'].numpy())
   if not base.endswith('_up'):
    tensor(base+'.w1ax_gain',params[key+'.log_gain'].exp().numpy())
  elif t.name in norms:tensor(t.name,params[norms[t.name]].numpy())
  else:
   kind=tensor_component(t.name);format=c.get('components',{}).get(kind)
   if format:
    weight=dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32);writer.add_tensor(t.name,encoded(weight,format),raw_dtype=FORMATS[format])
   else:writer.add_tensor(t.name,t.data,raw_dtype=t.tensor_type)
 if 'head' in c.get('components',{}):
  target=GGUFReader(c['evaluation']['target']['path']);t=next(t for t in target.tensors if t.name=='output.weight');weight=dequantize(t.data,t.tensor_type).reshape(tuple(t.shape[::-1])).astype(np.float32)
  writer.add_tensor('output.weight',encoded(weight,c['components']['head']),raw_dtype=FORMATS[c['components']['head']])
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 observed={t.name:t for t in GGUFReader(tmp).tensors}
 for t in reader.tensors:
  if t.name not in selected and t.name not in norms and tensor_component(t.name) not in c.get('components',{}):assert same_tensor(t,observed[t.name]),t.name
 for name,h in audit.items():assert raw_hash(observed[name].data)==h,name
 tmp.rename(destination)
 result=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)),config=c,progress=saved['progress'],lineage=saved['lineage'],protected_unchanged=not bool(c.get('components')),protected_exceptions=c.get('components',{}),private_head='head' in c.get('components',{}),tensor_hashes=audit)
 destination.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('destination',type=Path);a=ap.parse_args();r=export(a.checkpoint,a.destination);print(json.dumps({k:r[k] for k in ('path','sha256','bytes','progress')}))
