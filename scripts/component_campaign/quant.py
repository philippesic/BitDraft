"""Use the pinned GGML reference weight quantizer, with bounded row scratch."""
import ctypes as C
import numpy as np
from .common import OPT
from scripts.dflash_baseline.model import GGUFReader,dequantize
from gguf import GGMLQuantizationType as Type, GGML_QUANT_SIZES

def library():
 lib=C.CDLL(str(OPT/'bin/libggml-base.so.0.28.0'))
 lib.ggml_quantize_chunk.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p]
 lib.ggml_quantize_chunk.restype=C.c_size_t
 return lib

def encode(values,format):
 values=np.ascontiguousarray(values,dtype=np.float32)
 qt=Type[format];block,size=GGML_QUANT_SIZES[qt]
 if values.shape[-1]%block:raise ValueError('unaligned weight row')
 result=np.empty((*values.shape[:-1],values.shape[-1]//block*size),dtype=np.uint8)
 got=library().ggml_quantize_chunk(int(qt),values.ctypes.data,result.ctypes.data,0,int(np.prod(values.shape[:-1])),values.shape[-1],None)
 if got!=result.nbytes:raise ValueError('native quantizer byte count mismatch')
 return result

def target_tensor(target):
 reader=GGUFReader(target);lookup={t.name:t for t in reader.tensors}
 if 'output.weight' in lookup:
  out,emb=lookup['output.weight'],lookup['token_embd.weight']
  if out.tensor_type!=emb.tensor_type or not np.array_equal(out.data,emb.data):
   raise ValueError('target head/embedding not tied: explicit separate-source study required')
 return lookup.get('output.weight',lookup['token_embd.weight'])

def matrix_data(format):
 import json
 from .common import R
 path=R/'matrices'/f'{format}.bin'
 if path.exists():
  meta=json.load(open(path.with_suffix('.json')))
  return np.memmap(path,mode='r',dtype=np.uint8,shape=tuple(meta['packed_shape']))
 basis=R/'bases'/(format+'.gguf');reader=GGUFReader(basis)
 for tensor in reader.tensors:
  if tensor.name in ('output.weight','token_embd.weight') and tensor.tensor_type==Type[format]:return tensor.data
 raise ValueError('missing encoded component basis '+format)

def make_matrix(target,format,path):
 from .common import pin,save
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  receipt=__import__('json').load(open(path.with_suffix('.json')))
  if receipt['matrix']!=pin(path):raise ValueError('quantized matrix changed')
  return receipt
 t=target_tensor(target);rows,width=map(int,t.shape[::-1]);qt=Type[format];block,size=GGML_QUANT_SIZES[qt]
 dest=np.memmap(path.with_suffix('.partial'),mode='w+',dtype=np.uint8,shape=(rows,width//block*size))
 for start in range(0,rows,256):
  if t.tensor_type==qt:dest[start:start+256]=t.data[start:start+256] # EXACT packed Q6_K control
  else:dest[start:start+256]=encode(dequantize(t.data[start:start+256],t.tensor_type).reshape(-1,width),format)
 dest.flush();del dest;path.with_suffix('.partial').rename(path)
 receipt=dict(matrix=pin(path),source=pin(target),source_tensor=t.name,source_type=t.tensor_type.name,format=format,shape=[rows,width],packed_shape=[rows,width//block*size],exact_original_codes=qt==t.tensor_type,initializer='pinned target Q6_K; no higher-precision substitution')
 save(path.with_suffix('.json'),receipt);return receipt
