"""Lossless tensor deltas against the immutable paid B0 state, never frozen-buffer copies."""
import hashlib,json,os,pickle,struct,shutil
from pathlib import Path
import numpy as np
import torch
import zstandard as z
from .common import B,pin,save
from scripts.dflash_baseline.checkpoint_io import shuffle,unshuffle,unshuffle_stream

MAGIC=b'DFLASH_TENSOR_DELTA1\n';FOOTER=b'DFTEND1\n'

class Ref:
 def __init__(self,index):self.index=index

def memory_load(path):
 """Mmap decoded torch ZIP from a memfd; no /dev/shm size cap or disk transient."""
 path=Path(path)
 if path.suffix not in ('.bsz','.zst'):return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 fd=os.memfd_create('dflash-checkpoint',flags=0)
 try:
  with path.open('rb') as src,os.fdopen(os.dup(fd),'wb') as out:
   if path.suffix=='.bsz':
    for block in unshuffle_stream(src):out.write(block)
   else:z.ZstdDecompressor().copy_stream(src,out)
  return torch.load(f'/proc/self/fd/{fd}',map_location='cpu',weights_only=False,mmap=True)
 finally:os.close(fd)

def tensors(state):
 result={}
 def walk(value,path):
  if torch.is_tensor(value):result[path]=value
  elif isinstance(value,dict):
   for key,item in value.items():walk(item,path+(key,))
  elif isinstance(value,(tuple,list)):
   for index,item in enumerate(value):walk(item,path+(index,))
 walk(state,());return result

def bytes_of(tensor):return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()

def decoded_tensors(source,metadata,basis):
 refs=tensors(basis);decompressor=z.ZstdDecompressor()
 for record in metadata['tensors']:
  raw=np.empty(record['bytes'],dtype=np.uint8);old=refs.get(tuple(record['path']))
  if record['xor'] and (old is None or old.numel()*old.element_size()!=record['bytes'] or str(old.dtype)!=record['dtype']):raise ValueError('tensor delta basis layout changed')
  old_bytes=bytes_of(old) if record['xor'] else None
  cursor=0;h=hashlib.sha256()
  for chunk in record['chunks']:
   source.seek(chunk['offset']);data=unshuffle(decompressor.decompress(source.read(chunk['coded_bytes']),max_output_size=chunk['bytes']))
   block=np.frombuffer(data,dtype=np.uint8)
   if old_bytes is not None:block=np.bitwise_xor(block,old_bytes[cursor:cursor+len(block)])
   raw[cursor:cursor+len(block)]=block;h.update(block);cursor+=len(block)
  if cursor!=record['bytes'] or h.hexdigest()!=record['sha256']:raise ValueError('lossless tensor reconstruction failed')
  dtype=getattr(torch,record['dtype'].removeprefix('torch.'));value=torch.from_numpy(raw).view(dtype).reshape(record['shape']);yield value

def save_state(path,state,basis_path=None):
 basis_path=Path(basis_path or B/'checkpoint.pt.bsz');basis_pin=pin(basis_path);basis=memory_load(basis_path);old=tensors(basis)
 destination=Path(path).with_suffix('.pt.cpt');partial=destination.with_suffix('.partial')
 if destination.exists() or partial.exists():raise FileExistsError(destination)
 records=[];compressor=z.ZstdCompressor(level=3)
 try:
  with partial.open('xb') as out:
   out.write(MAGIC)
   def encode(value,path):
    if torch.is_tensor(value):
     raw=bytes_of(value);previous=old.get(path)
     compatible=previous is not None and previous.dtype==value.dtype and previous.shape==value.shape
     old_bytes=bytes_of(previous) if compatible else None
     record=dict(path=path,shape=list(value.shape),dtype=str(value.dtype),bytes=raw.size,xor=compatible,chunks=[],sha256=hashlib.sha256(raw).hexdigest())
     for start in range(0,raw.size,8<<20):
      if shutil.disk_usage(destination.parent).free<(512<<20):raise RuntimeError('tensor checkpoint hit512MiB free floor; prior state preserved')
      block=raw[start:start+(8<<20)];delta=np.bitwise_xor(block,old_bytes[start:start+len(block)]) if compatible else block
      coded=compressor.compress(shuffle(delta.tobytes()));record['chunks'].append(dict(offset=out.tell(),coded_bytes=len(coded),bytes=len(block)));out.write(coded)
     records.append(record);return Ref(len(records)-1)
    if isinstance(value,dict):return {k:encode(v,path+(k,)) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(encode(v,path+(i,)) for i,v in enumerate(value))
    if isinstance(value,list):return [encode(v,path+(i,)) for i,v in enumerate(value)]
    return value
   structure=encode(state,());metadata=dict(schema='dflash.lossless_tensor_delta.v1',basis=basis_pin,structure=structure,tensors=records)
   meta_offset=out.tell();encoded=z.ZstdCompressor(level=3).compress(pickle.dumps(metadata,protocol=5));out.write(encoded);out.write(struct.pack('<QQ',meta_offset,len(encoded))+FOOTER);out.flush();os.fsync(out.fileno())
  # Reconstruct every tensor digest before retiring any preceding full state.
  with partial.open('rb') as source:
   for value in decoded_tensors(source,metadata,basis):del value
  if pin(basis_path)!=basis_pin:raise ValueError('immutable basis changed')
  partial.rename(destination)
 except BaseException:
  partial.unlink(missing_ok=True);raise
 receipt=dict(format='dflash.lossless_tensor_delta.v1',archive=str(destination),archive_sha256=pin(destination)['sha256'],archive_bytes=destination.stat().st_size,basis=basis_pin,tensors=len(records),tensor_bytes=sum(r['bytes'] for r in records),lossless_verified=True,raw_materialized=False)
 save(destination.with_suffix('.json'),receipt);return receipt

def load(path):
 path=Path(path)
 if path.suffix!='.cpt':return memory_load(path)
 receipt=json.load(open(path.with_suffix('.json')))
 if pin(path)['sha256']!=receipt['archive_sha256'] or pin(receipt['basis']['path'])!=receipt['basis']:raise ValueError('checkpoint dependency changed')
 with path.open('rb') as source:
  if source.read(len(MAGIC))!=MAGIC:raise ValueError('tensor checkpoint version')
  source.seek(-24,2);tail=source.read(24)
  if tail[-8:]!=FOOTER:raise ValueError('tensor checkpoint footer')
  offset,length=struct.unpack('<QQ',tail[:16]);source.seek(offset);metadata=pickle.loads(z.ZstdDecompressor().decompress(source.read(length)))
  if metadata['basis']!=receipt['basis']:raise ValueError('tensor checkpoint basis mismatch')
  basis=memory_load(metadata['basis']['path']);values=list(decoded_tensors(source,metadata,basis))
 def decode(value):
  if isinstance(value,Ref):return values[value.index]
  if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
  if isinstance(value,tuple):return tuple(decode(v) for v in value)
  if isinstance(value,list):return [decode(v) for v in value]
  return value
 return decode(metadata['structure'])
