"""Lossless GGUF tensor-slice archives: reference frozen bytes once, retain all new bytes.

Whole-file raw dictionaries miss relocated large vocabulary tensors. This format
records authenticated file slices, including every header/padding byte, and
checks the complete reconstructed digest before retiring a new raw export.
"""
import hashlib,json,os,shutil
from pathlib import Path
import zstandard as z
from .common import R,pin,save
from scripts.dflash_baseline import native_archive as old

def paths(path):
 p=Path(path);return p.with_suffix('.gguf.slices.zst'),p.with_suffix('.gguf.slices.json')

def reference(path):
 p=Path(path);archive,receipt=paths(p)
 if not receipt.exists():return pin(p) if p.is_file() else old.reference(p)
 record=json.load(open(receipt));authenticate(record);return record['original']

def authenticate(record):
 if record['schema']!='dflash.authenticated_tensor_slices.v1':raise ValueError('archive schema')
 for ref in [record['archive'],*record['dependencies']]:
  if pin(ref['path'])!=ref:raise ValueError('archive dependency changed '+ref['path'])

def blocks(record):
 with open(record['archive']['path'],'rb') as archive:
  for segment in record['segments']:
   if segment['kind']=='basis':
    with open(segment['path'],'rb') as f:
     f.seek(segment['offset']);remaining=segment['bytes']
     while remaining:
      block=f.read(min(8<<20,remaining))
      if not block:raise ValueError('truncated basis')
      remaining-=len(block);yield block
   else:
    archive.seek(segment['offset']);coded=archive.read(segment['coded_bytes'])
    raw=z.ZstdDecompressor().decompress(coded,max_output_size=segment['bytes'])
    if len(raw)!=segment['bytes']:raise ValueError('truncated archive payload')
    if segment['kind']=='xor':
     import numpy as np
     with open(segment['path'],'rb') as basis:
      basis.seek(segment['basis_offset']);other=basis.read(segment['bytes'])
     if len(other)!=len(raw):raise ValueError('truncated XOR basis')
     raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(other,dtype=np.uint8)).tobytes()
    yield raw

def archive(path,basis=None):
 from scripts.dflash_baseline.model import GGUFReader
 p=Path(path);destination,receipt=paths(p)
 if receipt.exists():
  record=json.load(open(receipt));authenticate(record)
  if p.exists():
   if pin(p)!=record['original']:raise ValueError('changed native file')
   p.unlink()
  return record
 original=pin(p);catalog={};by_tensor={};dependencies={}
 # The three raw bases are permanent NEW campaign recovery dependencies.
 for base in sorted((R/'bases').glob('*.gguf')):
  if base.samefile(p):continue
  for t in GGUFReader(base).tensors:
   h=hashlib.sha256(t.data).hexdigest();catalog.setdefault((h,t.data.nbytes),dict(path=str(base.resolve()),offset=int(t.data_offset),bytes=t.data.nbytes))
   by_tensor.setdefault((t.name,int(t.tensor_type),t.data.nbytes),dict(path=str(base.resolve()),offset=int(t.data_offset),bytes=t.data.nbytes))
 reader=GGUFReader(p);tensors=sorted(reader.tensors,key=lambda t:t.data_offset);segments=[];cursor=0;compressor=z.ZstdCompressor(level=1)
 partial=destination.with_suffix('.partial')
 with p.open('rb') as src,partial.open('xb') as out:
  def payload(offset,size):
   src.seek(offset)
   while size:
    block=src.read(min(size,8<<20));coded=compressor.compress(block)
    segments.append(dict(kind='payload',offset=out.tell(),coded_bytes=len(coded),bytes=len(block)));out.write(coded);size-=len(block)
  for t in tensors:
   if t.data_offset>cursor:payload(cursor,int(t.data_offset)-cursor)
   n=t.data.nbytes;match=catalog.get((hashlib.sha256(t.data).hexdigest(),n))
   if match:
    segments.append(dict(kind='basis',**match))
    if match['path'] not in dependencies:dependencies[match['path']]=pin(match['path'])
   else:
    candidate=by_tensor.get((t.name,int(t.tensor_type),n))
    if candidate is None and t.name in ('output.weight','token_embd.weight'):
     candidate=by_tensor.get(('token_embd.weight' if t.name=='output.weight' else 'output.weight',int(t.tensor_type),n))
    if candidate:
     import numpy as np
     with open(candidate['path'],'rb') as basis:
      basis.seek(candidate['offset']);src.seek(int(t.data_offset));remaining=n;offset=0
      while remaining:
       count=min(remaining,8<<20);block=src.read(count);previous=basis.read(count)
       diff=np.bitwise_xor(np.frombuffer(block,dtype=np.uint8),np.frombuffer(previous,dtype=np.uint8)).tobytes();coded=compressor.compress(diff)
       segments.append(dict(kind='xor',path=candidate['path'],basis_offset=candidate['offset']+offset,offset=out.tell(),coded_bytes=len(coded),bytes=count));out.write(coded);remaining-=count;offset+=count
     if candidate['path'] not in dependencies:dependencies[candidate['path']]=pin(candidate['path'])
    else:payload(int(t.data_offset),n)
   cursor=int(t.data_offset)+n
  if original['bytes']>cursor:payload(cursor,original['bytes']-cursor)
 partial.rename(destination)
 record=dict(schema='dflash.authenticated_tensor_slices.v1',original=original,archive=pin(destination),dependencies=list(dependencies.values()),segments=segments,verified_full_byte_roundtrip=True)
 h=hashlib.sha256();size=0
 for block in blocks(record):h.update(block);size+=len(block)
 if h.hexdigest()!=original['sha256'] or size!=original['bytes']:raise ValueError('complete native archive roundtrip failed')
 save(receipt,record)
 if pin(p)!=original:raise ValueError('native source changed during archival')
 p.unlink();return record

def available(path):
 p=Path(path);a,r=paths(p);return p.exists() or (a.exists() and r.exists()) or old.available(p)

def hydrate(path):
 p=Path(path)
 if p.exists():return False
 archive,receipt=paths(p)
 if not receipt.exists():return old.hydrate(p)
 record=json.load(open(receipt));authenticate(record)
 if shutil.disk_usage(p.parent).free<record['original']['bytes']+(256<<20):raise RuntimeError('native hydration needs model bytes plus256MiB free')
 tmp=p.with_suffix('.gguf.hydrating');h=hashlib.sha256()
 try:
  with tmp.open('xb') as out:
   for block in blocks(record):out.write(block);h.update(block)
  if h.hexdigest()!=record['original']['sha256'] or tmp.stat().st_size!=record['original']['bytes']:raise ValueError('hydrated native bytes differ')
  tmp.rename(p)
 except BaseException:
  tmp.unlink(missing_ok=True);raise
 return True

def release(path,created):
 if not created:return
 p=Path(path);_,receipt=paths(p)
 if not receipt.exists():return old.release(p,created)
 record=json.load(open(receipt));authenticate(record)
 if pin(p)!=record['original']:raise ValueError('hydrated native source changed')
 p.unlink()
