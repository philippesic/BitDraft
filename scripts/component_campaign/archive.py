"""Lossless GGUF tensor-slice archives: reference frozen bytes once, retain all new bytes.

Whole-file raw dictionaries miss relocated large vocabulary tensors. This format
records authenticated file slices, including every header/padding byte, and
checks the complete reconstructed digest before retiring a new raw export.
"""
import hashlib,json,os,shutil,time
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
 dictionaries={}
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
    if segment.get('coded_source'):
     with open(segment['coded_source']['path'],'rb') as source:coded=source.read()
    else:archive.seek(segment['offset']);coded=archive.read(segment['coded_bytes'])
    if len(coded)!=segment['coded_bytes']:raise ValueError('truncated coded archive chunk')
    dictionary=None
    if segment.get('dictionary'):
     d=segment['dictionary'];key=(d['path'],d['offset'],d['bytes'])
     if key not in dictionaries:
      with open(d['path'],'rb') as source:source.seek(d['offset']);data=source.read(d['bytes'])
      if len(data)!=d['bytes']:raise ValueError('truncated header dictionary')
      dictionaries[key]=z.ZstdCompressionDict(data,dict_type=z.DICT_TYPE_RAWCONTENT)
     dictionary=dictionaries[key]
    raw=z.ZstdDecompressor(dict_data=dictionary).decompress(coded,max_output_size=segment['bytes'])
    if len(raw)!=segment['bytes']:raise ValueError('truncated archive payload')
    if segment['kind']=='xor':
     import numpy as np
     with open(segment['path'],'rb') as basis:
      basis.seek(segment['basis_offset']);other=basis.read(segment['bytes'])
     if len(other)!=len(raw):raise ValueError('truncated XOR basis')
     raw=np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(other,dtype=np.uint8)).tobytes()
    yield raw

def blob(coded):
 digest=hashlib.sha256(coded).hexdigest();folder=R/'archive-blobs'/digest[:2];folder.mkdir(parents=True,exist_ok=True);p=folder/(digest+'.zst')
 if not p.exists():
  tmp=folder/(digest+'.partial-'+str(os.getpid())+'-'+str(time.time_ns()))
  created_tmp=False
  try:
   with tmp.open('xb') as out:created_tmp=True;out.write(coded);out.flush();os.fsync(out.fileno())
   try:p.hardlink_to(tmp)
   except FileExistsError:pass
  finally:
   if created_tmp:tmp.unlink(missing_ok=True)
 ref=pin(p)
 if ref['sha256']!=digest or ref['bytes']!=len(coded):raise ValueError('immutable coded chunk differs')
 return ref

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
 # Raw bases remain permanent recovery dependencies. New payloads share immutable
 # compressed chunks; tokenizer/header bytes use the protected shared dictionary.
 for base in sorted((R/'bases').glob('*.gguf')):
  if base.samefile(p):continue
  for t in GGUFReader(base).tensors:
   h=hashlib.sha256(t.data).hexdigest();catalog.setdefault((h,t.data.nbytes),dict(path=str(base.resolve()),offset=int(t.data_offset),bytes=t.data.nbytes))
   by_tensor.setdefault((t.name,int(t.tensor_type),t.data.nbytes),dict(path=str(base.resolve()),offset=int(t.data_offset),bytes=t.data.nbytes))
 reader=GGUFReader(p);tensors=sorted(reader.tensors,key=lambda t:t.data_offset);segments=[];cursor=0;compressor=z.ZstdCompressor(level=1)
 header_path=R/'bases/shared.gguf';header_reader=GGUFReader(header_path);header_bytes=min(int(t.data_offset) for t in header_reader.tensors)
 with header_path.open('rb') as stream:header_dictionary=z.ZstdCompressionDict(stream.read(header_bytes),dict_type=z.DICT_TYPE_RAWCONTENT)
 header_compressor=z.ZstdCompressor(dict_data=header_dictionary,compression_params=z.ZstdCompressionParameters.from_level(1,window_log=24,strategy=z.STRATEGY_GREEDY))
 header_ref=dict(path=str(header_path),offset=0,bytes=header_bytes)
 partial=destination.with_suffix('.partial')
 with p.open('rb') as src,partial.open('xb') as out:
  def payload(offset,size):
   src.seek(offset)
   while size:
    block=src.read(min(size,8<<20));coded=(header_compressor if offset==0 else compressor).compress(block);ref=blob(coded);dependencies[ref['path']]=ref
    item=dict(kind='payload',offset=0,coded_bytes=len(coded),bytes=len(block),coded_source=ref)
    if offset==0:item['dictionary']=header_ref;dependencies[str(header_path)]=pin(header_path)
    segments.append(item);size-=len(block)
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
       ref=blob(coded);dependencies[ref['path']]=ref
       segments.append(dict(kind='xor',path=candidate['path'],basis_offset=candidate['offset']+offset,offset=0,coded_bytes=len(coded),bytes=count,coded_source=ref));remaining-=count;offset+=count
     if candidate['path'] not in dependencies:dependencies[candidate['path']]=pin(candidate['path'])
    else:payload(int(t.data_offset),n)
   cursor=int(t.data_offset)+n
  if original['bytes']>cursor:payload(cursor,original['bytes']-cursor)
 partial.rename(destination)
 record=dict(schema='dflash.authenticated_tensor_slices.v1',original=original,archive=pin(destination),dependencies=list(dependencies.values()),segments=segments,verified_full_byte_roundtrip=True,storage='Content-addressed immutable coded chunks; shared raw header dictionary; exact original GGUF bytes.')
 h=hashlib.sha256();size=0
 for block in blocks(record):h.update(block);size+=len(block)
 if h.hexdigest()!=original['sha256'] or size!=original['bytes']:raise ValueError('complete native archive roundtrip failed')
 save(receipt,record)
 if pin(p)!=original:raise ValueError('native source changed during archival')
 p.unlink();return record

def available(path):
 p=Path(path);a,r=paths(p)
 return p.exists() or (r.exists() and Path(json.load(open(r))['archive']['path']).is_file()) or old.available(p)

def hydrate(path):
 p=Path(path)
 if p.exists():return False
 archive,receipt=paths(p)
 legacy=not receipt.exists()
 if legacy:
  _,receipt=old.paths(p);record=json.load(open(receipt));old.authenticate(record)
  if record['original']['path']!=str(p.resolve()):raise ValueError('legacy native archive destination changed')
 else:record=json.load(open(receipt));authenticate(record)
 if shutil.disk_usage(p.parent).free<record['original']['bytes']+(256<<20):raise RuntimeError('native hydration needs model bytes plus256MiB free')
 tmp=p.with_suffix('.gguf.hydrating');h=hashlib.sha256();created_tmp=False
 try:
  with tmp.open('xb') as out:
   created_tmp=True
   if legacy:
    with open(record['archive']['path'],'rb') as source,z.ZstdDecompressor(dict_data=old.dictionary(record['basis']['path']),max_window_size=2**30).stream_reader(source) as decoded:
     for block in iter(lambda:decoded.read(8<<20),b''):out.write(block);h.update(block)
   else:
    for block in blocks(record):out.write(block);h.update(block)
  if h.hexdigest()!=record['original']['sha256'] or tmp.stat().st_size!=record['original']['bytes']:raise ValueError('hydrated native bytes differ')
  # Publish exclusively: never replace another owner's concurrently hydrated
  # model, and never remove a pre-existing partial after an exclusive-open error.
  p.hardlink_to(tmp);tmp.unlink()
 except BaseException:
  if created_tmp:tmp.unlink(missing_ok=True)
  raise
 return True

def release(path,created):
 if not created:return
 p=Path(path);_,receipt=paths(p)
 if not receipt.exists():return old.release(p,created)
 record=json.load(open(receipt));authenticate(record)
 if pin(p)!=record['original']:raise ValueError('hydrated native source changed')
 p.unlink()
