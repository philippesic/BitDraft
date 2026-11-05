"""Lossless GGUF archival against a pinned frozen GGUF; hydrate only while serving.

The original complete byte stream is recovered and authenticated. This changes
storage only, never tensors, runtime arithmetic, timed requests or checkpoints.
"""
import hashlib,json,os,shutil
from pathlib import Path
import zstandard as z
def sha(path):
 with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(path):
 p=Path(path);return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
def paths(path):
 p=Path(path);return p.with_suffix('.gguf.native.zst'),p.with_suffix('.gguf.native.json')
def dictionary(basis):return z.ZstdCompressionDict(Path(basis).read_bytes(),dict_type=z.DICT_TYPE_RAWCONTENT)
def authenticate(record):
 for ref in (record['archive'],record['basis']):
  if pin(ref['path'])!=ref:raise ValueError('native archive dependency changed '+ref['path'])
 if record['schema']!='dflash.lossless_gguf_dictionary.v1':raise ValueError('native archive version')
def reference(path):
 p=Path(path);_,receipt=paths(p);record=json.load(open(receipt));authenticate(record)
 if record['original']['path']!=str(p.resolve()):raise ValueError('native archive path changed')
 return record['original']
def available(path):
 p=Path(path);archive,receipt=paths(p);return p.is_file() or (archive.is_file() and receipt.is_file())
def hydrate(path):
 p=Path(path)
 if p.exists():return False
 archive,receipt=paths(p);record=json.load(open(receipt));authenticate(record)
 if record['original']['path']!=str(p.resolve()):raise ValueError('native archive destination changed')
 partial=p.with_suffix('.gguf.hydrating')
 if shutil.disk_usage(p.parent).free<record['original']['bytes']+2**30:raise RuntimeError('insufficient GGUF hydration headroom')
 try:
  with archive.open('rb') as inp,partial.open('xb') as out:z.ZstdDecompressor(dict_data=dictionary(record['basis']['path']),max_window_size=2**30).copy_stream(inp,out)
  observed=pin(partial)
  if any(observed[k]!=record['original'][k] for k in ('bytes','sha256')):raise ValueError('GGUF archive recovery changed bytes')
  partial.rename(p)
 except BaseException:
  if partial.exists():partial.unlink()
  raise
 return True
def release(path,created):
 if not created:return
 p=Path(path);_,receipt=paths(p);record=json.load(open(receipt));authenticate(record)
 if pin(p)!=record['original']:raise ValueError('hydrated GGUF changed before release')
 p.unlink()
def archive(path,basis,prepared=None,basis_pin=None):
 p=Path(path);basis=Path(basis)
 if p.resolve()==basis.resolve():raise ValueError('cannot archive the recovery basis')
 dest,receipt=paths(p)
 if dest.exists() and receipt.exists():
  record=json.load(open(receipt));authenticate(record)
  if record['basis']!=pin(basis) or (p.exists() and pin(p)!=record['original']):raise ValueError('existing native archival copy changed')
  if p.exists():p.unlink()
  return record
 if dest.exists() or receipt.exists():raise FileExistsError('incomplete native archive publication')
 original=pin(p);basis_pin=basis_pin or pin(basis);d=prepared or dictionary(basis)
 params=z.ZstdCompressionParameters.from_level(3,window_log=30,hash_log=27,chain_log=27,strategy=z.STRATEGY_GREEDY)
 compressor=z.ZstdCompressor(dict_data=d,compression_params=params);partial=dest.with_suffix('.partial')
 with p.open('rb') as inp,partial.open('xb') as out:compressor.copy_stream(inp,out)
 h=hashlib.sha256();size=0
 with partial.open('rb') as inp,z.ZstdDecompressor(dict_data=d,max_window_size=2**30).stream_reader(inp) as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
 if h.hexdigest()!=original['sha256'] or size!=original['bytes']:raise ValueError('native archive validation failed')
 if pin(p)!=original or pin(basis)!=basis_pin:raise ValueError('native source/basis changed during archival')
 partial.rename(dest);record=dict(schema='dflash.lossless_gguf_dictionary.v1',original=original,archive=pin(dest),basis=basis_pin,verified_full_byte_roundtrip=True,semantics='Storage container only; recover exact GGUF before native execution; timing excludes hydration; frozen source and original sidecar retained.')
 temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(receipt)
 if reference(p)!=original:raise ValueError('archive publication failed')
 p.unlink();return record
