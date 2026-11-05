"""Stream byte-shuffled lossless torch checkpoints without a raw disk temporary."""
import hashlib,json,os,shutil,struct
from pathlib import Path
import zstandard as z
from .common import pin,save
from scripts.dflash_baseline.checkpoint_io import MAGIC,shuffle,unshuffle_stream

class Writer:
 def __init__(self,out,path):self.out=out;self.path=path;self.pending=bytearray();self.digest=hashlib.sha256();self.count=0;self.compressor=z.ZstdCompressor(level=3)
 def write(self,block):
  self.digest.update(block);self.count+=len(block);self.pending.extend(block)
  while len(self.pending)>=8<<20:
   raw=bytes(self.pending[:8<<20]);del self.pending[:8<<20];self.chunk(raw)
  return len(block)
 def chunk(self,raw):
  if shutil.disk_usage(self.path.parent).free<(512<<20):raise RuntimeError('checkpoint hit512MiB physical free-space floor; prior state preserved')
  coded=self.compressor.compress(shuffle(raw));self.out.write(struct.pack('<QQ',len(raw),len(coded)));self.out.write(coded)
 def flush(self):self.out.flush()
 def finish(self):
  if self.pending:self.chunk(bytes(self.pending));self.pending.clear()
  self.flush()

def validate(path,expected_hash,expected_bytes):
 h=hashlib.sha256();size=0
 with Path(path).open('rb') as src:
  for block in unshuffle_stream(src):h.update(block);size+=len(block)
 if h.hexdigest()!=expected_hash or size!=expected_bytes:raise ValueError('lossless streamed checkpoint validation failed')

def save_state(path,state):
 import torch
 path=Path(path).with_suffix('.pt.bsz');partial=path.with_suffix('.partial')
 if path.exists() or partial.exists():raise FileExistsError(path)
 try:
  with partial.open('xb') as out:
   out.write(MAGIC);writer=Writer(out,partial);torch.save(state,writer);writer.finish();os.fsync(out.fileno())
  validate(partial,writer.digest.hexdigest(),writer.count);partial.rename(path)
 except BaseException:
  partial.unlink(missing_ok=True);raise
 receipt=dict(format='dflash.byte_shuffle_zstd.stream.v1',archive=str(path),archive_sha256=pin(path)['sha256'],archive_bytes=path.stat().st_size,raw_sha256=writer.digest.hexdigest(),raw_bytes=writer.count,lossless_verified=True,raw_materialized=False)
 save(path.with_suffix('.json'),receipt);return receipt

def convert_zst(source):
 source=Path(source);receipt=json.load(open(source.with_suffix('.json')))
 dest=source.with_suffix('.bsz');partial=dest.with_suffix('.partial')
 try:
  with source.open('rb') as src,partial.open('xb') as out,z.ZstdDecompressor().stream_reader(src) as decoded:
   out.write(MAGIC);writer=Writer(out,partial)
   for block in iter(lambda:decoded.read(8<<20),b''):writer.write(block)
   writer.finish();os.fsync(out.fileno())
  validate(partial,receipt['raw_sha256'],receipt['raw_bytes']);partial.rename(dest)
 except BaseException:
  partial.unlink(missing_ok=True);raise
 new=dict(receipt,format='dflash.byte_shuffle_zstd.stream.v1',archive=str(dest),archive_sha256=pin(dest)['sha256'],archive_bytes=dest.stat().st_size)
 save(dest.with_suffix('.json'),new)
 if pin(source)['sha256']!=receipt['archive_sha256']:raise ValueError('original checkpoint changed')
 source.unlink();return new

if __name__=='__main__':
 from .common import R
 receipt=json.load(open(R/'arms/H0-pilot/recovery.json'))
 converted=convert_zst(receipt['archive']);save(R/'arms/H0-pilot/recovery.json',converted)
 save(Path(converted['archive']).parent/'rolling.json',dict(arm='H0-pilot',update=32,receipt=converted))
 save(R/'checkpoint-storage-qualification.json',dict(status='LOSSLESS_STREAMED_SHUFFLE_CONVERSION_PASS',original=receipt,replacement=converted,physical_free_bytes=shutil.disk_usage('/mnt/c').free))
 print('CHECKPOINT_STORAGE_PASS',converted['archive_bytes'],flush=True)
