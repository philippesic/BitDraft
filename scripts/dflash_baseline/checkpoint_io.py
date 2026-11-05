"""Lossless archives with authenticated recovery; no frozen-buffer duplication."""
import hashlib,json,tempfile,struct
import numpy as np
from pathlib import Path
import torch
import zstandard as zstd

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):
 path=Path(path)
 if path.suffix not in ('.zst','.bsz'):return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
  with path.open('rb') as src:
   if path.suffix=='.bsz':
    for block in unshuffle_stream(src):tmp.write(block)
   else:zstd.ZstdDecompressor().copy_stream(src,tmp)
  tmp.flush();return torch.load(tmp.name,map_location='cpu',weights_only=False,mmap=True)
MAGIC=b'DFLASHBSZ1\n'
def shuffle(block):
 n=len(block)//4*4
 return np.frombuffer(block[:n],dtype=np.uint8).reshape(-1,4).T.copy().tobytes()+block[n:]
def unshuffle(block):
 n=len(block)//4*4
 return np.frombuffer(block[:n],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+block[n:]
def unshuffle_stream(src):
 if src.read(len(MAGIC))!=MAGIC:raise ValueError('bad byte-shuffled archive version')
 while header:=src.read(16):
  if len(header)!=16:raise ValueError('truncated archive header')
  raw_size,encoded_size=struct.unpack('<QQ',header);data=src.read(encoded_size)
  if len(data)!=encoded_size or raw_size>8<<20:raise ValueError('invalid archive chunk')
  block=zstd.ZstdDecompressor().decompress(data,max_output_size=raw_size)
  if len(block)!=raw_size:raise ValueError('bad archive raw size')
  yield unshuffle(block)
def archive(path):
 path=Path(path);dest=path.with_suffix(path.suffix+'.bsz');tmp=dest.with_suffix('.partial')
 if dest.exists():raise FileExistsError(dest)
 original=sha(path);compressor=zstd.ZstdCompressor(level=1)
 with path.open('rb') as src,tmp.open('wb') as out:
  out.write(MAGIC)
  for block in iter(lambda:src.read(8<<20),b''):
   coded=compressor.compress(shuffle(block));out.write(struct.pack('<QQ',len(block),len(coded)));out.write(coded)
 h=hashlib.sha256()
 with tmp.open('rb') as src:
  for block in unshuffle_stream(src):h.update(block)
 if h.hexdigest()!=original:raise ValueError('archive roundtrip changed checkpoint')
 tmp.rename(dest);receipt=dict(format='dflash.byte_shuffle_zstd.v1',source=str(path),raw_sha256=original,raw_bytes=path.stat().st_size,archive=str(dest),archive_sha256=sha(dest),archive_bytes=dest.stat().st_size,lossless_verified=True)
 dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');path.unlink();return receipt

class _DigestingWriter:
 def __init__(self,writer):self.writer=writer;self.digest=hashlib.sha256();self.count=0
 def write(self,data):
  size=self.writer.write(data)
  if size!=len(data):raise OSError('short compressed checkpoint write')
  self.digest.update(data);self.count+=size;return size
 def flush(self):self.writer.flush()

def save_streamed(path,state):
 """Atomically save exact torch state without materializing a second raw file."""
 import os
 path=Path(path);dest=path.with_suffix(path.suffix+'.zst');tmp=dest.with_suffix('.partial')
 if dest.exists() or tmp.exists():raise FileExistsError(dest)
 with tmp.open('xb') as out:
  with zstd.ZstdCompressor(level=1).stream_writer(out,closefd=False) as stream:
   recorded=_DigestingWriter(stream);torch.save(state,recorded)
  out.flush();os.fsync(out.fileno())
 observed=hashlib.sha256();count=0
 with tmp.open('rb') as source,zstd.ZstdDecompressor().stream_reader(source) as decoded:
  for block in iter(lambda:decoded.read(8<<20),b''):observed.update(block);count+=len(block)
 if observed.hexdigest()!=recorded.digest.hexdigest() or count!=recorded.count:raise ValueError('streamed checkpoint roundtrip changed bytes')
 tmp.rename(dest)
 receipt=dict(format='dflash.zstd.stream.v1',source=str(path),raw_sha256=observed.hexdigest(),raw_bytes=count,archive=str(dest),archive_sha256=sha(dest),archive_bytes=dest.stat().st_size,lossless_verified=True,raw_materialized=False)
 dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
