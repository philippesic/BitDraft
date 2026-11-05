"""Lossless archives with authenticated recovery; no frozen-buffer duplication."""
import hashlib,json,tempfile
from pathlib import Path
import torch
import zstandard as zstd

def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load(path):
 path=Path(path)
 if path.suffix!='.zst':return torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
  with path.open('rb') as src:zstd.ZstdDecompressor().copy_stream(src,tmp)
  tmp.flush();return torch.load(tmp.name,map_location='cpu',weights_only=False,mmap=True)
def archive(path):
 path=Path(path);dest=path.with_suffix(path.suffix+'.zst');tmp=dest.with_suffix('.partial')
 if dest.exists():raise FileExistsError(dest)
 original=sha(path)
 with path.open('rb') as src,tmp.open('wb') as out:zstd.ZstdCompressor(level=1,threads=2).copy_stream(src,out)
 h=hashlib.sha256()
 with tmp.open('rb') as src,zstd.ZstdDecompressor().stream_reader(src) as stream:
  for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
 if h.hexdigest()!=original:raise ValueError('archive roundtrip changed checkpoint')
 tmp.rename(dest);receipt=dict(source=str(path),raw_sha256=original,raw_bytes=path.stat().st_size,archive=str(dest),archive_sha256=sha(dest),archive_bytes=dest.stat().st_size,lossless_verified=True)
 dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');path.unlink();return receipt
