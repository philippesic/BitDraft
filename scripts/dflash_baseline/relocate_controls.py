"""Hash-verified relocation of two terminal owned recovery archives; canonical aliases remain."""
import hashlib,json,os,shutil,sys
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';dest=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/precision-campaign-20261104-01-control-archives')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert json.load(open(P/'runs/precision-foundation-20261104-01/state.json'))['exit_code']==0
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir();before=dict(wsl=shutil.disk_usage(P).free,windows=shutil.disk_usage('/mnt/c').free);receipts=[]
 if before['windows']<20*2**30:raise RuntimeError('insufficient Windows headroom for a safe relocation')
 for name in ('vanilla-matched','no-midpoint'):
  src=r/name/'checkpoint.pt.zst';target=dest/(name+'.pt.zst');tmp=target.with_suffix('.partial');old=src.stat();expected=json.load(open(r/name/'checkpoint.pt.json'))['archive_sha256']
  if src.is_symlink() or sha(src)!=expected:raise ValueError('source archive ownership/hash changed')
  with src.open('rb') as source,tmp.open('xb') as out:
   shutil.copyfileobj(source,out,16<<20);out.flush();os.fsync(out.fileno())
  if sha(tmp)!=expected:raise ValueError('relocation copy changed bytes')
  tmp.rename(target);current=src.stat()
  if (old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns):raise ValueError('source changed during relocation')
  alias=src.with_suffix('.pending-alias');alias.symlink_to(target);os.replace(alias,src)
  if sha(src)!=expected:raise ValueError('canonical recovery alias failed')
  receipts.append(dict(canonical_path=str(src),retained_archive=str(target),sha256=expected,bytes=current.st_size,lossless_copy_verified=True,canonical_alias_verified=True))
  print('Relocated terminal control',name,flush=True)
 after=dict(wsl=shutil.disk_usage(P).free,windows=shutil.disk_usage('/mnt/c').free)
 (r/'control-archive-relocation.json').write_text(json.dumps(dict(before_free_bytes=before,after_free_bytes=after,archives=receipts,scope='Owned terminal controls only; all checkpoint bytes retained and original paths preserved. No VHDX compaction or Windows reclaim claim. Concurrent task writes may affect observed deltas.'),indent=2)+'\n')
if __name__=='__main__':main()
