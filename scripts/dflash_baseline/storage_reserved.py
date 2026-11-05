"""Retain paid pilot states on the existing artifact volume before reserved trials."""
import json,os,shutil,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
def main():
 if json.load(open(P/'runs/precision-mutations-20261105-02/state.json')).get('exit_code')!=0:raise ValueError('wait for serial mutation job to finish')
 subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline/cache_zero.py'),'--after-all-adaptation'],cwd=ROOT,check=True)
 destination=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/precision-campaign-20261104-01-pilot-archives')
 destination.mkdir(exist_ok=True);before={n:shutil.disk_usage(p).free for n,p in [('wsl',P),('windows','/mnt/c')]};records=[]
 candidates=[R/'foundation-pilot'/name for name in ('checkpoint-32.pt.bsz','checkpoint-64.pt.bsz','checkpoint.pt.bsz')]
 needed=sum(p.stat().st_size for p in candidates if not p.is_symlink())
 if before['windows']-needed<8*2**30:raise RuntimeError('insufficient artifact-volume headroom')
 for source in candidates:
  expected=json.load(open(source.with_suffix('.json')))['archive_sha256'];target=destination/source.name
  if source.is_symlink():
   if source.resolve()!=target or pin(source)['sha256']!=expected:raise ValueError('existing recovery alias mismatch')
   records.append(dict(canonical=str(source),retained=pin(target),already_relocated=True));continue
  original=source.stat()
  if pin(source)['sha256']!=expected:raise ValueError('owned pilot archive changed')
  temporary=target.with_suffix('.partial')
  with source.open('rb') as inp,temporary.open('xb') as out:
   shutil.copyfileobj(inp,out,16<<20);out.flush();os.fsync(out.fileno())
  if pin(temporary)['sha256']!=expected:raise ValueError('archive copy mismatch')
  if target.exists():raise FileExistsError(target)
  temporary.rename(target);current=source.stat()
  if (original.st_dev,original.st_ino,original.st_size,original.st_mtime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns):raise ValueError('archive changed during copy')
  alias=source.with_suffix('.alias');alias.symlink_to(target);os.replace(alias,source)
  if pin(source)['sha256']!=expected:raise ValueError('recovery alias failed')
  records.append(dict(canonical=str(source),retained=pin(target),all_paid_state_preserved=True));print('Retained pilot archive relocated',source.name,flush=True)
 after={n:shutil.disk_usage(p).free for n,p in [('wsl',P),('windows','/mnt/c')]}
 save(R/'reserved-storage-relocation.json',dict(before_free_bytes=before,after_free_bytes=after,records=records,scope='Owned terminal pilot archives only; hash-verified full recoverability and canonical aliases. No VHDX compaction; no Windows reclamation claimed.'))
if __name__=='__main__':main()
