from pathlib import Path
import json,os,hashlib,shutil,subprocess,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-actual-training-20261103-01';source=P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda';dest=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-training-cache-20261103-01')
assert not source.is_symlink() and not dest.exists();query=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True);assert not query.strip()
for name in ['local-w1a8-modern-actual-reference-20261103-01','local-w1a8-modern-gpu-training-qualification-20261103-01']:
 s=json.loads((P/'runs'/name/'state.json').read_text());assert s['status']=='finished';assert not Path('/proc',str(s['pid'])).exists()
files=sorted(p for p in source.rglob('*') if p.is_file());assert all(not p.is_symlink() for p in files);total=sum(p.stat().st_size for p in files);assert shutil.disk_usage('/mnt/c').free-total>2*1024**3
before={'WSL':shutil.disk_usage(P).free,'C':shutil.disk_usage('/mnt/c').free};started=time.monotonic();dest.mkdir();records=[]
for p in files:
 d=dest/p.relative_to(source);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
 with d.open('rb+') as f:os.fsync(f.fileno())
 def sha(x):
  with x.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 digest=sha(p);assert sha(d)==digest;records.append({'relative':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':digest})
# Recheck every copied source before replacing any path. Never remove a unique byte.
for rec in records:
 p=source/rec['relative'];d=dest/rec['relative'];assert sha(p)==sha(d)==rec['sha256']
renamed=source.with_name('cuda.verified-old');source.rename(renamed);source.symlink_to(dest,target_is_directory=True)
for rec in records:assert sha(source/rec['relative'])==rec['sha256']
shutil.rmtree(renamed)
after={'WSL':shutil.disk_usage(P).free,'C':shutil.disk_usage('/mnt/c').free};result={'status':'VERIFIED_OWNED_CACHE_RELOCATED_WITH_PATH_PRESERVED','source':str(source),'destination':str(dest),'files':records,'bytes':total,'free_before':before,'free_after':after,'WSL_delta':after['WSL']-before['WSL'],'C_delta':after['C']-before['C'],'wall_seconds':time.monotonic()-started,'future_cache_limit_bytes':1073741824,'protected_models_and_data_untouched':True};(R/'cache-relocation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
