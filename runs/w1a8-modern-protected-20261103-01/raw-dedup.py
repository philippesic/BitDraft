from pathlib import Path
import json,os,stat,hashlib,shutil,time,uuid
R=Path('/home/philip/binary-eagle-decoding/runs/w1a8-modern-protected-20261103-01'); I=R/'raw-duplicate-inventory.json';d=json.loads(I.read_text());roots=[Path(x['path']) for x in d['roots']]
for run in ['local-w1a8-modern-balanced-quiet-20261103-01','local-w1a8-modern-nativefit3840-20261103-01','local-w1a8-modern-balanced3840-control-20261103-01']:
 state=json.loads((R.parent/run/'state.json').read_text());assert state['status']=='finished' and state['exit_code']==0
before={p:shutil.disk_usage(p).free for p in [str(R),'/mnt/c']};started=time.monotonic();changed=0;groups=0;dirs=set();retained=0
(R/'raw-dedup-plan.json').write_text(json.dumps({'inventory':str(I),'inventory_sha256':hashlib.sha256(I.read_bytes()).hexdigest(),'scope':[str(x) for x in roots],'operation':'Atomic wholefile byteverified hardlink consolidation; retain all paths and all unique bytes','readers':'owned GPU readers terminal0; source worker inventory completed; no new nativejob launched','before_free_bytes':before},indent=2)+'\n')
def check(meta,modified_links=0):
 p=Path(meta['path']);s=p.lstat();assert p.suffix=='.f32' and any(p.is_relative_to(r) for r in roots)
 assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size>=4096 and s.st_uid==1000
 assert s.st_dev==meta['st_dev'] and s.st_ino==meta['st_ino'] and s.st_mtime_ns==meta['mtime_ns'] and s.st_nlink==meta['st_nlink']+modified_links
 return p
for g in d['duplicate_groups']:
 files=g['files'];assert g['same_device'] and g['all_owned_by_philip'];repmeta=files[0];rep=check(repmeta);assert rep.stat().st_nlink==1
 with rep.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==g['sha256']
 linked=0
 for meta in files[1:]:
  target=check(meta);check(repmeta,linked);assert target.stat().st_nlink==1 and target.stat().st_size==g['bytes_per_file']==rep.stat().st_size
  digest=hashlib.sha256()
  with rep.open('rb') as a,target.open('rb') as b:
   while True:
    x=a.read(1024*1024);y=b.read(1024*1024);assert x==y
    if not x:break
    digest.update(y)
  assert digest.hexdigest()==g['sha256'];check(meta);check(repmeta,linked)
  temporary=target.with_name('.w1a8-dedup-'+uuid.uuid4().hex)
  try:
   os.link(rep,temporary,follow_symlinks=False);assert temporary.stat().st_ino==rep.stat().st_ino
   os.replace(temporary,target)
  finally:
   if temporary.exists():temporary.unlink()
  assert target.stat().st_ino==rep.stat().st_ino;linked+=1;changed+=1;dirs.add(target.parent)
 for meta in files:
  p=Path(meta['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==g['bytes_per_file']
  with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==g['sha256']
  retained+=1
 groups+=1
for directory in dirs:
 fd=os.open(directory,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
after={p:shutil.disk_usage(p).free for p in before};receipt={'status':'ALL_DUPLICATE_PATHS_BYTES_REVERIFIED','groups':groups,'redundant_inodes_consolidated':changed,'retained_paths_rehashed':retained,'unique_content_removed':False,'all_paths_retained':True,'old_metadata_preserved_in_inventory':True,'before_free_bytes':before,'after_free_bytes':after,'observed_free_byte_deltas':{p:after[p]-before[p] for p in before},'wall_seconds':time.monotonic()-started,'Windows_note':'WSL available bytes are separate from Windows backing-volume capacity; no VHDX compaction'}
(R/'raw-dedup-result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
