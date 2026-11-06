from pathlib import Path
import json,gzip,hashlib,os,shutil,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');jobs=[('local-w1a8-head-matched5760-20261103-01',C/'w1a8-head-pair-20261103-01/head5760/updates.jsonl'),('local-w1a8-head-false4320-20261103-01',C/'w1a8-selected-head-false4320-01/updates.jsonl')];reports=[];start=time.monotonic();before=shutil.disk_usage(C).free
for run,source in jobs:
 state=json.loads((P/'runs'/run/'state.json').read_text());assert state['status']=='finished' and not (Path('/proc')/str(state['pid'])).exists();assert source.is_file() and not source.is_symlink() and source.stat().st_uid==os.getuid();identity=source.stat();dest=source.with_suffix('.jsonl.gz');partial=dest.with_suffix('.gz.partial');assert not dest.exists() and not partial.exists();sha=hashlib.sha256();size=0
 with source.open('rb') as src,partial.open('xb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as encoded:
   while data:=src.read(1024*1024):
    sha.update(data);size+=len(data);encoded.write(data);assert dst.tell()<=8*1024**2,'bounded recovery archive exceeded8MiB'
  dst.flush();os.fsync(dst.fileno())
 with gzip.open(partial,'rb') as src:
  check=hashlib.sha256();count=0
  while data:=src.read(1024*1024):check.update(data);count+=len(data)
 assert count==size==identity.st_size and check.hexdigest()==sha.hexdigest() and source.stat().st_size==identity.st_size and source.stat().st_mtime_ns==identity.st_mtime_ns
 os.replace(partial,dest);archive_sha=hashlib.sha256(dest.read_bytes()).hexdigest();receipt=dict(status='COMPLETED_UPDATE_LOG_GZIP_ROUNDTRIP_EXACT',source=dict(path=str(source),bytes=size,sha256=sha.hexdigest()),archive=dict(path=str(dest),bytes=dest.stat().st_size,sha256=archive_sha),owner_terminal=state['exit_code'],paid_checkpoints_deleted=0,restore='gzip -dk '+str(dest),bounded_temporary_limit_bytes=8*1024**2)
 with source.with_suffix('.jsonl.archived.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 source.unlink();reports.append(receipt)
report=dict(status='COMPLETED_HEAD_LOGS_LOSSLESSLY_COMPRESSED',reports=reports,before_Cfree=before,after_Cfree=shutil.disk_usage(C).free,wall_seconds=time.monotonic()-start,training_guards_unchanged=True);(R/'completed-head-log-archives.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='reports'}))
