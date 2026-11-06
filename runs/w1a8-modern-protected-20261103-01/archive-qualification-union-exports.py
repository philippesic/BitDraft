from pathlib import Path
import json,hashlib,os,shutil,subprocess,uuid,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');base=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf';names=['qualification_learned','reachability_union']
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def durable(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
assert pin(base)['sha256']=='6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048'
assert (R/'head-real-qualification-root-audit.json').exists();started=time.monotonic();before=shutil.disk_usage(C).free;receipts=[]
for name in names:
 root=(C/'w1a8-selected-head-real-qualification-20261103-01' if name=='qualification_learned' else C/'w1a8-modern-protected-20261103-01/reachability-union3840-01');model=root/('learned.gguf' if name=='qualification_learned' else 'model.gguf');result=json.loads((root/'result.json').read_text());expected=result['learned_export'] if name=='qualification_learned' else result['model'];assert pin(model)==expected and not model.is_symlink() and model.stat().st_uid==os.getuid();assert model.stat().st_size==base.stat().st_size
 checkpoint=result['resume']['checkpoints'][-1] if name=='qualification_learned' else result['checkpoint'];assert pin(checkpoint['path'])==checkpoint
 folder=root/'compact-native-export';folder.mkdir(exist_ok=False);ranges=[];cursor=0;offset=0
 with base.open('rb') as a,model.open('rb') as b,(folder/'native-scales.bin').open('xb') as out:
  while True:
   left,right=a.read(4096),b.read(4096);assert len(left)==len(right)
   if not left:break
   if left!=right:
    out.write(right);ranges.append(dict(tensor='generic_changed_byte_chunk',model_offset=offset,payload_offset=cursor,bytes=len(right),sha256=hashlib.sha256(right).hexdigest()));cursor+=len(right)
   offset+=len(left)
  out.flush();os.fsync(out.fileno())
 assert 0<cursor<2*1024**2
 manifest=dict(schema='lossless_w1a8_native_export_chunk_archive_v1',base=pin(base),original=expected,ranges=ranges,payload_sha256=pin(folder/'native-scales.bin')['sha256'],payload_format='4096byte differing chunks, including learned O signs and scales; historical filename retained for verified restorer compatibility',paid_checkpoint=checkpoint,reason='Completed native retention-failing O sign candidates; preserve exact GGUF bytes and all unique paid checkpoints, release derived materializations only')
 durable(folder/'manifest.json',manifest);shutil.copyfile(R/'archive-head-export.py',folder/'restore.py');scratch=Path('/dev/shm')/('w1a8-O-restore-'+uuid.uuid4().hex)
 try:
  subprocess.run([str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(scratch)],check=True,stdout=subprocess.DEVNULL)
  restored=pin(scratch);assert restored['sha256']==expected['sha256'] and restored['bytes']==expected['bytes']
 finally:scratch.unlink(missing_ok=True)
 receipt=dict(status='EXACT_NATIVE_EXPORT_CHUNK_ARCHIVE_ROUNDTRIP_PASS',original=expected,archive_manifest=pin(folder/'manifest.json'),paid_checkpoint_unchanged=checkpoint,payload_bytes=cursor,all_unique_export_bytes_preserved=True,restore_command=[str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(model)])
 durable(model.with_suffix('.gguf.archived.json'),receipt);assert pin(model)==expected and pin(checkpoint['path'])==checkpoint and pin(base)==manifest['base'];model.unlink();receipts.append(receipt)
report=dict(status='QUALIFICATION_AND_UNION_EXPORTS_LOSSLESSLY_COMPACTED',receipts=receipts,before_free_bytes=before,after_free_bytes=shutil.disk_usage(C).free,wall_seconds=time.monotonic()-started,paid_checkpoints_deleted=0);durable(R/'qualification-union-export-chunk-archives.json',report);print(json.dumps({k:v for k,v in report.items() if k!='receipts'}))
