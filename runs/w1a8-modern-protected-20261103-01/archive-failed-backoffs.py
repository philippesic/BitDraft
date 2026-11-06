from pathlib import Path
import json,hashlib,os,shutil,subprocess,uuid,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');base=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf'
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def durable(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
plan=json.loads((R/'data-backoff-plan.json').read_text());state=json.loads((P/'runs/local-w1a8-data-backoff-native-20261103-01/state.json').read_text());assert state['status']=='finished' and state['exit_code']==0
assert not (Path('/proc')/str(state['pid'])).exists()
assert pin(base)['sha256']=='6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048'
parent=C/'w1a8-expanded-ce-resident-new24-3840-01/checkpoint-00003840.pt';paid=pin(parent);assert paid['sha256']=='f4a8d2d31b1016816827d38884f61bbbf881cf029a1a44cc48f94af71e838475'
start=time.monotonic();before=shutil.disk_usage(C).free;receipts=[]
for export in plan['exports']:
 model=Path(export['output']['path']);assert model.parent==C/'w1a8-data-backoff-20261103-01' and not model.is_symlink() and model.stat().st_uid==os.getuid() and pin(model)==export['output']
 folder=model.parent/'compact-exports'/export['id'];folder.mkdir(parents=True,exist_ok=False);ranges=[];cursor=0
 with model.open('rb') as src,(folder/'native-scales.bin').open('xb') as dst,base.open('rb') as ref:
  pos=0
  for patch in sorted(export['patches'],key=lambda x:x['offset']):
   gap=patch['offset']-pos;assert src.read(gap)==ref.read(gap)
   data=src.read(patch['bytes']);assert hashlib.sha256(data).hexdigest()==patch['sha256'];ref.seek(patch['bytes'],1);dst.write(data)
   ranges.append(dict(tensor=patch['name'],model_offset=patch['offset'],payload_offset=cursor,bytes=patch['bytes'],sha256=patch['sha256']));cursor+=patch['bytes'];pos=patch['offset']+patch['bytes']
  assert src.read()==ref.read();dst.flush();os.fsync(dst.fileno())
 manifest=dict(schema='lossless_w1a8_scale_export_archive_v1',base=pin(base),original=export['output'],ranges=ranges,payload_sha256=pin(folder/'native-scales.bin')['sha256'],paid_checkpoint=paid,export_recipe=export,reason='Failed preservation diagnostic, preserve every original export byte through exact scale patch; retain paid training state unchanged')
 durable(folder/'manifest.json',manifest);shutil.copyfile(R/'archive-head-export.py',folder/'restore.py');scratch=Path('/dev/shm')/('w1a8-backoff-restore-'+uuid.uuid4().hex)
 try:
  subprocess.run([str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(scratch)],check=True,stdout=subprocess.DEVNULL)
  restored=pin(scratch);assert restored['bytes']==export['output']['bytes'] and restored['sha256']==export['output']['sha256']
 finally:scratch.unlink(missing_ok=True)
 receipt=dict(status='EXACT_EXPORT_ROUNDTRIP_VERIFIED',original=export['output'],archive_manifest=pin(folder/'manifest.json'),all_unique_bytes_preserved=True,paid_checkpoint_unchanged=paid,restore_command=[str(P/'.venv/bin/python'),str(folder/'restore.py'),'restore',str(folder/'manifest.json'),str(model)])
 durable(model.with_suffix('.gguf.archived.json'),receipt);assert pin(model)==export['output'] and pin(base)==manifest['base'] and pin(parent)==paid;model.unlink();receipts.append(receipt)
report=dict(status='TWO_FAILED_BACKOFF_EXPORTS_LOSSLESSLY_COMPACTED',receipts=receipts,before_free_bytes=before,after_free_bytes=shutil.disk_usage(C).free,wall_seconds=time.monotonic()-start,paid_checkpoints_deleted=0)
durable(R/'failed-backoff-export-archives.json',report);print(json.dumps({k:v for k,v in report.items() if k!='receipts'}))
