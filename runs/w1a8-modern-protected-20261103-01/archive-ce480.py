from pathlib import Path
import os,sys,json,hashlib,shutil,uuid,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01')
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':digest(p),'bytes':Path(p).stat().st_size}
def restore(manifest,destination):
 m=json.loads(Path(manifest).read_text());base=Path(m['base']['path']);payload=Path(manifest).parent/'native-scales.bin';destination=Path(destination)
 assert not destination.exists() and pin(base)==m['base'] and digest(payload)==m['payload_sha256']
 with base.open('rb') as a,destination.open('xb') as b, payload.open('rb') as patches:
  shutil.copyfileobj(a,b,1024*1024)
  for r in m['ranges']:
   patches.seek(r['payload_offset']);data=patches.read(r['bytes']);assert hashlib.sha256(data).hexdigest()==r['sha256'];b.seek(r['model_offset']);b.write(data)
  b.flush();os.fsync(b.fileno())
 assert destination.stat().st_size==m['original']['bytes'] and digest(destination)==m['original']['sha256']
if len(sys.argv)==4 and sys.argv[1]=='restore':
 restore(sys.argv[2],sys.argv[3]);print('Exact GGUF restored');raise SystemExit
assert len(sys.argv)==1
started=time.monotonic();model=P/'runs/w1a8-modern-actual-training-20261103-01/ce01/model.gguf';base=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf';folder=O/'compact-exports/ce480';assert not folder.exists() and model.is_file() and not model.is_symlink() and model.stat().st_uid==1000
for name in ['local-w1a8-modern-balanced-quiet-20261103-01','local-w1a8-modern-body-half-20261103-01']:
 assert json.loads((R.parent/name/'state.json').read_text())['status']=='finished'
expected=json.loads((model.parent/'result.json').read_text())['model'];assert pin(model)==expected and pin(base)['sha256']=='6dacd68773153e7cb84dd415beb64a21b04ba7c77e2c6386673ad3cfc26dc048'
assert shutil.disk_usage('/dev/shm').free>=model.stat().st_size+4*1024**2
folder.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader,GGMLQuantizationType
def extract():
 b={t.name:t for t in GGUFReader(str(base)).tensors};a={t.name:t for t in GGUFReader(str(model)).tensors};assert a.keys()==b.keys();ranges=[];cursor=0
 with (folder/'native-scales.bin').open('xb') as payload:
  for name,t in sorted(a.items(),key=lambda x:x[1].data_offset):
   assert t.data_offset==b[name].data_offset and t.n_bytes==b[name].n_bytes and t.tensor_type==b[name].tensor_type and list(t.shape)==list(b[name].shape)
   if not name.endswith('.w1a1_scale'):continue
   assert t.tensor_type==GGMLQuantizationType.F32;data=t.data.tobytes();ranges.append({'tensor':name,'model_offset':int(t.data_offset),'payload_offset':cursor,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});payload.write(data);cursor+=len(data)
  payload.flush();os.fsync(payload.fileno())
 assert len(ranges)==9
 return ranges
ranges=extract()
with base.open('rb') as a,model.open('rb') as b:
 cursor=0
 for r in ranges:
  remaining=r['model_offset']-cursor
  while remaining:
   n=min(1024*1024,remaining);assert a.read(n)==b.read(n);remaining-=n
  a.seek(r['bytes'],1);b.seek(r['bytes'],1);cursor=r['model_offset']+r['bytes']
 assert a.read()==b.read()
manifest={'schema':'lossless_w1a8_scale_export_archive_v1','base':pin(base),'original':expected,'ranges':ranges,'payload_sha256':digest(folder/'native-scales.bin'),'paid_checkpoint':pin(model.parent/'checkpoint-00000480.pt'),'reason':'Superseded480update export; keep full checkpoint/optimizer and lossless native export archive. No training rejection or shortening.'}
metadata=folder/'manifest.json';metadata.write_text(json.dumps(manifest,indent=2)+'\n');shutil.copyfile(__file__,folder/'restore.py')
scratch=Path('/dev/shm')/('w1a8-export-roundtrip-'+uuid.uuid4().hex)
try:restore(metadata,scratch);assert scratch.stat().st_size==model.stat().st_size and digest(scratch)==digest(model)
finally:scratch.unlink(missing_ok=True)
assert pin(model)==expected and pin(base)==manifest['base'];before={k:shutil.disk_usage(k).free for k in ['.','/mnt/c']};model.unlink();after={k:shutil.disk_usage(k).free for k in before}
receipt={'status':'LOSSLESS_NATIVE_GGUF_ARCHIVED_ROUNDTRIP_VERIFIED','archive_manifest':pin(metadata),'original_model':expected,'paid_checkpoint_unchanged':manifest['paid_checkpoint'],'all_unique_model_bytes_preserved':True,'GGUF_materialization_retired':str(model),'restore_command':str(P/'.venv/bin/python')+' '+str(folder/'restore.py')+' restore '+str(metadata)+' '+str(model),'before_free_bytes':before,'after_free_bytes':after,'observed_free_byte_deltas':{k:after[k]-before[k] for k in before},'wall_seconds':time.monotonic()-started}
(model.parent/'model.gguf.archived.json').write_text(json.dumps(receipt,indent=2)+'\n');(R/'ce480-export-archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
