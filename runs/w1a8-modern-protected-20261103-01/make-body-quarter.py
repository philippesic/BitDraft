from pathlib import Path
import sys,json,hashlib,shutil,os,time
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01');sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader,GGMLQuantizationType
started=time.monotonic();initial=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf';body=O/'body3840-head-initial01.gguf';destination=O/'body3840-quarter-head-initial01.gguf';assert not destination.exists()
def pin(p):
 with Path(p).open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(Path(p).resolve()),'sha256':d,'bytes':Path(p).stat().st_size}
base={t.name:t for t in GGUFReader(str(initial)).tensors};trained={t.name:t for t in GGUFReader(str(body)).tensors};assert base.keys()==trained.keys()
for k in base:assert base[k].tensor_type==trained[k].tensor_type and base[k].data_offset==trained[k].data_offset and base[k].n_bytes==trained[k].n_bytes
pending=[]
for name,t in base.items():
 if not name.endswith('.w1a1_scale'):continue
 assert t.tensor_type==GGMLQuantizationType.F32
 original=np.array(t.data,dtype='<f4',copy=True);current=np.array(trained[name].data,dtype='<f4',copy=True)
 values=original if name=='output.w1a1_scale' else original*np.float32(0.75)+current*np.float32(0.25)
 assert np.isfinite(values).all() and (values>=0).all();pending.append((int(t.data_offset),values.astype('<f4').tobytes(),name))
assert len(pending)==9
plan={'hypothesis':'Quarter interpolation of8body nativeF32scale payloads towardinitializer; outputhead staysinitializer. No training; selectedbeforethismodelnativeevaluation.','initial':pin(initial),'body_full':pin(body),'coefficient':0.25,'output_head':'initializer','ranges':[{'offset':o,'bytes':len(v),'tensor':n,'sha256':hashlib.sha256(v).hexdigest()} for o,v,n in pending]}
(R/'body-quarter-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
assert shutil.disk_usage('/mnt/c').free>=2*1024**3+64*1024**2+initial.stat().st_size
assert shutil.disk_usage(R).free>=2*1024**3+64*1024**2
temporary=destination.with_name('.'+destination.name+'.tmp');assert not temporary.exists()
with initial.open('rb') as src,temporary.open('xb') as out:
 shutil.copyfileobj(src,out,1024*1024)
 for offset,payload,name in pending:out.seek(offset);out.write(payload)
 out.flush();os.fsync(out.fileno())
with initial.open('rb') as a,temporary.open('rb') as b:
 cursor=0
 for offset,payload,name in sorted(pending):
  remaining=offset-cursor
  while remaining:
   n=min(1024*1024,remaining);assert a.read(n)==b.read(n);remaining-=n
  a.seek(len(payload),1);assert b.read(len(payload))==payload;cursor=offset+len(payload)
 assert a.read()==b.read()
os.replace(temporary,destination);receipt={**plan,'status':'BODY_HALF_EXPORTED_NOT_NATIVE_QUALIFIED','output':pin(destination),'only_native_scale_payloads_changed':True,'wall_seconds':time.monotonic()-started};(R/'body-quarter-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
old=R/'head-rollback-cases01/cases.json';packet=json.loads(old.read_text());packet['models']['candidate']=receipt['output'];packet['model_choice']={'recipe':pin(R/'body-quarter-export.json'),'source_cases':pin(old)};out=R/'body-quarter-cases01';out.mkdir(exist_ok=False);(out/'cases.json').write_text(json.dumps(packet,separators=(',',':'))+'\n');print(json.dumps({'model':receipt['output'],'wall_seconds':receipt['wall_seconds']}))
