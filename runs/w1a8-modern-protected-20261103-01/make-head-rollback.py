from pathlib import Path
import sys,json,hashlib,shutil,os,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01');sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
from gguf import GGUFReader,GGMLQuantizationType
started=time.monotonic();initial=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf';trained=O/'nativefit3840-01/model.gguf';destination=O/'body3840-head-initial01.gguf';assert not destination.exists()
def pin(p):
 with Path(p).open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(Path(p).resolve()),'sha256':d,'bytes':Path(p).stat().st_size}
base={t.name:t for t in GGUFReader(str(initial)).tensors};new={t.name:t for t in GGUFReader(str(trained)).tensors};assert base.keys()==new.keys()
for key in base:assert base[key].tensor_type==new[key].tensor_type and base[key].data_offset==new[key].data_offset and base[key].n_bytes==new[key].n_bytes and list(base[key].shape)==list(new[key].shape)
t=base['output.w1a1_scale'];assert t.tensor_type==GGMLQuantizationType.F32 and t.n_bytes==50000*4;offset=t.data_offset;size=t.n_bytes
with initial.open('rb') as f:f.seek(offset);payload=f.read(size)
plan={'hypothesis':'Retain trained eight body scales, restore all50000outputscales toinitializer; diagnostic modelchoice only','initial':pin(initial),'trained':pin(trained),'paid_checkpoint':pin(O/'nativefit3840-01/checkpoint-00003840.pt'),'changed_native_tensor':'output.w1a1_scale','offset':offset,'bytes':size,'replacement_sha256':hashlib.sha256(payload).hexdigest(),'no_new_training_updates':True}
(R/'head-rollback-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
assert all(shutil.disk_usage(k).free>=2*1024**3+64*1024**2+size for k in [str(R),'/mnt/c']);temporary=destination.with_name('.'+destination.name+'.tmp');assert not temporary.exists()
with trained.open('rb') as src,temporary.open('xb') as out:shutil.copyfileobj(src,out,1024*1024);out.seek(offset);out.write(payload);out.flush();os.fsync(out.fileno())
with trained.open('rb') as a,temporary.open('rb') as b:
 for start,end in [(0,offset),(offset+size,trained.stat().st_size)]:
  a.seek(start);b.seek(start)
  while start<end:
   n=min(1024*1024,end-start);assert a.read(n)==b.read(n);start+=n
with temporary.open('rb') as f:f.seek(offset);assert f.read(size)==payload
os.replace(temporary,destination);receipt={**plan,'status':'HEAD_SCALE_ROLLBACK_EXPORTED_NOT_NATIVE_QUALIFIED','output':pin(destination),'all_outside_head_bytes_equal_trained':True,'head_bytes_equal_initializer':True,'wall_seconds':time.monotonic()-started};(R/'head-rollback-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
old=R/'actual-native-cases01/cases.json';packet=json.loads(old.read_text());basis=packet['models']['candidate'];packet['models']['candidate']=receipt['output'];packet['model_choice']={'recipe':pin(R/'head-rollback-export.json'),'source_cases':pin(old)};packet['selection']['classification_basis_model']=basis;packet['selection']['candidate_readouts_are_for_classification_basis_only']=True
for case in packet['cases']:case['selection_basis_model']=basis;case['selection_basis_gpu_prefix']=case.pop('gpu_candidate_prefix');case['classification_basis']='nativefit3840 before head rollback; not a rollback-model readout'
output=R/'head-rollback-cases01';output.mkdir(exist_ok=False);(output/'cases.json').write_text(json.dumps(packet,separators=(',',':'))+'\n');print(json.dumps({'model':receipt['output'],'wall_seconds':receipt['wall_seconds'],'cases':str(output/'cases.json')}))
