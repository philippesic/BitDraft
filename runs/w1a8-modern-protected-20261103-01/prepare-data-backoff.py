from pathlib import Path
import sys,json,hashlib,os,shutil,time,math
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');O=C/'w1a8-modern-protected-20261103-01';sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'));from gguf import GGUFReader,GGMLQuantizationType
initial=P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf';trained=C/'w1a8-expanded-ce-resident-new24-3840-01/model.gguf';oldfull=C/'w1a8-expanded-ce-control3840-01/model.gguf';oldquarter=O/'body3840-quarter-head-initial01.gguf';base={t.name:t for t in GGUFReader(str(initial)).tensors};learned={t.name:t for t in GGUFReader(str(trained)).tensors};old={t.name:t for t in GGUFReader(str(oldfull)).tensors};oldq={t.name:t for t in GGUFReader(str(oldquarter)).tensors}
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
assert base.keys()==learned.keys()==old.keys()==oldq.keys()
for name,t in base.items():
 assert all(x[name].tensor_type==t.tensor_type and x[name].n_bytes==t.n_bytes and x[name].data_offset==t.data_offset for x in [learned,old,oldq])
 if name.endswith('.w1a1_scale'):
  a=np.array(t.data,dtype='<f4',copy=True);z=np.array(old[name].data,dtype='<f4',copy=True);expected=a if name=='output.w1a1_scale' else a*np.float32(.75)+z*np.float32(.25);assert expected.tobytes()==oldq[name].data.tobytes(),'matchedoldquarterrecipechanged'
 else:assert t.data.tobytes()==learned[name].data.tobytes()==old[name].data.tobytes()==oldq[name].data.tobytes()
assert shutil.disk_usage('/mnt/c').free>=2*1024**3+64*1024**2+2*initial.stat().st_size+16*1024**2
exports=[]
for alpha,label in [(.25,'quarter'),(.125,'eighth')]:
 dst=C/f'w1a8-data-backoff-20261103-01/new-{label}-head-initial.gguf';dst.parent.mkdir(exist_ok=True);assert not dst.exists();patches=[]
 for name,t in base.items():
  if not name.endswith('.w1a1_scale'):continue
  assert t.tensor_type==GGMLQuantizationType.F32;a=np.array(t.data,dtype='<f4',copy=True);z=np.array(learned[name].data,dtype='<f4',copy=True);values=a if name=='output.w1a1_scale' else a*np.float32(1-alpha)+z*np.float32(alpha);assert np.isfinite(values).all() and (values>=0).all();patches.append((int(t.data_offset),values.astype('<f4').tobytes(),name))
 assert len(patches)==9;tmp=dst.with_name('.'+dst.name+'.tmp')
 with initial.open('rb') as src,tmp.open('xb') as out:
  shutil.copyfileobj(src,out,1024**2)
  for offset,payload,name in patches:out.seek(offset);out.write(payload)
  out.flush();os.fsync(out.fileno())
 with initial.open('rb') as a,tmp.open('rb') as b:
  cursor=0
  for offset,payload,name in sorted(patches):
   remaining=offset-cursor
   while remaining:
    n=min(1024**2,remaining);assert a.read(n)==b.read(n);remaining-=n
   a.seek(len(payload),1);assert b.read(len(payload))==payload;cursor=offset+len(payload)
  assert a.read()==b.read()
 os.replace(tmp,dst);exports.append({'id':'new_'+label,'output':pin(dst),'alpha':alpha,'head':'initializer','source':pin(trained),'only9declared_scale_ranges':True,'patches':[{'offset':x,'bytes':len(y),'name':n,'sha256':hashlib.sha256(y).hexdigest()} for x,y,n in patches]})
# Parent-only margins freeze a diagnostic subset before any backoff outcomes.
N=C/'w1a8-modern-data-native-comparison-20261103-01/control';records={};sources=[]
for f in sorted(N.glob('batch-*/parent/replay.json')):
 d=json.loads(f.read_text());assert d['native_successes_outside_parent_calibration']==0;sources.append(pin(f))
 for wid,v in d['windows'].items():
  margins=[s['gold_margin'] for s in v['steps'] if s['native_success']];assert margins and all(math.isfinite(x) and x>=0 for x in margins);records[wid]=(min(margins),v['case'])
assert len(records)==883;bygroup={}
for wid,(margin,case) in records.items():bygroup.setdefault(case['group_id'],[]).append((margin,wid))
selected={min(v)[1] for v in bygroup.values()};assert len(selected)<=96
for margin,wid in sorted((v[0],k) for k,v in records.items()):
 if len(selected)==96:break
 selected.add(wid)
assert len(selected)==96;cases=[records[k][1] for k in sorted(selected)];m=json.loads((C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json').read_text());template=json.loads((R/'actual-native-cases01/cases.json').read_text());filemap={};files=[];rows={}
for w in cases:
 for rid in w['prefix_raw_rows']+[w['seed_raw_row']]:
  if rid in rows:continue
  r=m['raw_rows'][rid];fid=r['file']
  if fid not in filemap:filemap[fid]=len(files);files.append(m['raw_files'][fid])
  rows[rid]=[rid,filemap[fid],r['row'],r['raw_sha256'],r['target_position']]
allmodels=[{'id':'old_quarter','output':pin(oldquarter)}]+exports;packetrefs=[]
for model in allmodels:
 packet={k:template[k] for k in ['schema','diagnostic_only','build','helper','runtime_libraries','cpu_library','cuda_library','target','draft_map','raw_row_columns']};packet.update(manifest=pin(C/'w1a8-modern-fit-expansion-20261103-01/union01/manifest.json'),models={'parent':m['parent'],'candidate':model['output']},files=files,raw_rows=list(rows.values()),cases=cases);f=R/f'data-backoff-{model["id"]}-cases01.json';assert not f.exists();blob=json.dumps(packet,separators=(',',':')).encode();assert len(blob)<=2*1024**2;f.write_bytes(blob);packetrefs.append({'id':model['id'],'packet':pin(f),'model':model['output'],'output':str(C/'w1a8-data-backoff-native-20261103-01'/model['id'])})
plan={'status':'DECLARED_BACKOFFS_AND_PARENT_MARGIN_SUBSET_FROZEN_NOT_NATIVE_QUALIFIED','exports':exports,'old_quarter_formula_byte_identity_confirmed':True,'parent_sources':sources,'selection':'one lowest-margin originalparent-native-success window/group, then global lowestparentmargins to96; no newcandidate outcomes','group_count':len(bygroup),'windows':96,'native_tokens':sum(w['native_prefix'] for w in cases),'packets':packetrefs,'scope':'targetedPhase1 preservationdiagnostic; passing96notfull956preservation orpredictorvalidity','source':pin(__file__)};(R/'data-backoff-plan.json').write_text(json.dumps(plan,indent=2)+'\n');print([(x['id'],x['model']['sha256']) for x in packetrefs])
