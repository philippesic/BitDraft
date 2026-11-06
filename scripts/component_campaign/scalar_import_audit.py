"""Confirm actual native F16 scalar reads, not file-header or CLI acceptance alone."""
import json,subprocess,sys
from .common import *
from .measure import completed
from .archive import hydrate,release
from scripts.dflash_baseline.model import GGUFReader

def fnv(data):
 value=14695981039346656037
 for x in memoryview(data).cast('B'):value=((value^int(x))*1099511628211)&((1<<64)-1)
 return value

def main():
 out=R/'qualification/scalar-import';model=Path(completed()['Sall']['model']);out.mkdir(exist_ok=True)
 if (out/'verified.json').exists():return
 save(out/'arms.json',dict(Sall=str(model)))
 if not (out/'replay/result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.replay',str(out/'arms.json'),'--out',str(out/'replay'),'--runtime',str(R/'runtime04'),'--reps','1','--audit'],cwd=ROOT,check=True)
 created=hydrate(model)
 try:
  reader=GGUFReader(model);expected={t.name:dict(bytes=t.data.nbytes,hash=fnv(t.data),group='norm' if 'norm' in t.name else 'scale' if t.name.endswith('.w1a1_scale') else 'midpoint' if t.name.endswith('.w1ax_midpoint') else 'gain') for t in reader.tensors if t.tensor_type.name=='F16' and ('norm' in t.name or t.name.endswith(('.w1a1_scale','.w1ax_midpoint','.w1ax_gain')))}
  clips=[k for k,f in reader.fields.items() if k.endswith('.w1ax.clip') and f.types[0].name=='UINT16']
  if len(clips)!=18:raise ValueError('physical half clip fields missing')
 finally:release(model,created)
 data=json.load(open(out/'replay/result.json'));observed=set();reads=0
 for cell in data['records']:
  for row in cell['result']['rows']:
   for item in row['audit']:
    if item.get('kind')!='half_metadata_read':continue
    name=item['source_name']
    if name not in expected or item['source_bytes']!=expected[name]['bytes'] or item['source_hash_fnv1a64']!=expected[name]['hash']:raise ValueError('actual native half metadata differs from exported F16 storage '+name)
    observed.add(name);reads+=1
 missing=set(expected)-observed
 if missing:raise ValueError('unobserved scalar F16 fields: '+','.join(sorted(missing)))
 clip_qualification=json.load(open(R/'qualification/clip-storage/result.json'))
 if clip_qualification['status']!='NATIVE_HALF_CLIP_STORAGE_PASS':raise ValueError('clip conversion/logit/cache proof missing')
 save(out/'verified.json',dict(status='ACTUAL_NATIVE_HALF_METADATA_READ_PASS',expected=expected,observed_fields=sorted(observed),reads=reads,clip_fields=clips,clip_conversion=pin(R/'qualification/clip-storage/result.json'),native=pin(out/'replay/result.json'),source=pin(__file__),scope='Actual graph consumers read F16 bytes equal to exported same-master half storage; source casts/multiplication retain F32 math. Clip bits use separately byte-equivalent F32-parameter qualification. Audit timings excluded; persistent model/logged workspace/sampled peak measured separately.'))
 print('SCALAR_NATIVE_IMPORT_VERIFIED',len(observed),flush=True)

if __name__=='__main__':main()
