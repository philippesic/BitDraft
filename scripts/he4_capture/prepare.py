"""Admit TRAIN texts with opaque identity exclusions; no historical features used."""
import json,hashlib,re,sys,unicodedata
from pathlib import Path
P=Path(sys.argv[1]); OUT=Path(sys.argv[2]) if len(sys.argv)>2 else P; OUT.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(P/'scripts/w1a8_generalization'))
from modern_fresh import identities,render

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def strings(d):
 if isinstance(d,dict):
  for k,v in d.items():
   if k not in ('messages','request','controls','expected','text','prompt','content'):yield from strings(v)
 elif isinstance(d,list):
  for v in d:yield from strings(v)
 elif isinstance(d,str):yield d
excluded=set();pins=[]
meta=P/'home/philip/binary-eagle-decoding'
files=[f for f in meta.rglob('*exclusions.json') if f.name!='actual_train-exclusions.json']+[f for f in meta.rglob('*.index.jsonl') if any(x in f.name for x in ('dev','final','sealed'))]
files+=list((P/'runs').rglob('consumption.json'))
files+=[P/'runs/data-cost-transfer-20261104-01/consumed-natural-inputs.json']
files+=list((P/'runs/w1a8-modern-protected-20261103-01').rglob('manifest.json'))
files+=list((P/'exclusion-metadata').glob('*.json'))
for f in files:
 docs=[json.loads(l) for l in f.read_text().splitlines()] if f.suffix=='.jsonl' else [read(f)]
 for d in docs:excluded.update(strings(d))
 pins.append(dict(path=str(f),sha256=sha(f)))
assert len(excluded)>1000
write(OUT/'exclusions.json',dict(policy='All dev/final/sealed opaque indexes; consumed evaluation identities including heldout24; never select by answer or output',sources=pins,identities=sorted(excluded)))
# Reserve all components/identities from non-optimizer roles.
selector=read(P/'runs/dspark-a8-full-pool-20261006-01/plans/role-selector.json')['selection']
for r in selector:
 if r['split']!='train':excluded.update(strings(r))
seen=set();records=[];rejected=[]
def add(row,pool):
 raw='\n'.join(m['content'] for m in row['messages'] if m['role']=='user')
 text=render(raw) if len(row['messages'])==1 else ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in row['messages'])+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
 keys=set(identities(raw).values())|set(identities(text).values())|{row.get(k,'') for k in ('id','prompt_id','group_id','source_group','source_topic','content_sha256')}
 keys.discard('')
 keys.discard(None)  # Missing topics carry no identity, just like empty topics.
 if keys&excluded or keys&seen:
  rejected.append(dict(id=row['id'],pool=pool,reason='excluded identity' if keys&excluded else 'duplicate identity',matched_identities=sorted(keys&(excluded if keys&excluded else seen))));return
 seen.update(keys)
 row.update(pool=pool,rendered_text=text,rendered_sha256=hashlib.sha256(text.encode()).hexdigest());records.append(row)
bank=P/'runs/data-cost-transfer-20261104-01/bank-inputs'
bank_manifest=read(bank/'bank.json')
assert sha(bank/'bank.json')==read(P/'runs/data-cost-transfer-20261104-01/train-exposure.json')['source_bank']['sha256']
for ref in bank_manifest['chunks']:
 assert sha(bank/Path(ref['path']).parent.name/'roster.json')==ref['sha256']
prior=read(P/'runs/component-campaign-20261105-01/broader-train-capture/manifest.json')
used={r['name'].removeprefix('broader-') for r in prior['records'] if r['name'].startswith('broader-')}
for f in sorted(bank.glob('chunk*/roster.json')):
 for row in read(f)['rows']:
  assert row['role']=='fit' and row['phase']=='training'
  if row['id'] in used:continue
  text=(f.parent/Path(row['rendered_text_path']).name).read_bytes()
  assert hashlib.sha256(text).hexdigest()==row['rendered_prompt_sha256']
  add(row,'unused_modern_reserved')
by_source={}
for r in selector:
 if r['split']=='train':by_source[(r['shard'],r['row'])]=r
for f in sorted((P/'data/continuous-w1ax/freeze-002').glob('train-*.jsonl')):
 if '.index.' in f.name:continue
 shard=int(f.stem.split('-')[1])
 for n,line in enumerate(f.open()):
  r=by_source.get((shard,n))
  if r is None:continue
  row=json.loads(line);assert row['id']==r['prompt_id'];row.update(r);add(row,'historical_optimizer')
write(OUT/'prompt-pool.json',dict(records=records,rejected=rejected,unused_modern=sum(r['pool']=='unused_modern_reserved' for r in records),historical=sum(r['pool']=='historical_optimizer' for r in records),prior_modern_used=sorted(used),selector_sha256=sha(P/'runs/dspark-a8-full-pool-20261006-01/plans/role-selector.json')))
print('admitted',len(records),'rejected',len(rejected))
