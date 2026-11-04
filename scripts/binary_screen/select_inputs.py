"""Freeze12 already-TRAIN groups and moderate-prefix windows before outcomes."""
from pathlib import Path
import sys
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require
R=P/'runs/binary-screen-20261104-01';bank=P/'runs/data-cost-transfer-20261104-01/registered-bank/registry.json'
registry=read(bank);groups={};refs={}
for shard in registry['shards']:
 ref=shard['manifest'];require(pin(ref['path'])==ref,'manifest changed');m=read(ref['path']);selected=set(shard['windows']);refs[ref['path']]=ref
 for w in m['windows']:
  if w['id'] in selected and w['role']=='fit' and w['max_depth']==2 and 32<=w['anchor']<=256:groups.setdefault((w['domain'],w['group_id']),[]).append((ref,w))
rows=[]
for domain in ['code','reasoning','prose']:
 eligible=sorted(k for k in groups if k[0]==domain);require(len(eligible)>=4,'insufficient TRAIN groups')
 for key in eligible[:4]:
  ref,w=min(groups[key],key=lambda x:(abs(x[1]['anchor']-128),x[1]['anchor'],x[1]['id']));rows.append(dict(manifest=ref,window=w['id'],group=w['group_id'],domain=domain,anchor=w['anchor']))
atomic(R/'quality-inputs.json',dict(schema='binary_preliminary_train_inputs_v1',registry=pin(bank),shards=pin(bank.with_name('shards.json')),selected=rows,scope='Four lexicographically first eligible TRAIN groups/domain; window nearest anchor128 within32..256 (anchor includes rendered prompt), depth2.12groups and24teacher-forced positions. Chosen without precision results; no new held-out consumption.'))
print([(x['domain'],x['anchor']) for x in rows])
