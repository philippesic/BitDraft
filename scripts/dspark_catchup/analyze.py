"""Recompute matched native catch-up costs and paired prompt/repetition uncertainty."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);out=R/'paired'
plain=json.loads((out/'plain/result.json').read_text())['records']
traced=json.loads((out/'traced/result.json').read_text())['records']
records={ (r['arm'],r['repetition'],r['id']):r for r in traced if not r['warmup'] }
summary=[]
for path in sorted((out/'traced').glob('rep-*/cost-trace.summary.json')):
 rows=json.loads(path.read_text())['rows']
 spans=[json.loads(x) for x in path.with_name('cost-trace.jsonl').read_text().splitlines()]
 for row in rows:
  if row['warmup']:continue
  record=records[(row['arm'],row['repetition'],row['id'])]
  bounds=json.loads(Path(record['response']['path']).with_suffix('.bounds.json').read_text())
  roots=[s for s in spans if s['depth']==0 and bounds['start_ns']<=s['start_ns'] and s['end_ns']<=bounds['end_ns']]
  beginnings=[s['start_ns'] for s in roots if s['name']=='generation_begin']
  if beginnings:
   row['generation_catchup_calls']=sum(s['name']=='draft_catchup' and s['start_ns']>=beginnings[0] for s in roots)
  summary.append(row)
result={'scope':'Qwen3.5 4B Q4_K_M, RTX5080, identical runtime, both depth2, reused DEV12. Host spans; no forced synchronization; five paired repetitions. Not historical DSpark W1A8.','responses':len(plain)+len(traced),'arms':{},'contrasts':{}}
for arm in sorted({r['arm'] for r in summary}):
 rows=[r for r in summary if r['arm']==arm];p=[r for r in plain if r['arm']==arm and not r['warmup']]
 rounds=sum(r['verification_rounds'] for r in rows);tokens=sum(r['completion_tokens'] for r in rows)
 costs={phase:{key:sum(r['root_ms'][phase].get(key,0) for r in rows) for key in sorted({k for r in rows for k in r['root_ms'][phase]})} for phase in ['prefill','generation']}
 gen=costs['generation'];calls=sum(r.get('generation_catchup_calls',0) for r in rows)
 result['arms'][arm]=dict(measured_requests=len(rows),tokens=tokens,rounds=rounds,tokens_per_round=tokens/rounds if rounds else None,root_ms=costs,generation_catchup_calls=calls,catchup_ms_per_call=gen.get('draft_catchup',0)/calls if calls else None,per_round_ms={k:v/rounds for k,v in gen.items()} if rounds else None,complete_draft_ms_per_round=sum(gen.get(k,0) for k in ['draft_generate','draft_catchup','draft_accept'])/rounds if rounds else None,prefill_catchup_ms_per_request=costs['prefill'].get('draft_catchup',0)/len(rows),plain_tokens_per_second=tokens/sum(r['full_http_wall_seconds'] for r in p),instrumentation_wall_change_percent=100*(sum(r['http_ms'] for r in rows)/sum(r['full_http_wall_seconds']*1000 for r in p)-1))
ids=sorted({r['id'] for r in summary});lookup={(r['arm'],r['repetition'],r['id']):r for r in summary};rng=np.random.default_rng(771)
# Paired resampling prompts and repetition blocks; exploratory, not untouched confirmation.
for metric in ['draft_catchup','draft_generate','complete_draft']:
 arrays=[]
 for arm in ['dspark_Q4KM_d2','dflash_Q4KM_d2']:
  num=np.array([[sum(lookup[(arm,rep,i)]['root_ms']['generation'].get(k,0) for k in (['draft_generate','draft_catchup','draft_accept'] if metric=='complete_draft' else [metric])) for i in ids] for rep in range(5)])
  den=np.array([[lookup[(arm,rep,i)]['verification_rounds'] for i in ids] for rep in range(5)])
  arrays.append((num,den))
 values=[]
 for _ in range(10000):
  rr=rng.integers(0,5,5);pp=rng.integers(0,len(ids),len(ids));idx=np.ix_(rr,pp)
  vals=[n[idx].sum()/d[idx].sum() for n,d in arrays];values.append(100*(vals[0]/vals[1]-1))
 result['contrasts'][metric]={'dspark_vs_dflash_percent':100*((arrays[0][0].sum()/arrays[0][1].sum())/(arrays[1][0].sum()/arrays[1][1].sum())-1),'paired_95_interval_percent':np.quantile(values,[.025,.975]).tolist()}
pairs=json.loads((out/'result.json').read_text())['pairs']
for arm in result['arms']:
 subset={(r['repetition'],r['id']):r for r in pairs if r['arm']==arm and not r['warmup']}
 a=np.array([[subset[(rep,i)]['traced_ms'] for i in ids] for rep in range(5)])
 b=np.array([[subset[(rep,i)]['plain_ms'] for i in ids] for rep in range(5)])
 vals=[]
 for _ in range(10000):
  idx=np.ix_(rng.integers(0,5,5),rng.integers(0,len(ids),len(ids)))
  vals.append(100*(a[idx].sum()/b[idx].sum()-1))
 result['arms'][arm]['instrumentation_wall_change_95_interval_percent']=np.quantile(vals,[.025,.975]).tolist()
(R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
