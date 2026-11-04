"""Recompute the six-prompt matched historical-target screen."""
from pathlib import Path
import json,sys,numpy as np
R=Path(sys.argv[1]);rows=[json.loads(s) for s in (R/'paired/records.jsonl').read_text().splitlines()]
assert json.loads((R/'paired/result.json').read_text())['status']=='MATCHED_ORIGINAL_DSPARK_DFLASH_PASS'
measured=[r for r in rows if not r['warmup']];arms=sorted({r['arm'] for r in measured});ids=sorted({r['prompt_id'] for r in measured});assert len(ids)==6
x={'scope':'Original Qwen3-4B F16, RTX5080, same retained runtime, greedy D5; six consumed DEV prompts, five repetitions. No architecture tuning or untouched confirmation.','responses':len(rows),'arms':{},'contrasts':{}}
lookup={(r['arm'],r['mode'],r['repetition'],r['prompt_id']):r for r in measured}
rng=np.random.default_rng(6071)
for arm in arms:
 plain=[r for r in measured if r['arm']==arm and r['mode']=='plain'];trace=[r for r in measured if r['arm']==arm and r['mode']=='trace'];assert len(plain)==len(trace)==30
 results=[r['result'] for r in plain];tok=sum(r['completion_tokens'] for r in results);counts={k:sum(r['speculative'][k] for r in results) for k in ['rounds','accepted','proposed']};wall=sum(r['request_wall_s'] for r in results);tw=sum(r['result']['request_wall_s'] for r in trace)
 d={'requests_per_mode':30,'tokens':tok,'plain_wall_seconds':wall,'plain_tokens_per_second':tok/wall,**counts,'emitted_tokens_per_round':tok/counts['rounds'] if counts['rounds'] else None,'acceptance_rate':counts['accepted']/counts['proposed'] if counts['proposed'] else None,'instrumentation_wall_change_percent':100*(tw/wall-1)}
 if counts['rounds']:
  costs={k:sum(r['cost_us'][k] for r in trace) for k in trace[0]['cost_us']};d['traced_cost_ms_per_round']={k:v/counts['rounds']/1000 for k,v in costs.items()};d['traced_complete_draft_ms_per_round']=sum(costs[k] for k in ['draft_us','process_us','accept_hook_us'])/counts['rounds']/1000
 x['arms'][arm]=d

def array(arm,mode,fn):return np.array([[fn(lookup[(arm,mode,rep,i)]) for i in ids] for rep in range(5)],dtype=float)
for arm in arms:
 t=array(arm,'trace',lambda r:r['result']['request_wall_s']);p=array(arm,'plain',lambda r:r['result']['request_wall_s']);vals=[]
 for _ in range(10000):
  idx=np.ix_(rng.integers(0,5,5),rng.integers(0,6,6));vals.append(100*(t[idx].sum()/p[idx].sum()-1))
 x['arms'][arm]['instrumentation_wall_change_95_interval_percent']=np.quantile(vals,[.025,.975]).tolist()
for base in ['dflash_q4km','dspark_q4km']:
 for metric in ['throughput','catchup','draft','complete_draft']:
  aa=[]
  for arm in ['dspark_w1a8_parent3504',base]:
   mode='plain' if metric=='throughput' else 'trace'
   num=array(arm,mode,lambda r:r['result']['completion_tokens'] if metric=='throughput' else (r['cost_us']['process_us'] if metric=='catchup' else (r['cost_us']['draft_us'] if metric=='draft' else sum(r['cost_us'][k] for k in ['draft_us','process_us','accept_hook_us']))))
   den=array(arm,mode,lambda r:r['result']['request_wall_s'] if metric=='throughput' else r['result']['speculative']['rounds']);aa.append((num,den))
  vals=[]
  for _ in range(10000):
   idx=np.ix_(rng.integers(0,5,5),rng.integers(0,6,6));ratios=[n[idx].sum()/d[idx].sum() for n,d in aa];vals.append(100*(ratios[0]/ratios[1]-1))
  mean=100*((aa[0][0].sum()/aa[0][1].sum())/(aa[1][0].sum()/aa[1][1].sum())-1)
  x['contrasts'][base+'/'+metric]={'w1a8_vs_control_percent':mean,'paired_95_interval_percent':np.quantile(vals,[.025,.975]).tolist()}
(R/'analysis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
