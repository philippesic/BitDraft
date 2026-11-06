"""Aggregate tokens/time and counts; paired prompt/repetition uncertainty, never seed uncertainty."""
import json,re
from pathlib import Path
import numpy as np

def summarize(result):
 data=json.load(open(result));rows=[x for x in data['rows'] if not x.get('warmup')];summary={}
 for arm in dict.fromkeys(x['arm'] for x in rows):
  v=[x for x in rows if x['arm']==arm];tokens=sum(x['tokens'] for x in v);seconds=sum(x['wall_seconds'] for x in v)
  native=lambda key:sum(x['native'].get(key,0) for x in v)
  rounds=native('rounds');proposed=native('proposed')
  summary[arm]=dict(responses=len(v),timing_repetitions=len({x['rep'] for x in v}),prompts=len({x['prompt'] for x in v}),output_tokens=tokens,request_wall_seconds=seconds,request_tok_s=tokens/seconds,accepted=native('accepted'),proposed=proposed,rounds=rounds,accepted_per_round=native('accepted')/rounds if rounds else None,emitted_per_round=tokens/rounds if rounds else None,acceptance=native('accepted')/proposed if proposed else None,correct=all(x['correct'] for x in v),instrumented=any(x['instrumented'] for x in v),full_draft_host_ms_per_round=sum(x.get('draft_full_ms',0) for x in v)/rounds if rounds and all('draft_full_ms' in x for x in v) else None)
  summary[arm].update(draft_prefill_host_ms_per_request=sum(x.get('draft_prefill_ms',0) for x in v)/len(v) if all('draft_prefill_ms' in x for x in v) else None,full_draft_all_phases_host_ms_per_round=sum(x.get('draft_all_phases_ms',0) for x in v)/rounds if rounds and all('draft_all_phases_ms' in x for x in v) else None,host_scope='Generation metric includes disjoint draft generate/catchup/accept root spans; prefill and all-phases totals also reported. Complete request wall includes prompt processing.')
 return summary

def paired_ratio(result,numerator,denominator,draws=2000):
 rows=[x for x in json.load(open(result))['rows'] if not x.get('warmup') and not x.get('instrumented')]
 mapping={(x['arm'],x['rep'],x['prompt']):x for x in rows};reps=sorted({x['rep'] for x in rows});prompts=sorted({x['prompt'] for x in rows})
 source=json.load(open(Path(result).parent/'prompts.json'));domains={x['id']:x.get('domain','capture') for x in source};strata={}
 for prompt in prompts:strata.setdefault(domains.get(prompt,'capture'),[]).append(prompt)
 for arm in (numerator,denominator):
  if any((arm,r,p) not in mapping for r in reps for p in prompts):raise ValueError('incomplete paired panel')
 def ratio(rs,ps):
  rates=[]
  for arm in (numerator,denominator):
   selected=[mapping[arm,r,p] for r in rs for p in ps];rates.append(sum(x['tokens'] for x in selected)/sum(x['wall_seconds'] for x in selected))
  return rates[0]/rates[1]
 rng=np.random.default_rng(61105);values=[ratio(rng.choice(reps,len(reps)).tolist(),[p for group in strata.values() for p in rng.choice(group,len(group)).tolist()]) for _ in range(draws)]
 return dict(ratio=ratio(reps,prompts),paired_95_interval=np.quantile(values,[.025,.975]).tolist(),draws=draws,scope='Paired complete prompts and process-cell repetitions; consumed/heldout cohort per source panel. Repeated greedy quality observations are not fresh samples or training seeds.')

def allocations(cell):
 text=(Path(cell)/'server.log').read_text(errors='replace');model={}
 for device,size in re.findall(r'load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB',text):model[device]=model.get(device,0)+float(size)*2**20
 caches=[];rot=[]
 for match in re.finditer(r'llama_kv_cache: size =\s+([\d.]+) MiB.*?K \(([^)]+)\):\s+([\d.]+) MiB, V \(([^)]+)\):\s+([\d.]+) MiB',text):
  caches.append(dict(total_reported_bytes=float(match[1])*2**20,k_type=match[2],k_bytes=float(match[3])*2**20,v_type=match[4],v_bytes=float(match[5])*2**20))
 for device,size in re.findall(r'init_rotation:\s+(\S+) rotation buffer size =\s+([\d.]+) MiB',text):rot.append(dict(device=device,reported_bytes=float(size)*2**20))
 compute=[dict(device=m[0],reported_bytes=float(m[1])*2**20) for m in re.findall(r'sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB',text)]
 return dict(model_buffers=model,cache_buffers=caches,rotation_buffers=rot,compute_buffer_history=compute,precision='Allocation log sizes have0.01MiB display rounding; tensor bytes and quantization block sizes provide exact persistent accounting separately. Target cache is first; draft base/SWA caches follow. Conversion/graph workspace and sampled peak are separate.')

def paired_host_ratio(result,numerator,denominator,draws=2000):
 rows=[x for x in json.load(open(result))['rows'] if not x.get('warmup') and x.get('instrumented')]
 mapping={(x['arm'],x['rep'],x['prompt']):x for x in rows};reps=sorted({x['rep'] for x in rows});prompts=sorted({x['prompt'] for x in rows})
 if any((arm,r,p) not in mapping for arm in (numerator,denominator) for r in reps for p in prompts):raise ValueError('incomplete paired host panel')
 def ratio(rs,ps):
  values=[]
  for arm in (numerator,denominator):
   chosen=[mapping[arm,r,p] for r in rs for p in ps];rounds=sum(x['native'].get('rounds',0) for x in chosen)
   if not rounds:raise ValueError('draft round count missing from host trace')
   values.append(sum(x['draft_full_ms'] for x in chosen)/rounds)
  return values[0]/values[1]
 rng=np.random.default_rng(61105);values=[ratio(rng.choice(reps,len(reps)).tolist(),rng.choice(prompts,len(prompts)).tolist()) for _ in range(draws)]
 return dict(ratio=ratio(reps,prompts),paired_95_interval=np.quantile(values,[.025,.975]).tolist(),draws=draws,scope='Complete instrumented prompt/process cells; disjoint full draft host milliseconds per native round. Timing uncertainty only, no independent training seed claim.')

def paired_replay_ratio(data,numerator,denominator,draws=2000):
 cells={}
 for rec in data['records']:
  values=[r['graph_event_ms'] for r in rec['result']['rows'] if not r['warmup']]
  cells[rec['arm'],rec['rep'],rec['domain']]=float(np.mean(values))
 reps=sorted({r for a,r,d in cells});domains=sorted({d for a,r,d in cells})
 if any((arm,r,d) not in cells for arm in (numerator,denominator) for r in reps for d in domains):raise ValueError('incomplete paired GPU replay panel')
 def ratio(rs,ds):return np.mean([cells[numerator,r,d] for r in rs for d in ds])/np.mean([cells[denominator,r,d] for r in rs for d in ds])
 rng=np.random.default_rng(61105);values=[ratio(rng.choice(reps,len(reps)).tolist(),rng.choice(domains,len(domains)).tolist()) for _ in range(draws)]
 return dict(ratio=float(ratio(reps,domains)),paired_95_interval=np.quantile(values,[.025,.975]).tolist(),draws=draws,scope='Paired process repetitions and whole captured-domain clusters;3neighboring contexts remain together. Fixed N3 GPU graph latency, not request throughput or training-seed uncertainty.')
