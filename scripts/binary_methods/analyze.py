from pathlib import Path
import json,sqlite3,statistics,collections,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/binary-methods-20261104-01';C=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
rows=[json.loads(l) for l in (R/'throughput-final/records.jsonl').read_text().splitlines() if l];arms=list(dict.fromkeys(r['arm'] for r in rows));out={'throughput':[],'kernels':[],'native':[]}
for arm in arms:
 rs=[r for r in rows if not r['warmup'] and r['arm']==arm];rr=[r['result'] for r in rs];n=sum(r['completion_tokens'] for r in rr);spec={k:sum(r['speculative'][k] for r in rr) for k in ['accepted','proposed','rounds']}
 rates=[]
 for rep in [0,1]:
  q=[r['result'] for r in rs if r['repetition']==rep];rates.append(sum(x['completion_tokens'] for x in q)/sum(x['server_predicted_ms'] for x in q)*1000)
 out['throughput'].append(dict(arm=arm,decode_tps=n/sum(r['server_predicted_ms'] for r in rr)*1000,request_tps=n/sum(r['request_wall_s'] for r in rr),acceptance=spec['accepted']/spec['proposed'],speculative=spec,decode_tps_per_repetition=rates,requests=len(rr),tokens=n))
for r in read(R/'kernels.json')['rows']:out['kernels'].append({**{k:r[k] for k in ['mode','m','k','n']},**{k+'_us':statistics.median(v) for k,v in r['timing_us'].items()}})
if (R/'native-trace-final.sqlite').exists():
 c=sqlite3.connect(R/'native-trace-final.sqlite');epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
 for r in [json.loads(l) for l in (R/'trace-final/records.jsonl').read_text().splitlines() if l]:
  if r['warmup']:continue
  plain=next(z for z in rows if not z['warmup'] and (z['arm'],z['repetition'],z['prompt_id'])==(r['arm'],r['repetition'],r['prompt_id']))
  assert plain['result']['generated_token_ids']==r['result']['generated_token_ids'] and plain['result']['speculative']==r['result']['speculative'], 'trace/plain mismatch'
  lo,hi=[v-epoch for v in r['bounds_unix_ns']];events=c.execute('select s.value,k.start,k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? order by k.start',(lo,hi)).fetchall();assert events
  groups=collections.defaultdict(lambda:dict(count=0,sum_us=0));stages=dict(quantize_pack=0,representation_prep=0,dot_scale=0)
  for name,start,end in events:
   key=name.split('(')[0];g=groups[key];g['count']+=1;g['sum_us']+=(end-start)/1000
   if any(x in name for x in ['w1ax_quantize','w1a1_pack_activations']):stage='quantize_pack'
   elif any(x in name for x in ['w1a8_make','w1a8_sum_codes','screen_unpack_a1','lut_transpose']):stage='representation_prep'
   elif any(x in name for x in ['imma_expanded','w1a8_dot','w1ax_integer_dot','w1a1_xor_popc','w1a1_binary_mma','kernel::_nqmv','lut_scale']):stage='dot_scale'
   else:continue
   stages[stage]+=(end-start)/1000
  allrounds=[json.loads(l) for l in (R/f'trace-final/0-{r["arm"]}/rounds.jsonl').read_text().splitlines()];taskids=list(dict.fromkeys(x['task_id'] for x in allrounds));assert len(taskids)==2;rounds=[x for x in allrounds if x['task_id']==taskids[-1]];counted=[x for x in rounds if x['status']=='complete' and not x['replay']];spec=dict(rounds=len(counted),accepted=sum(x['n_accepted'] for x in counted),proposed=sum(x['n_proposed'] for x in counted));assert spec==r['result']['speculative']
  emitted=[t for x in rounds if x['status']!='checkpoint_replay' for t in x['emitted_token_ids']];assert emitted in [r['result']['generated_token_ids'],r['result']['generated_token_ids'][1:]]
  stage_cpu={k:sum(x[k] for x in rounds) for k in ['draft_us','process_us','target_decode_sync_us','check_us','kv_repair_us','round_us']}
  out['native'].append(dict(arm=r['arm'],request_us=(hi-lo)/1000,all_kernel_sum_us=sum(v['sum_us'] for v in groups.values()),binary_stages_us=stages,binary_us_per_round=sum(stages.values())/spec['rounds'],speculative=spec,round_cpu_us=stage_cpu,binary_kernels={k:v for k,v in groups.items() if any(x in k for x in ['w1a','imma_expanded','screen_unpack','lut_transpose','kernel::_nqmv','lut_scale'])},top_kernels=sorted(groups.items(),key=lambda x:x[1]['sum_us'],reverse=True)[:12]))
if (R/'lutgemm.json').exists():out['lutgemm']=read(R/'lutgemm.json')
out['scope']='RTX5080, original Qwen3-4B F16 target, retained13-projection DSpark weights, D5;2 repetitions x3 reused DEV prompts,128 max tokens; A1 is runtime precision override with no training; timings are screen, not modern DFlash baseline evaluation.'
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['throughput'],indent=2))
