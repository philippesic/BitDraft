"""Summarize isolated timings and request-bounded native CUDA attribution."""
from pathlib import Path
import json,sqlite3,statistics,argparse
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();R=a.run
micro=json.loads((R/'micro/result.json').read_text());out={'micro':[],'native':[]}
for r in micro['records']:
 out['micro'].append({**{k:r[k] for k in ['mode','m','k','n']},**{k:statistics.median(r[k]) for k in ['pack_us','dot_us','pack_dot_us']}})
c=sqlite3.connect(R/'native-trace3.sqlite');epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
rows=[json.loads(x) for x in (R/'native3/records.jsonl').read_text().splitlines()]
measured=[r for r in rows if not r['warmup']]
assert len(measured)==2
assert measured[0]['result']['generated_token_ids']==measured[1]['result']['generated_token_ids']
assert measured[0]['result']['speculative']==measured[1]['result']['speculative']
for r in measured:
 lo,hi=[v-epoch for v in r['bounds_unix_ns']]
 events=c.execute('select s.value,k.start,k.end,k.gridX,k.gridY,k.blockX,k.blockY from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=? order by k.start',(lo,hi)).fetchall()
 assert events
 groups={}
 for name,start,end,*shape in events:
  key=name.split('(')[0] if '<' not in name else name.split('>')[0]+'>'
  g=groups.setdefault(key,{'count':0,'sum_us':0,'durations_us':[],'launch_shapes':[]})
  g['count']+=1;g['sum_us']+=(end-start)/1000;g['durations_us'].append((end-start)/1000)
  if shape not in g['launch_shapes']:g['launch_shapes'].append(shape)
 for g in groups.values():g['median_us']=statistics.median(g.pop('durations_us'))
 binary={k:v for k,v in groups.items() if any(s in k for s in ['w1a','imma_expanded'])}
 # Union of all kernel intervals measures time with any kernel active, not SM occupancy.
 union=0;left=right=None
 for _,start,end,*_ in events:
  if right is None:left,right=start,end
  elif start>right:union+=right-left;left,right=start,end
  else:right=max(right,end)
 union+=right-left
 out['native'].append({'arm':r['arm'],'request_us':(hi-lo)/1000,'all_kernel_sum_us':sum(g['sum_us'] for g in groups.values()),'kernel_active_union_us':union/1000,'binary_stages':binary,'top_kernels':sorted(groups.items(),key=lambda x:x[1]['sum_us'],reverse=True)[:12],'speculative':r['result']['speculative'],'output_tokens':r['result']['completion_tokens'],'scope':'One already-consumed DEV request after one warmup; instrumented duration, not throughput or SM occupancy; unix request bounds aligned to profiler session epoch.'})
out['checks']=len(micro['checks']);out['native_tokens_and_speculative_counters_equal']=True
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out['native']:
 print(r['arm'], 'request_us',round(r['request_us']), 'kernel_us',round(r['all_kernel_sum_us']))
 for k,v in r['binary_stages'].items():print(k,v)
