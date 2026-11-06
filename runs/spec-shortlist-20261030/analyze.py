from pathlib import Path
import json,statistics,collections
P=Path(__file__).resolve().parent
matrix=[]
def readrows(d):
 f=d/'requests.jsonl';return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
def key(r):return r['prompt'],r['rep']
def metrics(d,ref):
 rows=readrows(d);lookup={key(r):r for r in ref}
 exact=sum(r['result'].get('tokens')==lookup[key(r)]['result'].get('tokens') for r in rows if key(r) in lookup)
 tok=sum(r['result']['tokens_predicted'] for r in rows);elapsed=sum(r['elapsed'] for r in rows)
 decode_ms=sum(r['result'].get('timings',{}).get('predicted_ms',0) for r in rows)
 return {'requests':len(rows),'tokens':tok,'request_seconds':elapsed,'request_tps':tok/elapsed if elapsed else None,'server_decode_tps':tok*1000/decode_ms if decode_ms else None,'exact_outputs':exact,'checked_outputs':len(rows),'stopped_eos':sum((r['result'].get('stop_type')=='eos') for r in rows),'output_counts':[r['result']['tokens_predicted'] for r in rows],'draft_proposed':sum(r['result'].get('timings',{}).get('draft_n',0) for r in rows),'draft_accepted':sum(r['result'].get('timings',{}).get('draft_n_accepted',0) for r in rows),'cache_n_values':sorted(set(r['result'].get('timings',{}).get('cache_n',-1) for r in rows)),'sanity_checks':[{'prompt':r['prompt'],'rep':r['rep'],'pass':('295' if r['prompt']==2 else '17') in r['result'].get('content','')} for r in rows if r['prompt'] in [2,6]],'median_request_seconds':statistics.median([r['elapsed'] for r in rows]) if rows else None}
winners=[]
for size in ['4B','9B']:
 for fmt in ['Q4_K_M','NVFP4']:
  target=f'Qwen3.5-{size}-Q4_K_M.gguf' if fmt=='Q4_K_M' else f'qwen35-{size.lower()}-mtp-nvfp4.gguf';base=P/'measurements'/target
  refs=readrows(base/'no-spec/screen');available=[]
  for arch in ['none','mtp','draft-eagle3','draft-dflash','draft-dspark','draft-simple']:
   for precision in (['target'] if arch in ['none','mtp'] else ['BF16','Q8_0','Q4_K_M','NVFP4']):
    name={'none':'no-spec','mtp':'mtp'}.get(arch,arch+'-'+precision);d=base/name/'screen';f=d/'summary.json'
    row={'target_size':size,'target_precision':fmt,'architecture':arch,'draft_precision':precision,'row':name}
    if f.exists():
     row.update(json.loads(f.read_text()))
     if row['status']=='measured':row.update(metrics(d,refs));available.append(row)
    else:row.update(status='unavailable',reason='No compatible converted artifact/load result in this bounded operation; consult conversion and audit evidence.')
    matrix.append(row)
  # Screen does not qualify a throughput winner unless exact greedy outputs passed.
  qualified=[r for r in available if r.get('requests',0)>0 and r.get('exact_outputs')==r.get('checked_outputs')]
  winner=max(qualified,key=lambda r:r['request_tps']) if qualified else None
  leader_ref=readrows(base/'no-spec/leader');leader=[]
  for d in base.glob('*/leader'):
   if (d/'summary.json').exists():
    v=json.loads((d/'summary.json').read_text());v.update(metrics(d,leader_ref));leader.append(v)
  winners.append({'target_size':size,'target_precision':fmt,'screen_winner':winner,'leader_repeat':leader,'status':'provisional' if winner else 'blocked'})
(P/'matrix.json').write_text(json.dumps(matrix,indent=2));(P/'winners.json').write_text(json.dumps(winners,indent=2))
print(json.dumps(winners,indent=2))
