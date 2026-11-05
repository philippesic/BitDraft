"""Audit raw native responses, matched exposure, counters and paired instrumentation."""
import argparse,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.precision_benchmark import pin,save
P=Path('/home/philip/binary-eagle-decoding')
def read(p):return json.load(open(p))
def require(value,message):
 if not value:raise ValueError(message)
def signature(row):return {k:row[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}
def audit_panel(path):
 path=Path(path);result=read(path/'result.json');rows=result['rows'];prompts=read(path/'prompts.json');refs={};counts={};requests=[]
 for row in sorted(rows,key=lambda x:(x['rep'],x['arm']!='target',x['arm'],x.get('instrumented',False),x['prompt'])):
  traced=row.get('instrumented',False);cell=path/(f"{row['rep']:02d}-{row['arm']}"+('-traced' if traced else ''));response_path=cell/(row['prompt']+'-response.json');request_path=cell/(row['prompt']+'-request.json');response=read(response_path);request=read(request_path);release=read(cell/'release.json')
  require(release['reaped'],'unreleased server');require(0<len(response['tokens'])<=request['n_predict'],'output limit/token count');require(request['cache_prompt'] is False and request['temperature']==0 and request['seed']==1234,'sampling changed')
  if row['arm']=='target':refs.setdefault(row['prompt'],signature(response))
  require(row['prompt'] in refs and signature(response)==refs[row['prompt']],'native signature mismatch '+str(response_path));require(row['tokens']==len(response['tokens']) and row['correct'],'row token count/correctness differs')
  metrics=row.get('native',{})
  if metrics:
   require(metrics['accepted']==response['timings']['draft_n_accepted'] and metrics['proposed']==response['timings']['draft_n'],'native/server counters disagree')
   require(metrics['accepted']<=metrics['proposed'] and metrics['rounds']>0,'invalid speculative counts')
   prefix=metrics['prefix_survival_counts'];require(sum(prefix)==metrics['accepted'] and all(a>=b for a,b in zip(prefix,prefix[1:])) and all(0<=x<=metrics['rounds'] for x in prefix),'prefix survival counts invalid')
  key=(row['arm'],row['rep'],traced)
  if not row['warmup']:counts.setdefault(key,[]).append(row['prompt'])
  requests.append(dict(request=pin(request_path),response=pin(response_path),arm=row['arm'],rep=row['rep'],prompt=row['prompt'],instrumented=traced))
 for key,ids in counts.items():require(set(ids)=={x['id'] for x in prompts} and len(ids)==len(prompts),'incomplete/duplicate native cell '+str(key))
 if result.get('paired_trace_counters_equal'):
  for plain in [r for r in rows if not r['instrumented']]:
   traced=next(x for x in rows if x['instrumented'] and all(x[k]==plain[k] for k in ('arm','rep','prompt')))
   for k in ('accepted','proposed','rounds','prefix_survival_counts'):require(plain['native'].get(k)==traced['native'].get(k),'trace changed counters')
   require('host_spans_ms' in traced,'trace cost missing')
 return dict(status='RAW_NATIVE_PANEL_AUDITED',panel=pin(path/'result.json'),responses=len(rows),all_target_signatures_exact=True,all_cells_complete=True,paired_counters_unchanged=bool(result.get('paired_trace_counters_equal')),records=requests)
def campaign(root):
 root=Path(root);panels=[]
 for p in sorted(root.rglob('result.json')):
  value=read(p)
  if value.get('status')=='COMPLETE' and isinstance(value.get('rows'),list) and value['rows'] and 'arm' in value['rows'][0]:panels.append(audit_panel(p.parent))
 training=[];physical_updates=0;update_seconds=0
 for p in sorted(root.rglob('updates.jsonl')):
  records=[json.loads(x) for x in p.read_text().splitlines()];physical_updates+=len(records);update_seconds+=sum(x['seconds'] for x in records)
  windows={(x['sequence'],x['anchor']) for x in records};labels={(x['sequence'],x['anchor']+j) for x in records for j in range(15)}
  training.append(dict(path=str(p),updates=len(records),update_seconds=sum(x['seconds'] for x in records),supervised_positions=15*len(records),unique_anchor_windows=len(windows),unique_native_label_positions=len(labels),unique_captures=len({x['sequence'] for x in records}),position_semantics='Native label row p predicts token p+1; overlapping15-slot windows reuse labels. Supervised position count is exposure, not unique source tokens.'))
 matched=[]
 if (root/'adapted').exists():
  registry=read(root/'tranche.json');parent_end=read(root/'foundation/result.json')['progress']['cursor']
  for name,spec in registry['models'].items():
   if not spec['adapt']:continue
   out=root/'adapted'/name
   if not (out/'result.json').exists():continue
   result=read(out/'result.json');cfg=read(out/'resolved.json');progress=result['progress']
   require(progress['updates']==1024 and progress['supervised_tokens']==15360 and progress['cursor']==parent_end+1024,'matched exposure failure '+name)
   require(cfg['parent']==registry['parent'] and cfg['data']==registry['data'],'matched ancestry failure '+name)
   archive=read(out/'checkpoint.pt.json');require(archive['lossless_verified'] and pin(archive['archive'])['sha256']==archive['archive_sha256'],'archive changed '+name)
   matched.append(dict(arm=name,updates=1024,supervised_positions=15360,cursor=progress['cursor'],checkpoint_archive=pin(archive['archive']),config=pin(out/'resolved.json')))
 finalists=[]
 if (root/'finalist-training-confirmation.json').exists():
  from scripts.dflash_baseline.prototype import digest
  plan=read(root/'finalist-training-plan.json');confirmation=read(root/'finalist-training-confirmation.json');foundation=read(root/'foundation/result.json')['progress']
  require(confirmation['status']=='FOUR_MATCHED_2048_RECIPES_COMPLETE' and confirmation['plan']==pin(root/'finalist-training-plan.json'),'finalist lock changed')
  require(len(confirmation['records'])==4 and set(confirmation['records'])==set(plan['arms']),'four-family confirmation coverage')
  for name,record in confirmation['records'].items():
   arm=plan['arms'][name];out=Path(arm['out']);cfg=read(out/'resolved.json');result=read(out/'result.json');progress=result['progress']
   require(result['status']=='COMPLETE' and progress['updates']==2048 and progress['supervised_tokens']==30720 and progress['cursor']==foundation['cursor']+2048 and progress['order']==foundation['order'],'finalist matched exposure/order failure '+name)
   require(digest(cfg)==arm['resolved_hash'] and cfg['parent']==plan['parent'] and cfg['data']==plan['data'],'finalist matched parent/config failure '+name)
   for ref in (record['model'],record['checkpoint'],arm['config'],arm['resume']):require(pin(ref['path'])==ref,'finalist artifact changed '+ref['path'])
   finalists.append(dict(arm=name,updates=2048,supervised_positions=30720,cursor=progress['cursor'],checkpoint=record['checkpoint'],model=record['model'],reuse=arm['reuse_exact_matching_2048']))
 return dict(status='CAMPAIGN_CHECKPOINT_AUDITED',panels=panels,physical_training_updates=physical_updates,physical_training_update_seconds=update_seconds,supervised_positions=15*physical_updates,training=training,matched=matched,finalists=finalists,scope='Qualification and research updates are itemized; preserved vanilla training is separate. No phase extrapolation or missing-measurement zeros.')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();result=campaign(a.root);save(a.out,result);print(json.dumps({k:result[k] for k in ('status','physical_training_updates','physical_training_update_seconds')},indent=2))
