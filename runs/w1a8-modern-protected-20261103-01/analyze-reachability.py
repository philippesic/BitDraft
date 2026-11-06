from pathlib import Path
import json,hashlib,numpy as np
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';Q=R/'reachability-requests02';result=json.loads((Q/'result.json').read_text());records=[r for r in result['records'] if not r['warmup']];assert len(records)==420
sig=lambda r:(r['tokens'],r['content'],r['stop_type'],r['stopping_word'],r['truncated'])
seen={};matrix={};images={}
for row in records:
 raw=json.loads(Path(row['response']['path']).read_text());key=row['id'];s=sig(raw)
 if key in seen:assert s==seen[key]
 else:seen[key]=s
 assert len(raw['tokens'])==row['completion_tokens'];matrix[row['arm'],row['repetition'],row['id']]=row['full_http_wall_seconds']
for cell in sorted(Q.glob('rep-*')):
 release=json.loads((cell/'release.json').read_text());assert release['owned_process_group_absent'] and release['owned_process_reaped'];identity=json.loads((cell/'identity.json').read_text());assert not Path('/proc',str(identity['pid'])).exists()
 if not (cell/'loaded-images-warm.json').exists():continue
 for suffix in ['warm','final']:
  for ref in json.loads((cell/f'loaded-images-{suffix}.json').read_text())['images']:images[ref['path']]=ref
for path,ref in images.items():
 with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref['sha256']
ids=sorted(seen);rng=np.random.default_rng(192037);indices=rng.integers(0,len(ids),size=(20000,len(ids)));sums={arm:np.array([sum(matrix[arm,rep,i] for rep in range(5)) for i in ids]) for arm in {r['arm'] for r in records}};comparisons={}
for arm,controls in {'w1a8_union3840':['w1a8_nativefit3840','w1a8_initial','w1a8_quarter','dflash_Q4KM','target_only','eagle_Q4_0'],'w1a8_nativefit3840':['w1a8_initial','w1a8_quarter','dflash_Q4KM','target_only','eagle_Q4_0']}.items():
 comparisons[arm]={}
 for control in controls:
  draws=sums[control][indices].sum(1)/sums[arm][indices].sum(1);comparisons[arm][control]={'pooled_full_request_speedup':float(sums[control].sum()/sums[arm].sum()),'paired_prompt_95_interval':np.quantile(draws,[.025,.975]).tolist()}
a={'status':'ROOT_420_MATCHED_REACHABILITY_REQUESTS_AUDITED','token_text_stop_repeat_parity':True,'requests':420,'images_rehashed':len(images),'server_releases':len(list(Q.glob('rep-*/release.json'))),'all_raw_measured_responses_audited':420,'incomplete_Q4_diagnostic_responses':0,'comparisons':comparisons,'rates':[{k:x[k] for k in ['arm','tokens_per_second','repetitions']} for x in result['arms']], 'scope':'12reused modern-development groups,5reps;20kbootstrapresamples wholeprompts withallreps; no untouchedquality/release orscreenpredictorclaim'};(R/'native-analysis-reachability.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
