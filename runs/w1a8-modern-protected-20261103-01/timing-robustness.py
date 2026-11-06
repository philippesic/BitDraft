from pathlib import Path
import json,numpy as np
R=Path('runs/w1a8-modern-protected-20261103-01');F=R/'fresh-quarter-confirmation01'
for scope,resultpath,cand,control in [('fresh',F/'serving01/result.json','w1a8_quarter','w1a8_initial'),('reachability',R/'reachability-requests02/result.json','w1a8_union3840','w1a8_nativefit3840')]:
 d=json.loads(resultpath.read_text());rows=[r for r in d['records'] if not r['warmup'] and (scope!='fresh' or r['phase']=='performance')]
 meta={r['id']:r['family'] for r in json.loads((F/'inputs01/requests.json').read_text())} if scope=='fresh' else {r['id']:r['id'] for r in rows}
 groups=sorted(set(meta.values()));arr={a:np.array([[sum(r['full_http_wall_seconds'] for r in rows if r['arm']==a and r['repetition']==rep and meta[r['id']]==g) for g in groups] for rep in range(5)]) for a in [cand,control]};rng=np.random.default_rng(20261103);g=rng.integers(0,len(groups),size=(20000,len(groups)));rep=rng.integers(0,5,size=(20000,5));draw={a:t[rep[:,:,None],g[:,None,:]].sum((1,2)) for a,t in arr.items()};ratios=draw[control]/draw[cand]
 out={'scope':scope,'candidate':cand,'control':control,'groups':len(groups),'matched_repetitions':5,'pooled_ratio':float(arr[control].sum()/arr[cand].sum()),'paired_group_and_repetition_bootstrap95_interval':np.quantile(ratios,[.025,.975]).tolist(),'ratio_per_repetition':(arr[control].sum(1)/arr[cand].sum(1)).tolist(),'method':'paired resampling of complete group and complete repetition blocks,20000 draws; retain all measurements'}
 if scope=='reachability':
  out['native_vs_HTTP_components']={a:[{'repetition':rep,'http_wall':sum(r['full_http_wall_seconds'] for r in rows if r['arm']==a and r['repetition']==rep),'native_prompt_plus_decode_seconds':sum((r['timings']['prompt_ms']+r['timings']['predicted_ms'])/1000 for r in rows if r['arm']==a and r['repetition']==rep)} for rep in range(5)] for a in [cand,control]};dest=R/'reachability-timing-robustness.json'
 else:dest=F/'timing-robustness.json'
 dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
