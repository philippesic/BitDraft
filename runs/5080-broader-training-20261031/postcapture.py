from pathlib import Path
import os,time,json,subprocess,fcntl,importlib.util,hashlib,collections
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
for _ in range(300):
 q=R/'supervisor/runs/capture01/state.json'
 if q.exists() and json.loads(q.read_text())['status']!='running':break
 time.sleep(4)
s=json.loads(q.read_text());assert s.get('exit_code')==0,s
f=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
rows=[]
for label,env in [('detail',{}),('detail-no-gdn-mma',{'LD_PRELOAD':str(R/'disable-gdn-mma.so')})]:
 cmd=[str(R/'test-detail'),'-m',str(P/'runs/spec-parity-gpu-20261031/models/Qwen3.5-4B-Q4_K_M.gguf'),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(R/label)]
 with (R/(label+'.log')).open('x') as out:x=subprocess.run(cmd,env={**os.environ,**env},stdout=out,stderr=subprocess.STDOUT,timeout=300)
 rows.append({'label':label,'rc':x.returncode,'env':env});print('diagnostic',rows[-1],flush=True)
(R/'detail-results.json').write_text(json.dumps(rows,indent=2));f.close()
spec=importlib.util.spec_from_file_location('serving_data',R/'serving-data.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
d=data.build(W);fit=[w for w in d['windows'] if w['role']=='fit'];dev=[w for w in d['windows'] if w['role']=='development'];groups=collections.defaultdict(list)
for i,w in enumerate(fit):groups[w['group_id']].append(i)
assert len(groups)==240 and all(len(x)==4 for x in groups.values()),'capture must retain four valid windows per training group'
assert len({w['group_id'] for w in dev})==48 and len(dev)==192
# Domain-balanced pair order; disjoint group pairs, every window once per pass.
bydomain={d:[] for d in ('code','prose','reasoning')}
for g,indices in groups.items():bydomain[fit[indices[0]]['domain']].append(indices)
pairs=[]
for g in range(0,80,2):
 for j in range(4):
  for domain in ('code','prose','reasoning'):pairs.append([bydomain[domain][g][j],bydomain[domain][g+1][j]])
assert sorted(i for pair in pairs for i in pair)==list(range(960))
(W/'training-order.json').write_text(json.dumps({'pairs':pairs,'policy':'fixed domain-balanced adjacent-group pairs; each960fit windows exactly once per480-update pass'},indent=2))
import numpy as np
teacher=W/'teacher';teacher.mkdir(exist_ok=False);rr=[]
for i,w in enumerate(fit):
 raw=np.fromfile(w['teacher']['path'],dtype='<f4').reshape(7,151936);dest=teacher/f'{i:04d}.f32';raw[:5].tofile(dest);rr.append({'index':i,'labels':w['labels'][:5],'logits':data.pin(dest),'shape':[5,151936]})
m={'inputs':data.pin(W/'data/manifest.json'),'order':data.pin(W/'training-order.json'),'target_sha256':'05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57ccf2dc720a6','target_precision':'F16','target_kv_type':'F16','pairs':pairs,'producer_closed':True,'rows':rr,'source':'unchanged original4B F16 actual serving verifier; reached retained rows; inheritedTRAIN','capture':d['capture']}
# Explicit exact target identity, copied from the previously authenticated protocol.
m['target_sha256']=json.loads((P/'runs/unattended-training-preparation-20261028/gpu-resume-20261028/campaign-control124-config.json').read_text())['spec']['scientific_protocol']['target_sha256']
(teacher/'manifest.json').write_text(json.dumps(m,indent=2));(W/'data-ready.json').write_text(json.dumps({'manifest':data.pin(W/'data/manifest.json'),'order':data.pin(W/'training-order.json'),'teacher':data.pin(teacher/'manifest.json'),'fit_windows':len(fit),'development_windows':len(dev),'total_groups':len(groups)+48},indent=2));print('DATA_READY',len(fit),len(dev),flush=True)
