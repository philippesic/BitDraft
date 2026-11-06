from pathlib import Path
import os,json,hashlib,shutil,subprocess,shlex
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';S=R/'source02';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
assert not S.exists();shutil.copytree(R/'source01',S,ignore=shutil.ignore_patterns('__pycache__'))
p=S/'scripts/training_system/train.py';s=p.read_text();a="    if type(spec.get('data_cache_enabled', True)) is not bool:\n        raise ValueError('explicit boolean data cache choice required')\n";assert s.count(a)==1
s=s.replace(a,a+"    if spec.get('training_manifest'):\n        if spec.get('data_adapter', 'serving') != 'serving':\n            raise ValueError('separate training manifest requires serving data')\n        authenticate(spec['training_manifest'])\n")
a="ServingData(spec['manifest'], spec['order'], cached=";assert s.count(a)==1;s=s.replace(a,"ServingData(spec.get('training_manifest', spec['manifest']), spec['order'], cached=");p.write_text(s)
m=json.loads((S/'manifest.json').read_text());m['base_manifest']=pin(R/'source01/manifest.json');m['change']='Serving trainer can authenticate a distinct full training manifest and fixed evaluation manifest; uncached optimized numerical path; fresh user authority only. Actual CUDA continuation and native qualification required.'
for f in m['files']:
 p=S/f['path'];f.update(bytes=p.stat().st_size,sha256=sha(p))
write(S/'manifest.json',m)
d=json.loads((W/'data/manifest.json').read_text());groups={}
for w in d['windows']:
 if w['role']=='fit':groups.setdefault(w['domain'],[]);groups[w['domain']]+= [w['group_id']] if w['group_id'] not in groups[w['domain']] else []
selected={g for v in groups.values() for g in v[:8]};windows=[{**w,'source_window_index':i,'case_index':0} for i,w in enumerate(d['windows']) if w['role']=='development' or w['group_id'] in selected]
for i,w in enumerate(windows):w['case_index']=i
assert len(windows)==288
coverage={role:{'windows':len([w for w in windows if w['role']==role]),'groups':len({w['group_id'] for w in windows if w['role']==role})} for role in ['fit','development']}
ev={**d,'windows':windows,'coverage':coverage,'full_training_manifest':pin(W/'data/manifest.json'),'evaluation_policy':'Fixed before candidate updates: all192development windows plus first8 fit groups per domain (96windows); inherited TRAIN ancestry, no unseen claim.'};write(W/'evaluation-manifest02.json',ev)
cfg=json.loads((W/'qualification-config.json').read_text());spec=cfg['spec']
def rebase(v):
 if isinstance(v,dict):
  if set(v)>={'path','sha256'} and v['path'].startswith(str(R/'source01')+'/'):return {**v,**pin(S/Path(v['path']).relative_to(R/'source01'))}
  return {k:rebase(x) for k,x in v.items()}
 if isinstance(v,list):return [rebase(x) for x in v]
 return v
spec=rebase(spec);spec.update(training_manifest=pin(W/'data/manifest.json'),manifest=pin(W/'evaluation-manifest02.json'),checkpoint_root=str(W/'qualification-checkpoints03'),job_tag='broader-serving-source02-qualified')
write(W/'base-spec03.json',spec)
g=json.loads((W/'guard.json').read_text());k=json.loads((R/'interop-keeper.json').read_text());assert Path(k['interop']).exists();assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==k['boot'];assert Path('/proc/'+str(k['pid'])+'/stat').read_text().split()[21]==k['proc_start']
g['env'].update(PYTHONPATH=str(S/'scripts')+':'+str(P/'scripts'),WSL_INTEROP=k['interop']);g['inputs']=list(spec['sources'].values())+[spec[x] for x in ['manifest','training_manifest','order','teacher','controls','fidelity_gate','model_cache','model_cache_producer','source_checkpoint']]+list(spec['parent'].values());write(W/'guard03.json',g)
cfg.update(cli=str(S/'scripts/training_system/cli.py'),spec=spec,guard_template=str(W/'guard03.json'),run_root=str(W/'qualification-jobs03'));cfg['native']['manifest']=spec['manifest'];write(W/'qualification-config03.json',cfg)
os.environ['WSL_INTEROP']=k['interop']
cmd=[str(P/'.venv/bin/python'),'-B',str(S/'scripts/training_system/qualify.py'),'--config',str(W/'qualification-config03.json'),'--output',str(W/'qualification03')]
launch=['python3',str(R/'supervisor/scripts/remote_job.py'),'qualification03','--output-root',str(W/'supervisors'),'--']+cmd
subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s','broader-qualification03','-c',str(P),shlex.join(launch)],check=True)
print(json.dumps({'status':'QUALIFICATION_LAUNCHED','source':pin(S/'manifest.json'),'evaluation':coverage,'training_windows':len(d['windows'])}))
