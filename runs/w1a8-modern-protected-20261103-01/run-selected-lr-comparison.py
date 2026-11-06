from pathlib import Path
import os,json,subprocess,signal,time,hashlib,sys
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');trainer=W/'scripts/w1a8_generalization/modern_train_selected_sign_lr.py';parent=C/'w1a8-selected-sign-learn3840-01/checkpoint-00003840.pt';start=time.monotonic();stopping=False;records=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
plan=json.loads((R/'selected-lr-execution-plan.json').read_text());assert pin(trainer)==plan['trainer'];env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
for rate,name in [('0.001','w1a8-selected-sign-lr1e3-4320-01'),('0.003','w1a8-selected-sign-lr3e3-4320-01')]:
 assert not stopping;out=C/name;assert not out.exists();t=time.monotonic();argv=[str(P/'.venv/bin/python'),str(trainer),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--warm-selected',str(parent),'--out',str(out),'--sign-lr',rate,'--steps','4320','--allow-cuda'];print(json.dumps({'sign_lr':rate,'status':'starting','out':str(out)}),flush=True)
 with (R/(name+'.log')).open('x') as log:
  child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc',str(child.pid));write(R/(name+'-identity.json'),{'pid':child.pid,'pgid':os.getpgid(child.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()});code=child.wait()
 assert code==0,(rate,code);assert pin(trainer)==plan['trainer'];record={'sign_lr':rate,'exit_code':code,'wall_seconds':time.monotonic()-t,'out':str(out)}
 if rate=='0.001':
  import torch
  torch.set_num_threads(2);sys.path.insert(0,str(W/'scripts/w1a8_generalization'));from modern_train_selected_sign import checkpoint_equal
  actual=torch.load(out/'checkpoint-00004320.pt',map_location='cpu',weights_only=False);refroot=C/'w1a8-selected-sign-lr-control4320-01';expected=torch.load(refroot/'checkpoint-00004320.pt',map_location='cpu',weights_only=False);checks={k:checkpoint_equal(actual[k],expected[k]) for k in ['cursor','rng','cuda_rng','python_rng','exposure','selected_state','optimizer']};checks['nativeGGUF']=pin(out/'model.gguf')['sha256']==pin(refroot/'model.gguf')['sha256']
  for split in ['fit','development']:
   ar=json.loads((out/('after-'+split+'.json')).read_text())['rows'];er=json.loads((refroot/('after-'+split+'.json')).read_text())['rows'];keys=['id','role','native_prefix','actor_prefix','proposals','labels','supported','protected'];checks[split+'_decisions']=len(ar)==len(er) and all({k:x[k] for k in keys}=={k:y[k] for k in keys} for x,y in zip(ar,er))
  write(R/'selected-lr-control-parity.json',{'checks':checks,'all_exact':all(checks.values()),'new_migration':pin(out/'checkpoint-00004320.pt'),'original_strict_resume':pin(refroot/'checkpoint-00004320.pt')});assert all(checks.values()),checks;record['all_control_exact']=True;del actual,expected
 records.append(record);write(R/'selected-lr-progress.json',{'stages':records,'wall_seconds':time.monotonic()-start});print(json.dumps(record),flush=True)
write(R/'selected-lr-result.json',{'status':'MATCHED_MATURE_SIGN_LR_TRAINING_COMPLETE_NOT_NATIVE_QUALIFIED','stages':records,'wall_seconds':time.monotonic()-start,'scope':'capacitydiagnosis,notvalidatedscreenpredictor orbacklogscreen; nofreshDEVtraininglabels'})
