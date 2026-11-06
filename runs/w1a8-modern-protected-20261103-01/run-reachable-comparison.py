from pathlib import Path
import os,json,subprocess,signal,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01');trainer=W/'scripts/w1a8_generalization/modern_train_reachable.py';stopping=False

def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h,'bytes':p.stat().st_size}
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
plan=json.loads((R/'reachable-plan.json').read_text());assert pin(trainer)==plan['trainer']
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',CUBLAS_WORKSPACE_CONFIG=':4096:8');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
start=time.monotonic();results=[]
for mode,name in [('static','reachability-static3840-01'),('current-prefix-union','reachability-union3840-01')]:
 assert not stopping
 out=O/name;assert not out.exists();argv=[str(P/'.venv/bin/python'),str(trainer),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--warm-start',str(O/'all-replay01/checkpoint-00001920.pt'),'--ce-reachability',mode,'--scope','all','--retention','10','--steps','3840','--out',str(out),'--allow-cuda']
 print(json.dumps({'stage':mode,'status':'starting','out':str(out)}),flush=True);t=time.monotonic()
 with (R/(name+'.log')).open('x') as log:
  process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(process.pid)
  write(R/(name+'-identity.json'),{'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()});code=process.wait()
 assert code==0,(mode,code);assert pin(trainer)==plan['trainer'];result={'stage':mode,'exit_code':code,'wall_seconds':time.monotonic()-t,'output':str(out)}
 if mode=='static':
  import torch
  torch.set_num_threads(2)
  from sys import path
  path.insert(0,str(W/'scripts/w1a8_generalization'));from modern_train_protected import equal
  actual=torch.load(out/'checkpoint-00003840.pt',map_location='cpu',weights_only=False);expected=torch.load(O/'nativefit3840-01/checkpoint-00003840.pt',map_location='cpu',weights_only=False)
  fields=['cursor','scales','optimizer','rng','cuda_rng','python_rng','exposure'];checks={k:equal(actual[k],expected[k]) for k in fields};assert all(checks.values()),checks
  assert pin(out/'model.gguf')['sha256']==pin(O/'nativefit3840-01/model.gguf')['sha256'];result['exact_control_parity']=checks;result['native_export_equal']=True;write(R/'reachable-control-exact-parity.json',result)
 results.append(result);write(R/'reachable-progress.json',{'completed':results,'wall_seconds':time.monotonic()-start});print(json.dumps(result),flush=True)
write(R/'reachable-result.json',{'status':'MATCHED_REACHABILITY_TRAINING_COMPLETE','results':results,'wall_seconds':time.monotonic()-start,'fresh_cohort_usage':'none; old48fresh instances are consumed confirmation and cannot be reused as untouched candidate selection/test'})
