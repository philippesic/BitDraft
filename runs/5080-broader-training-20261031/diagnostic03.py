from pathlib import Path
import json,os,sys,time,subprocess,fcntl,resource,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');O=P/'runs/spec-parity-gpu-20261031'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
while True:
 try:s=json.loads((W/'supervisors/qualification04/state.json').read_text())
 except (OSError,ValueError):time.sleep(3);continue
 if s['status']!='running':break
 time.sleep(3)
assert s['exit_code']==0,('qualification did not finish',s)
k=json.loads((R/'interop-keeper.json').read_text());assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop']
rt=Path.home()/'.local/share/p-stack/runtime.json';d=json.loads(rt.read_text());sys.path.insert(0,str(rt.parent/'runtime'/d['source_digest']));os.environ['P_STACK_BROKER_COMMAND']=json.dumps(d['bridge']);from devicecheckout.project import before_costly_work
before_costly_work('benchmark_quiet')
f=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
results=[]
for name,binary,model,mods in [('trace-all','test-trace-all','Qwen3.5-4B-Q4_K_M.gguf',{}),('fa-q4','test-detail','Qwen3.5-4B-Q4_K_M.gguf',{'LD_PRELOAD':str(R/'fa-parity02.so'),'EAGLE_PARITY_FA_SERIAL':'1'}),('fa-mmv-q4','test-detail','Qwen3.5-4B-Q4_K_M.gguf',{'LD_PRELOAD':str(R/'fa-parity02.so')+':'+str(R/'mat-parity.so'),'EAGLE_PARITY_FA_SERIAL':'1','EAGLE_PARITY_MMVQ_SERIAL':'1'}),('fa-mmv-fp4','test-detail','qwen35-4b-mtp-nvfp4.gguf',{'LD_PRELOAD':str(R/'fa-parity02.so')+':'+str(R/'mat-parity.so'),'EAGLE_PARITY_FA_SERIAL':'1','EAGLE_PARITY_MMVQ_SERIAL':'1'})]:
 env={**os.environ,'GGML_CUDA_DISABLE_FUSION':'1',**mods};cmd=[str(R/binary),'-m',str(O/'models'/model),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(R/('diag03-'+name))]
 with (R/('diag03-'+name+'.log')).open('x') as log:x=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 result={'name':name,'returncode':x.returncode,'command':cmd,'environment':{k:env[k] for k in ['GGML_CUDA_DISABLE_FUSION',*mods]},'end':time.time()};results.append(result);print(json.dumps(result),flush=True)
(R/'diagnostic03-results.json').write_text(json.dumps(results,indent=2));f.close()
