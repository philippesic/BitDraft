from pathlib import Path
import os,json,time,subprocess,fcntl
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';O=P/'runs/spec-parity-gpu-20261031'
for _ in range(600):
 if (R/'detail-results.json').exists():break
 time.sleep(2)
assert (R/'detail-results.json').exists()
f=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
results=[]
for test,model in [('ub1','Qwen3.5-4B-Q4_K_M.gguf'),('ub1','qwen35-4b-mtp-nvfp4.gguf'),('trace','Qwen3.5-4B-Q4_K_M.gguf')]:
 label=test+'-'+model;cmd=[str(R/('test-'+test)),'-m',str(O/'models'/model),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(R/label)]
 with (R/(label+'.log')).open('x') as out:x=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=300)
 results.append({'test':test,'model':model,'rc':x.returncode});print(results[-1],flush=True)
f.close();(R/'extra-diagnostic-complete.json').write_text(json.dumps(results,indent=2))
