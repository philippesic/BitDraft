from pathlib import Path
import subprocess,json,os,time,signal,hashlib
P=Path('/home/philip/binary-eagle-decoding'); W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103'); R=P/'runs/w1a8-modern-protected-20261103-01'; O=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-protected-20261103-01')
started=time.monotonic(); results=[]; stopping=False
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def stop(signum,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
O.mkdir(parents=True,exist_ok=False)
for name,scope,coefficient in [('all-ce01','all',0),('all-replay01','all',10),('head-replay01','head',10)]:
 control=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']
 assert not stopping and not control.get('pause_requested') and not (R/'STOP').exists()
 argv=[str(P/'.venv/bin/python'),str(W/'scripts/w1a8_generalization/modern_train_protected.py'),'--manifest',str(P/'runs/w1a8-modern-actual-training-data-20261103-01/final01/manifest.json'),'--reference',str(P/'runs/w1a8-modern-actual-training-20261103-01/reference02/reference.json'),'--original-trainer',str(W/'scripts/w1a8_generalization/modern_train.py'),'--out',str(O/name),'--steps','1920','--retention',str(coefficient),'--scope',scope,'--allow-cuda']
 (R/(name+'-command.json')).write_text(json.dumps({'argv':argv,'environment':{k:os.environ[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_CACHE_PATH','CUDA_CACHE_MAXSIZE','CUBLAS_WORKSPACE_CONFIG']},'scope':'Matched1920 checkpoint; continuable horizon19200; fit-only protected replay'},indent=2)+'\n')
 process=subprocess.Popen(argv); q=Path('/proc')/str(process.pid)
 (R/(name+'-identity.json')).write_text(json.dumps({'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n')
 code=process.wait()
 if code: raise RuntimeError((name,code))
 result=json.loads((O/name/'result.json').read_text()); results.append(result)
 print(json.dumps({'arm':name,'status':result['status'],'updates':result['updates'],'trainer_seconds':result['trainer_seconds']}),flush=True)
 (R/'progress.json').write_text(json.dumps({'completed_results':results,'wall_seconds':time.monotonic()-started},indent=2)+'\n')
(R/'result.json').write_text(json.dumps({'status':'MATCHED_PROTECTED_TRIPLE_TRAINED_NATIVE_PENDING','results':results,'wall_seconds':time.monotonic()-started},indent=2)+'\n')
