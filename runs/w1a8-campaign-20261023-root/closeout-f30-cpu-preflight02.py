from pathlib import Path
import subprocess,resource,time,json,os
start=time.monotonic();cpu=time.process_time();u=resource.getrusage(resource.RUSAGE_CHILDREN)
P=Path('/home/philip/binary-eagle-decoding');O=P/'runs/w1a8-campaign-20261023-root';S=O/'closeout-f30-runtime02-source';T=O/'closeout-f30-runtime02-trainer';out=O/'closeout-f30-cpu-preflight02';out.mkdir()
r=subprocess.run([str(P/'.venv/bin/python'),'-B',str(S/'run.py'),'--cpu-preflight','--trainer-source',str(T)],capture_output=True,text=True,timeout=180,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
(out/'stdout.txt').write_text(r.stdout);(out/'stderr.txt').write_text(r.stderr);v=json.loads(r.stdout)if r.returncode==0 else None;z=resource.getrusage(resource.RUSAGE_CHILDREN);cost={'CPU_seconds':time.process_time()-cpu+z.ru_utime+z.ru_stime-u.ru_utime-u.ru_stime,'whole_seconds':time.monotonic()-start}
ok=r.returncode==0 and v['status']=='F30_RUNTIME02_CPU_PREFLIGHT_PASS'and v['GPU_calls']==0 and v['real_model_reads']==0 and cost['CPU_seconds']<120 and cost['whole_seconds']<180
(out/'result.json').write_text(json.dumps({'status':'ROOT_PC_F30_CPU_PREFLIGHT_PASS'if ok else 'INCOMPLETE','returncode':r.returncode,'GPU_calls':0,'real_model_reads':0,'cost':cost,'preflight':v},indent=2)+'\n')
raise SystemExit(0 if ok else 1)
