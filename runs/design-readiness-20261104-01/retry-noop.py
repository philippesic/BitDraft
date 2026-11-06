from pathlib import Path
import subprocess,json,time,os,sys
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';C=Path(json.loads((R/'freeze.json').read_text())['output']);t=time.monotonic()
argv=[str(P/'.venv/bin/python'),str(P/'scripts/design_readiness/fixed_objective.py'),'--recipe','control','--final-updates','5808','--out',str(C/'noop-new02'),'--freeze',str(R/'freeze.json'),'--allow-cuda']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
child=subprocess.Popen(argv,cwd=P,env=env);p=Path('/proc')/str(child.pid);(R/'noop-migrated02-identity.json').write_text(json.dumps(dict(pid=child.pid,pgid=os.getpgid(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),argv=argv),indent=2)+'\n');code=child.wait();assert code==0,code
import torch
sys.path.insert(0,str(P/'scripts/w1a8_generalization'));from modern_train_protected import equal
old=torch.load(C/'noop-original/checkpoint-00005808.pt',map_location='cpu',weights_only=False);new=torch.load(C/'noop-new02/checkpoint-00005808.pt',map_location='cpu',weights_only=False)
checks={key:equal(old[key],new[key]) for key in ['cursor','scales','optimizer','rng','cuda_rng','python_rng']};result=dict(status='NOOP48_EXACT' if all(checks.values()) else 'NOOP48_MISMATCH',checks=checks,wall_seconds=time.monotonic()-t,original_physical_updates=48,migrated_physical_updates=48,first_failure='Root module path rejected by exact union source identity before updates; retry reuses admitted pinned source checkout.')
(R/'noop-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert all(checks.values()),checks
