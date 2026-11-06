from pathlib import Path
import importlib.util,cProfile,pstats,json,sys,time,os
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';C=Path(json.loads((R/'freeze.json').read_text())['output']);source=P/'scripts/design_readiness/fixed_objective.py'
os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',CUBLAS_WORKSPACE_CONFIG=':4096:8')
spec=importlib.util.spec_from_file_location('_current_profile',source);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
sys.argv=[str(source),'--recipe','control','--final-updates','5808','--out',str(C/'profile-control5808'),'--freeze',str(R/'freeze.json'),'--allow-cuda']
profile=cProfile.Profile();start=time.perf_counter();profile.enable();module.main();profile.disable();profile.dump_stats(str(R/'current-control.prof'));elapsed=time.perf_counter()-start
stats=pstats.Stats(profile);rows=[]
for (filename,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():rows.append(dict(file=filename,line=line,function=name,primitive_calls=cc,total_calls=nc,exclusive_seconds=tt,inclusive_seconds=ct))
rows.sort(key=lambda r:r['inclusive_seconds'],reverse=True)
import torch
sys.path.insert(0,str(P/'scripts/design_readiness'));from checkpoint_compare import compare
old=torch.load(C/'noop-new02/checkpoint-00005808.pt',map_location='cpu',weights_only=False);new=torch.load(C/'profile-control5808/checkpoint-00005808.pt',map_location='cpu',weights_only=False);checks=compare(old,new,['contract','cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure']);assert all(checks.values()),checks
result=dict(status='PROFILED_CONTROL_BYTE_EXACT',wall_seconds=elapsed,checks=checks,top_inclusive=rows[:35],scope='One profiled48-update fixed control. Inclusive function times overlap and must not be summed; profiling times are not clean throughput measurements.')
(R/'profile-control-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
