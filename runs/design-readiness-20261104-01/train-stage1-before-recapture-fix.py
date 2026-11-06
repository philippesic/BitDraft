from pathlib import Path
import subprocess,json,time,os,sys
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';C=Path(json.loads((R/'freeze.json').read_text())['output']);PY=P/'.venv/bin/python'
assert json.loads((R/'noop-result.json').read_text())['status']=='NOOP48_EXACT'
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-gpu-training-qualification-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
start=time.monotonic();stages=[]
def run(name,argv):
 t=time.monotonic();child=subprocess.Popen(argv,cwd=P,env=env);p=Path('/proc')/str(child.pid);(R/(name+'-identity.json')).write_text(json.dumps(dict(pid=child.pid,pgid=os.getpgid(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),argv=argv),indent=2)+'\n');code=child.wait();row=dict(stage=name,exit_code=code,wall_seconds=time.monotonic()-t,argv=argv);stages.append(row);(R/'stage1-processes.json').write_text(json.dumps(stages,indent=2)+'\n');assert code==0,(name,code);return row
run('capture-two-fit',[str(PY),str(P/'scripts/design_readiness/capture_probe.py'),'--output',str(R/'capture-probe01'),'--allow-cuda'])
for recipe in ['control','depth','smooth','combined']:
 out=C/(recipe+'6720')
 argv=[str(PY),str(P/'scripts/design_readiness/fixed_objective.py'),'--recipe',recipe,'--final-updates','6720','--out',str(out),'--freeze',str(R/'freeze.json'),'--allow-cuda']
 run(recipe+'6720',argv)
 result=json.loads((out/'result.json').read_text());assert result['cursor']==6720 and sum(result['changed_scale_elements'].values())>0
 (R/'stage1-progress.json').write_text(json.dumps(dict(stages=stages,latest=recipe,wall_seconds=time.monotonic()-start),indent=2)+'\n')
run('split-resume',[str(PY),str(P/'scripts/design_readiness/fixed_objective.py'),'--recipe','control','--final-updates','6720','--resume',str(C/'control6720/checkpoint-00006240.pt'),'--out',str(C/'control-split6720'),'--freeze',str(R/'freeze.json'),'--allow-cuda'])
import torch
sys.path.insert(0,str(P/'scripts/w1a8_generalization'));from modern_train_protected import equal
left=torch.load(C/'control6720/checkpoint-00006720.pt',map_location='cpu',weights_only=False);right=torch.load(C/'control-split6720/checkpoint-00006720.pt',map_location='cpu',weights_only=False)
checks={key:equal(left[key],right[key]) for key in ['cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure']}
proof=dict(status='SPLIT480_RESUME_EXACT' if all(checks.values()) else 'SPLIT_RESUME_MISMATCH',checks=checks,scope='Exact current objective schema and actual480 resumed updates. Wall and allocator/cache counters deliberately not equated.')
(R/'split-resume-result.json').write_text(json.dumps(proof,indent=2)+'\n');assert all(checks.values()),checks
(R/'stage1-result.json').write_text(json.dumps(dict(status='CAPTURE_TRAINING_AND_SPLIT_RESUME_PASS_NOT_NATIVE_ADMITTED',stages=stages,wall_seconds=time.monotonic()-start),indent=2)+'\n');print('STAGE1 COMPLETE',flush=True)
