import sys,json
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01');S=R/'source';sys.path.insert(0,str(S))
from runner import pin,atomic
py=str(P/'.venv/bin/python');freeze=P/'runs/design-readiness-20261104-01/freeze.json'
plan=dict(schema='contender_plan_v1',budget_seconds=5400,candidate_budgets={'reference':5400},grace_seconds=180,gpu=dict(control=str(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'),lock=str(P/'runs/contender-gpu.lock')),inputs=[pin(p) for p in [S/'runner.py',S/'current_adapter.py',freeze,P/'scripts/design_readiness/fixed_objective.py']],jobs=[])
def add(id,stage,argv,result,status,timeout=900,extra=None):
 jobs=plan['jobs'];job=dict(id=id,candidate='reference',stage=stage,depends_on=[jobs[-1]['id']] if jobs else [],timeout_seconds=timeout,argv=argv,cwd=str(P),checks=[dict(path=str(result),keys=['status'],equals=status)])
 if extra:job.update(extra)
 jobs.append(job)
add('continuous',0,[py,str(P/'scripts/design_readiness/fixed_objective.py'),'--recipe','control','--final-updates','5808','--out',str(C/'continuous'),'--freeze',str(freeze),'--allow-cuda'],C/'continuous/result.json','TRAINED_EXPORTED_NOT_NATIVE_ADMITTED')
common=[py,str(S/'current_adapter.py')]
add('interrupt',1,common+['interrupt','--project',str(P),'--output',str(R/'interruption'),'--training-output',str(C/'interrupted'),'--freeze',str(freeze)],R/'interruption/result.json','INTERRUPTION_CHECKPOINT_SAVED')
add('resume',1,common+['resume','--project',str(P),'--output',str(R/'resume'),'--training-output',str(C/'resumed'),'--freeze',str(freeze),'--interruption',str(R/'interruption/result.json')],R/'resume/result.json','RESUMED_AND_EXPORTED')
add('exact',2,[py,str(P/'scripts/design_readiness/checkpoint_compare.py'),str(C/'continuous/checkpoint-00005808.pt'),str(C/'resumed/checkpoint-00005808.pt'),'--fields','contract','cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure','--out',str(R/'exact.json')],R/'exact.json','DECLARED_FIELDS_BYTE_EXACT',120)
add('native',2,common+['benchmark','--project',str(P),'--output',str(R/'native'),'--models',str(P/'runs/design-readiness-20261104-01/models-stage2-quiet.json'),'--inputs',str(P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json'),'--results',str(C/'continuous/result.json'),str(C/'resumed/result.json')],R/'native/result.json','CONTENDER_PANEL_AUDITED',1800)
atomic(R/'plan.json',plan)
