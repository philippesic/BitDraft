import sys,json
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';S=R/'source-v2';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01');sys.path.insert(0,str(S))
from runner import atomic,pin,read
freeze=read(P/'runs/design-readiness-20261104-01/freeze.json');freeze['workflow_qualification']='Bounded immutable audit snapshots and equivalent F32 shards; original objective/data/order unchanged.'
freeze['sources'] += [pin(S/name) for name in ['runner.py','log_stream.py','snapshot_audit.py','sharded_rows.py','optimized_current.py']]+[pin(R/'shards/manifest.json')]
atomic(R/'optimized-freeze.json',freeze)
py=str(P/'.venv/bin/python');plan=dict(schema='contender_plan_v1',budget_seconds=7200,candidate_budgets={'reference':7200},grace_seconds=180,prior_state=pin(R/'execution/state.json'),gpu=dict(control=str(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'),lock=str(P/'runs/contender-gpu.lock')),inputs=[pin(R/'optimized-freeze.json')]+[pin(S/n) for n in ['runner.py','log_stream.py','snapshot_audit.py','sharded_rows.py','optimized_current.py','current_adapter.py']],jobs=[])
def job(id,argv,result,status,timeout=900):
 jobs=plan['jobs'];jobs.append(dict(id=id,candidate='reference',stage=0,depends_on=[jobs[-1]['id']] if jobs else [],timeout_seconds=timeout,argv=argv,cwd=str(P),checks=[dict(path=str(result),keys=['status'],equals=status)]))
job('optimized', [py,str(S/'optimized_current.py'),'--project',str(P),'--shards',str(R/'shards/manifest.json'),'--recipe','control','--final-updates','5808','--out',str(C/'optimized'),'--freeze',str(R/'optimized-freeze.json'),'--allow-cuda'],C/'optimized/result.json','TRAINED_EXPORTED_NOT_NATIVE_ADMITTED')
job('optimized_exact',[py,str(P/'scripts/design_readiness/checkpoint_compare.py'),str(C/'continuous/checkpoint-00005808.pt'),str(C/'optimized/checkpoint-00005808.pt'),'--fields','cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure','--out',str(R/'optimized-exact.json')],R/'optimized-exact.json','DECLARED_FIELDS_BYTE_EXACT',120)
job('optimized_native',[py,str(S/'current_adapter.py'),'benchmark','--project',str(P),'--output',str(R/'optimized-native'),'--models',str(P/'runs/design-readiness-20261104-01/models-stage2-quiet.json'),'--inputs',str(P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json'),'--results',str(C/'optimized/result.json'),'--tier','smoke'],R/'optimized-native/result.json','SMOKE_PASS_NO_RANKING',600)
job('native_retry',[py,str(S/'current_adapter.py'),'benchmark','--project',str(P),'--output',str(R/'native-retry'),'--models',str(P/'runs/design-readiness-20261104-01/models-stage2-quiet.json'),'--inputs',str(P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json'),'--results',str(C/'continuous/result.json'),str(C/'resumed/result.json')],R/'native-retry/result.json','CONTENDER_PANEL_AUDITED',1800)
atomic(R/'optimized-plan.json',plan)
