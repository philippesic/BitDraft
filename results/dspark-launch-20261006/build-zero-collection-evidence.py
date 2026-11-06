import json,hashlib
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');OLD=R/'runs/checkouts/dspark-full-training-20261006-01';OUT=R/'runs/dspark-a8-full-pool-20261006-01';D=R/'results/dspark-launch-20261006/zero-native-repair1'
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
plan=json.loads((OUT/'timed-plan-v3/timed-evaluation-plan.json').read_text());status=json.loads((D.parent/'zero-native-repair1-status.json').read_text());assert status['measurement_complete'] and status['clean_records']==480 and status['diagnostic_records']==96 and status['strict_quality_gate']=='FAIL' and status['optimizer_updates']==0
keys=['candidate','target','target_policy','runtime','protocol','prompts','gpu_uuid'];bindings={k:plan[k] for k in keys};bindings.update(initial_model=plan['initial']['model'],initial_audit=plan['initial']['audit'],q4_model=plan['control']['model'])
models={plan['candidate']:plan['initial']['model'],'initial':plan['initial']['model'],'eagle_q4':plan['control']['model']};stages=[]
for rep,diag in [(r,False) for r in range(5)]+[(0,True)]:
 for cell in [plan['candidate'],'initial','eagle_q4','target_only']:
  stage=D/('diagnostic' if diag else f'rep-{rep:02d}')/cell
  v={'cell':cell,'repetition':rep,'diagnostic':diag,'process':pin(stage/'process.json'),'lineage':pin(stage/'process-lineage.json'),'server_log':pin(stage/'server.log'),'resource_return':pin(stage/'resource-return.json'),'model':plan['target'] if cell=='target_only' else models[cell]}
  if diag and cell in [plan['candidate'],'initial']:v.update(dispatch=pin(stage/'actual-dispatch.json'),export_audit=plan['initial']['audit'])
  stages.append(v)
v={'schema':'native_zero_measurement_evidence_v1','progress':pin(D/'progress.json'),'bindings':bindings,'hardware':status['resource_return'],'failed_status':pin(D.parent/'zero-native-repair1-status.json'),'zero_preparation':pin(OUT/'packet/initial-prepare-receipt.json'),'model_ancestry':models,'stages':stages,'producer_sources':{name:pin(OLD/name) for name in ['scripts/evaluate_nine_model_timed_checkpoint.py','scripts/benchmark_native_eagle.py','src/w1a1_eagle/nine_model_pipeline.py']},'original_supervisor_state':pin(OLD/'runs/dspark-a8-zero-native-20261006-02/state.json')}
p=D.parent/'zero-native-collection-evidence-v1.json';assert not p.exists();p.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');print(json.dumps({'evidence':pin(p),'progress':v['progress'],'stage_count':len(stages),'native_quality':'FAILED','continuation_authorized':False}))
