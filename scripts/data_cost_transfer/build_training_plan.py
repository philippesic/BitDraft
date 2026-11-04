"""Freeze the exact bounded data-transfer trial before the first optimizer update."""
from pathlib import Path
import copy,sys
from datetime import datetime,timezone
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/data-cost-transfer-20261104-01';S=Path(__file__).parent
sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,atomic,require
from train import derive

def main():
 bank=R/'registered-bank';q=read(bank/'qualification.json');require(q['status']=='BROADER_TRAIN_STREAM_QUALIFIED' and q['groups']==192,'complete admitted192 bank required')
 require(not (R/'training-freeze.json').exists(),'training already frozen')
 source=P/'scripts/design_readiness/fixed_objective.py';derived=R/'fixed_data_trainer.py';require(derived.read_text()==derive(source.read_text()),'trainer derivation changed')
 oldshards=P/'runs/contender-infra-20261104-01/shards/manifest.json'
 base=read(P/'runs/design-readiness-20261104-01/freeze.json');f={k:copy.deepcopy(base[k]) for k in ('parent','sources','hardware','candidate')};f.update(schema='fixed_data_transfer_freeze_v1',frozen_at=datetime.now(timezone.utc).isoformat(),original_training_freeze=pin(P/'runs/design-readiness-20261104-01/freeze.json'),architecture_precision_held_fixed=True);f['recipes']={'control':{'depth_weights':[1.,1.],'label_smoothing':0.}}
 refs=[S/'train.py',S/'training_bank.py',derived,oldshards,bank/'registry.json',bank/'shards.json',bank/'qualification.json',R/'training-recipe.json',R/'train-exposure.json']+[P/'scripts/contender_pipeline'/n for n in ['runner.py','snapshot_audit.py','sharded_rows.py','manifest_stream.py']]
 f['sources'] += [pin(p) for p in refs];f['data_transfer']=dict(recipe=pin(R/'training-recipe.json'),new_bank=pin(bank/'qualification.json'),max_unique_updates_per_arm=1920,native_checkpoints=[6240,6720,7680],untouched_final_inputs_accessed=False,old_data_schedule='unchanged',broader_data_schedule='1old+1new CE windows per update; replay unchanged')
 atomic(R/'training-freeze.json',f)
 C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/data-cost-transfer-20261104-01');py=str(P/'.venv/bin/python');jobs=[]
 def add(ident,stage,args,result,status,timeout=1200,candidate='training',artifacts=()):
  jobs.append(dict(id=ident,candidate=candidate,stage=stage,depends_on=[jobs[-1]['id']] if jobs else [],timeout_seconds=timeout,argv=[py,*map(str,args)],cwd=str(P),checks=[dict(path=str(result),keys=['status'],equals=status)],artifacts=list(map(str,artifacts))))
 def training(name,arm,final,stage,resume=None):
  out=C/name;args=[S/'train.py','--data-arm',arm,'--original-shards',oldshards,'--derived-source',derived,'--recipe','control','--final-updates',str(final),'--out',out,'--freeze',R/'training-freeze.json','--allow-cuda']
  if arm=='broader':args += ['--registry',bank/'registry.json','--shards',bank/'shards.json']
  if resume:args += ['--resume',resume]
  add(name,stage,args,out/'result.json','TRAINED_EXPORTED_NOT_NATIVE_ADMITTED',timeout=3600 if final==7680 else 1200,artifacts=[out/f'checkpoint-{final:08d}.pt',out/'model.gguf']);return out
 old=training('old-qualification48','old',5808,0);new=training('broader-qualification48','broader',5808,0);split=training('broader-split24','broader',5784,0);resumed=training('broader-resumed48','broader',5808,0,split/'checkpoint-00005784.pt')
 add('state-proof',1,[S/'check_states.py','--old',old/'result.json','--reference','/mnt/c/Users/philip/binary-eagle-decoding-artifacts/contender-infra-20261104-01/continuous/result.json','--continuous',new/'result.json','--resumed',resumed/'result.json','--out',R/'state-proof.json'],R/'state-proof.json','REFERENCE_AND_BROADER_RESUME_EXACT',timeout=300)
 models=P/'runs/contender-infra-20261104-01/native-retry/models-quiet.json';inputs=P/'runs/contender-infra-20261104-01/native-retry/serving/inputs.json'
 def compare(name,results,stage,tier='contender'):
  out=R/name;add(name,stage,[S/'native_compare.py','benchmark','--project',P,'--output',out,'--models',models,'--inputs',inputs,'--results',*results,'--tier',tier],out/'result.json','SMOKE_PASS_NO_RANKING' if tier=='smoke' else 'CONTENDER_PANEL_AUDITED',timeout=1800 if tier=='contender' else 600)
 compare('data-smoke',[old/'result.json',new/'result.json'],2,'smoke')
 old_final=training('old-final1920','old',7680,3,old/'checkpoint-00005808.pt');new_final=training('broader-final1920','broader',7680,3,new/'checkpoint-00005808.pt')
 endpoints=[]
 for cursor in [6240,6720]:
  for arm,root in [('old',old_final),('broader',new_final)]:
   name=f'{arm}-export{cursor}';out=C/name;add(name,4,[S/'export_checkpoint.py','--checkpoint',root/f'checkpoint-{cursor:08d}.pt','--out',out],out/'result.json','TRAINED_EXPORTED_NOT_NATIVE_ADMITTED',timeout=300);endpoints.append(out/'result.json')
 compare('early-checkpoint-panel',endpoints,5)
 compare('final-checkpoint-panel',[old_final/'result.json',new_final/'result.json'],6)
 natural=R/'natural-final-panel';add('natural-final-panel',7,[S/'consumed_panel.py','--models',R/'final-checkpoint-panel/models-quiet.json','--inputs',R/'consumed-natural-inputs.json','--output',natural,'--allow-cuda'],natural/'audit.json','RAW_REQUEST_MODEL_RUNTIME_METRICS_AND_RELEASE_AUDIT_PASS',timeout=2400)
 sources=[pin(S/n) for n in ['build_training_plan.py','train.py','training_bank.py','check_states.py','native_compare.py','consumed_panel.py','export_checkpoint.py','native_log_text.py']]
 atomic(R/'training-plan.json',dict(schema='contender_plan_v1',budget_seconds=18000,candidate_budgets=dict(training=18000),grace_seconds=180,gpu=read(R/'capture-plan.json')['gpu'],inputs=sources+[pin(R/'training-freeze.json'),pin(bank/'qualification.json'),pin(models),pin(inputs),pin(R/'consumed-natural-inputs.json')],jobs=jobs))
 print('FROZEN',len(jobs),'jobs; first launch through2 for qualification only')
if __name__=='__main__':main()
