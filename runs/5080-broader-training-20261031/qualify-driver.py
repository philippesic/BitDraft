from pathlib import Path
import os,sys,json,time,subprocess,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';S=R/'source01';OLD=P/'runs/unattended-training-preparation-20261028/gpu-resume-20261028';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
sys.path[:0]=[str(S/'scripts'),str(P/'scripts')];os.environ['PYTHONPATH']=str(S/'scripts')+':'+str(P/'scripts')
from training_system.io import read,pin,atomic_json,authenticate
for _ in range(600):
 if (W/'data-ready.json').exists():break
 q=R/'supervisor/runs/postcapture02/state.json'
 if q.exists() and read(q)['status']!='running':raise RuntimeError('data preparation stopped: '+str(read(q)))
 time.sleep(3)
assert (R/'extra-diagnostic-complete.json').exists(), 'diagnostic gate incomplete'
ready=read(W/'data-ready.json');cfg=read(OLD/'campaign-control124-config.json');spec=cfg['spec']
def rebase(value):
 if isinstance(value,dict):
  if set(value)>={'path','sha256'} and value['path'].startswith(str(OLD/'source26')+'/'):
   q=S/Path(value['path']).relative_to(OLD/'source26');return {**value,**pin(q)}
  return {k:rebase(v) for k,v in value.items()}
 if isinstance(value,list):return [rebase(x) for x in value]
 return value
spec=rebase(spec)
for k in ['data_adapter','historical_pairs','training_inputs','training_order']:spec.pop(k,None)
GiB=1024**3
spec.update(manifest=ready['manifest'],order=ready['order'],teacher=ready['teacher'],data_cache_enabled=False,target_updates=8,deadline_unix=1793505600,max_invocation_train_seconds=60,checkpoint_root=str(W/'qualification-checkpoints'),checkpoint_interval=0,job_tag='broader-control-qualification',stop_path=str(W/'STOP'),retention_policy='latest_verified',retention_references_complete=True,retention_references=[],export_policy='milestones')
spec['storage_budget']={'schema':'training_job_storage_budget_v1','max_job_bytes':20*GiB,'max_attempt_bytes':8*GiB,'max_attempt_write_bytes':12*GiB,'checkpoint_bound_bytes':4*GiB,'model_bound_bytes':GiB,'max_files':20000,'execution_backed_by_host':False,'volumes':{'nativeC':{'path':'/mnt/c','floor_bytes':10*GiB,'reservation_bytes':10*GiB,'backing_reservation_bytes':0}}}
guard=read(OLD/'campaign-control124-guard.json');guard.update(authority='user_5080_broader_training_20261031',deadline_unix=spec['deadline_unix'],minimum_wsl_free_bytes=2*GiB,minimum_windows_free_bytes=10*GiB,stop_paths=[str(W/'STOP')],wall_seconds=3600,max_train_seconds=840)
guard['env']={**guard['env'],'PYTHONPATH':os.environ['PYTHONPATH']}
guard['inputs']=list(spec['sources'].values())+[spec[k] for k in ['parent'] if False]+[spec[k] for k in ['manifest','order','teacher','controls','fidelity_gate','model_cache','model_cache_producer','source_checkpoint']]+list(spec['parent'].values())
gp=W/'guard.json';atomic_json(gp,guard);atomic_json(W/'base-spec.json',spec)
config={**cfg,'cli':str(S/'scripts/training_system/cli.py'),'guard_template':str(gp),'spec':spec,'run_root':str(W/'qualification-jobs'),'objectives':['control','dpal'],'complete_pair_memory_probe':False,'max_memory_probe_processes_per_objective':0,'maximum_pause_continuations':2,'crash_recovery':True,'host_storage_path':'/mnt/c','execution_backed_by_host':True,'native':{**cfg['native'],'manifest':ready['manifest']}}
cp=W/'qualification-config.json';atomic_json(cp,config)
from training_system.train import validate_spec
validate_spec(spec);print('QUALIFICATION_CONFIG_VALIDATED',str(cp),flush=True)
cmd=[sys.executable,'-B',str(S/'scripts/training_system/qualify.py'),'--config',str(cp),'--output',str(W/'qualification')]
subprocess.run(cmd,check=True);print('QUALIFICATION_COMPLETE',flush=True)
