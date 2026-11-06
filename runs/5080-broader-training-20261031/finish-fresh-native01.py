"""Complete interrupted final native stage; no training invocation or updates."""
from pathlib import Path
import sys,os,subprocess,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');S=R/'source03';sys.path.insert(0,str(S/'scripts'))
from training_system.io import read,pin,authenticate,atomic_json
from training_system.promotion import verified_metrics
old=read(W/'supervisors/fresh-training01/state.json');assert old['status']=='interrupted' and old['exit_code']==-15
root=W/'fresh-training01';state=read(root/'state.json');assert set(state['native'])=={'96','4800'}
job=read(Path(state['job'])/'job.json');assert job['status']=='COMPLETE' and job['owner'] is None
finish=read(authenticate(job['finish']));assert finish['status']=='EXPOSURE_COMPLETE' and finish['local_updates']==14400
ga=read(Path(job['attempts'][-1]['guard_output'])/'result.json');assert ga['returncode']==0 and not ga.get('error')
authenticate(finish['model']);authenticate(finish['checkpoint'])
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
k=read(R/'interop-keeper02.json');assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop'];os.environ['CUDA_CACHE_PATH']=str(W/'fresh-qualification01/cuda')
snapshot=R/'fresh-training-interrupted-state01.json';assert not snapshot.exists();atomic_json(snapshot,state)
config=root/'native-config-14400.json';c=read(config);assert c['model']==finish['model'] and c['manifest']==job['spec']['manifest']
native=root/'native-14400-recovery01';assert not native.exists()
g=read(root/'native-guard-config-14400.json');g.update(output=str(root/'native-guard-14400-recovery01'),progress_paths=[str(native/'native-0000/rows.jsonl')]);g['inputs'] += [pin(__file__),pin(snapshot),pin(W/'supervisors/fresh-training01/state.json')];gp=root/'native-guard-config-14400-recovery01.json';atomic_json(gp,g)
subprocess.run([sys.executable,'-B',str(S/'scripts/shortlist_pilots/guard.py'),'--config',str(gp),'--',sys.executable,'-B',str(S/'scripts/training_system/native.py'),'--config',str(config),'--output',str(native),'--batch-cases','144'],check=True)
nr=pin(native/'result.json');metrics,proof=verified_metrics(nr,model=finish['model'],manifest=job['spec']['manifest'])
assert read(Path(state['job'])/'job.json')==job
state['native']['14400']=nr;state['events'] += [{'action':'native_complete_independently_verified','target':14400,'metrics':metrics,'proof':proof,'unix':time.time(),'operational_recovery':True},{'action':'all_planned_milestones_complete_after_native_only_recovery','unix':time.time()}];state['status']='COMPLETE';atomic_json(root/'state.json',state)
receipt={'scope':'Final native-only operational completion after SIGTERM; training already complete and unchanged.','old_supervisor':pin(W/'supervisors/fresh-training01/state.json'),'old_curve_snapshot':pin(snapshot),'completed_curve':pin(root/'state.json'),'job_finish':job['finish'],'native_result':nr,'optimizer_updates':0,'TRAIN_seconds':0,'job_unchanged':True,'guard':pin(root/'native-guard-14400-recovery01/result.json')};atomic_json(R/'fresh-training-completion-recovery01.json',receipt)
print('FRESH_NATIVE_ONLY_COMPLETION',metrics['development']['prefix5_sum'],flush=True)
