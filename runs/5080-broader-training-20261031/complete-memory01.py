"""Add the missing real all-pair memory proof; reuse closed exact qualification."""
from pathlib import Path
import os,sys,time,subprocess,json
P=Path('/home/philip/binary-eagle-decoding')
R=P/'runs/5080-broader-training-20261031'
S=R/'source02'
W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
O=W/'qualification-memory01'
sys.path.insert(0,str(S/'scripts'))
from training_system.io import read,pin,authenticate,atomic_json
from training_system.memory_probe import validate_slices
from training_system.device_proofs import _proofs
from training_system.resources import admit_capacity
k=read(R/'interop-keeper.json');assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop']
original=pin(W/'qualification04/result.json');q=read(authenticate(original))
assert all(q[k] is True for k in ['fresh_process_resume_exact','native_outputs_exact','source_authenticated'])
O.mkdir(exist_ok=False)
started=time.time();completed={}
for objective in ['control','dpal']:
 record=q['records'][objective];proofs=record['qualification_proofs']
 source=proofs['source'];spec=read(authenticate(source))
 checkpoint=record['resumed']['checkpoint'];authenticate(checkpoint)
 contract=read(authenticate(proofs['runtime_contract']));pairs=contract['data']['pairs']
 assert len(pairs)==480
 cursor=0;slices=[];refs=[]
 for ordinal in range(16):
  if time.time()-started>5400:raise TimeoutError('finite memory extension wall ceiling reached')
  import shutil
  admit_capacity({'WSL':{'free_bytes':shutil.disk_usage(R).free,'floor_bytes':2*1024**3,'reservation_bytes':256*1024**2},'nativeC':{'free_bytes':shutil.disk_usage(W).free,'floor_bytes':10*1024**3,'reservation_bytes':3*1024**3}})
  dest=O/(objective+'-%02d'%ordinal)
  guard=read(W/'guard03.json')
  guard.update(output=str(O/(objective+'-%02d-guard'%ordinal)),max_train_seconds=0,wall_seconds=1200,deadline_unix=spec['deadline_unix'],progress_paths=[str(dest/'progress.json')])
  guard['inputs']=guard['inputs']+[source,checkpoint,pin(__file__)]
  gp=O/(objective+'-%02d-guard-config.json'%ordinal);atomic_json(gp,guard)
  command=[sys.executable,'-B',str(S/'scripts/shortlist_pilots/guard.py'),'--config',str(gp),'--',sys.executable,'-B',str(S/'scripts/training_system/train.py'),'--spec',source['path'],'--output',str(dest),'--resume',checkpoint['path'],'--memory-probe-start',str(cursor),'--allow-cuda']
  with (O/(objective+'-%02d-dispatch.log'%ordinal)).open('x') as out:
   subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT)
  ref=pin(dest/'memory-probe.json');item=read(ref['path'])
  assert item['start_pair']==cursor and cursor<item['next_pair']<=len(pairs)
  slices.append(item);refs.append(ref);cursor=item['next_pair']
  atomic_json(O/'progress.json',{'objective':objective,'next_pair':cursor,'total_pairs':len(pairs),'completed':completed,'optimizer_updates':0,'training_exposure_updates':0})
  print(json.dumps({'objective':objective,'covered_pairs':cursor,'total_pairs':len(pairs),'probe_work_seconds':item['probe_work_seconds']}),flush=True)
  if cursor==len(pairs):break
 if cursor!=len(pairs):raise RuntimeError('bounded memory process ceiling exhausted; proof remains incomplete')
 peak=validate_slices(slices,pairs=pairs,checkpoint=checkpoint,state_key=record['adapter_state_key'],precision=record['precision'],hardware=record['hardware'],spec=spec,sampler_plan=contract['data'].get('sampler_memory_plan'))
 dest=O/(objective+'-memory-coverage.json')
 atomic_json(dest,{'schema':'complete_training_pair_memory_qualification_v1','slices':refs,'source':source,'runtime_contract':proofs['runtime_contract'],'checkpoint':checkpoint,'pairs':pairs,'peak_reserved_bytes':peak,'actual_cuda':True,'all_declared_training_pairs_executed':True,'optimizer_updates':0,'training_exposure_updates':0,'probe_work_seconds':sum(x['probe_work_seconds'] for x in slices),'probe_full_job_wall_seconds':sum(x['full_job_wall_seconds'] for x in slices),'scope':'unchanged production nonupdating forward/backward over every declared pair at retained8; exact full-state digest and RNG restoration; short8-update optimizer peaks remain separate'})
 proofs['memory_coverage']=pin(dest)
 completed[objective]={'covered_pairs':cursor,'processes':len(slices),'peak_reserved_bytes':peak}
q['admission_extension']={'schema':'missing_memory_boundary_extension_v1','original_qualification':original,'driver':pin(__file__),'completed':completed,'optimizer_updates':0,'training_exposure_updates':0,'wall_seconds':time.time()-started,'scope':'reuse authenticated closed state/export/native proofs; add only actual missing all-pair memory coverage; original sealed qualification remains unchanged'}
output=W/'qualification-complete05.json';atomic_json(output,q)
for objective in ['control','dpal']:_proofs(pin(output),objective)
atomic_json(O/'result.json',{'status':'COMPLETE','qualification':pin(output),'completed':completed,'optimizer_updates':0,'training_exposure_updates':0,'wall_seconds':time.time()-started})
print('COMPLETE_PAIR_MEMORY_PROOFS_VALIDATED',flush=True)
