"""Operational resume: verify real same-boot admission, restore paid batch, then unchanged lifecycle."""
import argparse,fcntl,json,os,sys,time,uuid
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007'
sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import GATES,LinuxResources,SubprocessRunner,atomic_json,deferred_termination,require,require_available,resource_gate,stop_signals
from w1a1_eagle.block_shard_lifecycle import ShardRequired,open_provider,pin,restore_requested_shards,validate_controller_config
from run_nine_model_lane import validate_lane,run_training_lifecycle
from evaluate_nine_model_timed_checkpoint import validate_plan as validate_timed_plan
from train_nine_model_qat import require_admission

def main():
 p=argparse.ArgumentParser()
 for key in ['lane','run-dir','availability','supervisor-state','admission-receipt']:p.add_argument('--'+key,type=Path,required=True)
 for key in ['lane-sha256','admission-sha256','admission-boot']:p.add_argument('--'+key,required=True)
 a=p.parse_args()
 lane,files,plan=validate_lane(a.lane,a.lane_sha256);validate_timed_plan(lane['timed_evaluation_plan'],lane,plan,files)
 lease=require_available(a.availability,a.lane_sha256)
 def authorization():
  if json.loads(Path(lane['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False:raise InterruptedError('RTX5080 paused')
  require(json.loads(a.availability.read_text())==lease,'availability ownership changed')
  require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==a.admission_boot,'admission boot changed')
  if (a.run_dir/'STOP').exists():raise InterruptedError('lane STOP requested')
 authorization()
 sup=json.loads(a.supervisor_state.read_text())
 require(sup.get('pid')==os.getpid() and sup.get('supervisor_pid')==os.getppid() and sup.get('status')=='running' and sup.get('stop_grace_seconds',0)>=90 and os.environ.get('TMUX'),'detached90-second supervisor required')
 require(lease['gpu_uuid']==lane['gpu_uuid'],'lease GPU differs')
 previous=json.loads((a.run_dir/'state.json').read_text())
 require(previous['bundle_sha256']==a.lane_sha256 and previous['candidate']==lane['candidate'] and previous.get('status')!='training_complete','resume lane/endpoint differs')
 state=dict(schema='nine_model_lane_state_v1',bundle_sha256=a.lane_sha256,candidate=lane['candidate'],campaign_complete=False,previous_attempt=previous)
 for key in ['pending_training','evaluation_receipt','evaluations','pending_shard_resume','shard_phases']:
  if key in previous:state[key]=previous[key]
 state['operator_resume_wrapper']=pin(Path(__file__))
 state_path=a.run_dir/'state.json';runner=SubprocessRunner(C,environment=lane.get('environment'),authorization=authorization);resources=LinuxResources(lane['gpu_uuid']);baseline=None
 def release():
  proof=resources.require_released(runner.process_groups,runner.process_identities);observed=resources.snapshot();resource_gate(observed,baseline,lane['resource_policy'])
  require(proof['other_context_pids']==[] and observed['dxg_holders']==[],'foreign GPU contexts forbid continuation')
  state.update(owned_release=proof,resource_return=observed);return proof
 lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('a') as owner:
  fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   with stop_signals():
    baseline=resources.snapshot();resource_gate(baseline,baseline,lane['resource_policy']);state.update(resource_baseline=baseline)
    for process_path in a.run_dir.glob('**/process.json'):
     process=json.loads(process_path.read_text());require(process.get('kernel_identity') is not None,'prior kernel identity absent');runner.process_groups.append(process['pgid']);runner.process_identities.append(process['kernel_identity'])
    for path in a.run_dir.glob('**/process-lineage.json'):runner.process_identities.extend(json.loads(path.read_text())['kernel_identities'])
    release()
    admission_record={'path':str(a.admission_receipt),'sha256':a.admission_sha256};admission=json.loads(files.check(admission_record).read_text())
    require(admission.get('schema')=='nine_model_stage_receipt_v1' and admission.get('status')=='PASS' and admission.get('artifact_kind')=='production' and admission.get('bundle_sha256')==a.lane_sha256 and admission.get('gpu_uuid')==lane['gpu_uuid'] and admission.get('compute_capability')==[12,0] and admission.get('optimizer_updates')==0 and admission['gates']==dict.fromkeys(sorted(GATES),'PASS'),'genuine six-gate admission differs')
    require(admission.get('stage')=='admission' and admission['resource_baseline']['boot_id']==a.admission_boot and admission['resource_return']['boot_id']==a.admission_boot,'admission aggregate boot/stage differs')
    origin=a.admission_receipt.parent/'admission';processes=list(origin.glob('**/process.json'));require(len(processes)>=4,'admission producer provenance absent')
    for process_path in processes:
     process=json.loads(process_path.read_text());require(process['kernel_identity']['boot_id']==a.admission_boot,'admission producer boot differs')
     exit_path=process_path.parent/'process-exit.json';exit_state=json.loads(exit_path.read_text());require(exit_state.get('exit_code')==0 and exit_state.get('cleanup_status')=='PASS','admission producer did not exit cleanly')
    selected=admission['candidate_admissions'][lane['candidate']];admission_path=files.check(selected)
    require_admission(admission_path,a.lane_sha256,lane['config']['sha256'],candidate=lane['candidate'],gpu_uuid=lane['gpu_uuid'])
    state['admission_reference']=admission_record;state['admission_reuse_kind']='actual_same_source_config_GPU_boot_reference_not_new_execution'
    spec=json.loads(files.check(lane['config']).read_text());validate_controller_config(spec['shard_lifecycle'],spec['data'],lane['source'])
    checkpoint_record=pin(a.run_dir/'training/checkpoints/latest.json');checkpoint=json.loads(Path(checkpoint_record['path']).read_text());budget_record=pin(a.run_dir/'training/budget-used.json');budget=json.loads(Path(budget_record['path']).read_text())
    require(checkpoint['committed'] and checkpoint['cursor']['step']==11161 and checkpoint['cursor']['elapsed_seconds']==budget['training_seconds']==4310.6791790700045 and budget['active_attempt'] is None and budget['max_seconds']==43200 and budget['source_sha256']==a.lane_sha256,'paid checkpoint/ledger differs')
    require(pin(Path(checkpoint['path']))['sha256']==checkpoint['sha256']=='1c9bf9fca335740cc791cd3281e22d5fe252cf5703f0eecb3b7f4508106404f1','paid payload SHA differs')
    attempt=a.run_dir/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True,exist_ok=False)
    state.update(status='restoring_paid_resume_shards',attempt=str(attempt),admission=selected);atomic_json(state_path,state)
    provider=open_provider(spec['data']);paid_cursor=provider.restore_cursor(checkpoint['cursor']['data_cursor']);request=None
    try:
     provider.reserve_batch(paid_cursor,2,reservation=checkpoint['cursor']['batch_reservation'])
    except ShardRequired as error:
     request=dict(plan_sha256=provider.sha256,shard_ids=list(error.shard_ids),reservation=error.reservation)
    finally:provider.close()
    if request is not None:
     authorization();provider=open_provider(spec['data'])
     try:phases=restore_requested_shards(spec['shard_lifecycle'],request,cache=provider.cache,run_dir=a.run_dir,attempt=attempt,runner=runner,release=release,authorization=authorization,source=lane['source'])
     finally:provider.close()
     state.setdefault('shard_phases',[]).extend(phases)
    provider=open_provider(spec['data'])
    try:
     paid_cursor=provider.restore_cursor(checkpoint['cursor']['data_cursor'])
     provider.reserve_batch(paid_cursor,2,reservation=checkpoint['cursor']['batch_reservation'])
    finally:provider.close()
    require(pin(a.run_dir/'training/checkpoints/latest.json')==checkpoint_record and pin(a.run_dir/'training/budget-used.json')==budget_record,'preparation mutated paid state')
    release();state.update(status='training',paid_resume_batch_verified=True);atomic_json(state_path,state)
    run_training_lifecycle(lane,{'path':str(a.lane),'sha256':a.lane_sha256},a.run_dir,attempt,admission_path,a.supervisor_state,state,state_path,files,runner,resources,release,authorization,resume=True)
  except BaseException as error:
   with deferred_termination():
    state.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',failure=dict(type=type(error).__name__,reason=str(error)),checkpoints_deleted=False)
    if baseline is not None:
     try:release()
     except BaseException as cleanup:state['cleanup_failure']=str(cleanup)
   raise
  finally:atomic_json(state_path,state)
 print(json.dumps(state))
if __name__=='__main__':main()
