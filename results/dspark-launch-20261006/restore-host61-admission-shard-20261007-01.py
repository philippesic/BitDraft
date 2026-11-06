import json,sys,time,os,fcntl
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';O=R/'runs/dspark-a8-full-pool-20261006-01/training-memory-r1-lane';D=R/'results/dspark-launch-20261006';W=R/'runs/host61-admission-shard-restore-20261007-01';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,SubprocessRunner,atomic_json,process_identity
from w1a1_eagle.block_shard_lifecycle import pin,open_provider,restore_requested_shards
lane_path=O.parent/'lane-source-memory-r1.json';lane=json.loads(lane_path.read_text());assert pin(lane_path)['sha256']=='0a7d98fc5f8dff8eb89abb9dd966332776c9142beacc543bd3e26053a70902ce'
assert not W.exists();W.mkdir()
cfg=json.loads(Path(lane['config']['path']).read_text());source_status=O/'attempts/56a21a6394424f08aa7e5bad2f73dd9b/admission/dspark_a8/backward/status.json';request=json.loads(source_status.read_text())['shard_request'];assert request['shard_ids']==['chunk-0547']
old_budget=pin(O/'training/budget-used.json');old_latest=pin(O/'training/checkpoints/latest.json');b=json.loads((O/'training/budget-used.json').read_text());assert b['active_attempt'] is None and b['training_seconds']==4310.6791790700045
lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
with lock.open('a') as owner:
 fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def authorization():
  assert json.loads(Path(lane['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is False,'GPU paused'
  assert not (O/'STOP').exists() and not (W/'STOP').exists(),'stop requested'
 authorization()
 resources=LinuxResources(lane['gpu_uuid']);snapshot=resources.snapshot();assert snapshot['host_available_bytes']>=4294967296 and not snapshot['dxg_holders'];assert os.statvfs(R).f_bavail*os.statvfs(R).f_frsize>=10737418240
 runner=SubprocessRunner(W,environment=lane.get('environment'),authorization=authorization)
 def release():
  proof=resources.require_released(runner.process_groups,runner.process_identities);snap=resources.snapshot();assert not proof['other_context_pids'] and not snap['dxg_holders'];return proof
 release()
 provider=open_provider(cfg['data']);cache=provider.cache
 initial=dict(schema='dspark_admission_shard_restore_v1',status='restoring',shard='chunk-0547',original_resident=list(cache.state['resident']),request_source=pin(source_status),lane=pin(lane_path),original_budget=old_budget,original_latest=old_latest,owner=process_identity(os.getpid()),resource_snapshot=snapshot,trainer_seconds_charged=0)
 atomic_json(W/'status.json',initial)
 try:
  phases=restore_requested_shards(cfg['shard_lifecycle'],request,cache=cache,run_dir=W,attempt=W/'attempt',runner=runner,release=release,authorization=authorization,source=lane['source'])
  assert pin(O/'training/budget-used.json')==old_budget and pin(O/'training/checkpoints/latest.json')==old_latest
  initial.update(status='PASS',phases=phases,final_resident=list(cache.state['resident']),owned_release=release(),budget_and_paid_cursor_unchanged=True,completed_unix=time.time())
 except BaseException as e:
  initial.update(status='FAIL',failure=dict(type=type(e).__name__,reason=str(e)),completed_unix=time.time());atomic_json(W/'status.json',initial);raise
 finally:
  provider.close()
 atomic_json(W/'status.json',initial);dest=D/'host61-admission-shard-restored-20261007-01.json';assert not dest.exists();atomic_json(dest,initial);print(json.dumps(initial))
