import os,json,sys
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';OUT=R/'runs/dspark-a8-full-pool-20261006-01';P=OUT/'packet-v3';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,SubprocessRunner,atomic_json,process_identity,stop_signals
from w1a1_eagle.block_shard_lifecycle import ShardCache,open_provider
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
runtime=json.loads((OUT/'runtime.inference-v2.json').read_text());cmd=json.loads((P/'commands.json').read_text())['initial_prepare'];assert cmd[0]==str(R/'.venv/bin/python') and cmd[1]==str(C/'scripts/train_nine_model_qat.py')
owner=process_identity(os.getpid());lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock';resources=LinuxResources(runtime['gpu_uuid']);stop=R/'results/dspark-launch-20261006/NEW_ZERO_STOP'
def authorization():
 if stop.exists() or json.loads(Path(runtime['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False:raise InterruptedError('paused or STOP')
r=SubprocessRunner(C,dict(runtime['environment'],CUDA_VISIBLE_DEVICES=runtime['gpu_uuid']),authorization=authorization);s={'schema':'dspark_source_cutover_zero_operation_v1','owner':owner,'status':'PENDING','optimizer_updates':0}
config=json.loads((P/'configs/dspark_a8.json').read_text());provider=open_provider(config['data']);cache=provider.cache
with owner_lock(lock,owner),stop_signals():
 try:
  authorization();release=resources.require_released([],[]);initial=resources.snapshot();assert not release['other_context_pids'] and not initial['dxg_holders'];assert initial['host_available_bytes']>=4*1024**3 and initial['gpu_free_bytes']>=1024**3
  if cache.state['owner']:raise RuntimeError('existingcacheowner_no_handoff')
  cache.acquire('root-source-zero','student');s['initial_resources']=initial
  with wall_cap(1800):r.run(cmd,directory=P/'initial-prepare-process',stop_path=stop,wall_seconds=1800)
  receipt=json.loads((P/'initial-prepare-receipt.json').read_text());assert receipt['optimizer_updates']==0 and receipt['status']=='PASS';s.update(status='PASS',receipt=receipt)
 except BaseException as e:s.update(status='FAIL',failure={'type':type(e).__name__,'message':str(e)});raise
 finally:
  proof=resources.require_released(r.process_groups,r.process_identities);s['release']=proof;s['resource_return']=resources.snapshot()
  if cache.state['owner'] and cache.state['owner']['owner']=='root-source-zero':cache.release('root-source-zero',released_proof=proof)
  atomic_json(R/'results/dspark-launch-20261006/source-zero-v3-operation.json',s)
print(json.dumps({'status':s['status'],'optimizer_updates':0}))
