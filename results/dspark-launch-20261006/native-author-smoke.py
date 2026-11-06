import sys,json,os,subprocess
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-full-training-20261006-01';OUT=R/'runs/dspark-a8-full-pool-20261006-01';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import Files,LinuxResources,SubprocessRunner,atomic_json,process_identity,stop_signals
from w1a1_eagle.block_shard_lifecycle import pin
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
f=Files();runtime=json.loads((OUT/'runtime.inference-v2.json').read_text());e=json.loads((OUT/'packet/initial-export-receipt.json').read_text());o=R/'results/dspark-launch-20261006/native-author-smoke-v2.json';assert not o.exists()
for p in [e['model'],e['audit'],runtime['inputs']['block_native_binary'],runtime['inputs']['target'],runtime['inference_build_provenance']]:f.check(p)
observer=LinuxResources(runtime['gpu_uuid']);owner=process_identity(os.getpid());lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
def authorization():
 if json.loads(Path(runtime['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False:raise InterruptedError('GPU paused')
r=SubprocessRunner(C,dict(runtime['environment'],CUDA_VISIBLE_DEVICES=runtime['gpu_uuid']),authorization=authorization)
s={'schema':'dspark_author_smoke_operation_v1','owner':owner,'optimizer_updates':0,'source':'5eb221ec53cb4317aef963e3bdec3cc69401eca4','status':'PENDING'}
with owner_lock(lock,owner),stop_signals():
 try:
  authorization();initial_release=observer.require_released([],[]);initial_resources=observer.snapshot()
  if initial_release['other_context_pids'] or initial_resources['dxg_holders']:raise RuntimeError('foreignGPUcontextbeforefirstserver')
  s['initial_resource_preflight']={'release':initial_release,'resources':initial_resources}
  proof=R/'results/dspark-launch-20261006/native-author-model-proof-v2.json'
  command=['/usr/bin/env','DSPARK_REQUIRE_AUTHOR_LAYOUT=1',str(R/'.venv/bin/python'),str(C/'scripts/check_block_binary_native.py'),'--binary',runtime['inputs']['block_native_binary']['path'],'--model',e['model']['path'],'--export',e['audit']['path'],'--gpu-layers','999','--require-cuda','--target',runtime['inputs']['target']['path'],'--target-sha256',runtime['inputs']['target']['sha256'],'--receipt',str(proof)]
  with wall_cap(180):r.run(command,wall_seconds=180,stop_path=R/'results/dspark-launch-20261006/NATIVE_AUTHOR_STOP',directory=R/'results/dspark-launch-20261006/native-author-smoke-process-v2')
  observed=json.loads(proof.read_text());assert observed['passed'] and observed['author_reference_layout'] and observed['draft_n_batch']==observed['draft_n_ubatch']==32 and observed['oversized_noise_guard_checked'];s.update(status='PASS',proof=pin(proof))
 except BaseException as err:s.update(status='FAIL',failure={'type':type(err).__name__,'message':str(err)});raise
 finally:s['release']=observer.require_released(r.process_groups,r.process_identities);s['resource_return']=observer.snapshot();atomic_json(o,s)
print(json.dumps(s))
