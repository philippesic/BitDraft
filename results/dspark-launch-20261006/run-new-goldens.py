import json,sys,os,subprocess,hashlib
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';P=O/'packet-v3';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,SubprocessRunner,atomic_json,process_identity,stop_signals
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
rt=json.loads((O/'runtime.inference-v2.json').read_text());cmd=json.loads((P/'commands.json').read_text())['native_goldens'];stop=P/'STOP';obs=LinuxResources(rt['gpu_uuid']);owner=process_identity(os.getpid())
def auth():
 if stop.exists() or json.loads(Path(rt['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False:raise InterruptedError('paused or STOP')
r=SubprocessRunner(C,dict(rt['environment'],CUDA_VISIBLE_DEVICES=rt['gpu_uuid']),authorization=auth);s={'schema':'dspark_v3_golden_operation_v1','owner':owner,'optimizer_updates':0,'status':'PENDING'}
with owner_lock(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock',owner),stop_signals():
 try:
  auth();proof=obs.require_released([],[]);base=obs.snapshot();assert proof['other_context_pids']==[] and base['dxg_holders']==[]
  with wall_cap(180):r.run(cmd,directory=P/'native-goldens-process',stop_path=stop,wall_seconds=180)
  raw=P/'native-goldens-process/stdout.log';rows=[json.loads(l) for l in raw.read_text().splitlines() if l.startswith('{')];assert len(rows)==3
  f=P/'native-goldens-receipts.jsonl';assert not f.exists();f.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in rows));s.update(status='PASS',receipts={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 except BaseException as e:s.update(status='FAIL',failure={'type':type(e).__name__,'message':str(e)});raise
 finally:s['release']=obs.require_released(r.process_groups,r.process_identities);s['resources']=obs.snapshot();atomic_json(R/'results/dspark-launch-20261006/source-v3-goldens-operation.json',s)
print(json.dumps(s))
