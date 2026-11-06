import json,sys,signal,os
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding');C=ROOT/'runs/checkouts/dspark-full-training-20261006-01';OUT=ROOT/'runs/dspark-a8-full-pool-20261006-01';P=OUT/'packet'
sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import Files,LinuxResources,SubprocessRunner,atomic_json,process_identity,stop_signals
from w1a1_eagle.block_shard_lifecycle import pin
from evaluate_paused_eagle_checkpoint import owner_lock,wall_cap
import evaluate_nine_model_timed_checkpoint as timed
# Enforce the already-selected separate clean3/diagnostic4 protocol.
base_native_command=timed.native_command
command_calls=[0]
def protocol_native_command(*args):
 command=base_native_command(*args);diagnostic=command_calls[0]>=20;command_calls[0]+=1
 return command+["--log-verbosity","4" if diagnostic else "3"]
timed.native_command=protocol_native_command
evaluate=timed.evaluate
from prepare_nine_model_lane_endpoint import validate_protocol
runtime=json.loads((OUT/'runtime.packet.json').read_text());export=json.loads((P/'initial-export-receipt.json').read_text());protocol_path=OUT/'operator-inputs/native-protocol-packet/native-primary24.protocol.json';protocol=json.loads(protocol_path.read_text());validate_protocol(protocol,{'evaluation_wall_seconds':protocol['evaluation_wall_seconds']})
model={'model':export['model'],'audit':export['audit']}
plan={'candidate':'dspark_a8','config':export['config'],'gpu_uuid':runtime['gpu_uuid'],'target_policy':protocol['target_policy'],'resource_policy':runtime['resource_policy'],'environment':runtime['environment'],'gpu_control_path':runtime['gpu_control_path'],'target':runtime['inputs']['target'],'initial':model,'runtime':{k:v for k,v in runtime['inputs'].items() if k=='binary' or k.startswith('runtime_library_')},'control':{'frozen_original':True,'precision':'Q4_0','model':{'path':str(ROOT/'runs/binary-scale-fitting-5080-artifacts-20260927/results/scale-models/Qwen3-4B-eagle3-q4_0.gguf'),'sha256':'2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280'}},'prompts':protocol['primary_prompts'],'protocol':pin(protocol_path),'development_admission':{'path':str(ROOT/'results/dspark-launch-20261006/development-disjoint/evidence.json'),'sha256':'7700c2947036e9d89284150aa5a6e04bc67e369a687475a2ecf76ed31731356c'},'frozen_lane':{'scope':'zero-hour_pretraining_measurement_not_paid_lane','initial_checkpoint':export['checkpoint']},'remaining_candidates':['dflash_a8','eagle_a1','dspark_a1','dflash_a1']}
files=Files()
for rec in [plan['target'],plan['control']['model'],plan['prompts'],plan['development_admission'],*model.values(),*plan['runtime'].values()]:files.check(rec)
dir=ROOT/'results/dspark-launch-20261006/zero-native';stop=dir.parent/'ZERO_EVAL_STOP';observer=LinuxResources(plan['gpu_uuid']);observer.require_released([],[])
def authorization():
 if json.loads(Path(plan['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False or stop.exists():raise InterruptedError('paused or STOP')
runner=SubprocessRunner(C,dict(plan['environment'],CUDA_VISIBLE_DEVICES=plan['gpu_uuid']),authorization=authorization)
owner=process_identity(os.getpid());lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock'
status={'schema':'dspark_zero_measurement_status_v1','optimizer_updates':0,'current_and_initial_same_exact_model':True,'source_plan':plan,'owner':owner,'strict_quality_gate':'PENDING','measurement_complete':False}
atomic_json(dir.parent/'zero-native-plan.json',plan)
try:
 with owner_lock(lock,owner),stop_signals(),wall_cap(protocol['evaluation_wall_seconds']+60):
  result=evaluate(plan,files,protocol,model,dir,stop,authorization,observer,runner)
  status.update(measurement_complete=True,strict_quality_gate='PASS',report=result)
except BaseException as e:
 status.update(failure={'type':type(e).__name__,'message':str(e)})
 if (dir/'progress.json').exists():
  progress=json.loads((dir/'progress.json').read_text());status['clean_records']=len(progress['clean_records']);status['diagnostic_records']=len(progress['diagnostic_records']);status['measurement_complete']=status['clean_records']==480 and status['diagnostic_records']==96
  if 'greedy verifier token sequence differs' in str(e):status['strict_quality_gate']='FAIL'
 raise
finally:
 status['owned_release']=observer.require_released(runner.process_groups,runner.process_identities);status['resource_return']=observer.snapshot();atomic_json(dir.parent/'zero-native-status.json',status)
