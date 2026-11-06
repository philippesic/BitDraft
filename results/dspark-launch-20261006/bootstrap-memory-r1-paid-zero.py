import json,sys,copy,math
from pathlib import Path
import numpy as np, torch
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';OLD=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,SubprocessRunner,atomic_json,process_identity,stop_signals
from w1a1_eagle.block_shard_lifecycle import pin,open_provider
from run_nine_model_lane import validate_lane
from evaluate_paused_eagle_checkpoint import owner_lock
from export_nine_model_lane_candidate import load_checkpoint
lane_path=O/'lane-source-memory-r1.json';lane,files,admission=validate_lane(lane_path,pin(lane_path)['sha256']);spec=json.loads(Path(lane['config']['path']).read_text());root=O/'training-memory-r1-lane';training=root/'training';assert not root.exists()
obs=LinuxResources(lane['gpu_uuid']);owner=process_identity(__import__('os').getpid());stop=O/'MEMORY_BOOTSTRAP_STOP'
def auth():
 if stop.exists() or json.loads(Path(lane['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is not False:raise InterruptedError('paused')
runner=SubprocessRunner(C,lane['environment'],authorization=auth);cache=open_provider(spec['data']).cache;receipt=O/'memory-r1-paid-zero-receipt.json';assert not receipt.exists()
report={'schema':'dspark_memory_repair_paid_zero_bootstrap_v1','status':'PENDING','owner':owner,'lane':pin(lane_path),'rollback_to_step':0}
with owner_lock(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock',owner),stop_signals():
 try:
  auth();p=obs.require_released([],[]);snap=obs.snapshot();assert p['other_context_pids']==[] and snap['dxg_holders']==[];assert cache.state['owner'] is None;cache.acquire('root-memory-r1-bootstrap','student')
  cmd=[str(R/'.venv/bin/python'),str(C/'scripts/train_nine_model_qat.py'),'--config',lane['config']['path'],'--run-dir',str(training),'--bundle-sha256',pin(lane_path)['sha256'],'--stage-name','dspark_a8/recovery-committed-zero','--completion-output',str(receipt),'--allow-cuda','--prepare-only']
  runner.run(cmd,directory=O/'memory-r1-paid-zero-process',stop_path=stop,wall_seconds=1800)
  old_receipt=json.loads((OLD/'runs/dspark-a8-twelve-hour-20261007-01/lane/training/checkpoints/latest.json').read_text());new_receipt=json.loads((training/'checkpoints/latest.json').read_text());assert old_receipt['cursor']['step']==new_receipt['cursor']['step']==0
  assert pin(Path(old_receipt['path']))['sha256']==old_receipt['sha256'] and pin(Path(new_receipt['path']))['sha256']==new_receipt['sha256']
  carry_input=json.loads((D/'memory-r1-accounting-carry.json').read_text());assert pin(OLD/'runs/dspark-a8-twelve-hour-20261007-01/lane/training/checkpoints/latest.json')==carry_input['original_valid_checkpoint_receipt']
  old=load_checkpoint(Path(old_receipt['path']));new=load_checkpoint(Path(new_receipt['path']))
  def equal(a,b):
   if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
   if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
   if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
   if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
   return type(a)==type(b) and a==b
  for key in ['linears','optimizer','rng','cursor','contract','runtime']:assert equal(old[key],new[key]),'committed_zero_equivalence:'+key
  assert not new['optimizer']['state'];del old,new
  carry=json.loads((D/'memory-r1-accounting-carry.json').read_text());original=json.loads(Path(carry['original_budget']['path']).read_text());assert pin(Path(carry['original_budget']['path']))==carry['original_budget'];assert original['active_attempt'] is None and original['training_seconds']==carry['charged_seconds']==4.706928438999967 and original['max_seconds']==43200
  budget=training/'budget-used.json';assert not budget.exists();atomic_json(budget,dict(schema='continuous_training_budget_v1',source_sha256=pin(lane_path)['sha256'],max_seconds=43200,training_seconds=carry['charged_seconds'],active_attempt=None))
  state=root/'state.json';assert not state.exists();atomic_json(state,dict(schema='nine_model_lane_state_v1',candidate='dspark_a8',bundle_sha256=pin(lane_path)['sha256'],status='recovery_prepared',campaign_complete=False,rollback_provenance=pin(D/'memory-r1-accounting-carry.json')))
  report.update(status='PASS',old_checkpoint={'path':old_receipt['path'],'sha256':old_receipt['sha256']},new_checkpoint={'path':new_receipt['path'],'sha256':new_receipt['sha256']},exact_committed_zero_tensors_optimizer_rng_cursor_contract_runtime=True,lost_step2_exact_resume=False,charged_seconds=carry['charged_seconds'],new_derived_budget_initial=pin(budget),original_budget_immutable=carry['original_budget'],run_dir=str(root))
 except BaseException as e:report.update(status='FAIL',failure={'type':type(e).__name__,'message':str(e)});raise
 finally:
  proof=obs.require_released(runner.process_groups,runner.process_identities);report['release']=proof;report['resources']=obs.snapshot()
  if cache.state['owner'] and cache.state['owner']['owner']=='root-memory-r1-bootstrap':cache.release('root-memory-r1-bootstrap',released_proof=proof)
  atomic_json(D/'memory-r1-paid-zero-bootstrap.json',report)
print(json.dumps({'status':report['status'],'charged_seconds':report.get('charged_seconds'),'committed_zero_exact':report.get('exact_committed_zero_tensors_optimizer_rng_cursor_contract_runtime')}))
