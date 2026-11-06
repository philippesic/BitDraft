import json,sys,time,subprocess,shlex
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';O=R/'runs/dspark-a8-full-pool-20261006-01';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json
from w1a1_eagle.block_shard_lifecycle import pin,open_provider
from run_nine_model_lane import validate_lane
p=O/'lane-source-memory-r1.json';sha=pin(p)['sha256'];lane,files,plan=validate_lane(p,sha)
bootstrap=json.loads((D/'memory-r1-paid-zero-bootstrap.json').read_text());assert bootstrap['status']=='PASS' and bootstrap['exact_committed_zero_tensors_optimizer_rng_cursor_contract_runtime'] is True and bootstrap['charged_seconds']==4.706928438999967 and bootstrap['lost_step2_exact_resume'] is False
root=O/'training-memory-r1-lane';b=json.loads((root/'training/budget-used.json').read_text());assert b['source_sha256']==sha and b['max_seconds']==43200 and b['active_attempt'] is None and b['training_seconds']==4.706928438999967
obs=LinuxResources(lane['gpu_uuid']);proof=obs.require_released([],[]);snap=obs.snapshot();assert not proof['other_context_pids'] and not snap['dxg_holders'];assert json.loads(Path(lane['gpu_control_path']).read_text())['rtx5080']['pause_requested'] is False
assert open_provider(json.loads(Path(lane['config']['path']).read_text())['data']).cache.state['owner'] is None
run='dspark-a8-memory-r1-training-20261007-01';run_root=C/'runs'/run;assert not run_root.exists()
lease=O/'launch-lease-memory-r1.json';assert not lease.exists();t=time.time();atomic_json(lease,dict(schema='nine_model_gpu_lease_v1',host='rtx5080',user_announced_available=True,sole_owner=True,pause_requested=False,bundle_sha256=sha,gpu_uuid=lane['gpu_uuid'],granted_unix=t,expires_unix=t+300,prelaunch_release=proof,prelaunch_resource_snapshot=snap))
cmd=[str(R/'.venv/bin/python'),str(C/'scripts/remote_job.py'),run,'--stop-grace-seconds','90','--','/usr/bin/env','PYTHONPATH='+str(R/'third_party/llama.cpp/gguf-py'),str(R/'.venv/bin/python'),str(C/'scripts/run_nine_model_lane.py'),'--lane',str(p),'--lane-sha256',sha,'--run-dir',str(root),'--availability',str(lease),'--supervisor-state',str(run_root/'state.json'),'--start','--resume']
subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s','root-'+run,shlex.join(cmd)],check=True)
atomic_json(D/'memory-r1-training-launch.json',dict(schema='dspark_memory_r1_training_launch_v1',run=run,source_checkout=str(C),lane=pin(p),run_dir=str(root),supervisor_state=str(run_root/'state.json'),charged_carry_seconds=b['training_seconds'],command=cmd,stage='fresh_admission_pending_no_repair_paid_updates_observed'))
print(json.dumps({'run':run,'lane':pin(p),'run_dir':str(root),'carry_seconds':b['training_seconds'],'stage':'admission_pending'}))
