import json,sys,time,subprocess,shlex
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-research-training-92516f8d-20261006';O=R/'runs/dspark-a8-full-pool-20261006-01';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json
from w1a1_eagle.block_shard_lifecycle import pin,open_provider
from run_nine_model_lane import validate_lane
lane_path=O/'lane-source-v4.json';sha=pin(lane_path)['sha256'];lane,files,plan=validate_lane(lane_path,sha)
run='dspark-a8-twelve-hour-20261007-01';run_root=C/'runs'/run;assert not run_root.exists()
prior=json.loads((C/'runs/dspark-a8-v4-prelaunch-20261007-03/state.json').read_text());assert prior['status']=='finished' and prior['exit_code']==0
obs=LinuxResources(lane['gpu_uuid']);proof=obs.require_released([],[]);snap=obs.snapshot();assert proof['other_context_pids']==[] and snap['dxg_holders']==[]
control=json.loads(Path(lane['gpu_control_path']).read_text());assert control['rtx5080']['pause_requested'] is False
spec=json.loads(Path(lane['config']['path']).read_text());provider=open_provider(spec['data']);assert provider.cache.state['owner'] is None
lease_path=O/'launch-lease-v4.json';assert not lease_path.exists();t=time.time()
atomic_json(lease_path,dict(schema='nine_model_gpu_lease_v1',host='rtx5080',user_announced_available=True,sole_owner=True,pause_requested=False,bundle_sha256=sha,gpu_uuid=lane['gpu_uuid'],granted_unix=t,expires_unix=t+300,standing_authorization=lane['authorization'],prelaunch_release=proof,prelaunch_resource_snapshot=snap))
cmd=[str(R/'.venv/bin/python'),str(C/'scripts/remote_job.py'),run,'--stop-grace-seconds','90','--','/usr/bin/env','PYTHONPATH='+str(R/'third_party/llama.cpp/gguf-py'),str(R/'.venv/bin/python'),str(C/'scripts/run_nine_model_lane.py'),'--lane',str(lane_path),'--lane-sha256',sha,'--run-dir',str(run_root/'lane'),'--availability',str(lease_path),'--supervisor-state',str(run_root/'state.json'),'--start']
subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s','root-'+run,shlex.join(cmd)],check=True)
atomic_json(D/'training-launch-v4.json',dict(schema='dspark_training_launch_v4',run_id=run,source_checkout=str(C),lane=pin(lane_path),tmux_socket='binary-eagle-runtime',tmux_session='root-'+run,command=cmd,phase='controller_started_admission_pending',positive_optimizer_updates_unobserved=True))
print(json.dumps({'run':run,'lane':pin(lane_path),'phase':'admission_pending'}))
