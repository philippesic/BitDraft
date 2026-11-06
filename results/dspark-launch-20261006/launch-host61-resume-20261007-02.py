import json,sys,time,subprocess,shlex,shutil,hashlib,fcntl
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';O=R/'runs/dspark-a8-full-pool-20261006-01';root=O/'training-memory-r1-lane';D=R/'results/dspark-launch-20261006';sys.path[:0]=[str(C/'src'),str(C/'scripts')]
from w1a1_eagle.nine_model_pipeline import LinuxResources,atomic_json
from w1a1_eagle.block_shard_lifecycle import pin,open_provider
from w1a1_eagle.block_training import protect_block_checkpoint
restored=json.loads((D/'host61-admission-shard-restored-20261007-01.json').read_text()); assert restored['status']=='PASS' and restored['budget_and_paid_cursor_unchanged'] is True and restored['owned_release']['other_context_pids']==[]
lane_path=O/'lane-source-memory-r1.json';sha=pin(lane_path)['sha256'];assert sha=='0a7d98fc5f8dff8eb89abb9dd966332776c9142beacc543bd3e26053a70902ce';lane=json.loads(lane_path.read_text())
assert subprocess.run(['git','-C',str(C),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()=='968ab3d1bd45ab70b2488bf0f5e05efa6fc19363'
assert pin(C/'scripts/train_nine_model_qat.py')['sha256']=='5c1fe880aa47eaaa40df9f280c0294612467016ecad24a62ca9d637ed601d665'
lock=Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock';lock.parent.mkdir(parents=True,exist_ok=True)
with lock.open('a') as owner:
 fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 b=json.loads((root/'training/budget-used.json').read_text());cp=json.loads((root/'training/checkpoints/latest.json').read_text());state=json.loads((root/'state.json').read_text())
 assert b['source_sha256']==sha and b['max_seconds']==43200 and b['active_attempt'] is None and b['training_seconds']==4310.6791790700045
 assert cp['committed'] is True and cp['cursor']['step']==11161 and cp['cursor']['elapsed_seconds']==b['training_seconds']
 assert cp['sha256']=='1c9bf9fca335740cc791cd3281e22d5fe252cf5703f0eecb3b7f4508106404f1' and pin(Path(cp['path']))['sha256']==cp['sha256']
 assert state['bundle_sha256']==sha and state['candidate']=='dspark_a8' and state['status']!='training_complete' and not (root/'STOP').exists()
 ids=[dict(pid=2645,start_ticks=480460,boot_id='206feba6-9535-4483-892a-9807057589ee'),dict(pid=2646,start_ticks=480468,boot_id='206feba6-9535-4483-892a-9807057589ee')];pgids=[2646]
 for p in root.glob('**/process.json'):
  x=json.loads(p.read_text());assert x.get('kernel_identity');ids.append(x['kernel_identity']);pgids.append(x['pgid'])
 for p in root.glob('**/process-lineage.json'): ids.extend(json.loads(p.read_text())['kernel_identities'])
 obs=LinuxResources(lane['gpu_uuid']);proof=obs.require_released(pgids,ids);snap=obs.snapshot();assert not proof['other_context_pids'] and not snap['dxg_holders']
 assert snap['host_available_bytes']>=4294967296
 backup=D/'host61-resume-20261007-02-original-state';assert not backup.exists();backup.mkdir()
 oldjob=C/'runs/dspark-a8-memory-r1-training-20261007-01/state.json'
 for name,p in [('old-supervisor.json',oldjob),('lane-state.json',root/'state.json'),('trainer-status.json',root/'training/status.json'),('budget-used.json',root/'training/budget-used.json'),('latest-original.json',root/'training/checkpoints/latest.json'),('cache-state.json',R/'runs/dspark-a8-train-cache-20261006-01/cache-state.json')]: shutil.copyfile(p,backup/name)
 protected=protect_block_checkpoint(cp,'stop');assert protected['sha256']==cp['sha256'] and protected['cursor']==cp['cursor']
 cfg=json.loads(Path(lane['config']['path']).read_text());provider=open_provider(cfg['data']);cache=provider.cache;before_owner=cache.state['owner']
 if before_owner is not None: cache.release(before_owner['owner'],released_proof=proof)
 assert cache.state['owner'] is None
 control_path=Path(lane['gpu_control_path']);control=json.loads(control_path.read_text());control['rtx5080']['pause_requested']=False;control['rtx5080']['updated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());atomic_json(control_path,control)
 run='dspark-a8-host61-resume-20261007-02';run_root=C/'runs'/run;assert not run_root.exists()
 lease=O/'launch-lease-host61-resume-20261007-02.json';assert not lease.exists();t=time.time();atomic_json(lease,dict(schema='nine_model_gpu_lease_v1',host='rtx5080',user_announced_available=True,sole_owner=True,pause_requested=False,bundle_sha256=sha,gpu_uuid=lane['gpu_uuid'],granted_unix=t,expires_unix=t+300,prelaunch_release=proof,prelaunch_resource_snapshot=snap))
 cmd=[str(R/'.venv/bin/python'),str(C/'scripts/remote_job.py'),run,'--stop-grace-seconds','90','--','/usr/bin/env','PYTHONPATH='+str(R/'third_party/llama.cpp/gguf-py'),str(R/'.venv/bin/python'),str(C/'scripts/run_nine_model_lane.py'),'--lane',str(lane_path),'--lane-sha256',sha,'--run-dir',str(root),'--availability',str(lease),'--supervisor-state',str(run_root/'state.json'),'--start','--resume']
 subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s','root-'+run,shlex.join(cmd)],check=True)
 receipt=dict(schema='dspark_host61_resume_launch_v1',run=run,source_checkout=str(C),source_commit='968ab3d1bd45ab70b2488bf0f5e05efa6fc19363',lane=pin(lane_path),run_dir=str(root),supervisor_state=str(run_root/'state.json'),budget_seconds_preserved=b['training_seconds'],remaining_trainer_seconds=43200-b['training_seconds'],checkpoint_before_resume={'path':cp['path'],'sha256':cp['sha256'],'step':11161,'payload_directly_verified':True,'protected':protected['retention']['protected']},original_state_backup=str(backup),old_owned_release=proof,resource_snapshot=snap,released_stale_cache_owner=before_owner,command=cmd,stage='fresh_admission_pending_no_new_optimizer_updates_observed',observed_unix=time.time())
 dest=D/'host61-resume-launch-20261007-02.json';assert not dest.exists();atomic_json(dest,receipt);print(json.dumps(receipt))
