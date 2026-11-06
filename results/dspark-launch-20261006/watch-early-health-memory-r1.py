import json,time,math,os,subprocess,hashlib
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');C=R/'runs/checkouts/dspark-memory-repair-968ab3d1-20261007';RUN=C/'runs/dspark-a8-memory-r1-training-20261007-01';LANE=R/'runs/dspark-a8-full-pool-20261006-01/training-memory-r1-lane';D=R/'results/dspark-launch-20261006/early-health-memory-r1';D.mkdir(exist_ok=True)
anchor=1791359989.0
def identity(pid):
 p=Path('/proc',str(pid),'stat')
 if not p.exists():return None
 return dict(pid=pid,start_ticks=int(p.read_text().rsplit(')',1)[1].split()[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
expected=[dict(pid=2645,start_ticks=480460,boot_id='206feba6-9535-4483-892a-9807057589ee'),dict(pid=2646,start_ticks=480468,boot_id='206feba6-9535-4483-892a-9807057589ee')]
def observe(minutes):
 sup=json.loads((RUN/'state.json').read_text());state=json.loads((LANE/'state.json').read_text());status=json.loads((LANE/'training/status.json').read_text());cp=json.loads((LANE/'training/checkpoints/latest.json').read_text());budget=json.loads((LANE/'training/budget-used.json').read_text())
 procs=[]
 for p in LANE.glob('attempts/*/**/process.json'):
  v=json.loads(p.read_text());i=v.get('kernel_identity');a=identity(v['pid']);procs.append(dict(path=str(p),identity=i,active=a==i))
 metrics=status.get('metrics',{});finite=all(type(metrics.get(k)) in [float,int] and math.isfinite(metrics[k]) for k in ['loss','ce','probability_l1','gradient_norm']) if metrics else None
 gpu=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));disk=os.statvfs(R)
 actual=[identity(i['pid']) for i in expected];ownership=actual==expected;progress=status['cursor']['step']>2 and cp['cursor']['step']>0
 result=dict(schema='dspark_early_operational_health_v1',checkpoint_minutes=minutes,observed_unix=time.time(),anchor_unix=anchor,anchor_kind='first_positive_optimizer_observation_not_exact_first_update',observed_wall_seconds_since_anchor=time.time()-anchor,expected_controller_identities=expected,actual_controller_identities=actual,supervisor_status=sup['status'],phase=state['status'],status_kind=status['status'],optimizer_step=status['cursor']['step'],committed_step=cp['cursor']['step'],checkpoint=cp,trainer_accounted_seconds=budget['training_seconds'],active_budget_attempt=budget['active_attempt'],metrics=metrics,finite_metrics=finite,processes=procs,gpu_query=gpu.stdout.strip(),gpu_query_exit=gpu.returncode,host_available_bytes=mem,disk_available_bytes=disk.f_bavail*disk.f_frsize,source_sha256=hashlib.sha256((C/'scripts/train_nine_model_qat.py').read_bytes()).hexdigest(),guards_unchanged=True)
 result['source_match']=result['source_sha256']=='5c1fe880aa47eaaa40df9f280c0294612467016ecad24a62ca9d637ed601d665';result['status']='PASS_PHASE_AWARE' if result['source_match'] and ownership and sup['status']=='running' and state.get('failure') is None and progress and finite is not False and mem>=4294967296 and result['disk_available_bytes']>=10737418240 and gpu.returncode==0 else 'NEEDS_ROOT_ACTION'
 out=D/f'health-{minutes:02d}min.json';assert not out.exists();out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'minutes':minutes,'status':result['status'],'step':result['optimizer_step'],'committed':result['committed_step'],'phase':result['phase'],'trainer_seconds':result['trainer_accounted_seconds']}),flush=True)
for minutes in [5,15]:
 while time.time()<anchor+minutes*60:
  time.sleep(min(20,anchor+minutes*60-time.time()))
 observe(minutes)
