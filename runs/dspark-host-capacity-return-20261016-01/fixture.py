"""One bounded real host-capacity hold; no GPU APIs or scientific settings."""
from pathlib import Path
import datetime,hashlib,json,mmap,os,sys,time,signal
RUN=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane')
OP=RUN/'direct-runtime/direct-dspark-bfb4b78cf9fd460b81426282e48e1d5c'
OUT=Path(__file__).parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(OP/'source/src'))
from w1a1_eagle.direct_training_runtime import metadata,live_metadata,checked,current_control,write_new,status_snapshot
runtime=checked({'path':str(OP/'runtime.json'),'sha256':'bfd8c2263f4324ce002bbf3f117b1a99bb7c0feccbb5f77ee873aa175ab721cf'})
config=checked(runtime['config']);lane=checked(config['lane']);owned=checked({'path':str(OP/'owned-identity.json'),'sha256':'58cd1aae43a6828484889ec0e11fe0594c928ff9d2fa7761d0c75943ac42b466'})
def identity(pid):
 p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return {'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':pid,'start_ticks':int(f[19])}
def authorize():
 current_control(lane)
 assert not(RUN/'STOP').exists() and not(RUN/'training/STOP').exists()
 for role,row in owned.items():assert identity(row['pid'])==row
 sup=live_metadata(runtime['supervisor_state'])[0];assert sup['status']=='running' and sup.get('received_signal') is None
 return True
def available():
 return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
def rss():
 return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
def stop(signum,frame):raise InterruptedError('capacity fixture signal '+str(signum))
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
report={'schema':'real_host_capacity_hold_v1','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':identity(os.getpid()),'runtime_sha256':'bfd8c2263f4324ce002bbf3f117b1a99bb7c0feccbb5f77ee873aa175ab721cf','gpu_accessed':False,'settings_changed':False,'bytes_reserved':5*1024**3,'minimum_host_available_bytes':8*1024**3,'hold_limit_seconds':300,'status':'NOT_PROVEN'}
allocation=None
try:
 authorize();budget,bsha=live_metadata(RUN/'training/budget-used.json');status=status_snapshot(config)
 assert budget['active_attempt'] is not None,'fixture requires actual active TRAIN, never loader'
 assert status['current_updates']>21523,'new optimizer progress required before fixture'
 state=live_metadata(RUN/'state.json')[0];baseline=state['resource_baseline'];policy=lane['resource_policy']
 threshold=max(policy['host_floor_bytes'],baseline['host_available_bytes']-policy['host_return_tolerance_bytes']);before=available()
 assert before-report['bytes_reserved']>=report['minimum_host_available_bytes'],'insufficient bounded host-memory headroom; NO_ALLOCATION'
 allocation=mmap.mmap(-1,report['bytes_reserved']);chunk=b'x'*(1024**2)
 for offset in range(0,report['bytes_reserved'],len(chunk)):
  allocation[offset:offset+len(chunk)]=chunk
  if offset%(128*1024**2)==0:authorize()
 held_at=time.time();held_available=available();assert held_available>=report['minimum_host_available_bytes']
 report.update(held_at_unix=held_at,host_before=before,host_held=held_available,rss_held=rss(),threshold=threshold,baseline=baseline,policy=policy,budget_before_sha256=bsha)
 write_new(OUT/'holding.json',dict(report,status='HOLDING_WAITING_FOR_ONE_NORMAL_EVAL_REQUEST'))
 deadline=time.monotonic()+300;waits=[]
 while time.monotonic()<deadline:
  authorize();path=RUN/'resource-handoff-progress.json'
  if path.exists():
   progress=live_metadata(path)[0]
   assert progress.get('controller_pid')==owned['controller']['pid']
   waits=[x for x in progress.get('observations',[]) if x.get('observed_unix',0)>=held_at and x.get('status')=='WAITING_FOR_RESOURCES' and x.get('deficits_bytes',{}).get('host_available_bytes',0)>0]
   if len(waits)>=2 and waits[-1].get('capacity_wait_wall_seconds',0)>0:
    inactive,sha=live_metadata(RUN/'training/budget-used.json');latest,lsha=live_metadata(RUN/'training/checkpoints/latest.json')
    assert inactive['active_attempt'] is None and latest['cursor']['step']>21523
    assert inactive['training_seconds']==latest['cursor']['elapsed_seconds']
    report.update(status='REAL_WAIT_OBSERVED',wait_observations=waits,budget_while_waiting=inactive,budget_while_waiting_sha256=sha,latest_while_waiting_sha256=lsha)
    break
  time.sleep(1)
 else:raise TimeoutError('No two genuine host-deficit resource waits; release without PASS')
except BaseException as error:
 report['error']={'type':type(error).__name__,'reason':str(error)}
finally:
 if allocation is not None:allocation.close()
 report['released_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();report['host_after_release']=available();report['rss_after_release']=rss()
 out=OUT/'fixture-result.json';write_new(out,report);raw=out.read_bytes();print(json.dumps({'path':str(out),'sha256':hashlib.sha256(raw).hexdigest(),'status':report['status'],'error':report.get('error')}),flush=True)
if report.get('error'):raise SystemExit(1)
