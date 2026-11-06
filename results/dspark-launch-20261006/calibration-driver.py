"""Root-owned serial preparation; zero optimizer updates."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/home/philip/binary-eagle-decoding')
CHECKOUT=ROOT/'runs/checkouts/dspark-data-capture-20261006-01'
sys.path.insert(0,str(CHECKOUT/'src'));sys.path.insert(0,str(CHECKOUT/'scripts'))
import capture_nine_model_train_data as capture
from w1a1_eagle.nine_model_pipeline import LinuxResources
PLANS=ROOT/'runs/dspark-a8-full-pool-20261006-01/plans'
OUT=PLANS.parent
SCHEDULE=PLANS/'full-pool-schedule.json'
assert capture.file_sha256(SCHEDULE)=='7abf663b1d0cf46fe714bfe5c6b071e852b9e2d876159a3e4c3e2754b6e90d44'
schedule=json.loads(SCHEDULE.read_text())
chunks=[c for c in schedule['chunks'] if c['phase']=='calibration']
assert len(chunks)==14 and sum(len(c['prompt_ids']) for c in chunks)==144
chunks += [next(c for c in schedule['chunks'] if c['phase']=='train')]
resources=LinuxResources('GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e')
start=time.monotonic();completed=[]
progress=ROOT/'results/dspark-launch-20261006/calibration-driver-progress.json'
def unique_bytes(directories):
 seen=set();total=0
 for directory in directories:
  for p in directory.rglob('*'):
   if p.is_file():
    s=p.stat();k=(s.st_dev,s.st_ino)
    if k not in seen:seen.add(k);total+=s.st_size
 return total
for chunk in chunks:
 phase=chunk['phase'];ordinal=chunk['phase_index']
 output=OUT/f'{phase}-{ordinal:04d}'
 pin=chunk['plans']['remote'];plan=PLANS/pin['path']
 assert capture.file_sha256(plan)==pin['sha256']
 if output.exists():
  report=json.loads((output/'capture-report.json').read_text())
  assert ordinal==0 and phase=='calibration' and report['status']=='PASS'
  assert report['completed_prompt_count']==len(chunk['prompt_ids']) and report['producer_closed'] is True
 else:
  resources.require_released([],[])
  observed=resources.snapshot()
  assert observed['host_available_bytes']>=4*1024**3
  cal_bytes=unique_bytes(list(OUT.glob('calibration-*')))
  if phase=='calibration':assert cal_bytes+8*1024**3<=64*1024**3,(cal_bytes,'cal cap')
  import shutil
  # Reserved future TRAIN cache, checkpoint generations, exports, metadata and free floor.
  assert shutil.disk_usage(OUT).free-8*1024**3>=104*1024**3,'next writer/resource reserve'
  mapping=dict(zip(chunk['prompt_ids'],chunk['chain_ids'],strict=True))
  report=capture.run_capture(plan,pin['sha256'],output,execute=True,chain_ids=mapping)
  assert report['status']=='PASS',report.get('failure')
  assert report['completed_prompt_count']==len(chunk['prompt_ids']) and report['producer_closed'] is True
 resources.require_released([],[])
 completed.append({'chunk_id':chunk['chunk_id'],'phase':phase,'phase_index':ordinal,'prompts':report['completed_prompt_count'],'wall_seconds':report['elapsed_wall_seconds'],'retained_bytes':report['retained_bytes'],'report':{'path':str(output/'capture-report.json'),'sha256':capture.file_sha256(output/'capture-report.json')},'manifests':report['manifests']})
 state={'schema':'dspark_serial_calibration_preparation_v1','status':'RUNNING' if len(completed)<len(chunks) else 'PASS','phase':'calibration_and_first_train_capture','optimizer_updates':0,'driver_sha256':capture.file_sha256(Path(__file__)),'capture_source':'4ba3f88b','schedule_sha256':capture.file_sha256(SCHEDULE),'elapsed_wall_seconds':time.monotonic()-start,'completed':completed,'calibration_raw_bytes':unique_bytes(list(OUT.glob('calibration-*'))),'calibration_cap_bytes':64*1024**3}
 tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(progress)
 print(json.dumps({'chunk_id':chunk['chunk_id'],'phase':phase,'prompts':report['completed_prompt_count'],'capture_wall_seconds':report['elapsed_wall_seconds'],'retained_bytes':report['retained_bytes'],'completed_chunks':len(completed)}),flush=True)
