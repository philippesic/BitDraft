from pathlib import Path
import json,hashlib,os,subprocess,time,signal,datetime
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');pair=C/'w1a8-head-pair-20261103-01';root=pair/'native96-retry02';started=time.monotonic();stopping=False;results=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
training=read(R/'head-matched5760-result.json');assert training['status']=='MATCHED1920_CAPACITY_TRAINING_COMPLETE_NOT_NATIVE_ADMITTED'
original=read(R/'data-backoff-plan.json');template_entry=original['packets'][0];assert pin(template_entry['packet']['path'])==template_entry['packet'];template=read(template_entry['packet']['path']);assert len(template['cases'])==96 and sum(w['native_prefix'] for w in template['cases'])==113
root.mkdir(exist_ok=False);packets=[]
for arm in ['control','head']:
 stage=next(x for x in training['stages'] if x['arm']==arm);d=read(Path(stage['out'])/'result.json');assert d['model']==stage['model'] and pin(d['model']['path'])==d['model'];packet={**template,'models':{**template['models'],'candidate':d['model']}};f=root/(arm+'-cases.json');f.write_text(json.dumps(packet,separators=(',',':')));assert read(f)['cases']==template['cases'];packets.append(dict(arm=arm,model=d['model'],packet=pin(f)))
plan=dict(status='FROZEN_PARENT_MARGIN96_CHECK_BEFORE_ANY_HEAD_NATIVE_OUTCOME',original_selection=pin(R/'data-backoff-plan.json'),selection=original['selection'],training=pin(R/'head-matched5760-result.json'),source=pin(__file__),packets=packets,scope='preservation diagnostic113positives/96windows across60groups; fail establishes failureatthischoice, passrequiresfull956; notvalidatedquickpredictor')
(R/'head-native96-plan02.json').write_text(json.dumps(plan,indent=2)+'\n')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','LD_LIBRARY_PATH':str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin')};env.pop('LD_PRELOAD',None)
for entry in packets:
 arm=entry['arm'];out=root/arm;out.mkdir()
 for role in ['parent','candidate']:
  assert not stopping;argv=[str(R/'actual-native-build01/modern_actual_native_replay'),'--cases',entry['packet']['path'],'--model',role,'--out',str(out/role),'--allow-cuda']
  if role=='candidate':argv+=['--calibration',str(out/'parent/replay.json')]
  with (out/(role+'.log')).open('x') as log:
   child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(child.pid);identity=dict(pid=child.pid,pgid=os.getpgid(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/(role+'-identity.json')).write_text(json.dumps(identity,indent=2)+'\n');assert child.wait()==0,(arm,role,child.returncode)
  f=out/role/'replay.json';d=read(f);assert d['status']=='ACTUAL_NATIVE_REPLAY_COMPLETE' and d['native_reference_tokens']==d['calibrated_reference_tokens']==113 and d['native_successes_outside_parent_calibration']==0 and len(d['windows'])==96
  if role=='parent':assert d['lost_calibrated_reference_tokens']==0
  result=dict(arm=arm,role=role,lost=d['lost_calibrated_reference_tokens'],successes=113,windows=96,wall_seconds=d['wall_seconds'],replay=pin(f));results.append(result);print(json.dumps(result),flush=True)
(R/'head-native96-result.json').write_text(json.dumps(dict(status='MATCHED_HEAD_CAPACITY_NATIVE96_DIAGNOSTIC_COMPLETE',plan=pin(R/'head-native96-plan02.json'),results=results,wall_seconds=time.monotonic()-started,scope=plan['scope']),indent=2)+'\n')
