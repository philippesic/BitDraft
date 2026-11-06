from pathlib import Path
import json,hashlib,os,subprocess,time,signal,datetime
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts');root=C/'w1a8-added-replay-pair-20261103-01/native96';started=time.monotonic();stopping=False
signal.signal(signal.SIGTERM,lambda s,f:globals().__setitem__('stopping',True));signal.signal(signal.SIGINT,lambda s,f:globals().__setitem__('stopping',True))
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
training=read(R/'added-replay-pair-result.json');assert training['status']=='ADDED_FIT_REPLAY_TRAINED_NOT_NATIVE_ADMITTED';stage=next(x for x in training['stages'] if x['added_replay']=='true');model=stage['model'];assert pin(model['path'])==model
oldplan=read(R/'head-native96-plan02.json');control=next(x for x in oldplan['packets'] if x['arm']=='control');assert pin(control['packet']['path'])==control['packet'];template=read(control['packet']['path']);oldresults=read(R/'head-native96-result.json');cached=[x for x in oldresults['results'] if x['arm']=='control'];assert len(cached)==2
for x in cached:assert pin(x['replay']['path'])==x['replay']
root.mkdir(exist_ok=False);packet={**template,'models':{**template['models'],'candidate':model}};path=root/'cases.json';path.write_text(json.dumps(packet,separators=(',',':')));plan=dict(status='ADDED_REPLAY_ON_IDENTICAL_FROZEN96_CASES',packet=pin(path),model=model,cached_control=cached,cached_control_packet=control['packet'],selection=oldplan['selection'],source=pin(__file__),scope='unchangedcorrectnesscontrol evidence reused, NOT timingcontrol; newparent calibration becausepacketmodelpin changes; 113positives/96windows not full956')
(R/'added-replay-native96-plan.json').write_text(json.dumps(plan,indent=2)+'\n');env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','LD_LIBRARY_PATH':str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin')};env.pop('LD_PRELOAD',None);results=[]
for role in ['parent','candidate']:
 assert not stopping;args=[str(R/'actual-native-build01/modern_actual_native_replay'),'--cases',str(path),'--model',role,'--out',str(root/role),'--allow-cuda']
 if role=='candidate':args+=['--calibration',str(root/'parent/replay.json')]
 with (root/(role+'.log')).open('x') as log:
  proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(proc.pid);i=dict(pid=proc.pid,pgid=os.getpgid(proc.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(),at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(root/(role+'-identity.json')).write_text(json.dumps(i,indent=2)+'\n');assert proc.wait()==0,(role,proc.returncode)
 f=root/role/'replay.json';v=read(f);assert v['status']=='ACTUAL_NATIVE_REPLAY_COMPLETE' and v['native_reference_tokens']==v['calibrated_reference_tokens']==113 and v['native_successes_outside_parent_calibration']==0 and len(v['windows'])==96
 if role=='parent':assert v['lost_calibrated_reference_tokens']==0
 result=dict(arm='added_replay',role=role,lost=v['lost_calibrated_reference_tokens'],successes=113,windows=96,wall_seconds=v['wall_seconds'],replay=pin(f));results.append(result);print(json.dumps(result),flush=True)
(R/'added-replay-native96-result.json').write_text(json.dumps(dict(status='ADDED_REPLAY_NATIVE96_COMPLETE',results=results,cached_control=cached,plan=pin(R/'added-replay-native96-plan.json'),wall_seconds=time.monotonic()-started,scope=plan['scope']),indent=2)+'\n')
