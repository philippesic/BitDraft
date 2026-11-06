from pathlib import Path
import json,os,subprocess,time,hashlib,signal
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';plan=json.loads((R/'data-native-comparison-plan.json').read_text());start=time.monotonic();stopping=False;allresults=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def pin(f):
 f=Path(f).resolve()
 with f.open('rb') as x:h=hashlib.file_digest(x,'sha256').hexdigest()
 return {'path':str(f),'bytes':f.stat().st_size,'sha256':h}
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','LD_LIBRARY_PATH':str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin')};env.pop('LD_PRELOAD',None)
for arm in plan['arms']:
 manifest=json.loads(Path(arm['packet_manifest']['path']).read_text());assert pin(arm['packet_manifest']['path'])==arm['packet_manifest'];out=Path(arm['output']);out.mkdir(parents=True,exist_ok=False);results=[];armstart=time.monotonic()
 for i,entry in enumerate(manifest['packets']):
  assert not stopping;assert pin(entry['packet']['path'])==entry['packet'];batch=out/f'batch-{i:02d}';batch.mkdir()
  for role in ['parent','candidate']:
   assert not stopping;argv=[str(R/'actual-native-build01/modern_actual_native_replay'),'--cases',entry['packet']['path'],'--model',role,'--out',str(batch/role),'--allow-cuda']
   if role=='candidate':argv+=['--calibration',str(batch/'parent/replay.json')]
   with (batch/(role+'.log')).open('x') as log:
    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(child.pid);(batch/(role+'-identity.json')).write_text(json.dumps({'pid':child.pid,'pgid':os.getpgid(child.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');assert child.wait()==0,(arm['arm'],i,role,child.returncode)
   f=batch/role/'replay.json';d=json.loads(f.read_text());assert d['status']=='ACTUAL_NATIVE_REPLAY_COMPLETE' and d['native_successes_outside_parent_calibration']==0
   if role=='parent':assert d['native_reference_tokens']==d['calibrated_reference_tokens'] and d['lost_calibrated_reference_tokens']==0
   result={'arm':arm['arm'],'batch':i,'role':role,'windows':len(d['windows']),'native_reference_tokens':d['native_reference_tokens'],'calibrated_reference_tokens':d['calibrated_reference_tokens'],'lost_calibrated_reference_tokens':d['lost_calibrated_reference_tokens'],'wall_seconds':d['wall_seconds'],'replay':pin(f)};results.append(result);print(json.dumps(result),flush=True)
  (out/'progress.json').write_text(json.dumps({'results':results,'wall_seconds':time.monotonic()-armstart},indent=2)+'\n')
 summary={role:{k:sum(x[k] for x in results if x['role']==role) for k in ['windows','native_reference_tokens','calibrated_reference_tokens','lost_calibrated_reference_tokens']} for role in ['parent','candidate']};assert summary['parent']['windows']==summary['candidate']['windows']==883 and summary['parent']['native_reference_tokens']==summary['candidate']['native_reference_tokens']==956
 (out/'result.json').write_text(json.dumps({'status':'ALL956_NATIVE_PRESERVATION_EVALUATED','arm':arm['arm'],'model':manifest['model'],'packets':arm['packet_manifest'],'results':results,'summary':summary,'wall_seconds':time.monotonic()-armstart,'scope':'allold549+new407 fixed-source histories; no servingthroughput/freshclaim'},indent=2)+'\n');allresults.append({'arm':arm['arm'],'summary':summary,'result':pin(out/'result.json'),'wall_seconds':time.monotonic()-armstart})
(R/'data-native-comparison-result.json').write_text(json.dumps({'status':'MATCHED_OLD_NEW_DATA_FULL956_NATIVE_COMPLETE','arms':allresults,'wall_seconds':time.monotonic()-start,'scope':'nativepreservation only, not fullrequesttiming oruntouchedgeneralization'},indent=2)+'\n')
