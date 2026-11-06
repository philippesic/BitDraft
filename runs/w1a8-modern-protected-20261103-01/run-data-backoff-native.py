from pathlib import Path
import json,subprocess,os,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';plan=json.loads((R/'data-backoff-plan.json').read_text());start=time.monotonic();results=[]
def pin(f):
 f=Path(f).resolve()
 with f.open('rb') as x:h=hashlib.file_digest(x,'sha256').hexdigest()
 return {'path':str(f),'bytes':f.stat().st_size,'sha256':h}
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','CUDA_CACHE_PATH':str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),'CUDA_CACHE_MAXSIZE':'1073741824','LD_LIBRARY_PATH':str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin')};env.pop('LD_PRELOAD',None)
for entry in plan['packets']:
 assert pin(entry['packet']['path'])==entry['packet'];out=Path(entry['output']);out.mkdir(parents=True,exist_ok=False)
 for role in ['parent','candidate']:
  argv=[str(R/'actual-native-build01/modern_actual_native_replay'),'--cases',entry['packet']['path'],'--model',role,'--out',str(out/role),'--allow-cuda']
  if role=='candidate':argv+=['--calibration',str(out/'parent/replay.json')]
  with (out/(role+'.log')).open('x') as log:
   child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(child.pid);(out/(role+'-identity.json')).write_text(json.dumps({'pid':child.pid,'pgid':os.getpgid(child.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');assert child.wait()==0,(entry['id'],role,child.returncode)
  d=json.loads((out/role/'replay.json').read_text());assert len(d['windows'])==96 and d['native_successes_outside_parent_calibration']==0
  if role=='parent':assert d['native_reference_tokens']==d['calibrated_reference_tokens'] and d['lost_calibrated_reference_tokens']==0
  result={'id':entry['id'],'role':role,'native_successes':d['native_reference_tokens'],'lost':d['lost_calibrated_reference_tokens'],'wall_seconds':d['wall_seconds'],'replay':pin(out/role/'replay.json')};results.append(result);print(json.dumps(result),flush=True)
(R/'data-backoff-native-result.json').write_text(json.dumps({'status':'PARENT_MARGIN_96_WINDOW_DIAGNOSTIC_COMPLETE_NOT_FULL956_OR_THROUGHPUT','results':results,'selection':pin(R/'data-backoff-plan.json'),'wall_seconds':time.monotonic()-start,'scope':'pass96doesnotestablishfullpreservation; anylostnativeposition isactualfailureatthischoice'},indent=2)+'\n')
