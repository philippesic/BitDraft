from pathlib import Path
import os,subprocess,json,signal,time,hashlib
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';B=R/'actual-native-build01/modern_actual_native_replay';M=R/'selected-lr3e3-all-native-cases01/manifest.json';manifest=json.loads(M.read_text());output=R/'selected-lr3e3-all-native01';output.mkdir(exist_ok=False);start=time.monotonic();stopping=False;results=[]
os.environ.update(CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DECODE_CONSISTENT='32',GGML_W1AX_ACT_BITS='8',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',LD_LIBRARY_PATH=str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin'));os.environ.pop('LD_PRELOAD',None)
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
for index,entry in enumerate(manifest['packets']):
 batch=output/f'batch-{index:02d}';batch.mkdir()
 for role in ['parent','candidate']:
  assert not stopping
  packet=Path(entry['packet']['path'])
  with packet.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['packet']['sha256']
  argv=[str(B),'--cases',str(packet),'--model',role,'--out',str(batch/role),'--allow-cuda']
  if role=='candidate':argv+=['--calibration',str(batch/'parent/replay.json')]
  with (batch/(role+'.log')).open('x') as log:
   process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc')/str(process.pid)
   (batch/(role+'-identity.json')).write_text(json.dumps({'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n')
   code=process.wait();assert code==0,(index,role,code)
  d=json.loads((batch/role/'replay.json').read_text());results.append({'batch':index,'role':role,'windows':len(d['windows']),'native_reference_tokens':d['native_reference_tokens'],'calibrated_reference_tokens':d['calibrated_reference_tokens'],'lost_calibrated_reference_tokens':d['lost_calibrated_reference_tokens'],'native_successes_outside_parent_calibration':d['native_successes_outside_parent_calibration'],'wall_seconds':d['wall_seconds']})
  print(json.dumps(results[-1]),flush=True)
 (output/'progress.json').write_text(json.dumps({'completed':results,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
summary={role:{k:sum(d[k] for d in results if d['role']==role) for k in ['windows','native_reference_tokens','calibrated_reference_tokens','lost_calibrated_reference_tokens','native_successes_outside_parent_calibration']} for role in ['parent','candidate']}
assert summary['parent']['windows']==summary['candidate']['windows']==511 and summary['parent']['native_reference_tokens']==summary['candidate']['native_reference_tokens']==549
(output/'result.json').write_text(json.dumps({'status':'ALL549_NATIVE_PRESERVATION_EVALUATED','partition':str(M),'results':results,'summary':summary,'wall_seconds':time.monotonic()-start,'scope':'All549 original emitted native success positions across36groups; reconstructed fixed-source histories, not serving/freshquality'},indent=2)+'\n')
