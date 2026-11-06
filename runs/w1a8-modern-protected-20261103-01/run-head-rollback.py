from pathlib import Path
import os,subprocess,json,signal,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';B=R/'actual-native-build01/modern_actual_native_replay';stopping=False;results=[];start=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DECODE_CONSISTENT='32',GGML_W1AX_ACT_BITS='8',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',LD_LIBRARY_PATH=str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin'));os.environ.pop('LD_PRELOAD',None)
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
for role in ['parent','candidate']:
 assert not stopping
 argv=[str(B),'--cases',str(R/'head-rollback-cases01/cases.json'),'--model',role,'--out',str(R/('head-rollback-'+role+'01')),'--allow-cuda']
 if role=='candidate':argv+=['--calibration',str(R/'head-rollback-parent01/replay.json')]
 process=subprocess.Popen(argv);q=Path('/proc')/str(process.pid)
 (R/('head-rollback-'+role+'-identity.json')).write_text(json.dumps({'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n')
 code=process.wait();assert code==0,(role,code)
 result=json.loads((R/('head-rollback-'+role+'01')/'replay.json').read_text());results.append({k:result[k] for k in ['model_role','native_reference_tokens','calibrated_reference_tokens','lost_calibrated_reference_tokens','wall_seconds']})
(R/'head-rollback-native-result.json').write_text(json.dumps({'status':'HEAD_ROLLBACK_NATIVE_DIAGNOSTIC_COMPLETE','results':results,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
