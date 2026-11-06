from pathlib import Path
import os,json,subprocess,signal,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';F=R/'selected-sign-native01';B=R/'actual-native-build01/modern_actual_native_replay';M=json.loads((F/'manifest.json').read_text());Q=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-selected-sign-qualification-20261103-01');stopping=False;start=time.monotonic();records=[]
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DECODE_CONSISTENT='32',GGML_W1AX_ACT_BITS='8',CUDA_CACHE_PATH=str(P/'runs/local-w1a8-modern-serving-trace-20261103-01/cuda'),CUDA_CACHE_MAXSIZE='1073741824',LD_LIBRARY_PATH=str(P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin'));env.pop('LD_PRELOAD',None)
for entry in M['packets']:
 folder=F/entry['kind'];folder.mkdir(exist_ok=False)
 for role in ['parent','candidate']:
  assert not stopping;out=folder/role;argv=[str(B),'--cases',entry['packet']['path'],'--model',role,'--out',str(out),'--allow-cuda']
  if role=='candidate':argv += ['--calibration',str(folder/'parent/replay.json')]
  with (folder/(role+'.log')).open('x') as log:
   p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);q=Path('/proc',str(p.pid));(folder/(role+'-identity.json')).write_text(json.dumps({'pid':p.pid,'pgid':os.getpgid(p.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()},indent=2)+'\n');assert p.wait()==0,(entry['kind'],role)
  native=json.loads((out/'replay.json').read_text());source=Q/'before-fit.json' if role=='parent' else Path(entry['GPU_readout']['path']);gpu={r['id']:r for r in json.loads(source.read_text())['rows']};details=[]
  for wid,w in native['windows'].items():
   for s in w['steps']:details.append({'window':wid,'depth':s['depth'],'native_proposed':s['proposed'],'GPU_proposed':gpu[wid]['proposals'][s['depth']],'equal':s['proposed']==gpu[wid]['proposals'][s['depth']]})
  record={'kind':entry['kind'],'role':role,'native_successes':native['native_reference_tokens'],'lost_old_successes':native['lost_calibrated_reference_tokens'],'native_GPU_reached_proposals':len(details),'native_GPU_matches':sum(d['equal'] for d in details),'details':details,'wall_seconds':native['wall_seconds']};records.append(record);print(json.dumps(record),flush=True)
 (F/'progress.json').write_text(json.dumps({'records':records,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
assert not stopping
p=subprocess.run([str(P/'.venv/bin/python'),str(R/'qualify-selected-sign-http.py')]);assert p.returncode==0,p.returncode
(F/'result.json').write_text(json.dumps({'status':'SELECTED_SIGN_REPLAY_AND28_HTTP_ADMISSION_COMPLETE','records':records,'wall_seconds':time.monotonic()-start,'scope':'two reused FIT windows, two model exports; changed sign is forced packing diagnostic, not a learned capacity claim; no repeated performance/fresh quality'},indent=2)+'\n')
