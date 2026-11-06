from pathlib import Path
import hashlib,json,os,subprocess,time,signal
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-native-replay-20261103-01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
control=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
prepared=read(R/'prepared.json');build=read(prepared['build']['path'])
refs=[prepared[k] for k in ['cases','draft_map','draft_model','target','baseline','build']]+[build['executable']]+build['candidate_binaries']
for a in prepared['ancestry']:refs += [a['admission'],a['capture']]+a['features']
for ref in refs:assert pin(ref['path'])==ref,ref['path']
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']}
env={**clean,**build['environment'],'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2'}
argv=[build['executable']['path'],'--target',prepared['target']['path'],'--draft',prepared['draft_model']['path'],'--cases',prepared['cases']['path'],'--out',str(R/'native01')]+build['required_backend_argv']
write(R/'freeze01.json',{'source':pin(__file__),'prepared':pin(R/'prepared.json'),'argv':argv,'environment':build['environment'],'pins':refs,'scope':prepared['scope']})
started=time.monotonic();process=None
try:
 with (R/'native01-stdout.log').open('x') as output,(R/'native01-stderr.log').open('x') as error:
  process=subprocess.Popen(argv,stdout=output,stderr=error,env=env)
  q=Path('/proc')/str(process.pid)
  write(R/'native01-identity.json',{'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
  code=process.wait(timeout=240);assert code==0,code
 result=read(R/'native01/replay.json');assert result['status']=='NATIVE_REPLAY_COMPLETE'
 maps=(R/'native01/loaded-library-maps.txt').read_text();actual=sorted({line.split()[-1] for line in maps.splitlines() if '/libggml' in line or '/libllama' in line});expected=sorted(ref['path'] for ref in build['candidate_binaries'] if '/libggml' in ref['path'] or '/libllama.so' in ref['path'])
 assert set(actual).issubset(expected),actual
 assert any('libggml-cuda.so' in path for path in actual) and any('libllama.so' in path for path in actual)
 for path in actual:assert pin(path) in refs
 write(R/'result01.json',{'status':'NATIVE_REPLAY_EXECUTED','wall_seconds':time.monotonic()-started,'result':pin(R/'native01/replay.json'),'mapped_images':[pin(x) for x in actual],'teacher_rows':sum(c['teacher']['rows'] for c in result['cases'].values()),'recurrent_rows':sum(c['recurrent']['rows'] for c in result['cases'].values()),'CPU_comparison_qualified':False,'scope':prepared['scope']})
 print(json.dumps(read(R/'result01.json')),flush=True)
except BaseException as error:
 if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
 write(R/'failure01.json',{'error':repr(error),'wall_seconds':time.monotonic()-started});raise
finally:
 write(R/'release01.json',{'child_reaped':process is None or process.poll() is not None,'returncode':process.returncode if process else None})
