from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,time,urllib.request
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-serving-trace-20261103-03'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
F=read(P/'runs/w1a8-modern-native-admission-20261103-01/freeze.json');B=P/'runs/w1a8-modern-packed-runtime-20261103-01/build02/bin';build=read(F['candidate_build']['path']);server=pin(B/'llama-server')
refs=[F['target'],F['w1a8_initialization'],F['candidate_build']]+build['binaries'];assert all(pin(r['path'])==r for r in refs)
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']}
env={**clean,'LD_LIBRARY_PATH':str(B),'GGML_CUDA_DECODE_CONSISTENT':'32','GGML_W1AX_ACT_BITS':'8','LLAMA_TRACE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'}
inputs={x['id']:x for x in read(P/'runs/qwen35-correctness-matrix-20261101-01/pilot-inputs/requests.json')};started=time.monotonic();records=[]
write(R/'freeze.json',{'source':pin(__file__),'baseline':F['baseline'],'build':F['candidate_build'],'target':F['target'],'draft':F['w1a8_initialization'],'pins':refs,'scope':'Actual serving accepted-prefix diagnostic with existing debug logs, not performance/quality/generalization'})
for pid,index in [(2,3)]:
 assert not read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'].get('pause_requested') and not (R/'STOP').exists()
 cell=R/f'prompt-{pid:02d}';cell.mkdir();process=None
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 argv=[str(B/'llama-server'),'-m',F['target']['path'],'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-t','4','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port',str(port),'--metrics','--no-webui','--no-warmup','--spec-type','draft-eagle3','-md',F['w1a8_initialization']['path'],'-ngld','99','--cache-type-k-draft','f16','--cache-type-v-draft','f16','--spec-draft-n-max','2','--spec-draft-n-min','0','--spec-draft-p-min','0','--spec-draft-sampling','greedy','-lv','5']
 write(cell/'command.json',{'argv':argv,'environment':{k:v for k,v in env.items() if k.startswith(('LD_','GGML_','LLAMA_','CUDA_','OMP_'))}})
 trace=cell/'trace';trace.mkdir();env.update(read(P/'runs/w1a8-modern-trace-build-20261103-02/build.json')['environment']);env['W1A8_TRACE_DIR']=str(trace);env['W1A8_TRACE_MAP']=str(P/'runs/w1a8-modern-native-replay-20261103-01/draft-map.ids');write(cell/'trace-environment.json',{k:v for k,v in env.items() if k.startswith(('LD_','W1A8_','GGML_','CUDA_','LLAMA_'))})
 try:
  with (cell/'server.log').open('x') as log:
   process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
   q=Path('/proc')/str(process.pid);write(cell/'identity.json',{'pid':process.pid,'pgid':os.getpgid(process.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
   for _ in range(240):
    assert process.poll() is None
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
      if response.status==200:break
    except Exception:time.sleep(.1)
   else:raise RuntimeError('startup timeout')
   (cell/'loaded-library-maps.txt').write_text((q/'maps').read_text())
   payload={'prompt':inputs[pid]['prompt'],'n_predict':128,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False};write(cell/'request.json',payload)
   req=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=120) as response:result=json.load(response)
   write(cell/'response.json',result)
   gold=read(P/f'runs/w1a8-modern-native-admission-20261103-01/frozen_target/response-{index:02d}.json')
   for key in ['tokens','content','stop_type','stopping_word','truncated']:assert result[key]==gold[key],key
 finally:
  if process is not None:
   if process.poll() is None:process.terminate()
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
   write(cell/'release.json',{'child_reaped':process.poll() is not None,'returncode':process.returncode})
 records.append({'prompt_id':pid,'response':pin(cell/'response.json'),'log':pin(cell/'server.log'),'token_text_stop_parity':True,'accepted':result['timings'].get('draft_n_accepted'),'proposed':result['timings'].get('draft_n')});print(json.dumps(records[-1]),flush=True)
write(R/'result.json',{'status':'ACTUAL_SERVING_TRACE_COLLECTED','records':records,'wall_seconds':time.monotonic()-started,'trace_parsed':False})
