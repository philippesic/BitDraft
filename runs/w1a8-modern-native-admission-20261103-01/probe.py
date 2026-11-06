"""Development-only native loader/runtime smoke; no performance winner claim."""
from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,time,urllib.request
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-native-admission-20261103-01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
baseline=read(P/'experiments/w1a8-standing-20261102/baseline.json');frozen=read(P/'runs/qwen35-correctness-matrix-20261101-01/final-freeze.json')
native=P/'runs/w1a8-modern-packed-runtime-20261103-01';build=read(native/'build03-result.json');assert build['status']=='COMPILED_NOT_GPU_QUALIFIED'
candidate_server=pin(native/'build02/bin/llama-server');target=baseline['rows'][0]['target_pin'];dflash=baseline['rows'][0]['draft_pin'];ordinary=next(x for x in frozen['arms'] if x['target_id']=='4B-Q4_K_M' and x['id']=='eagle3-Q4_0-d2')['draft'];initial=read(P/'runs/w1a8-modern-packed-init-20261103-01/result.json')['model']
for ref in [target,dflash,ordinary,initial,frozen['server']]+[x for x in frozen['pins'] if '.so' in x['path']]+build['binaries']:assert pin(ref['path'])['sha256']==ref['sha256'],ref['path']
arms=[('frozen_target','frozen','none',None),('candidate_target','candidate','none',None),('frozen_dflash','frozen','draft-dflash',dflash),('candidate_dflash','candidate','draft-dflash',dflash),('candidate_ordinary_eagle','candidate','draft-eagle3',ordinary),('candidate_w1a8_initial','candidate','draft-eagle3',initial)]
prompts={x['id']:x for x in read(P/'runs/qwen35-correctness-matrix-20261101-01/pilot-inputs/requests.json')};ids=[1,2,4]
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']}
write(R/'freeze.json',{'source':pin(__file__),'baseline':pin(P/'experiments/w1a8-standing-20261102/baseline.json'),'candidate_build':pin(native/'build03-result.json'),'candidate_patch':pin(native/'packed-w1a8-v3.patch'),'target':target,'dflash':dflash,'ordinary_eagle':ordinary,'w1a8_initialization':initial,'prompt_ids':ids,'output_limit':128,'repetitions':1,'diagnostic':True,'training_updates':0,'scope':'Existing development inputs only; loader/target/runtime correctness smoke, no repeated-throughput/quality/generalization or trained-model claim'})
records=[];started=time.monotonic()
try:
 for name,runtime,spec,draft in arms:
  control=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
  cell=R/name;cell.mkdir();server=frozen['server'] if runtime=='frozen' else candidate_server
  env={**clean,'GGML_CUDA_DECODE_CONSISTENT':'32','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':frozen['library_path'] if runtime=='frozen' else str(native/'build02/bin')}
  if name=='candidate_w1a8_initial':env['GGML_W1AX_ACT_BITS']='8'
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  argv=[server['path'],'-m',target['path'],'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-t','4','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port',str(port),'--metrics','--no-webui','--no-warmup','--spec-type',spec]
  if draft:argv+=['-md',draft['path'],'-ngld','99','--cache-type-k-draft','f16','--cache-type-v-draft','f16','--spec-draft-n-max','2','--spec-draft-n-min','0','--spec-draft-p-min','0','--spec-draft-sampling','greedy']
  if name=='candidate_w1a8_initial':argv+=['-lv','4']
  write(cell/'command.json',{'argv':argv,'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','CUDA_','LD_','OMP_'))}})
  process=None
  try:
   with (cell/'server.log').open('x') as log:
    process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);q=Path('/proc')/str(process.pid);write(cell/'identity.json',{'pid':process.pid,'pgid':process.pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'start_ticks':int((q/'stat').read_text().rsplit(')',1)[1].split()[19]),'cmdline':(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()})
    for _ in range(240):
     assert process.poll() is None, name+' exited during startup'
     try:
      with urllib.request.urlopen('http://127.0.0.1:%d/health'%port,timeout=1) as response:
       if response.status==200:break
     except Exception:time.sleep(.25)
    else:raise RuntimeError('startup timeout')
    maps=(q/'maps').read_text();(cell/'loaded-library-maps.txt').write_text(maps)
    cuda=sorted({x.split()[-1] for x in maps.splitlines() if 'libggml-cuda.so' in x});expected=baseline['cuda']['path'] if runtime=='frozen' else str((native/'build02/bin/libggml-cuda.so.0.28.0').resolve());assert cuda==[expected]
    for index,pid in enumerate([0,1]+ids):
     payload={'prompt':prompts[pid]['prompt'],'n_predict':128,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False};request=urllib.request.Request('http://127.0.0.1:%d/completion'%port,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});t=time.monotonic()
     with urllib.request.urlopen(request,timeout=120) as response:result=json.load(response)
     seconds=time.monotonic()-t;write(cell/('request-%02d.json'%index),payload);write(cell/('response-%02d.json'%index),result)
     if index>=2:
      row={'arm':name,'prompt_id':pid,'response':pin(cell/('response-%02d.json'%index)),'request_wall_seconds_diagnostic':seconds,'completion_tokens':len(result['tokens']),'tokens_drafted':result.get('timings',{}).get('draft_n'),'tokens_accepted':result.get('timings',{}).get('draft_n_accepted')};records.append(row)
    print(json.dumps({'arm':name,'status':'requests_complete','measured_requests':3}),flush=True)
  finally:
   if process is not None:
    if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    write(cell/'release.json',{'owned_server_reaped':process.poll() is not None,'returncode':process.returncode})
 write(R/'requests.json',records)
 def signature(response):
  return (response['tokens'],response['content'],response['stop_type'],response['stopping_word'],response['truncated'])
 refs={x['prompt_id']:signature(read(x['response']['path'])) for x in records if x['arm']=='frozen_target'}
 assert len(records)==18 and len(refs)==3
 for row in records:assert signature(read(row['response']['path']))==refs[row['prompt_id']],(row['arm'],row['prompt_id'])
 log=(R/'candidate_w1a8_initial/server.log').read_text();assert 'complete nine-linear fixed row W1A8' in log
 assert log.count('W1A8 packed ')==9,'exactly nine packed loader receipts required'
 write(R/'result.json',{'status':'DEVELOPMENT_NATIVE_SMOKE_PASS','requests':18,'warmup_requests':12,'arms':[x[0] for x in arms],'token_text_stop_parity':True,'nine_packed_loader_receipts':True,'training_updates':0,'diagnostic':True,'quality_qualified':False,'repeated_performance_qualified':False,'wall_seconds':time.monotonic()-started,'records':records})
 print(json.dumps({'status':'DEVELOPMENT_NATIVE_SMOKE_PASS','requests':18,'wall_seconds':time.monotonic()-started}),flush=True)
except BaseException as error:
 write(R/'failure.json',{'error':repr(error),'completed_records':records,'wall_seconds':time.monotonic()-started});raise
