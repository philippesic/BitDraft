"""Bounded Nsight CUDA/NVTX attribution on three consumed DEV requests per model.

No performance claim: profiler overhead is excluded from the uninstrumented panel.
"""
from pathlib import Path
import argparse,json,os,signal,socket,subprocess,sys,time,urllib.request
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(P/'scripts/w1a8_generalization')]
from runner import pin,read,atomic,require,identity,identity_alive,descendants,group_live
from modern_requests import signature
NSYS='/usr/local/cuda-13.1/bin/nsys'
PROFILER_EXTRA_IMAGE={'path': '/usr/lib/x86_64-linux-gnu/libutil.so.1', 'bytes': 14408, 'sha256': '1d5fdfec85a9c4ca1e7a8c3125aac07c92d551b902a02fdd6bbdf8c66c9e4c4a'}

def request(port,route,payload=None):
 r=urllib.request.Request(f'http://127.0.0.1:{port}/{route}',data=json.dumps(payload).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=120) as response:return json.load(response)
def guard():
 c=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];require(not c.get('pause_requested') and not c.get('stop_all_requested'),'GPU paused')
def profile(a):
 require(not a.out.exists(),'new profiler directory required');a.out.mkdir(parents=True)
 models=read(a.models);inputs=read(a.inputs);prior=read(a.command.parent.parent/'result.json');reference_counters={r['id']:r for r in prior['records'] if r['arm']==a.arm and r['repetition']==0};arm=next(x for x in models['arms'] if x['id']==a.arm)
 for ref in [models['target'],arm['model'],arm['runtime']['server']]+arm['runtime']['libraries']:require(pin(ref['path'])==ref,'native profile dependency differs')
 original=read(a.command);argv=original['argv'];require(argv[0]==arm['runtime']['server']['path'] and argv[argv.index('-md')+1]==arm['model']['path'],'command/model differs')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 argv[argv.index('--port')+1]=str(port)
 env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1AX_','W1A8_','DSPARK_','DFLASH_','EAGLE_','LLAMA_','OMP_'))}
 env.update(arm['runtime']['environment']);env.update(LD_PRELOAD=str(a.library.resolve()),DECODE_COST_TRACE=str(a.out/'host-spans.jsonl'))
 command=[NSYS,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','--output='+str(a.out/'cuda'),*argv]
 chosen=[]
 for domain in ('code','reasoning','prose'):
  matches=[x for x in inputs['rows'] if domain in x['id']];require(matches,'domain missing');chosen.append(sorted(matches,key=lambda x:x['id'])[0])
 atomic(a.out/'freeze.json',dict(source=pin(__file__),models=pin(a.models),inputs=pin(a.inputs),arm=arm,original_command=pin(a.command),argv=command,library=pin(a.library),requests=chosen,scope='Diagnostic node-level CUDA trace; first consumed DEV case/domain, two excluded warmups; no output-dependent selection'))
 process=None;server=None;records=[]
 try:
  guard()
  with (a.out/'profiler.log').open('x') as log:
   process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owner=identity(process.pid);atomic(a.out/'identity.json',owner)
   deadline=time.monotonic()+120
   while time.monotonic()<deadline:
    guard();require(process.poll() is None,'profiler exited before server startup')
    matches=[]
    # Nsight may launch from a non-main thread or collector. Authenticate the
    # unique frozen command and our injected environment, not PPID alone.
    for proc in Path('/proc').iterdir():
     if not proc.name.isdigit():continue
     try:
      if (proc/'exe').resolve()!=Path(argv[0]).resolve():continue
      parts=(proc/'cmdline').read_bytes().split(b'\0')
      if b'--port' not in parts or parts[parts.index(b'--port')+1]!=str(port).encode():continue
      require(parts[1:len(argv)]==[x.encode() for x in argv[1:]],'profiler server arguments differ')
      environment=(proc/'environ').read_bytes().split(b'\0')
      selected={k.decode():v.decode() for item in environment if b'=' in item for k,v in [item.split(b'=',1)] if k in (b'DECODE_COST_TRACE',b'LD_PRELOAD')}
      atomic(a.out/'discovery-environment.json',dict(pid=int(proc.name),selected=selected))
      require(selected.get('DECODE_COST_TRACE')==str(a.out/'host-spans.jsonl') and str(a.library.resolve()) in selected.get('LD_PRELOAD','').replace(':',' ').split(),'profiler server ownership environment differs')
      ref=identity(int(proc.name));require(ref['start_ticks']>=owner['start_ticks'],'preexisting server cannot belong to this trace');matches.append(ref)
     except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    require(len(matches)<=1,'ambiguous profiler server identity')
    if matches:server=matches[0]
    if server:
     try:request(port,'health');break
     except Exception:pass
    time.sleep(.2)
   else:raise TimeoutError('profile server startup exceeded120s')
   atomic(a.out/'server-identity.json',server)
   maps=(Path('/proc')/str(server['pid'])/'maps').read_text();(a.out/'loaded-maps.txt').write_text(maps)
   expected={str(Path(x['path']).resolve()):x for x in [arm['runtime']['server']]+arm['runtime']['libraries']+[pin(a.library),PROFILER_EXTRA_IMAGE]};images=[]
   for line in maps.splitlines():
    parts=line.split(maxsplit=5)
    if len(parts)!=6 or not parts[5].startswith('/') or not ('x' in parts[1] or '.so' in parts[5]):continue
    path=str(Path(parts[5]).resolve());ref=pin(path)
    if path in expected:require(ref==expected[path],'loaded native image differs')
    else:require(path.startswith(('/opt/nvidia/nsight-systems/','/usr/local/cuda-13.1/')) or Path(path).name.startswith(('libcupti','libnvperf','libToolsInjection','libnvToolsExt')),'unexpected profiler image: '+path)
    if ref not in images:images.append(ref)
   atomic(a.out/'loaded-images.json',images)
   for index,item in enumerate(inputs['warmups']+chosen):
    guard();payload=read(item['request']['path']);require(pin(item['request']['path'])==item['request'],'request changed');require(pin(item['reference']['path'])==item['reference'],'gold changed')
    begin=time.monotonic_ns();response=request(port,'completion',payload);end=time.monotonic_ns();require(signature(response)==signature(read(item['reference']['path'])),'instrumented output differs')
    expected=reference_counters[item['id']];require(response['timings']['draft_n']==expected['draft_proposed'] and response['timings']['draft_n_accepted']==expected['draft_accepted'],'profiler changed draft decisions')
    path=a.out/f'response-{index:02d}.json';atomic(path,response);records.append(dict(id=item['id'],warmup=index<2,start_ns=begin,end_ns=end,response=pin(path),request=item['request'],token_text_stop_parity=True))
   require(identity_alive(server),'server identity disappeared');os.kill(server['pid'],signal.SIGINT);require(process.wait(timeout=120)==0,'nsys export failed')
 finally:
  if server is not None and identity_alive(server):
   os.kill(server['pid'],signal.SIGINT);deadline=time.monotonic()+10
   while identity_alive(server) and time.monotonic()<deadline:time.sleep(.1)
   if identity_alive(server):os.kill(server['pid'],signal.SIGKILL)
  if process is not None:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   atomic(a.out/'release.json',dict(owner=owner,server=server,returncode=process.returncode,owned_process_reaped=process.poll() is not None,owned_process_group_absent=not group_live(process.pid),server_identity_absent=server is None or not identity_alive(server)))
 require((a.out/'cuda.nsys-rep').is_file(),'missing Nsight report')
 subprocess.run([NSYS,'export','--type=sqlite','--output='+str(a.out/'cuda.sqlite'),str(a.out/'cuda.nsys-rep')],check=True,timeout=120,stdout=subprocess.DEVNULL)
 atomic(a.out/'result.json',dict(status='NSIGHT_NATIVE_OUTPUT_AND_TRACE_PASS',records=records,trace=pin(a.out/'cuda.nsys-rep'),sqlite=pin(a.out/'cuda.sqlite'),release=pin(a.out/'release.json'),scope='Instrumented diagnostic kernel attribution only'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--command',type=Path,required=True);p.add_argument('--arm',choices=['paid5760','dflash_Q4KM'],required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();require(a.allow_cuda,'explicit GPU required');profile(a)
