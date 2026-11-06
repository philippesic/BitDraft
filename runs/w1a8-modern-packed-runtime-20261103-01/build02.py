"""Bounded CPU/CUDA compilation; no model or CUDA device is initialized."""
from pathlib import Path
import hashlib,json,os,signal,shutil,subprocess,time

P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-packed-runtime-20261103-01'
SOURCE=R/'source';BUILD=R/'build02';FLOOR=2*1024**3
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def birth(pid):return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','GIT_CEILING_DIRECTORIES':str(SOURCE),'TMPDIR':str(R/'compiler-tmp02')}
Path(env['TMPDIR']).mkdir(exist_ok=True)
commands=[['cmake','-S',str(SOURCE),'-B',str(BUILD),'-G','Ninja',
 '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120',
 '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc','-DGGML_CUDA=ON',
 '-DCMAKE_CUDA_FLAGS=-I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include',
 '-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_APP=OFF',
 '-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_BUILD_SERVER=ON','-DLLAMA_BUILD_UI=OFF',
 '-DLLAMA_USE_PREBUILT_UI=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_COMMIT=aff0cbd-w1a8',
 '-DLLAMA_BUILD_NUMBER=0'],['cmake','--build',str(BUILD),'--target','llama-server','-j','2']]
freeze={'source_origin':pin(R/'source-origin.json'),'source_patch':pin(R/'packed-w1a8.patch'),
 'recipe':pin(__file__),'cuda_header_compatibility':pin(R/'cuda-header-compatibility.json'),'previous_failure':pin(R/'build-failure.json'),'commands':commands,'environment_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','GIT_CEILING_DIRECTORIES','TMPDIR']},
 'minimum_free_bytes':FLOOR,'scope':'Compile isolated candidate backend/server carrying existing frozen correctness source plus packed ops; no native correctness/performance admission'}
write(R/'build02-freeze.json',freeze);phases=[];minimum=shutil.disk_usage(R).free
try:
 for index,argv in enumerate(commands):
  assert shutil.disk_usage(R).free>FLOOR+128*1024**2,'insufficient headroom before compilation'
  started=time.monotonic()
  with (R/('build02-phase-%d.log'%index)).open('x') as log:
   proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(R/'active-builder02.json',{'pid':proc.pid,'pgid':proc.pid,'start_ticks':birth(proc.pid),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'argv':argv})
   try:
    while proc.poll() is None:
     free=shutil.disk_usage(R).free;minimum=min(minimum,free)
     if free<FLOOR or (R/'STOP-BUILD').exists():
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=30)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
      raise RuntimeError('build stopped to preserve storage floor or requested stop')
     time.sleep(.5)
    if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,argv)
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
  phases.append({'command':argv,'wall_seconds':time.monotonic()-started,'exit_code':proc.returncode})
 binaries=[pin(p) for p in sorted((BUILD/'bin').iterdir()) if p.is_file() and not p.is_symlink()]
 result={'status':'COMPILED_NOT_GPU_QUALIFIED','phases':phases,'binaries':binaries,'minimum_observed_free_bytes':minimum,
 'free_bytes_after':{p:shutil.disk_usage(p).free for p in ['.','/mnt/c']},'gpu_seconds':0,'native_execution_admitted':False}
 write(R/'build02-result.json',result);print(json.dumps({'status':result['status'],'phases':phases,'minimum_observed_free_bytes':minimum}),flush=True)
except BaseException as error:
 write(R/'build02-failure.json',{'error':repr(error),'completed_phases':phases,'minimum_observed_free_bytes':minimum,'free_bytes':shutil.disk_usage(R).free})
 raise
