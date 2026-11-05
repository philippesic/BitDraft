"""Bounded native target/Q4/packed comparison; not a performance benchmark."""
import json,os,subprocess,time,urllib.request,socket,signal,sys
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/dflash-modern-native-20261104-03'
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def request(port,path,data=None):
 req=urllib.request.Request(f'http://127.0.0.1:{port}/{path}',data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as f:return json.load(f)
x=json.load(open(P/'runs/w1a8-modern-generalization-capture-20261103-02/fit-code-00/capture-command.json'))
ids=[int(v) for v in Path(x['argv'][x['argv'].index('--tokens')+1]).read_text().replace(',',' ').split()]
tag=sys.argv[1] if len(sys.argv)>1 else "03"
w1=Path(sys.argv[2]) if len(sys.argv)>2 else P/"runs/dflash-modern-20261104-02/zero.gguf"
results={}
for arm,model in [('target',None),('q4',P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),('w1',w1)]:
 assert not json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']
 cell=R/('smoke-'+tag+'-'+arm);cell.mkdir(exist_ok=False)
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1A8_','LLAMA_','CUDA_','LD_','OMP_'))}
 env.update(LD_LIBRARY_PATH=str(R/'bin'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
 argv=[str(R/'bin/llama-server'),'-m',str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-t','4','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port',str(port),'--metrics','--no-webui','--no-warmup','--spec-type','draft-dflash' if model else 'none','-lv','4']
 if model:argv+=['-md',str(model),'-ngld','99','--cache-type-k-draft','f16','--cache-type-v-draft','f16','--spec-draft-n-max','2','--spec-draft-n-min','0','--spec-draft-p-min','0','--spec-draft-sampling','greedy']
 save(cell/'command.json',dict(argv=argv,environment=env))
 with (cell/'server.log').open('w') as log:
  process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   for i in range(120):
    if process.poll() is not None:raise RuntimeError(f'{arm} server exited {process.returncode}')
    try:request(port,'health');break
    except Exception:time.sleep(1)
   else:raise RuntimeError('server startup timeout')
   (cell/'maps.txt').write_text(Path(f'/proc/{process.pid}/maps').read_text())
   t=time.perf_counter();out=request(port,'completion',dict(prompt=ids,n_predict=32,temperature=0,seed=1234,cache_prompt=False,return_tokens=True));elapsed=time.perf_counter()-t
   save(cell/'response.json',out)
   results[arm]=dict(tokens=out.get('tokens'),timings=out.get('timings'),wall_seconds=elapsed)
   save(R/f'smoke-{tag}-progress.json',results)
  finally:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
   save(cell/'release.json',dict(returncode=process.returncode,reaped=True))
assert results['target']['tokens'] and results['target']['tokens']==results['q4']['tokens']==results['w1']['tokens']
save(R/f'smoke-{tag}-result.json',dict(correct=True,results=results,scope='single TRAIN prompt;32 tokens;no throughput claim'))
print('Native smoke passed',flush=True)
