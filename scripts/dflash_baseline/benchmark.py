"""Serialized matched consumed-DEV panel; run only after training releases GPU."""
import json,os,subprocess,time,urllib.request,socket,sys,hashlib
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/dflash-modern-native-20261104-03';TRAIN=P/'runs/dflash-modern-train-20261104-14/artifacts'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def request(port,path,data=None):
 req=urllib.request.Request(f'http://127.0.0.1:{port}/{path}',data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
def main():
 out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
 assert json.load(open(TRAIN/'result.json'))['status']=='TRAINING_ALLOCATION_COMPLETE'
 arms={'target':None,'q4':P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf','zero':P/'runs/dflash-modern-20261104-02/zero.gguf'}
 for label in ('300','900','1800'):
  dest=out/f'trained-{label}.gguf'
  subprocess.run([sys.executable,str(Path(__file__).with_name('export_trained.py')),str(TRAIN/f'checkpoint-{label}.pt'),str(dest)],check=True)
  arms[label]=dest
 prompts={}
 for p in sorted((P/'runs/dflash-modern-capture-20261104-04/data').glob('development-*')):
  meta=json.load(open(p/'capture/capture.json'));prompts[p.name]=meta['input_tokens'][:meta['prompt_tokens']]
 save(out/'prompts.json',prompts);rows=[];reference={}
 for rep in range(7):
  order=list(arms) if rep%2==0 else list(reversed(arms))
  for arm in order:
   assert not json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']
   cell=out/f'{rep}-{arm}';cell.mkdir();model=arms[arm]
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
      if process.poll() is not None:raise RuntimeError(f'{arm} server exited')
      try:request(port,'health');break
      except Exception:time.sleep(1)
     else:raise RuntimeError('startup timeout')
     for name,ids in prompts.items():
      payload=dict(prompt=ids,n_predict=256,temperature=0,seed=1234,cache_prompt=False,return_tokens=True)
      save(cell/(name+'-request.json'),payload);start=time.perf_counter();response=request(port,'completion',payload);elapsed=time.perf_counter()-start
      save(cell/(name+'-response.json'),response)
      tokens=response['tokens']
      if arm=='target':reference.setdefault(name,tokens)
      correct=tokens==reference[name]
      rows.append(dict(rep=rep,warmup=rep<2,arm=arm,prompt=name,correct=correct,wall_seconds=elapsed,tokens=len(tokens),timings=response.get('timings')))
      save(out/'progress.json',rows)
      if not correct:raise RuntimeError(f'Target mismatch {arm} {name}')
    finally:
     process.terminate()
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:process.kill();process.wait()
     save(cell/'release.json',dict(reaped=True,returncode=process.returncode))
   print('Completed',rep,arm,flush=True)
 save(out/'result.json',dict(status='COMPLETE',rows=rows))
if __name__=='__main__':main()
