"""Finite concurrency-one Qwen3.5 comparison; results are provisional screening."""
from pathlib import Path
import json,subprocess,os,time,urllib.request,urllib.error,fcntl,hashlib,signal,sys,socket
P=Path(__file__).resolve().parent; ROOT=P.parent.parent
os.environ['LD_PRELOAD']=str(P/'invariant-attn.so')
os.environ['GGML_CUDA_DISABLE_FUSION']='1'
runtime=json.loads((Path.home()/'.local/share/p-stack/runtime.json').read_text());sys.path.insert(0,str(Path.home()/'.local/share/p-stack/runtime'/runtime['source_digest']));os.environ.setdefault('P_STACK_BROKER_COMMAND',json.dumps(runtime['bridge']))
from devicecheckout.project import before_costly_work,check_pause
before_costly_work('benchmark_quiet')
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']['pause_requested']
assert not (P/'STOP').exists()
# Explicit deterministic chat rendering, including a closed empty thinking section.
def prompt(text):return '<|im_start|>system\nYou are a helpful assistant. Answer directly and accurately.<|im_end|>\n<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
questions=[
 'Explain why the sky appears blue in two short paragraphs.',
 'Write a Python function that returns the first non-repeated character in a string. Include tests.',
 'A store has 24 boxes of 18 pencils and sells 137 pencils. How many remain? Explain briefly.',
 'Compare an SQL inner join with a left join using a small example.',
 'Summarize the following notes in five bullets: '+('The team ships weekly. Changes require tests. Customers value reliable releases. The service stores encrypted backups. Incidents get written reviews. '*20),
 'Find the bug and provide a corrected implementation: def unique(xs):\n    out=[]\n    for x in xs:\n        if x not in out: out.append(x)\n        return out',
 'Use the following record to answer the question. '+('Sensor A reports 14 degrees. Sensor B reports 19 degrees. The calibration offset is 2 degrees for B only. '*45)+'What is the corrected reading for B? Explain the calculation.',
 'Write a clear plan for migrating a small SQLite database to PostgreSQL with little downtime.'
]
(P/'prompts.json').write_text(json.dumps([{'id':i,'prompt':prompt(q),'sha256':hashlib.sha256(prompt(q).encode()).hexdigest()} for i,q in enumerate(questions)],indent=2))
server=None
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
def request(path,data=None):
 body=None if data is None else json.dumps(data).encode();req=urllib.request.Request('http://127.0.0.1:18935'+path,data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
def stop():
 global server
 if server is not None:
  if server.poll() is None:server.terminate()
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  server=None

def runrow(target,name,spectype,draft=None,phase='invariant',qs=questions):
 global server
 out=P/'measurements'/target/name/phase;
 if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
 out.mkdir(parents=True,exist_ok=False)
 argv=[str(P/'build/bin/llama-server'),'-m',str(P/'models'/target),'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port','18935','--spec-type',spectype,'--no-webui','--no-warmup']
 if draft:argv+=['-md',str(P/'models'/draft),'-ngld','99','--spec-draft-n-max','16']
 if spectype=='draft-mtp':argv+=['--spec-draft-n-max','3']
 (out/'command.json').write_text(json.dumps(argv));log=(out/'server.log').open('w');rows=[]
 try:
  check_pause();assert not (P/'STOP').exists();server=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+150
  while True:
   if server.poll() is not None:raise RuntimeError('server failed '+str(server.returncode))
   try:
    if request('/health').get('status')=='ok':break
   except (OSError,ValueError):pass
   if time.monotonic()>deadline:raise TimeoutError('load timeout')
   time.sleep(.25)
  # Same warmup for all arms, excluded from measured requests.
  request('/completion',{'prompt':prompt('Say hello.'),'n_predict':32,'temperature':0,'seed':1234,'cache_prompt':False})
  mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader'],text=True);(out/'gpu.txt').write_text(mem)
  for rep in range(2):
   order=list(range(len(qs)));order=order if rep==0 else list(reversed(order))
   for i in order:
    check_pause();assert not (P/'STOP').exists();payload={'prompt':prompt(qs[i]),'n_predict':256,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'stream':False}
    t=time.perf_counter();result=request('/completion',payload);elapsed=time.perf_counter()-t
    row={'prompt':i,'rep':rep,'elapsed':elapsed,'result':result};rows.append(row)
    with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  n=sum(x['result']['tokens_predicted'] for x in rows);elapsed=sum(x['elapsed'] for x in rows)
  summary={'target':target,'row':name,'phase':phase,'tokens':n,'elapsed':elapsed,'request_tps':n/elapsed,'requests':len(rows),'status':'measured','gpu':mem}
 except Exception as e:summary={'target':target,'row':name,'phase':phase,'status':'failed','error':repr(e)}
 finally:stop();log.close()
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return summary

try:
 for size in ['4B','9B']:
  for fmt in ['Q4_K_M','NVFP4']:
   target=f'Qwen3.5-{size}-Q4_K_M.gguf' if fmt=='Q4_K_M' else f'qwen35-{size.lower()}-mtp-nvfp4.gguf'
   qs=questions
   runrow(target,'no-spec','none',qs=qs)
   if size=='4B':runrow(target,'mtp','draft-mtp',qs=qs)
   else:runrow(target,'dflash-Q4_K_M','draft-dflash',f'qwen35-{size.lower()}-dflash-Q4_K_M.gguf',qs=qs)
finally:stop()
