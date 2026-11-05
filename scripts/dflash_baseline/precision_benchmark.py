"""Matched native panels with authentic consumed DEV prompts and complete draft cost."""
import argparse,hashlib,json,os,re,signal,socket,subprocess,sys,threading,time,urllib.request
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');ROOT=Path(__file__).resolve().parents[2]
def pin(path):
 p=Path(path);return dict(path=str(p.resolve()),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),bytes=p.stat().st_size)
def save(path,x):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)
def auth(ref):
 if pin(ref['path'])['sha256']!=ref['sha256']:raise ValueError('pin mismatch '+ref['path'])
 return json.load(open(ref['path']))
def request(port,path,data=None):
 req=urllib.request.Request(f'http://127.0.0.1:{port}/{path}',data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
def pause():
 if json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']:raise RuntimeError('user GPU pause requested')
def cohort(name):
 if name=='capture3':
  return [dict(id=p.parent.parent.name,payload=dict(prompt=(m:=json.load(open(p)))['input_tokens'][:m['prompt_tokens']],n_predict=256,temperature=0,seed=1234,cache_prompt=False,return_tokens=True)) for p in sorted((P/'runs/dflash-modern-capture-20261104-04/data').glob('development-*/capture/capture.json'))]
 refs=json.load(open(P/'runs/data-cost-transfer-20261104-01/consumed-natural-inputs.json'));auth(refs['manifest']);marker=auth(refs['marker'])
 if not marker['evaluated']:raise ValueError('natural DEV consumption not authenticated')
 items=refs['rows']
 if name=='natural12':
  items=[x for domain in ('code','reasoning','prose') for x in [x for x in items if x['domain']==domain][:4]]
  if len(items)!=12:raise ValueError('natural12 domain mapping')
 return [dict(id=x['id'],payload=auth(x['request']),source=x['request'],domain=x['domain']) for x in items]
STAT=re.compile(r'#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+).*?dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms')
def stats(text):
 matches=list(STAT.finditer(text))
 if not matches:return None
 m=matches[-1];values=[int(m[i]) for i in range(1,8)]+[float(m[i]) for i in range(8,11)]
 return dict(zip(('begin_calls','rounds','accept_calls','drafts','accepted_rounds','proposed','accepted','begin_ms','draft_ms','accept_ms'),values))
def delta(now,previous):return {k:v-(previous or {}).get(k,0) for k,v in now.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arms',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',choices=['capture3','natural12','natural24'],default='capture3');ap.add_argument('--reps',type=int,default=1);ap.add_argument('--warmups',type=int,default=0);ap.add_argument('--prompts',type=Path);ap.add_argument('--trace',action='store_true');ap.add_argument('--paired-trace',action='store_true');a=ap.parse_args()
 arms=json.load(open(a.arms));prompts=json.load(open(a.prompts)) if a.prompts else cohort(a.cohort);a.out.mkdir(exist_ok=False);save(a.out/'prompts.json',prompts)
 (a.out/'hardware.txt').write_text(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','-q'],text=True))
 if next(iter(arms))!='target':raise ValueError('target must be first for correctness')
 save(a.out/'freeze.json',dict(arms={k:pin(v) if v else None for k,v in arms.items()},source=pin(__file__),runtime={n:pin(a.runtime/'bin'/n) for n in ('libllama.so.0.8.0','libggml-base.so.0.28.0','libggml-cuda.so.0.28.0','libggml-cpu.so.0.28.0','libllama-common.so.0.8.0','libllama-server-impl.so','libggml.so.0.28.0','llama-server')},target=pin(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),baseline_registry=pin(ROOT/'experiments/w1a8-standing-20261102/baseline.json'),build=pin(a.runtime/'build.json'),trace_library=pin(P/'runs/data-cost-transfer-20261104-01/libdecode-cost-trace.so') if (a.trace or a.paired_trace) else None,cohort='heldout24' if a.prompts else a.cohort,prompt_source=pin(a.prompts) if a.prompts else None,precision='Qwen3.5-4B Q4_K_M target;F16 KV;depth2;greedy;5080',sampling='target sample and match;temperature0',memory='nvidia-smi sampled device used MiB;includes display/unrelated baseline;not exact allocator peak'))
 stop=False
 def stopping(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 reference={};rows=[]
 for rep in range(a.reps+a.warmups):
  order=list(arms) if rep%2==0 else list(reversed(arms))
  queue=[(label,traced) for label in order for traced in ([False,True] if rep%2==0 else [True,False])] if a.paired_trace else [(label,a.trace) for label in order]
  for label,traced in queue:
   pause()
   if stop:raise RuntimeError('supervisor stop')
   model=arms[label];cell=a.out/(f'{rep:02d}-{label}'+('-traced' if traced else ''));cell.mkdir()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1A8_','LLAMA_','CUDA_','LD_','OMP_'))};env.update(LD_LIBRARY_PATH=str(a.runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
   argv=[str(a.runtime/'bin/llama-server'),'-m',str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-t','4','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port',str(port),'--metrics','--no-webui','--no-warmup','--spec-type','draft-dflash' if model else 'none','-lv','4']
   if model:argv+=['-md',str(model),'-ngld','99','--cache-type-k-draft','f16','--cache-type-v-draft','f16','--spec-draft-n-max','2','--spec-draft-n-min','0','--spec-draft-p-min','0','--spec-draft-sampling','greedy']
   if traced:env.update(LD_PRELOAD=str(P/'runs/data-cost-transfer-20261104-01/libdecode-cost-trace.so'),DECODE_COST_TRACE=str(cell/'cost-trace.jsonl'))
   save(cell/'command.json',dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith(('GGML_','CUDA_','LD_','OMP_','DECODE_')) or k in ('PATH','LANG')}))
   done=threading.Event();memory=[]
   def sample():
    while not done.is_set():
     try:memory.append(dict(monotonic=time.monotonic(),MiB=int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())))
     except Exception:pass
     done.wait(.2)
   watcher=threading.Thread(target=sample,daemon=True);watcher.start()
   with (cell/'server.log').open('w') as log:
    process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(cell/'ownership.json',dict(pid=process.pid,stat=Path(f'/proc/{process.pid}/stat').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text(),inherited_supervised_group=os.getpgrp()))
    try:
     for _ in range(120):
      if process.poll() is not None:raise RuntimeError(f'{label} server exited {process.returncode}')
      try:request(port,'health');break
      except Exception:time.sleep(1)
     else:raise RuntimeError('startup timeout')
     (cell/'library-maps.txt').write_text(Path(f'/proc/{process.pid}/maps').read_text())
     previous=None
     for item in prompts:
      pause()
      if stop:raise RuntimeError('supervisor stop')
      payload=dict(item['payload']);payload['cache_prompt']=False;payload['return_tokens']=True
      save(cell/(item['id']+'-request.json'),payload);start_ns=time.monotonic_ns();start=time.perf_counter();response=request(port,'completion',payload);elapsed=time.perf_counter()-start;end_ns=time.monotonic_ns();save(cell/(item['id']+'-response.json'),response)
      tokens=response['tokens']
      signature={k:response[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}
      if label=='target':reference.setdefault(item['id'],signature)
      correct=signature==reference[item['id']]
      # Completion logs precede response; one short flush grace if required.
      text=(cell/'server.log').read_text(errors='replace');now=stats(text)
      metrics=delta(now,previous) if now else {};previous=now
      prefix=re.findall(r'acc per pos = \(([^)]+)\)',text)
      if model and prefix:
       rates=[float(x) for x in prefix[-1].split(',')];n=metrics['rounds']
       metrics['prefix_survival_counts']=[round(x*n) for x in rates];metrics['prefix_count_method']='per-request rounded rate * exact rounds;rounding error<0.5 at<=256 rounds'
      row=dict(instrumented=traced,bounds_ns=[start_ns,end_ns],rep=rep,warmup=rep<a.warmups,arm=label,prompt=item['id'],correct=correct,wall_seconds=elapsed,tokens=len(tokens),timings=response.get('timings'),native=metrics)
      rows.append(row);save(a.out/'progress.json',rows)
      if not correct:raise RuntimeError('target mismatch '+label+' '+item['id'])
    finally:
     process.terminate()
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:process.kill();process.wait()
     done.set();watcher.join(timeout=5);save(cell/'memory.json',memory);save(cell/'release.json',dict(reaped=True,returncode=process.returncode))
   if traced:
    spans=[json.loads(line) for line in (cell/'cost-trace.jsonl').read_text().splitlines()]
    for row in [x for x in rows if x['rep']==rep and x['arm']==label and x['instrumented']]:
     begin,end=row['bounds_ns'];roots=sorted([x for x in spans if x['depth']==0 and begin<=x['start_ns']<=x['end_ns']<=end],key=lambda x:x['start_ns'])
     if not roots or any(x['end_ns']>y['start_ns'] for x,y in zip(roots,roots[1:])):raise RuntimeError('trace roots missing/overlap')
     markers=[x['start_ns'] for x in roots if x['name']=='generation_begin']
     boundary=markers[0] if model else next(x['end_ns'] for x in roots if x['name']=='target_sync')
     if model and (len(markers)!=1 or not any(x['name']=='draft_catchup' for x in roots)):raise RuntimeError('trace catchup/generation hook missing')
     costs={'prefill':{},'generation':{}}
     for x in roots:
      phase='prefill' if x['start_ns']<boundary else 'generation';v=costs[phase];v[x['name']]=v.get(x['name'],0)+(x['end_ns']-x['start_ns'])/1e6
     row['host_spans_ms']=costs;row['draft_full_ms']=sum(costs['generation'].get(n,0) for n in ('draft_generate','draft_catchup','draft_accept'));row['target_verify_and_sample_ms']=sum(costs['generation'].get(n,0) for n in ('target_process','target_sync','verifier_sample','target_sample'))
     row['trace_scope']='Disjoint host root spans; async GPU cost may be charged at later synchronization. Instrumented diagnostics, paired plain speed when present.'
   save(a.out/'progress.json',rows)
   print('Completed',rep,label,'traced' if traced else 'plain',flush=True)
 if a.paired_trace:
  for plain in [x for x in rows if not x['instrumented']]:
   traced=next(x for x in rows if x['instrumented'] and all(x[k]==plain[k] for k in ('rep','arm','prompt')))
   if any(plain['native'].get(k)!=traced['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')):raise RuntimeError('instrumentation changed native counters')
 save(a.out/'result.json',dict(status='COMPLETE',rows=rows,paired_trace_counters_equal=a.paired_trace))
if __name__=='__main__':main()
