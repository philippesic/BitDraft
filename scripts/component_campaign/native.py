"""Matched native panels with authentic consumed DEV prompts and complete draft cost."""
import argparse,hashlib,json,os,re,signal,socket,subprocess,sys,threading,time,urllib.request
from collections import Counter
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');TRACE=Path(os.environ.get('EXEC_TRACE_LIBRARY',str(P/'runs/data-cost-transfer-20261104-01/libdecode-cost-trace.so')));ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.component_campaign.archive import reference as archived_reference,hydrate,release
from scripts.dflash_baseline.native_counters import STAT,delta,reconcile
def pin(path):
 p=Path(path)
 if p.suffix=='.gguf' and not p.exists():return archived_reference(p)
 return dict(path=str(p.resolve()),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),bytes=p.stat().st_size)
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
def heldout(path,arms,runtime):
 audit=json.load(open(path.parent/'audit.json'));consumed=json.load(open(path.parent/'consumption.json'))
 if audit['role']!='HELDOUT_AFTER_SHORTLIST_LOCK' or consumed['selection']!=audit['selection']:raise ValueError('heldout selection/consumption missing')
 selected=auth(audit['selection'])
 if selected['status']!='SHORTLIST_LOCKED_BEFORE_HELDOUT' or selected['heldout_opened'] is not False:raise ValueError('shortlist did not precede heldout')
 if selected['target']!=pin(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'):raise ValueError('heldout target differs from measured lane')
 expected={'target':None,'q4':selected['baseline'],**selected.get('controls',{}),**selected['candidates']}
 if set(expected)!=set(arms):raise ValueError('heldout arms differ from frozen shortlist')
 for name,ref in expected.items():
  if (ref is None and arms[name] is not None) or (ref is not None and pin(arms[name])['sha256']!=ref['sha256']):raise ValueError('heldout model changed')
 for name,ref in selected['runtime'].items():
  if pin(runtime/'bin'/name)['sha256']!=ref['sha256']:raise ValueError('heldout runtime changed')
 rows=json.load(open(path))
 if len(rows)!=24 or Counter(x['domain'] for x in rows)!=Counter(code=8,reasoning=8,prose=8) or {x['group_id'] for x in rows}!={x['group_id'] for x in consumed['records']}:raise ValueError('heldout cohort coverage changed')
 return rows
def cohort(name):
 if name=='capture3':
  data_pin=json.load(open(P/'runs/precision-campaign-20261104-01/qualification-config.json'))['data'];manifest=auth(data_pin);records=sorted([r for r in manifest['records'] if r['split']=='consumed_DEV'],key=lambda r:r['name'])
  if len(records)!=3:raise ValueError('capture DEV coverage changed')
  rows=[]
  for r in records:
   m=auth(r['metadata']);rows.append(dict(id=r['name'],domain=next(d for d in ('code','reasoning','prose') if d in r['name']),source=r['metadata'],payload=dict(prompt=m['input_tokens'][:m['prompt_tokens']],n_predict=256,temperature=0,seed=1234,cache_prompt=False,return_tokens=True)))
  return rows
 refs=json.load(open(P/'runs/data-cost-transfer-20261104-01/consumed-natural-inputs.json'));auth(refs['manifest']);marker=auth(refs['marker'])
 if not marker['evaluated']:raise ValueError('natural DEV consumption not authenticated')
 items=refs['rows']
 if name=='natural12':
  items=[x for domain in ('code','reasoning','prose') for x in [x for x in items if x['domain']==domain][:4]]
  if len(items)!=12:raise ValueError('natural12 domain mapping')
 return [dict(id=x['id'],payload=auth(x['request']),source=x['request'],domain=x['domain']) for x in items]
def stats(text):
 matches=list(STAT.finditer(text))
 if not matches:return None
 m=matches[-1];values=[int(m[i]) for i in range(1,8)]+[float(m[i]) for i in range(8,11)]
 return dict(zip(('begin_calls','rounds','accept_calls','drafts','accepted_rounds','proposed','accepted','begin_ms','draft_ms','accept_ms'),values))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arms',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',choices=['capture3','natural12','natural24'],default='capture3');ap.add_argument('--reps',type=int,default=1);ap.add_argument('--warmups',type=int,default=0);ap.add_argument('--prompts',type=Path);ap.add_argument('--cohort-label');ap.add_argument('--trace',action='store_true');ap.add_argument('--paired-trace',action='store_true');ap.add_argument('--profile',action='store_true');ap.add_argument('--limit',type=int);ap.add_argument('--output-cap',type=int);ap.add_argument('--runtime-map',type=Path);ap.add_argument('--draft-cache-map',type=Path);ap.add_argument('--stress-length',type=int);ap.add_argument('--target',type=Path,default=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf');ap.add_argument('--depth',type=int,default=2);a=ap.parse_args()
 arms=json.load(open(a.arms));prompts=heldout(a.prompts,arms,a.runtime) if a.prompts else cohort(a.cohort);a.out.mkdir(exist_ok=False);save(a.out/'prompts.json',prompts)
 if a.prompts and (a.depth!=2 or a.target!=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'):raise ValueError('Frozen heldout contract is 4B Q4 depth2')
 if a.stress_length:
  if a.cohort!='capture3' or not 0<a.stress_length<=3840:raise ValueError('stress uses consumed capture token IDs and<=3840 positions')
  for item in prompts:
   ids=item['payload']['prompt'];item['payload']['prompt']=(ids*((a.stress_length+len(ids)-1)//len(ids)))[:a.stress_length]
   item['payload']['ignore_eos']=True
   item['id']+='-synthetic-context-'+str(a.stress_length)
  save(a.out/'context-stress.json',dict(positions=a.stress_length,construction='Repeated already-consumed DEV token sequence; context/cache stress, not natural heldout quality.',stopping='ignore_eos=true in every matched arm to exercise full256-token reserve and sequential cache growth',source=a.cohort))
 if a.limit:prompts=prompts[:a.limit]
 if a.output_cap:
  for x in prompts:x['payload']['n_predict']=min(x['payload']['n_predict'],a.output_cap)
 save(a.out/'prompts.json',prompts)
 runtime_map=json.load(open(a.runtime_map)) if a.runtime_map else {}
 draft_cache_map=json.load(open(a.draft_cache_map)) if a.draft_cache_map else {}
 original_runtime=a.runtime
 (a.out/'hardware.txt').write_text(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','-q'],text=True))
 if next(iter(arms))!='target':raise ValueError('target must be first for correctness')
 save(a.out/'freeze.json',dict(arms={k:pin(v) if v else None for k,v in arms.items()},source=pin(__file__),counter_parser=pin(ROOT/'scripts/dflash_baseline/native_counters.py'),runtime={n:pin(a.runtime/'bin'/n) for n in ('libllama.so.0.8.0','libggml-base.so.0.28.0','libggml-cuda.so.0.28.0','libggml-cpu.so.0.28.0','libllama-common.so.0.8.0','libllama-server-impl.so','libggml.so.0.28.0','llama-server')},target=pin(a.target),baseline_registry=pin(ROOT/'experiments/w1a8-standing-20261102/baseline.json'),build=pin(a.runtime/'build.json'),trace_library=pin(TRACE) if (a.trace or a.paired_trace) else None,cohort=a.cohort_label or ('heldout24' if a.prompts else a.cohort),cohort_source=pin(a.prompts.parent/'audit.json') if a.prompts else pin(P/'runs/dflash-modern-capture-20261104-04/data/result.json') if a.cohort=='capture3' else pin(P/'runs/data-cost-transfer-20261104-01/consumed-natural-inputs.json'),prompt_source=pin(a.prompts) if a.prompts else None,precision=f'{a.target.name};targetF16 KV;draftKV per declared arm map;depth{a.depth};greedy;5080',draft_cache_types=draft_cache_map,synthetic_context_positions=a.stress_length,warmup_policy=dict(per_server_cell=a.warmups,measured_repetitions=a.reps,same_process=True,source='first consumed cohort prompts, repeated only for excluded warmup'),sampling='target sample and match;temperature0',memory='nvidia-smi sampled device used MiB;includes display/unrelated baseline;not exact allocator peak'))
 stop=False
 def stopping(*_):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 reference={};rows=[];counter_reconciliations=[]
 for rep in range(a.reps):
  order=list(arms) if rep%2==0 else list(reversed(arms))
  queue=[(label,traced) for label in order for traced in ([False,True] if rep%2==0 else [True,False])] if a.paired_trace else [(label,a.trace) for label in order]
  for label,traced in queue:
   pause()
   if stop:raise RuntimeError('supervisor stop')
   a.runtime=Path(runtime_map.get(label,original_runtime))
   model=arms[label];cell=a.out/(f'{rep:02d}-{label}'+('-traced' if traced else ''));cell.mkdir()
   with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','W1A8_','LLAMA_','CUDA_','LD_','OMP_'))};env.update(LD_LIBRARY_PATH=str(a.runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
   argv=[str(a.runtime/'bin/llama-server'),'-m',str(a.target),'-ngl','99','-c','4096','-np','1','-b','512','-ub','512','-t','4','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--host','127.0.0.1','--port',str(port),'--metrics','--no-webui','--no-warmup','--spec-type','draft-dflash' if model else 'none','-lv','4']
   if model:argv+=['-md',str(model),'-ngld','99','--cache-type-k-draft',draft_cache_map.get(label,['f16','f16'])[0],'--cache-type-v-draft',draft_cache_map.get(label,['f16','f16'])[1],'--spec-draft-n-max',str(a.depth),'--spec-draft-n-min','0','--spec-draft-p-min','0','--spec-draft-sampling','greedy']
   if traced:env.update(LD_PRELOAD=str(TRACE),DECODE_COST_TRACE=str(cell/'cost-trace.jsonl'),EXEC_PROPOSAL_TRACE=str(cell/'proposals.jsonl'))
   save(cell/'command.json',dict(runtime=str(a.runtime),runtime_build=pin(a.runtime/'build.json'),argv=argv,environment={k:v for k,v in env.items() if k.startswith(('GGML_','CUDA_','LD_','OMP_','DECODE_')) or k in ('PATH','LANG')}))
   if a.profile and model:
    argv=['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx','--cuda-graph-trace=node','--force-overwrite=false','-o',str(cell/'profile')]+argv
    save(cell/'profiler-command.json',dict(argv=argv,scope='Nsight instrumented format diagnostic; never an uninstrumented request timing'))
   created_native=hydrate(model) if model else False
   done=threading.Event();memory=[];server_pid=None
   def sample():
    while not done.is_set():
     try:
      values=[v.strip() for v in subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,utilization.gpu,clocks.current.sm,power.draw,temperature.gpu,pstate','--format=csv,noheader,nounits'],text=True).strip().split(',')]
      entry=dict(monotonic=time.monotonic(),MiB=int(values[0]),pstate=values[5])
      if server_pid is not None:
       try:
        host=Path(f'/proc/{server_pid}/status').read_text()
        entry.update({key:int(re.search(r'^'+key+r':\s+(\d+) kB',host,re.M)[1])*1024 for key in ('VmRSS','VmHWM')})
       except (OSError,TypeError):pass
       try:
        residency=Path(f'/proc/{server_pid}/smaps_rollup').read_text()
        entry.update({key+'_bytes':int(re.search(r'^'+key+r':\s+(\d+) kB',residency,re.M)[1])*1024 for key in ('Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty')})
       except (OSError,TypeError):pass
      for key,value in zip(('utilization_percent','sm_clock_MHz','power_W','temperature_C'),values[1:5]):
       try:entry[key]=float(value)
       except ValueError:entry[key]=None
      memory.append(entry)
     except Exception:pass
     done.wait(.2)
   watcher=threading.Thread(target=sample,daemon=True);watcher.start()
   with (cell/'server.log').open('w') as log:
    try:
     process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);server_pid=process.pid
    except BaseException:
     done.set();watcher.join(timeout=5)
     if model:release(model,created_native)
     raise
    save(cell/'ownership.json',dict(pid=process.pid,stat=Path(f'/proc/{process.pid}/stat').read_text(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text(),inherited_supervised_group=os.getpgrp()))
    try:
     for _ in range(120):
      if process.poll() is not None:raise RuntimeError(f'{label} server exited {process.returncode}')
      try:request(port,'health');break
      except Exception:time.sleep(1)
     else:raise RuntimeError('startup timeout')
     actual_cache=re.findall(r'llama_kv_cache: size =.*?K \(([^)]+)\):.*?V \(([^)]+)\):',(cell/'server.log').read_text(errors='replace'))
     expected_cache=draft_cache_map.get(label,['f16','f16'])
     if not actual_cache or list(actual_cache[0])!=['f16','f16'] or (model and (len(actual_cache)<2 or any(list(x)!=expected_cache for x in actual_cache[1:]))):raise ValueError('Actual native target/draft cache allocation differs from frozen precision '+label)
     save(cell/'cache-allocation.json',dict(actual=actual_cache,target=['f16','f16'],draft=expected_cache if model else None,policy='Observed native allocation after health readiness, before requests; target immutable, draft-only formats.'))
     (cell/'library-maps.txt').write_text(Path(f'/proc/{process.pid}/maps').read_text())
     geometry=[]
     for item in prompts:
      payload=item['payload'];prompt=payload['prompt']
      ids=prompt if isinstance(prompt,list) else request(port,'tokenize',dict(content=prompt,add_special=True,parse_special=True))['tokens']
      if not (0<len(ids) and len(ids)+payload['n_predict']<=4096):raise RuntimeError('frozen prompt exceeds native context; no truncation/replacement')
      geometry.append(dict(id=item['id'],prompt_tokens=len(ids),output_reserve=payload['n_predict']))
     save(cell/'geometry.json',geometry)
     previous=None
     items=[dict(prompts[i%len(prompts)],id='warmup-'+str(i)+'-'+prompts[i%len(prompts)]['id'],_warmup=True) for i in range(a.warmups)]+prompts
     for item in items:
      pause()
      if stop:raise RuntimeError('supervisor stop')
      payload=dict(item['payload']);payload['cache_prompt']=False;payload['return_tokens']=True
      save(cell/(item['id']+'-request.json'),payload);start_ns=time.monotonic_ns();start=time.perf_counter();response=request(port,'completion',payload);elapsed=time.perf_counter()-start;end_ns=time.monotonic_ns();save(cell/(item['id']+'-response.json'),response)
      tokens=response['tokens']
      signature={k:response[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}
      if label=='target':reference.setdefault(item['id'],signature)
      correct=signature==reference[item['id']]
      # Provisional live log read; final reaped log is reconciled before cell acceptance.
      text=(cell/'server.log').read_text(errors='replace');now=stats(text)
      metrics=delta(now,previous) if now else {};previous=now
      prefix=re.findall(r'acc per pos = \(([^)]+)\)',text)
      if model and prefix and metrics:
       rates=[float(x) for x in prefix[-1].split(',')];n=metrics['rounds']
       metrics['prefix_survival_counts']=[round(x*n) for x in rates];metrics['prefix_count_method']='per-request rounded rate * exact rounds;rounding error<0.5 at<=256 rounds'
      row=dict(instrumented=traced,profiled=bool(a.profile and model),bounds_ns=[start_ns,end_ns],rep=rep,warmup=item.get('_warmup',False),arm=label,prompt=item['id'],correct=correct,wall_seconds=elapsed,tokens=len(tokens),timings=response.get('timings'),native=metrics)
      rows.append(row);save(a.out/'progress.json',rows)
      if not correct:raise RuntimeError('target mismatch '+label+' '+item['id'])
    finally:
     process.terminate()
     try:process.wait(timeout=90 if a.profile else 15)
     except subprocess.TimeoutExpired:process.kill();process.wait()
     done.set();watcher.join(timeout=5);save(cell/'memory.json',memory);save(cell/'release.json',dict(reaped=True,returncode=process.returncode))
     if model:release(model,created_native)
   if model:
    corrections=reconcile(cell,[x for x in rows if x['rep']==rep and x['arm']==label and x['instrumented']==traced])
    if corrections:
     counter_reconciliations.extend(corrections);print('Reconciled delayed native log rows:',len(corrections),rep,label,traced,flush=True)
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
     row['draft_prefill_ms']=sum(costs['prefill'].get(n,0) for n in ('draft_generate','draft_catchup','draft_accept'))
     row['draft_all_phases_ms']=row['draft_prefill_ms']+row['draft_full_ms']
     row['trace_scope']='Disjoint host root spans; async GPU cost may be charged at later synchronization. Instrumented diagnostics, paired plain speed when present.'
   save(a.out/'progress.json',rows)
   print('Completed',rep,label,'traced' if traced else 'plain',flush=True)
 if a.paired_trace:
  for plain in [x for x in rows if not x['instrumented']]:
   traced=next(x for x in rows if x['instrumented'] and all(x[k]==plain[k] for k in ('rep','arm','prompt')))
   if any(plain['native'].get(k)!=traced['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')):raise RuntimeError('instrumentation changed native counters')
 save(a.out/'result.json',dict(status='COMPLETE',rows=rows,paired_trace_counters_equal=a.paired_trace,counter_reconciliations=counter_reconciliations))
if __name__=='__main__':main()
