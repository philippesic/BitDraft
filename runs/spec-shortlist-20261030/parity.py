"""Bounded first-divergence diagnostics against target-only next-token probabilities."""
from pathlib import Path
import json,sys,subprocess,time,urllib.request,fcntl,os,signal
P=Path(__file__).resolve().parent;ROOT=P.parent.parent
runtime=json.loads((Path.home()/'.local/share/p-stack/runtime.json').read_text());sys.path.insert(0,str(Path.home()/'.local/share/p-stack/runtime'/runtime['source_digest']));os.environ.setdefault('P_STACK_BROKER_COMMAND',json.dumps(runtime['bridge']))
from devicecheckout.project import before_costly_work,check_pause
before_costly_work('benchmark_quiet');lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prompts=json.loads((P/'prompts.json').read_text());results=[]
def req(path,data=None):
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18935'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=30) as r:return json.load(r)
for group in (P/'measurements').iterdir():
 refpath=group/'no-spec/screen/requests.jsonl'
 if not refpath.exists():continue
 refs={(r['prompt'],r['rep']):r for r in map(json.loads,refpath.read_text().splitlines())};tests=[]
 for arm in group.iterdir():
  if arm.name.startswith('no-spec'):continue
  f=arm/'screen/requests.jsonl'
  if not f.exists():continue
  for r in map(json.loads,f.read_text().splitlines()):
   if r['rep']!=0:continue
   a=refs[r['prompt'],0]['result']['tokens'];b=r['result']['tokens'];idx=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
   if idx is not None:
    tests.append({'arm':arm.name,'prompt':r['prompt'],'first_divergence':idx,'reference_token':a[idx],'candidate_token':b[idx],'prefix_output_tokens':a[:idx]});break
 if not tests:continue
 argv=json.loads((group/'no-spec/screen/command.json').read_text());log=(P/(group.name+'.parity-server.log')).open('w');server=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 try:
  deadline=time.monotonic()+120
  while True:
   if server.poll() is not None:raise RuntimeError('parity server load failed')
   try:
    if req('/health').get('status')=='ok':break
   except (OSError,ValueError):pass
   if time.monotonic()>deadline:raise TimeoutError('load timeout')
   time.sleep(.25)
  for test in tests:
   check_pause();assert not (P/'STOP').exists()
   inp=req('/tokenize',{'content':prompts[test['prompt']]['prompt'],'add_special':False,'parse_special':True})['tokens']+test.pop('prefix_output_tokens')
   response=req('/completion',{'prompt':inp,'n_predict':1,'temperature':0,'seed':1234,'cache_prompt':False,'return_tokens':True,'n_probs':10})
   test.update(target=group.name,result=response);results.append(test);(P/'parity-diagnostics.json').write_text(json.dumps(results,indent=2))
 finally:
  server.terminate()
  try:server.wait(timeout=20)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  log.close()
print('First-divergence diagnostics',len(results))
