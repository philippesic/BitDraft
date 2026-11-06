from pathlib import Path
import json,os,sys,time,subprocess,fcntl,hashlib,re,resource
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');O=P/'runs/spec-cuda-fix-20261031';V=R/'verification04'
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
while True:
 try:s=json.loads((W/'supervisors/qualification04/state.json').read_text())
 except (OSError,ValueError):time.sleep(3);continue
 if s['status']!='running':break
 time.sleep(3)
assert s['exit_code']==0,('qualification did not finish',s)
k=json.loads((R/'interop-keeper.json').read_text());assert Path(k['interop']).exists();os.environ['WSL_INTEROP']=k['interop']
rt=Path.home()/'.local/share/p-stack/runtime.json';d=json.loads(rt.read_text());sys.path.insert(0,str(rt.parent/'runtime'/d['source_digest']));os.environ['P_STACK_BROKER_COMMAND']=json.dumps(d['bridge']);from devicecheckout.project import before_costly_work
before_costly_work('benchmark_quiet')
f=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
V.mkdir(exist_ok=False)
for name in ['build','patched-build','models']:(V/name).symlink_to(O/name,target_is_directory=True)
env={**os.environ,'GGML_CUDA_DECODE_CONSISTENT':'32','LD_LIBRARY_PATH':str(O/'patched-build/bin')+':'+str(O/'build/bin')};env.pop('LD_PRELOAD',None)
backend=O/'patched-build/bin/libggml-cuda.so.0.28.0';expected='ce3cb04b2fb1145e9950e5924b61333da61d18d2ed6fd0458446c015ac8e508c'
assert hashlib.file_digest(backend.open('rb'),'sha256').hexdigest()==expected
results=[]
for model in ['Qwen3.5-4B-Q4_K_M.gguf','qwen35-4b-mtp-nvfp4.gguf']:
 dest=V/('regression-'+model);cmd=[str(O/'test-regression'),'-m',str(O/'models'/model),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(dest)]
 with (V/(model+'.stdout.log')).open('x') as out:r=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=300)
 assert r.returncode==0,(model,r.returncode)
 matches=[x for p in dest.glob('*.log') for x in re.findall(r'spec parity: checks=(\d+) greedy_flips=(\d+) max_abs=([^\s]+)',p.read_text())];assert len(matches)==2 and sum(int(x[0]) for x in matches)==510 and all(int(x[1])==0 and float(x[2])==0 for x in matches)
 results.append({'model':model,'checks':510,'max_abs':0,'greedy_flips':0,'command':cmd});print(json.dumps(results[-1]),flush=True)
f.close()
source=(O/'validate-patched.py').read_text();assert source.count("for size in ['4B','9B']:")==1;source=source.replace("for size in ['4B','9B']:","for size in ['4B']:").replace("'127.0.0.1:18935'","'127.0.0.1:18936'").replace("127.0.0.1:18935","127.0.0.1:18936").replace("'18935'","'18936'");(V/'validate-patched-4b.py').write_text(source)
subprocess.run([sys.executable,'-B',str(V/'validate-patched-4b.py')],env=env,check=True,timeout=1200)
pairs=[]
for target in sorted((V/'measurements').iterdir()):
 base=[json.loads(l) for l in (target/'no-spec/patched/requests.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in (target/'mtp/patched/requests.jsonl').read_text().splitlines()];a={(r['prompt'],r['rep']):r for r in base};b={(r['prompt'],r['rep']):r for r in rows};assert len(a)==len(b)==32 and a.keys()==b.keys();assert all(r['result']['timings']['cache_n']==0 for r in base+rows);bad=[k for k in a if a[k]['result']['tokens']!=b[k]['result']['tokens']];assert not bad,(target,bad);pairs.append({'model':target.name,'comparisons':32,'exact':32,'prompt_cache_hits':0})
assert len(pairs)==2
result={'schema':'independent_4b_cuda_correctness_v1','backend':{'path':str(backend),'sha256':expected},'mode':'GGML_CUDA_DECODE_CONSISTENT=32; F16 KV; FAon; concurrency1','native_regressions':results,'full_requests':pairs,'correctness_pass':True,'scope':'bounded greedy Qwen3.5 4B Q4_K_M/NVFP4 with maximum32verificationbatch including anchor; compare against same-mode target-only; no universal or stochastic claim','completed_unix':time.time()};(R/'verification04-result.json').write_text(json.dumps(result,indent=2));print('INDEPENDENT_4B_CORRECTNESS_PASS',flush=True)
