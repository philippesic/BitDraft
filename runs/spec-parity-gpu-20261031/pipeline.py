# Native CUDA differential test and focused original full-request reproduction.
from pathlib import Path
import subprocess,json,os,sys
P=Path(__file__).resolve().parent;OLD=P.parent/'spec-shortlist-20261030';S=OLD/'runtime';B=OLD/'build/bin'
runtime=json.loads((Path.home()/'.local/share/p-stack/runtime.json').read_text());sys.path.insert(0,str(Path.home()/'.local/share/p-stack/runtime'/runtime['source_digest']));os.environ.setdefault('P_STACK_BROKER_COMMAND',json.dumps(runtime['bridge']))
from devicecheckout.project import before_costly_work,check_pause
before_costly_work('benchmark_quiet')
import fcntl
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']['pause_requested']
assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()=='792fcf4e7501972b595d3aeeb85153f9926f6aef'
assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
for name in ['models','build']:
 if not (P/name).exists():(P/name).symlink_to(OLD/name,target_is_directory=True)
case=(S/'tests/llama-context/cases-rs.cpp').read_text()
case=case.replace('std::vector<ctx_test_ptr> make_rs_cases()', (P/'parity-case.inc').read_text()+'\nstd::vector<ctx_test_ptr> make_rs_cases()')
case=case.replace('    std::vector<ctx_test_ptr> cases;', '    std::vector<ctx_test_ptr> cases;\n    cases.push_back(std::make_unique<case_spec_parity>());')
(P/'cases-rs.cpp').write_text(case)
sources=[S/'tests/test-llama-context.cpp',*[x for x in (S/'tests/llama-context').glob('*.cpp') if x.name!='cases-rs.cpp'],P/'cases-rs.cpp']
argv=['g++','-O2','-std=c++17','-pthread',*[f'-I{x}' for x in [S/'include',S/'ggml/include',S/'common',S/'src',S/'tests/llama-context']],*[str(x) for x in sources],'-L'+str(B),'-Wl,-rpath,'+str(B),'-lllama-common','-lllama','-lggml','-lggml-base','-o',str(P/'test-parity')]
(P/'compile-command.json').write_text(json.dumps(argv));subprocess.run(argv,check=True)
results=[]
for model in ['Qwen3.5-4B-Q4_K_M.gguf','qwen35-4b-mtp-nvfp4.gguf','Qwen3.5-9B-Q4_K_M.gguf','qwen35-9b-mtp-nvfp4.gguf']:
 check_pause()
 cmd=[str(P/'test-parity'),'-m',str(P/'models'/model),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(P/('native-'+model))]
 with (P/(model+'.native.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 results.append({'model':model,'returncode':r.returncode,'command':cmd});(P/'native-results.json').write_text(json.dumps(results,indent=2))
 print('native',model,r.returncode,flush=True)
lock.close()
subprocess.run([sys.executable,str(P/'focused.py')],check=True)
