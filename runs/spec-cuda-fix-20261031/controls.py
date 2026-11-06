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

for label,envextra in [('no-fusion',{'GGML_CUDA_DISABLE_FUSION':'1'}),('no-graphs',{'GGML_CUDA_DISABLE_GRAPHS':'1'})]:
 cmd=[str(P/'test-parity'),'-m',str(P/'models/Qwen3.5-4B-Q4_K_M.gguf'),'-ngl','99','-t','4','-fa','on','--test','spec-parity','--logs',str(P/(label+'-logs'))]
 with (P/(label+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,**envextra),check=True,timeout=240)
