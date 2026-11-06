from pathlib import Path
import subprocess,json,os,sys,shutil,time
P=Path(__file__).resolve().parent;OLD=P.parent/'spec-shortlist-20261030';S=OLD/'runtime'
runtime=json.loads((Path.home()/'.local/share/p-stack/runtime.json').read_text());sys.path.insert(0,str(Path.home()/'.local/share/p-stack/runtime'/runtime['source_digest']));os.environ.setdefault('P_STACK_BROKER_COMMAND',json.dumps(runtime['bridge']))
from devicecheckout.project import before_costly_work
before_costly_work('benchmark_quiet')
import fcntl
lock=(Path.home()/'.config/binary-eagle-decoding/rtx5080-campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()=='792fcf4e7501972b595d3aeeb85153f9926f6aef'
assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
assert shutil.disk_usage(P).free>3_000_000_000
cmd=['cmake','-S',str(P/'patched-source'),'-B',str(P/'patched-build'),'-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_FLAGS=-I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_COMMON=OFF','-DLLAMA_BUILD_MTMD=OFF','-DLLAMA_BUILD_APP=OFF']
(P/'configure-command.json').write_text(json.dumps(cmd));subprocess.run(cmd,check=True)
subprocess.run(['cmake','--build',str(P/'patched-build'),'--target','ggml-cuda','-j','6'],check=True)
(P/'build-complete.json').write_text(json.dumps({'finished':time.time()}))
