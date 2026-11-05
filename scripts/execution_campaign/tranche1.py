"""Build and qualify the first exact execution alternatives serially."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F,B,pin
R=P/'runs/execution-campaign-20261105-01'
variants=[('copy',dict(copy=True)),('bound256',dict(bound=True)),('both256',dict(copy=True,bound=True)),('both32',dict(copy=True,bound=True,threads=32)),('cross',dict(copy=True,bound=True,threads=32,dot='cross')),('mma',dict(copy=True,bound=True,threads=32,dot='mma'))]
for name,options in variants:
 if json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))['rtx5080']['pause_requested']:raise RuntimeError('user pause')
 r=build(R/('runtime-'+name),**options);print('BUILT',name,flush=True)
 if name in ('copy','bound256','both256'):continue
 out=r/'op-check';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
 cmd=['c++','-O2','-std=c++17','-ffp-contract=off','-I'+str(S/'ggml/include'),str(ROOT/'scripts/dflash_baseline/precision_op_check.cpp'),'-L'+str(r/'bin'),'-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-Wl,-rpath,'+str(r/'bin'),'-o',str(out)]
 subprocess.run(cmd,check=True)
 env={**os.environ,'LD_LIBRARY_PATH':str(r/'bin'),'GGML_CUDA_DECODE_CONSISTENT':'32'}
 for n in (1,3,8,9):
  with (r/f'ops-N{n}.jsonl').open('w') as log:subprocess.run([str(out),str(r/f'maps-N{n}.txt'),'fused',str(n)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('QUALIFIED',name,n,flush=True)
