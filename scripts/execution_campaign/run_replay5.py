from pathlib import Path
import json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,B,pin
from scripts.dflash_baseline.native_archive import hydrate,release
R=P/'runs/execution-campaign-20261105-01';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
exe=R/'replay-v5';timer=R/'libgraph-timer.so'
cmd=['c++','-O2','-std=c++17','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-I'+str(S/'vendor/nlohmann'),str(ROOT/'scripts/execution_campaign/replay.cpp'),'-L'+str(F/'bin'),'-lllama','-lggml-base','-lggml','-ldl','-Wl,-rpath,'+str(F/'bin'),'-o',str(exe)]
if not exe.exists():subprocess.run(cmd,check=True)
cmd2=['c++','-shared','-fPIC','-O2','-I/usr/local/cuda/include',str(ROOT/'scripts/execution_campaign/graph_timer.cpp'),'-L/usr/local/cuda/lib64','-lcudart','-ldl','-o',str(timer)]
if not timer.exists():subprocess.run(cmd2,check=True)
arms=json.load(open(R/'arms.json'));dest=R/'replay-tranche5';dest.mkdir(exist_ok=True)
model=arms['w1'];created=hydrate(model)
try:
 for name in ['original','combined','combined-head','combined-last','silu-debug','q4','q4opt']:
  runtime=F if name in ('original','q4') else R/('runtime-'+('combined-last' if name=='q4opt' else name))
  for audit in [1,0]:
   out=dest/(name+('-audit' if audit else '-clean')+'.json')
   if out.exists():continue
   env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_'))};env.update(LD_LIBRARY_PATH=str(runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
   if name in ('combined-head','combined-last','q4opt'):env['EXEC_REQUEST_MASKS']='1'
   if not audit:env['LD_PRELOAD']=str(timer)
   capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture';meta=json.load(open(capture/'capture.json'));anchor=meta['prompt_tokens']+5
   argv=[str(exe),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),arms['q4'] if name.startswith('q4') else model,str(capture),str(anchor),'1' if audit else '7',str(out),str(audit)]
   with (dest/(name+f'-{audit}.log')).open('w') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   print('REPLAY',name,audit,flush=True)
finally:release(model,created)
