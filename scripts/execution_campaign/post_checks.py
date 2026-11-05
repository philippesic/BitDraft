from pathlib import Path
import json,os,subprocess,sys,shutil,sqlite3
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
from scripts.dflash_baseline.native_archive import hydrate,release
R=P/'runs/execution-campaign-20261105-01';r=R/'post-checks';r.mkdir(exist_ok=False)
subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign/reproduce.py'),str(R/'runtime-reproduction01')],check=True)
# All-output diagnostics and embedding requests must retain anchor work/outputs.
s=(ROOT/'scripts/execution_campaign/replay.cpp').read_text().replace('auto cp=llama_context_default_params();','auto cp=llama_context_default_params();cp.embeddings=getenv("EXEC_EMBEDDINGS");')
(r/'replay-source.cpp').write_text(s);S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';exe=r/'replay';cmd=['c++','-O2','-std=c++17','-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-I'+str(S/'vendor/nlohmann'),str(r/'replay-source.cpp'),'-L'+str(F/'bin'),'-lllama','-lggml-base','-lggml','-ldl','-Wl,-rpath,'+str(F/'bin'),'-o',str(exe)];subprocess.run(cmd,check=True)
arms=json.load(open(R/'arms.json'));created=hydrate(arms['w1']);records=[]
try:
 for mode in ['all_outputs','embeddings']:
  for name,runtime in [('original',F),('retained',R/'runtime-combined-last02'),('reproduced',R/'runtime-reproduction01')]:
   capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture';anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5;env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_','EXEC_'))};env.update(LD_LIBRARY_PATH=str(runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
   if mode=='embeddings':env['EXEC_EMBEDDINGS']='1'
   out=r/(mode+'-'+name+'.json')
   with out.with_suffix('.log').open('w') as log:subprocess.run([str(exe),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),arms['w1'],str(capture),str(anchor),'1',str(out),'1'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  a=json.load(open(r/(mode+'-original.json')))['rows']
  for name in ['retained','reproduced']:
   b=json.load(open(r/(mode+'-'+name+'.json')))['rows'];assert all(x['outputs']==y['outputs'] and x['cache_hash_fnv1a64']==y['cache_hash_fnv1a64'] for x,y in zip(a,b));records.append(dict(mode=mode,variant=name,all_logits_including_anchor_and_cache_exact=True))
finally:release(arms['w1'],created)
(r/'fallback-reproduction.json').write_text(json.dumps(dict(checks=records,build=pin(R/'runtime-reproduction01/build.json'),sources=[pin(r/'replay-source.cpp'),pin(ROOT/'scripts/execution_campaign/post_checks.py')]),indent=2)+'\n')
# Observe actual cuBLAS FP16 dispatch, without requiring privileged GPU counters.
out=r/'fp16-dispatch';subprocess.run(['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda','-o',str(out),sys.executable,str(ROOT/'scripts/execution_campaign/fp16_dispatch.py')],stdout=(r/'fp16-profile.log').open('w'),stderr=subprocess.STDOUT,check=True)
subprocess.run(['nsys','export','--type','sqlite','--output',str(out.with_suffix('.sqlite')),str(out.with_suffix('.nsys-rep'))],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True);c=sqlite3.connect(out.with_suffix('.sqlite'));kernels=c.execute('select distinct s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName').fetchall();names=[x[0] for x in kernels if any(s in x[0].lower() for s in ['gemm','gemv','h168','h884'])];(r/'fp16-dispatch.json').write_text(json.dumps(dict(actual_cublas_kernel_names=names,scope='Actual N3 gate/up and down cuBLAS F16-code/F32-acc adapter; no timing claims from profiler; matrix instruction mechanism only claimed if disassembly confirms it.'),indent=2)+'\n');print('POST_CHECKS_COMPLETE',names,flush=True)
