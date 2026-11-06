from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-output-cap-20261103-01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d}
bp=P/'runs/w1a8-modern-capture-build-20261103-09/build.json';b=read(bp);baseline=read(P/'experiments/w1a8-standing-20261102/baseline.json');target=baseline['rows'][0]['target_pin']
for ref in b['runtime_libraries']+b['runtime_sources']+b['included_local_headers']+[b['source'],b['builder'],b['executable'],b['cuda_overlay'],b['baseline'],target]:assert pin(ref['path'])['sha256']==ref['sha256']
refs={}
for line in (P/'runs/qwen35-correctness-matrix-20261101-01/pilot-01/requests.jsonl').open():
 x=json.loads(line)
 if x['target_id']=='4B-Q4_K_M' and x['family']=='target_only' and x['phase']=='performance':refs[x['prompt_id']]=x['result']['tokens'][:9]
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']};env={**clean,**b['environment'],'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'}
(R/'freeze.json').write_text(json.dumps({'source':pin(__file__),'build':pin(bp),'target':target,'scope':'Existing pilot development: context output reservation1/3/512, same fixed history; no training or final release claim'},indent=2))
started=time.monotonic();results={}
for pid,previous in [(2,P/'runs/w1a8-modern-cadence-20261103-01'),(4,P/'runs/w1a8-modern-cadence-20261103-02')]:
 prefix='prompt-%02d'%pid;tokens=previous/(prefix+'-tokens/tokens.ids');fixed=previous/(prefix+'-continuation.ids');rows={}
 for name,cap,features in [('control1',1,False),('features1',1,True),('features3',3,True),('features512',512,True)]:
  control=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
  out=R/(prefix+'-'+name);argv=[b['executable']['path'],'--model',target['path'],'--tokens',str(tokens),'--out',str(out),*b['required_backend_argv'],'--decode-tokens',str(fixed),'--prefill-chunk','512','--decode-chunk','1','--output-cap',str(cap),'--threads','4','--taps','0,2,16,29','--max-output-mib','128']
  if not features:argv+=['--no-features']
  (R/(prefix+'-'+name+'-command.json')).write_text(json.dumps({'argv':argv,'environment':b['environment']},indent=2))
  with (R/(prefix+'-'+name+'.log')).open('x') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
  m=read(out/'capture.json');logits=np.fromfile(out/'logits.f32',dtype='<f4').reshape(9,248320);assert np.isfinite(logits).all() and logits.argmax(-1).tolist()==m['greedy_ids']
  assert m['output_cap']==cap and m['input_tokens']==[int(t) for t in tokens.read_text().split()]+refs[pid][:8]
  mapped=sorted({l.split()[-1] for l in (out/'loaded-library-maps.txt').read_text().splitlines() if 'libggml-cuda.so' in l});assert mapped==[baseline['cuda']['path']]
  mismatch=[{'row':i,'native':token,'server':refs[pid][i],'server_margin':float(logits[i,refs[pid][i]]-logits[i].max())} for i,token in enumerate(m['greedy_ids']) if token!=refs[pid][i]]
  rows[name]={'metadata':pin(out/'capture.json'),'mismatches':mismatch,'files':[pin(p) for p in sorted(out.glob('*.f32'))]}
  print(json.dumps({'prompt':pid,'case':name,'mismatches':mismatch}),flush=True)
 results[str(pid)]=rows
(R/'result.json').write_text(json.dumps({'status':'COMPLETE','results':results,'wall_seconds':time.monotonic()-started,'training_updates':0},indent=2))
