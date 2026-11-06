from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np
P=Path('/home/philip/binary-eagle-decoding')
R=P/'runs/w1a8-modern-capture-qualification-20261103-01'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p);return {'path':str(p.resolve()),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
bp=P/'runs/w1a8-modern-capture-build-20261103-05/build.json';b=read(bp);baseline=read(P/'experiments/w1a8-standing-20261102/baseline.json');target=baseline['rows'][0]['target_pin'];assert pin(target['path'])['sha256']==target['sha256'];assert pin(b['executable']['path'])['sha256']==b['executable']['sha256']
for ref in b['runtime_libraries']+[b['cuda_overlay'],b['source'],b['builder']]:assert pin(ref['path'])['sha256']==ref['sha256']
reference=None
for line in (P/'runs/qwen35-correctness-matrix-20261101-01/pilot-01/requests.jsonl').open():
 row=json.loads(line)
 if row['target_id']=='4B-Q4_K_M' and row['family']=='target_only' and row['phase']=='performance' and row['prompt_id']==0:reference=row['result']['tokens'][:9];break
assert reference is not None
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']};env={**clean,**b['environment'],'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'}
root=R/'captures';root.mkdir(exist_ok=False);cases={};started=time.monotonic()
for name,chunk,features in [('control512',512,False),('features512',512,True),('repeat512',512,True),('features1',1,True)]:
 ctl=read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'];assert not ctl.get('pause_requested') and not (R/'STOP').exists()
 out=root/name;argv=[b['executable']['path'],'--model',target['path'],'--tokens',str(R/'tokens/tokens.ids'),'--out',str(out),*b['required_backend_argv'],'--greedy-steps','8','--prefill-chunk',str(chunk),'--threads','4','--max-output-mib','32'];argv+=[] if features else ['--no-features']
 (root/(name+'-command.json')).write_text(json.dumps({'argv':argv,'environment':b['environment']},indent=2))
 with (root/(name+'.log')).open('w') as log:subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 d=read(out/'capture.json');assert d['status']=='native_capture_complete' and d['logits_shape']==[9,248320] and d['feature_shape']==[48,2560]
 logits=np.fromfile(out/'logits.f32',dtype='<f4').reshape(d['logits_shape']);assert np.isfinite(logits).all() and logits.argmax(-1).tolist()==d['greedy_ids'];assert d['input_tokens'][40:]==d['greedy_ids'][:8]
 cuda=sorted({line.split()[-1] for line in (out/'loaded-library-maps.txt').read_text().splitlines() if 'libggml-cuda.so' in line});assert cuda==[baseline['cuda']['path']],cuda
 cases[name]={'metadata':pin(out/'capture.json'),'greedy_ids':d['greedy_ids'],'server_reference_match':d['greedy_ids']==reference,'files':[pin(p) for p in out.glob('*.f32')]}
 print(json.dumps({'case':name,'server_reference_match':d['greedy_ids']==reference}),flush=True)
control=cases['control512']['greedy_ids'];assert cases['features512']['greedy_ids']==control and cases['repeat512']['greedy_ids']==control
numeric={}
for tap in [2,16,29]:
 a=np.fromfile(root/'features512'/('layer-%d.f32'%tap),dtype='<f4').reshape(48,2560);r=np.fromfile(root/'repeat512'/('layer-%d.f32'%tap),dtype='<f4').reshape(48,2560);c=np.fromfile(root/'features1'/('layer-%d.f32'%tap),dtype='<f4').reshape(48,2560);assert np.isfinite(a).all() and np.isfinite(c).all() and np.array_equal(a,r)
 diff=np.asarray(a,dtype=np.float64)-c;rrms=float(np.linalg.norm(diff)/np.linalg.norm(a));bad=float(np.linalg.norm(a-a[::-1])/np.linalg.norm(a));numeric[str(tap)]={'repeat_byteexact':True,'chunk_max_abs':float(np.abs(diff).max()),'chunk_relative_rms':rrms,'reversed_row_relative_rms':bad,'known_wrong_row_order_rejected':bad>.005}
 assert bad>.005
result={'scope':'bounded native teacher capture on one existing development prompt, Qwen3.5-4B Q4_K_M only; not trained W1A8 or broad correctness/quality qualification','passed_matched_history_gate':all(cases[k]['server_reference_match'] for k in ['control512','features512','repeat512']),'chunk1_matches_reference':cases['features1']['server_reference_match'],'cases':cases,'numeric':numeric,'source':pin(__file__),'build':pin(bp),'tokens':pin(R/'tokens/tokens.ids'),'target':target,'wall_seconds':time.monotonic()-started,'training_updates':0}
assert result['passed_matched_history_gate']
(R/'probe-result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['cases','numeric','source','target','tokens','build']}),flush=True)
