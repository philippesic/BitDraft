from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-actual-embedding-20261103-02'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':d}
b=read(P/'runs/w1a8-modern-capture-build-20261103-09/build.json');baseline=read(P/'experiments/w1a8-standing-20261102/baseline.json');target=baseline['rows'][0]['target_pin'];assert pin(target['path'])==target;assert not read(Path.home()/'.config/binary-eagle-decoding/gpu-control.json')['rtx5080'].get('pause_requested');tokens=[int(x) for x in (R/'tokens.ids').read_text().split()];assert len(tokens)==len(set(tokens))==113
for ref in [b['source'],b['builder'],b['executable']]+b['runtime_libraries']+[b['cuda_overlay']]:assert pin(ref['path'])==ref
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};env={**clean,**b['environment'],'CUDA_VISIBLE_DEVICES':'0'}
argv=[b['executable']['path'],'--model',target['path'],'--tokens',str(R/'tokens.ids'),'--out',str(R/'capture'),*b['required_backend_argv'],'--taps','0','--logit-positions',str(len(tokens)-1),'--prefill-chunk','512','--decode-chunk','1','--threads','4','--output-cap','1','--max-output-mib','16'];(R/'freeze.json').write_text(json.dumps({'source':pin(__file__),'tokens':pin(R/'tokens.ids'),'target':target,'build':pin(P/'runs/w1a8-modern-capture-build-20261103-09/build.json'),'argv':argv,'environment':b['environment'],'scope':'Native target layer0 get_rows for all69 observed actual EAGLE input IDs; logits are diagnostic only, no teacher/admission/training claim'},indent=2)+'\n');t=time.monotonic()
with (R/'stdout.log').open('x') as out,(R/'stderr.log').open('x') as err:subprocess.run(argv,env=env,stdout=out,stderr=err,check=True,timeout=120)
m=read(R/'capture/capture.json');assert m['feature_shape']==[113,2560] and m['taps']==[0] and m['input_tokens']==tokens
(R/'result.json').write_text(json.dumps({'status':'NATIVE_EMBEDDING_ROWS_CAPTURED','wall_seconds':time.monotonic()-t,'capture':pin(R/'capture/capture.json'),'layer0':pin(R/'capture/layer-0.f32'),'unique_ids':113},indent=2)+'\n');print('Native F32 embeddings captured for113actual EAGLE input IDs',flush=True)
