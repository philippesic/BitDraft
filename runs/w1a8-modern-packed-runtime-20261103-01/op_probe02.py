from pathlib import Path
import hashlib,json,os,subprocess,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-packed-runtime-20261103-01';B=R/'build02/bin'
def pin(path):
 p=Path(path).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
control=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080'];assert not control.get('pause_requested') and not (R/'STOP').exists()
libraries=[pin(B/name) for name in ['libggml-base.so.0.28.0','libggml-cpu.so.0.28.0','libggml-cuda.so.0.28.0']]
clean={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','DSPARK_','DFLASH_','EAGLE_','W1AX_','LLAMA_ARG_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','NVIDIA_TF32_OVERRIDE']};env={**clean,'LD_LIBRARY_PATH':str(B),'GGML_CUDA_DECODE_CONSISTENT':'32','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'2'}
freeze={'source':pin(__file__),'fixture_source':pin(R/'op_check.cpp'),'executable':pin(R/'op_check'),'libraries':libraries,'runtime_patch':pin(R/'packed-w1a8-v3.patch'),'scope':'Real CUDA fixed W1A8 operation qualification only; no model/serving/performance or training claim'}
(R/'op02-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');started=time.monotonic()
with (R/'op02-stdout.jsonl').open('x') as output,(R/'op02-stderr.log').open('x') as error:
 subprocess.run([str(R/'op_check'),str(R/'op02-library-maps.txt')],env=env,stdout=output,stderr=error,check=True,timeout=180)
lines=[json.loads(line) for line in (R/'op02-stdout.jsonl').read_text().splitlines()];assert len(lines)==9 and lines[-1]['status']=='CPU_CUDA_FIXED_W1A8_OP_PASS'
assert all(x['codes_scales_products_byteexact'] and x['admission_negatives_pass'] for x in lines[:-1]);assert sum(x['backend']=='CUDA' for x in lines[:-1])==4
mapped=sorted({line.split()[-1] for line in (R/'op02-library-maps.txt').read_text().splitlines() if 'libggml' in line});assert mapped==sorted(x['path'] for x in libraries),mapped
for ref in libraries+[freeze['executable']]:assert pin(ref['path'])==ref
result={'status':'CPU_CUDA_FIXED_W1A8_OP_PASS','cases':lines[:-1],'scope':freeze['scope'],'wall_seconds':time.monotonic()-started,'native_model_admitted':False,'training_updates':0,'library_maps':pin(R/'op02-library-maps.txt')}
(R/'op02-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':8,'wall_seconds':result['wall_seconds']}),flush=True)
