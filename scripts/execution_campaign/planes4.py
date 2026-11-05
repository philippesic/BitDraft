from pathlib import Path
import json,subprocess,sys,os,statistics
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F,pin
from scripts.execution_campaign.extend import relink
from scripts.dflash_baseline.native_archive import hydrate,release
R=P/'runs/execution-campaign-20261105-01';r=build(R/'runtime-planes4',copy=True,bound=True,threads=256)
old=P/'runs/w1a8-algorithms-20261020/residual30-native03/ggml/src/ggml-cuda/w1a8_algorithms.cuh';text=old.read_text();code=text[text.index('static __global__ void w1a8_make_planes'):text.index('// Sixteen exact signed sums')];code=code.replace('w1a8_make_planes','exec_a4_make_planes').replace('w1a8_dot_planes','exec_a4_dot_planes').replace('bit < 8','bit < 4').replace('*8+bit','*4+bit').replace('*8;','*4;').replace('bit < 7','bit < 3').replace('128*__popc(w & p[7])','8*__popc(w & p[3])').replace('float * output, int32_t * raw_dots)', 'float * output, int32_t * raw_dots,const float*mid,const float*affine_sums)').replace('output[token*m+row] = weighted * act_scales[token];','output[token*m+row]=exec_affine(sum,weight_scales[row],act_scales[token],mid?mid[row]:0,affine_sums[token]);')
cu=(R/'runtime-tiled/w1a1.cu').read_text().replace('// Four output rows per block',code+'\n// Four output rows per block');needle='    w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>('
cu=cu.replace(needle,'''    if(m==9216 && k==2560 && n==3 && ggml_get_op_params_i32(dst,2)==4){
        ggml_cuda_pool_alloc<uint32_t> planes(ctx.pool(),n*(k/32)*4);ggml_cuda_pool_alloc<int32_t> raw(ctx.pool(),n*(k/32));
        exec_a4_make_planes<<<dim3((k/32+7)/8,n),256,0,ctx.stream()>>>((const int8_t*)packed,k,k/32,planes.get(),raw.get());
        exec_a4_dot_planes<3><<<(m+3)/4,dim3(32,4),0,ctx.stream()>>>((const uint32_t*)dst->src[0]->data,(const float*)dst->src[1]->data,planes.get(),raw.get(),(const float*)(packed+layout.scale_offset),m,k,k/32,n,(float*)dst->data,nullptr,dst->src[4]?(const float*)dst->src[4]->data:nullptr,(const float*)(packed+layout.scale_offset+n));
    } else w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>(''')
cpp=(R/'runtime-combined-last02/src/models/dflash.cpp').read_text();relink(r,{'w1a1.cu':cu,'src/models/dflash.cpp':cpp})
for name in ['libllama-common.so.0.8.0','libllama-common.so','libllama-common.so.0']:
 f=r/'bin'/name;f.unlink();f.symlink_to(R/'runtime-head02/bin/libllama-common.so.0.8.0')
(r/'algorithm-ancestry.json').write_text(json.dumps(dict(source=pin(old),semantics='A4 low4two-complement planes, raw integer word sum; no F32 sum division; extra plane conversion and midpoint epilogue charged'),indent=2)+'\n')
env={**os.environ,'LD_LIBRARY_PATH':str(r/'bin'),'GGML_CUDA_DECODE_CONSISTENT':'32'}
with (r/'ops-N3.log').open('w') as log:subprocess.run([str(R/'runtime-both32/op-check'),str(r/'ops.maps'),'fused','3'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
arms=json.load(open(R/'arms.json'));created=hydrate(arms['w1']);capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture';anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5;env.update(LD_PRELOAD=str(R/'libgraph-timer.so'),EXEC_REQUEST_MASKS='1')
try:
 with (r/'replay.log').open('w') as log:subprocess.run([str(R/'replay-v6'),str(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'),arms['w1'],str(capture),str(anchor),'7',str(r/'replay.json'),'0'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
finally:release(arms['w1'],created)
a=json.load(open(R/'replay-tranche6/original-clean.json'))['rows'];b=json.load(open(r/'replay.json'))['rows'];assert all(x['outputs'][1:]==y['outputs'] and x['cache_hash_fnv1a64']==y['cache_hash_fnv1a64'] for x,y in zip(a,b));rs=[x for x in b if not x['warmup']];print('A4_PLANES_NATIVE_PASS',statistics.median(x['graph_event_ms'] for x in rs),flush=True)
