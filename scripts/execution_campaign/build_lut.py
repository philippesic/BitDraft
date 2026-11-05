from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F,pin
from scripts.execution_campaign.extend import relink
R=P/'runs/execution-campaign-20261105-01';r=build(Path(sys.argv[1]),copy=True,bound=True,threads=256)
cu=(R/'runtime-tiled/w1a1.cu').read_text();header=P/'runs/w1a8-algorithms-20261020/residual30-native03/ggml/src/ggml-cuda/w1a8_algorithms.cuh';s=header.read_text();a=s.index('static __global__ void w1a8_make_lut4');b=s.index('__device__ __forceinline__ uint32_t spread_u8');code=s[a:b].replace('float * output, int32_t * raw_dots)', 'float * output, int32_t * raw_dots, const float*mid,const float*affine_sums)').replace('output[token*m+row] = weighted * act_scales[token];','output[token*m+row] = exec_affine(sum,weight_scales[row],act_scales[token],mid?mid[row]:0,affine_sums[token]);')
cu=cu.replace('// Four output rows per block',code+'\n// Four output rows per block')
needle='    w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>('
cu=cu.replace(needle,'''    if (m==9216 && k==2560 && n==3) {
        ggml_cuda_pool_alloc<int16_t> table(ctx.pool(),n*k*4);
        w1a8_make_lut4<<<dim3((k/4*16+255)/256,n),256,0,ctx.stream()>>>((const int8_t*)packed,k,k/4,table.get());
        w1a8_dot_lut4<3><<<(m+3)/4,dim3(32,4),0,ctx.stream()>>>((const uint32_t*)dst->src[0]->data,(const float*)dst->src[1]->data,table.get(),(const float*)(packed+layout.scale_offset),m,k,k/32,n,(float*)dst->data,nullptr,dst->src[4]?(const float*)dst->src[4]->data:nullptr,(const float*)(packed+layout.scale_offset+n));
    } else w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>(''')
relink(r,{'w1a1.cu':cu})
d=json.load(open(r/'build.json'));d['algorithm_ancestry']=pin(header);(r/'build.json').write_text(json.dumps(d,indent=2)+'\n')
