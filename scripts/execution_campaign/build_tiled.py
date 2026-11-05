from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P
from scripts.execution_campaign.extend import relink
r=build(Path(sys.argv[1]),copy=True,bound=True,threads=256)
cu=(r/'w1a1.cu').read_text();cu=cu.replace('// Four output rows per block',(ROOT/'scripts/execution_campaign/tiled.cuh').read_text()+'\n// Four output rows per block')
a=cu.index('        exec_known_bound<256><<<');b=cu.index('\n    } else w1ax_quantize',a)
cu=cu[:a]+'''        const int tiles=(k+255)/256;ggml_cuda_pool_alloc<int> partial(ctx.pool(),tiles*n);
        const int bits=ggml_get_op_params_i32(dst,0);const float clip=ggml_get_op_params_f32(dst,2);const float*g=dst->src[1]?(const float*)dst->src[1]->data:nullptr;
        #define TILED(B) exec_tiled_codes<B><<<dim3(tiles,n),256,0,ctx.stream()>>>((const float*)acts->data,g,k,bits,clip,codes,partial.get(),tiles)
        switch(bits){case 1:TILED(1);break;case 2:TILED(2);break;case 4:TILED(4);break;case 8:TILED(8);break;default:GGML_ABORT("unsupported width");}
        #undef TILED
        exec_tiled_finish<<<n,32,0,ctx.stream()>>>(partial.get(),tiles,bits,clip,scales,scales+n);'''+cu[b:]
relink(r,{'w1a1.cu':cu})
