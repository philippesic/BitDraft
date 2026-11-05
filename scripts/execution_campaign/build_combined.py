from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,pin
from scripts.execution_campaign.extend import relink
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--head',action='store_true');ap.add_argument('--last-ffn',action='store_true');a=ap.parse_args();R=P/'runs/execution-campaign-20261105-01';r=build(a.out,copy=True,bound=True,threads=256)
cu=(R/'runtime-tiled/w1a1.cu').read_text().replace('// Four output rows per block',(ROOT/'scripts/execution_campaign/joint.cuh').read_text()+'\n// Four output rows per block');cu+='\n'+(ROOT/'scripts/execution_campaign/joint_host.cuh').read_text();backend=(R/'runtime-joint/ggml/src/ggml-cuda/ggml-cuda.cu').read_text();changes={'w1a1.cu':cu,'ggml/src/ggml-cuda/ggml-cuda.cu':backend}
cpp=(R/('runtime-lastffn02' if a.last_ffn else 'runtime-head02')/'src/models/dflash.cpp').read_text() if (a.head or a.last_ffn) else (r/'dflash.cpp').read_text()
changes['src/models/dflash.cpp']=cpp.replace('ggml_mul(ctx0,ggml_silu(ctx0,gate),up)','ggml_mul(ctx0,up,ggml_silu(ctx0,gate))')
relink(r,changes)
if a.head or a.last_ffn:
 for name in ['libllama-common.so.0.8.0','libllama-common.so','libllama-common.so.0']:
  f=r/'bin'/name;f.unlink();f.symlink_to(R/'runtime-head02/bin/libllama-common.so.0.8.0')
 d=json.load(open(r/'build.json'));d['common_overlay_dependency']=pin(R/'runtime-head02/build.json');d['common_library']=pin(r/'bin/libllama-common.so.0.8.0');(r/'build.json').write_text(json.dumps(d,indent=2)+'\n')
