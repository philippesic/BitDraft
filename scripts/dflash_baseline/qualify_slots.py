"""Close exact16-slot gained-stream qualification after the formal quiet block."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.precision_benchmark import pin,save,pause
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-04';base=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
def main():
 if not (R/'confirmation-dev12/result.json').exists():raise ValueError('respect formal measurement quiet window')
 pause();out=R/'slot16-qualification';out.mkdir(exist_ok=False);binary=out/'op-check';source=ROOT/'scripts/dflash_baseline/precision_op_check.cpp'
 command=['c++','-std=c++17','-O2','-ffp-contract=off',str(source),'-I'+str(runtime/'ggml/include'),'-I'+str(base/'ggml/include'),'-L'+str(runtime/'bin'),'-Wl,-rpath,'+str(runtime/'bin'),'-Wl,-rpath-link,'+str(runtime/'bin'),'-l:libggml.so.0.28.0','-l:libggml-base.so.0.28.0','-l:libggml-cuda.so.0.28.0','-l:libggml-cpu.so.0.28.0','-o',str(binary)]
 subprocess.run(command,check=True,timeout=120);save(out/'build.json',dict(command=command,source=pin(source),binary=pin(binary),header=pin(runtime/'ggml/include/ggml.h'),runtime=pin(runtime/'build.json'),compiler=subprocess.check_output(['c++','--version'],text=True),scope='Additional operation-only test binary; production runtime and all measured models unchanged.'))
 pause()
 with (out/'ops.jsonl').open('w') as stdout,(out/'stderr.log').open('w') as stderr:subprocess.run([str(binary),str(out/'library-maps.txt'),'fused','16'],env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'bin')),stdout=stdout,stderr=stderr,check=True,timeout=300)
 rows=[json.loads(x) for x in (out/'ops.jsonl').read_text().splitlines() if x.lstrip().startswith('{')];cases=[x for x in rows if 'N' in x]
 if len(cases)!=36 or not all(x['N']==16 and x['backend']=='CUDA' and x['gained_codes_scales_products_byteexact'] and x['admission_negatives_pass'] for x in cases):raise ValueError('full16-slot CUDA qualification failed')
 expected={(k,m,bits,mode,16) for k,m in ((32,7),(2560,9216),(9216,2560)) for bits in (1,2,4,8) for mode in (0,1,2)}
 if {(x['K'],x['M'],x['bits'],x['mode'],x['N']) for x in cases}!=expected or rows[-1].get('status')!='CUDA_AFFINE_PRECISION_OP_PASS':raise ValueError('incomplete16-slot case coverage')
 save(out/'result.json',dict(status='GAINED_STREAM_CUDA36_N16_BYTEEXACT',cases=cases,raw=pin(out/'ops.jsonl'),build=pin(out/'build.json'),maps=pin(out/'library-maps.txt'),scope='All four logical widths and three bound modes; zero/subnormal/RNE-tie inputs; nonzero row midpoints; actual gate/up/down geometries at16 slots. Not a kernel speed benchmark.'))
if __name__=='__main__':main()
