from pathlib import Path
import json,os,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
R=P/'runs/execution-campaign-20261105-01';r=R/'fusion-fixtures';r.mkdir(exist_ok=True);S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';exe=r/'fusion_check'
cmd=['c++','-O2','-I'+str(S/'ggml/include'),str(ROOT/'scripts/execution_campaign/fusion_check.cpp'),'-L'+str(F/'bin'),'-lggml','-lggml-base','-lggml-cuda','-Wl,-rpath,'+str(F/'bin'),'-o',str(exe)];subprocess.run(cmd,check=True)
records=[]
for name in ['original','silu04']:
 runtime=F if name=='original' else R/('runtime-'+name);env={**os.environ,'LD_LIBRARY_PATH':str(runtime/'bin'),'GGML_CUDA_DECODE_CONSISTENT':'32'}
 with (r/(name+'.log')).open('w') as log:subprocess.run([str(exe),str(r/(name+'.bin'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 records.append(dict(variant=name,output=pin(r/(name+'.bin')),fusion_count=(r/(name+'.log')).read_text().count('EXEC_SILU_FUSION')))
a=(r/'original.bin').read_bytes();b=(r/'silu04.bin').read_bytes();records.append(dict(packed_streams_byteexact=a==b,differing_bytes=sum(x!=y for x,y in zip(a,b)),scope='12actualCUDA fixtures:A1/A2/A4/A8,N1/3/8,gains,subnormal gate values,wide SiLU inputs; compare untouched native unary/mul/pack with actual fused dispatch'))
(r/'result.json').write_text(json.dumps(records,indent=2)+'\n');print(records[-1],flush=True)
