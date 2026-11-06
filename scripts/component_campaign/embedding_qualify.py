"""Full-vocabulary native CPU/CUDA lookups on mask/special/common/rare IDs."""
import argparse,json,os,subprocess
from collections import Counter
import numpy as np
from .common import *
from scripts.dflash_baseline.model import GGUFReader,dequantize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--format',choices=['Q6_K','Q4_K','Q3_K','Q5_K']);a=ap.parse_args()
 extra=a.format if a.format not in (None,'Q6_K','Q4_K') else None
 formats=[extra] if extra else ['Q6_K','Q4_K']
 out=R/'qualification'/('embedding-lookup-'+extra if extra else 'embedding-lookup')
 if (out/'result.json').exists():return
 out.mkdir(exist_ok=False);source=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';runtime=R/'runtime03';exe=out/'embedding-fixture'
 command=['c++','-O2','-std=c++17','-I'+str(source/'ggml/include'),'-I'+str(source/'ggml/src'),str(ROOT/'scripts/component_campaign/embedding_fixture.cpp'),'-L'+str(runtime/'bin'),'-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-Wl,-rpath,'+str(runtime/'bin'),'-o',str(exe)]
 subprocess.run(command,check=True)
 metadata=json.load(open(R/'freeze.json'));source_model=GGUFReader(metadata['B0_config']['source']['path']);special=[]
 for key,f in source_model.fields.items():
  if key.startswith('tokenizer.ggml.') and key.endswith('_token_id'):
   value=f.contents()
   if isinstance(value,(int,np.integer)):special.append(int(value))
 data=json.load(open(metadata['B0_config']['data']['path']));counts=Counter()
 for record in data['records']:
  if record['split']=='TRAIN':counts.update(json.load(open(record['metadata']['path']))['input_tokens'])
 ids=sorted(set([0,1,100001,200001,248319,*special,*[k for k,_ in counts.most_common(6)],*sorted([k for k,v in counts.items() if v==1])[:6]]));np.array(ids,dtype='<i4').tofile(out/'ids.i32')
 if any(x<0 or x>=248320 for x in ids):raise ValueError('invalid selected vocabulary ID')
 records=[]
 for format in formats:
  basis=R/'bases'/(format+'.gguf');tensor=next(t for t in GGUFReader(basis).tensors if t.name=='output.weight');reference=dequantize(tensor.data[ids],tensor.tensor_type).reshape(len(ids),2560)
  if tuple(map(int,tensor.shape))!=(2560,248320):raise ValueError('vocabulary capacity changed')
  for backend in ('cpu','cuda'):
   dest=out/(format+'-'+backend);argv=[str(exe),backend,str(int(tensor.tensor_type)),str(basis),str(tensor.data_offset),'248320','2560',str(out/'ids.i32'),str(len(ids)),str(dest)+'.f32',str(dest)+'.json']
   subprocess.run(argv,env=dict(os.environ,LD_LIBRARY_PATH=str(runtime/'bin'),GGML_CUDA_DECODE_CONSISTENT='32'),check=True)
   observed=np.fromfile(str(dest)+'.f32',dtype='<f4').reshape(reference.shape);error=float(np.abs(observed-reference).max())
   if error>2e-7 or not np.isfinite(observed).all():raise ValueError('native embedding lookup differs '+format+' '+backend)
   records.append(dict(format=format,backend=backend,max_absolute_error=error,allocation=json.load(open(str(dest)+'.json')),output=pin(str(dest)+'.f32')))
 save(out/'result.json',dict(status='FULL_VOCAB_CPU_CUDA_LOOKUP_PASS',selected_ids=ids,roles='mask/special/common TRAIN/rare TRAIN/unseen-high/final vocabulary row',target_immutable=True,Q6_exact_codes=True if 'Q6_K' in formats else None,records=records,source=pin(__file__),fixture=pin(ROOT/'scripts/component_campaign/embedding_fixture.cpp')))
 print('EMBEDDING_LOOKUP_QUALIFICATION_PASS',flush=True)

if __name__=='__main__':main()
