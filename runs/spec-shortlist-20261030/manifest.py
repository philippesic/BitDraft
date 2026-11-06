from pathlib import Path
import sys,json,hashlib,subprocess,collections,shutil,time,os
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;sys.path.insert(0,str(P/'runtime/gguf-py'))
from gguf import GGUFReader
v={'at':time.time(),'free_bytes':{'wsl':shutil.disk_usage(P).free,'windows_backing_volume':shutil.disk_usage('/mnt/c').free},'models':[],'environment':{}}
for name,argv in {'gpu':['nvidia-smi','-q'],'compiler':['c++','--version'],'cuda':['/usr/local/cuda-13.1/bin/nvcc','--version'],'remote_revision':['git','rev-parse','HEAD'],'remote_dirty':['git','diff','--stat'],'native_revision':['git','-C',str(P/'runtime'),'rev-parse','HEAD']}.items():
 r=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True);v['environment'][name]=r.stdout
for f in [P/'build/CMakeCache.txt',P/'build/bin/llama-server',ROOT/'scripts/remote_job.py',ROOT/'results/cuda-glibc-compat/include/cuda_fp4.hpp']:
 if f.exists():v['environment'][str(f)]={'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest(),'bytes':f.stat().st_size}
for f in sorted((P/'models').glob('*.gguf')):
 try:
  r=GGUFReader(str(f));hist=collections.Counter();bybytes=collections.Counter();tensors=[]
  for t in r.tensors:
   typ=t.tensor_type.name;hist[typ]+=1;bybytes[typ]+=t.n_bytes;tensors.append({'name':t.name,'type':typ,'shape':t.shape.tolist()})
  def val(field):
   try:return field.contents()
   except:return str(field)
  m={'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest(),'tensor_types':dict(hist),'bytes_by_type':dict(bybytes),'tensors':tensors,'metadata':{k:val(x) for k,x in r.fields.items() if any(s in k for s in ['architecture','file_type','nextn','target','block_count','draft','head_count','hidden_state','tokenizer.chat_template'])}}
  v['models'].append(m)
 except Exception as e:v['models'].append({'path':str(f),'error':repr(e)})
(P/'environment-model-manifest.json').write_text(json.dumps(v,indent=2,default=str));print(json.dumps({'models':len(v['models']),'free':v['free_bytes']}))
