from pathlib import Path
import sys,subprocess,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
R=P/'runs/execution-campaign-20261105-01';r=R/'edge-checks';r.mkdir(exist_ok=True);S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
s=(ROOT/'scripts/dflash_baseline/precision_op_check.cpp').read_text().replace('#include <algorithm>','#include <algorithm>\n#include <cstdlib>')
s=s.replace('const float clip=1.25f;', 'const float clip=std::getenv("EXEC_TEST_CLIP")?std::strtof(std::getenv("EXEC_TEST_CLIP"),nullptr):1.25f;')
s=s.replace('0.0f : std::sin(float(j)*.137f)', '(j%2?-0.0f:0.0f) : std::sin(float(j)*.137f)')
s=s.replace('std::clamp(int(std::nearbyint(normalized)), -qmax, qmax)', 'int(std::nearbyint(std::clamp(normalized,-float(qmax),float(qmax))))')
s=s.replace('const auto raw=acts;', '''if(const char*bad=std::getenv("EXEC_BAD")){std::string kind=bad;if(kind=="nan")acts[0]=std::numeric_limits<float>::quiet_NaN();if(kind=="inf")acts[0]=std::numeric_limits<float>::infinity();if(kind=="overflow"){acts[0]=std::numeric_limits<float>::max();gains[0]=2;}if(kind=="gainzero")gains[0]=0;if(kind=="gainnan")gains[0]=std::numeric_limits<float>::quiet_NaN();}
    const auto raw=acts;''')
s=s.replace('ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);', '''const bool strided=std::getenv("EXEC_STRIDED");
    ggml_tensor * storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k+(strided?32:0),n);
    ggml_tensor * a=strided?ggml_view_2d(ctx,storage,k,n,(k+32)*4,0):storage;''')
s=s.replace('ggml_backend_tensor_set(a, raw.data(), 0, raw.size()*sizeof(float));','''if(strided){std::vector<float> padded(size_t(k+32)*n,123.0f);for(int t=0;t<n;++t)std::copy_n(raw.data()+t*k,k,padded.data()+t*(k+32));ggml_backend_tensor_set(storage,padded.data(),0,padded.size()*4);}else ggml_backend_tensor_set(a,raw.data(),0,raw.size()*4);''')
s=s.replace('size_t(k)*n/4 + n);','size_t(k)*n/4 + 2*n);')
s=s.replace('require(std::memcmp(observed.data()', '''for(int t=0;t<n;++t){int sum=0;for(int j=0;j<k;++j)sum+=expected_codes[t*k+j];float value=float(sum)*beta[t];require(std::memcmp(packed_words.data()+size_t(k)*n/4+n+t,&value,4)==0,"scaled code sum differs");}
    require(std::memcmp(observed.data()''')
src=r/'edge_check.cpp';src.write_text(s);exe=r/'edge_check';cmd=['c++','-O2','-std=c++17','-ffp-contract=off','-I'+str(S/'ggml/include'),str(src),'-L'+str(F/'bin'),'-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-Wl,-rpath,'+str(F/'bin'),'-o',str(exe)];subprocess.run(cmd,check=True)
receipts=[]
for variant in ['both256','tiled','cross256','mma256','narrow']:
 runtime=R/('runtime-'+variant);env={**os.environ,'LD_LIBRARY_PATH':str(runtime/'bin'),'GGML_CUDA_DECODE_CONSISTENT':'32'}
 for label,options,n in [('base',{},3),('strided',{'EXEC_STRIDED':'1'},3),('tinyclip',{'EXEC_TEST_CLIP':'1e-40'},3),('hugeclip',{'EXEC_TEST_CLIP':'3.402823466e38'},3)]:
  with (r/(variant+'-'+label+'.jsonl')).open('w') as log:result=subprocess.run([str(exe),str(r/(variant+'-'+label+'.maps')),'fused',str(n)],env={**env,**options},stdout=log,stderr=subprocess.STDOUT)
  receipts.append(dict(variant=variant,fixture=label,returncode=result.returncode,expected_success=True));print('EDGE',receipts[-1],flush=True)
  if result.returncode:raise RuntimeError('edge fixture failed '+str(receipts[-1]))
 for kind in ['nan','inf','overflow','gainzero','gainnan']:
  with (r/(variant+'-reject-'+kind+'.log')).open('w') as log:result=subprocess.run([str(exe),str(r/'bad.maps'),'fused','3'],env={**env,'EXEC_BAD':kind},stdout=log,stderr=subprocess.STDOUT)
  log=(r/(variant+'-reject-'+kind+'.log')).read_text();trapped=result.returncode!=0 and any(x in log.lower() for x in ['illegal instruction','device-side assert','unspecified launch failure'])
  receipts.append(dict(variant=variant,fixture=kind,returncode=result.returncode,expected_rejection=trapped));print('REJECT',receipts[-1],flush=True)
  if not trapped:raise RuntimeError('nonfinite was not rejected '+kind)
 (r/'result.json').write_text(json.dumps(dict(source=pin(src),command=cmd,checks=receipts),indent=2)+'\n')
