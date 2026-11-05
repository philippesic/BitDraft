from pathlib import Path
import sys,os,json,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,pin
R=P/'runs/execution-campaign-20261105-01';r=R/'joint-overflow-checks';r.mkdir(exist_ok=False);S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';s=(R/'edge-checks03/edge_check.cpp').read_text()
s=s.replace('ggml_tensor * alpha =','ggml_tensor * w2=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,k/32,m);\n    ggml_tensor * alpha =')
s=s.replace('require(ggml_backend_supports_op(backend, pack)', '''ggml_tensor*out2=ggml_w1ax_mul_mat_shared(ctx,w2,alpha,pack,k,bits);out2->src[4]=mid;
    std::vector<float> expected2(size_t(m)*n);for(int t=0;t<n;++t)for(int r=0;r<m;++r){int sum=0,qsum=0;for(int j=0;j<k;++j){int sign=(signs[size_t(r)*(k/32)+j/32]>>(j%32))&1?1:-1;sum-=sign*int(expected_codes[t*k+j]);qsum+=expected_codes[t*k+j];}float weighted=float(sum)*scales[r],middle=float(qsum)*beta[t];expected2[t*m+r]=weighted*beta[t]+midpoints[r]*middle;}
    require(ggml_backend_supports_op(backend, pack)''')
s=s.replace('ggml_build_forward_expand(graph, out);','ggml_build_forward_expand(graph, out);ggml_build_forward_expand(graph,out2);')
s=s.replace('ggml_backend_tensor_set(w, signs.data(),', 'auto signs2=signs;for(auto&v:signs2)v=~v;ggml_backend_tensor_set(w2,signs2.data(),0,signs2.size()*4);\n    ggml_backend_tensor_set(w, signs.data(),')
s=s.replace('require(std::memcmp(observed.data(), expected.data(), expected.size()*sizeof(float)) == 0, "native integer-dot products differ from independent reference");', '''std::vector<float> observed2(observed.size());ggml_backend_tensor_get(out2,observed2.data(),0,observed2.size()*4);
    auto equivalent=[](const std::vector<float>&a,const std::vector<float>&b){for(size_t i=0;i<a.size();++i)if(memcmp(&a[i],&b[i],4)!=0 && !(std::isnan(a[i])&&std::isnan(b[i])))return false;return true;};
    require(equivalent(observed,expected)&&equivalent(observed2,expected2),"joint affine products differ from independent integer reference");
    if(const char*p=getenv("EXEC_DUMP_BINARY")){std::ofstream f(p,std::ios::binary|std::ios::app);f.write((char*)packed_words.data(),packed_words.size()*4);f.write((char*)observed.data(),observed.size()*4);f.write((char*)observed2.data(),observed2.size()*4);}''')
src=r/'joint_check.cpp';src.write_text(s);exe=r/'joint_check';cmd=['c++','-O2','-std=c++17','-ffp-contract=off','-I'+str(S/'ggml/include'),str(src),'-L'+str(F/'bin'),'-lggml','-lggml-base','-lggml-cpu','-lggml-cuda','-Wl,-rpath,'+str(F/'bin'),'-o',str(exe)];subprocess.run(cmd,check=True);checks=[]
for variant in ['original','optimized']:
 runtime=F if variant=='original' else R/'runtime-combined-last02';env={**os.environ,'LD_LIBRARY_PATH':str(runtime/'bin'),'GGML_CUDA_DECODE_CONSISTENT':'32'}
 for n in [1,3,8,9]:
  out=r/f'{variant}-N{n}.log'
  with out.open('w') as log:subprocess.run([str(exe),str(r/f'{variant}-N{n}.maps'),'fused',str(n)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  checks.append(dict(variant=variant,n=n,all_bits_modes_joint_affine_pass=True,joint_dispatch_count=out.read_text().count('EXEC_JOINT_FUSION')));print(checks[-1],flush=True)
 # Max finite clip can overflow the original affine output: compare GPU bytes, including NaN payloads, directly.
 out=r/(variant+'-maxclip.log');dump=r/(variant+'-maxclip.bin')
 with out.open('w') as log:subprocess.run([str(exe),str(r/(variant+'-maxclip.maps')),'fused','3'],env={**env,'EXEC_TEST_CLIP':'3.402823466e38','EXEC_DUMP_BINARY':str(dump)},stdout=log,stderr=subprocess.STDOUT,check=True)
assert (r/'original-maxclip.bin').read_bytes()==(r/'optimized-maxclip.bin').read_bytes(),'overflow/NaN original GPU bytes changed'
(r/'result.json').write_text(json.dumps(dict(checks=checks,max_finite_clip_gpu_outputs_byteexact=True,source=pin(src),command=cmd,note='Independent integer CPU reference; finite output bytes exact. CPU/GPU NaN payloads are not equated; original/optimized actual GPU payloads explicitly byte-compared.'),indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign/final_replay.py')],check=True)
