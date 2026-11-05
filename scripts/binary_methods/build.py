from pathlib import Path
import json,subprocess,os,hashlib,difflib,sys
P=Path('/home/philip/binary-eagle-decoding');C=Path(__file__).resolve().parent;R=P/'runs/binary-methods-20261104-01';B=R/('native-lut' if '--sharedlut' in sys.argv else 'native');S=P/'runs/w1a8-algorithms-20261020/residual30-native03';OLD=P/'runs/w1a8-algorithms-20261020/input-gains-build01';BASE=P/'runs/shortlist-throughput-20261027/native/build/bin'
def read(p):return json.loads(p.read_text())
def pin(p):return dict(path=str(p),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
B.mkdir(exist_ok=True);(B/'bin').mkdir(exist_ok=True)
for p in BASE.iterdir():
 if p.is_file() and not p.name.startswith('libggml-cuda') and not (B/'bin'/p.name).exists():(B/'bin'/p.name).symlink_to(p.resolve())
s=(S/'ggml/src/ggml-cuda/w1a1.cu').read_text();original=s
s=s.replace('#include "w1a1.cuh"','#include "w1a1.cuh"\n#include "bmma.cuh"\n#include "lutgemm_exact.cuh"\n__global__ void screen_unpack_a1(const uint32_t*p,int8_t*q,int k,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<k*n)q[i]=((p[i/32]>>(i%32))&1)?1:-1;}')
for i in [0,2]:s=s.replace(f'const int bits = ggml_get_op_params_i32(dst, {i});',f'const int bits = getenv("GGML_SCREEN_A1") ? 1 : ggml_get_op_params_i32(dst, {i});')
a='    w1a1_xor_popc<<<dim3((unsigned) ((m - 1)/4 + 1), token_blocks), dim3(32, 4), 0, stream>>>('
b='''    const char * screen = getenv("GGML_SCREEN_A1_METHOD");
    if (screen && strcmp(screen,"bmma")==0 && !midpoints) {
        w1a1_binary_mma<<<dim3((unsigned)((m+7)/8),(unsigned)((n+7)/8)),32,0,stream>>>((const uint32_t*)weights->data,(const float*)scales->data,packed_ptr,scale_ptr,m,n,k,words,(float*)dst->data);
    } else if (screen && strcmp(screen,"tensor")==0 && n==7 && k%128==0 && !midpoints) {
        ggml_cuda_pool_alloc<int8_t> q(ctx.pool(),n*k);
        screen_unpack_a1<<<(unsigned)((n*k+255)/256),256,0,stream>>>(packed_ptr,q.ptr,k,n);
        imma_expanded<4,false><<<dim3((unsigned)((m+15)/16)),128,0,stream>>>((const uint32_t*)weights->data,(const float*)scales->data,q.ptr,nullptr,scale_ptr,m,k,words,(float*)dst->data,nullptr);
    } else
'''+a
assert s.count(a)==1;s=s.replace(a,b)

if '--sharedlut' in sys.argv:
 needle='            if ((use_planes || use_lut || use_mma) && bits == 8'
 insert='            if (algorithm && strcmp(algorithm,"sharedlut")==0 && bits==8 && k%128==0 && m%2==0 && !midpoints) {\n                ggml_cuda_pool_alloc<uint32_t> wt(ctx.pool(),(size_t)m*words);\n                lut_transpose<<<(unsigned)((m*words+255)/256),256,0,stream>>>((const uint32_t*)weights->data,wt.ptr,m,words);\n                CUDA_CHECK(cudaMemsetAsync(dst->data,0,m*n*sizeof(float),stream));\n                const int kt=k<=4096?64:128, mt=k<=4096?2048:512;\n                if(kt==64)kernel::_nqmv<64><<<dim3((m+mt-1)/mt,k/64,n),256,0,stream>>>(wt.ptr,nullptr,code_ptr,(float*)dst->data,m,k,1,mt,k);\n                else kernel::_nqmv<128><<<dim3((m+mt-1)/mt,k/128,n),256,0,stream>>>(wt.ptr,nullptr,code_ptr,(float*)dst->data,m,k,1,mt,k);\n                lut_scale<<<(unsigned)((m*n+255)/256),256,0,stream>>>((float*)dst->data,(const float*)scales->data,scale_ptr,m,n);\n            } else\n'
 assert s.count(needle)==1;s=s.replace(needle,insert+needle)
 needle='    if (screen && strcmp(screen,"bmma")==0 && !midpoints) {'
 insert='    if(screen && strcmp(screen,"sharedlut")==0 && k%128==0 && m%2==0 && !midpoints){\n        ggml_cuda_pool_alloc<uint32_t> wt(ctx.pool(),(size_t)m*words);\n        ggml_cuda_pool_alloc<int8_t> q(ctx.pool(),(size_t)n*k);\n        lut_transpose<<<(unsigned)((m*words+255)/256),256,0,stream>>>((const uint32_t*)weights->data,wt.ptr,m,words);\n        screen_unpack_a1<<<(unsigned)((n*k+255)/256),256,0,stream>>>(packed_ptr,q.ptr,k,n);\n        CUDA_CHECK(cudaMemsetAsync(dst->data,0,m*n*sizeof(float),stream));\n        const int kt=k<=4096?64:128,mt=k<=4096?2048:512;\n        if(kt==64)kernel::_nqmv<64><<<dim3((m+mt-1)/mt,k/64,n),256,0,stream>>>(wt.ptr,nullptr,q.ptr,(float*)dst->data,m,k,1,mt,k);\n        else kernel::_nqmv<128><<<dim3((m+mt-1)/mt,k/128,n),256,0,stream>>>(wt.ptr,nullptr,q.ptr,(float*)dst->data,m,k,1,mt,k);\n        lut_scale<<<(unsigned)((m*n+255)/256),256,0,stream>>>((float*)dst->data,(const float*)scales->data,scale_ptr,m,n);\n    } else\n'
 assert s.count(needle)==1;s=s.replace(needle,insert+needle)
 
else:
 s=s.replace('#include \"lutgemm_exact.cuh\"\n','')
(B/'w1a1.cu').write_text(s);(C/'native.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='w1a1.cu',tofile='w1a1.cu')))
prior=read(OLD/'complete-repaired.json');cmd=next(c['argv'] for c in prior['commands'] if c.get('unit','').endswith('w1a1.cu'))
cmd=[v.replace(str(P/'runs/w1a8-algorithms-20261020/input-gains-native01'),str(S)) for v in cmd];cmd[cmd.index('-c')+1]=str(B/'w1a1.cu');cmd[cmd.index('-o')+1]=str(B/'w1a1.o');cmd+=['-I'+str(S/'ggml/src/ggml-cuda'),'-I'+str(C)]
commands=[cmd]
with (B/'compile.log').open('w') as f:subprocess.run(cmd,cwd=B,stdout=f,stderr=subprocess.STDOUT,check=True)
# Use the exact previously linked object list from the linker dependency manifest.
dep=(P/'runs/w1a8-algorithms-20261020/residual30-build03/cuda.d').read_text().split('\n\n')[0].replace('\\\n',' ').split(':',1)[1].split()
objects=[str(B/'w1a1.o') if v.endswith('/w1a1.cu.o') else v for v in dep if v.endswith('.o') and v.startswith(str(P))];assert len(objects)>50 and str(B/'w1a1.o') in objects
cmd=['/usr/bin/g++','-shared','-o',str(B/'bin/libggml-cuda.so.0.25.1'),'-Wl,-soname,libggml-cuda.so.0',*objects,'-L'+str(B/'bin'),'-lggml-base','-L/usr/local/cuda-13.1/lib64','-lcudart','-lcublas','-lcublasLt','-L/usr/lib/wsl/lib','-lcuda','-Wl,-rpath,'+str(B/'bin')+':/usr/local/cuda-13.1/lib64'];commands.append(cmd)
with (B/'link.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
for n in ['libggml-cuda.so','libggml-cuda.so.0']:
 if not (B/'bin'/n).exists():(B/'bin'/n).symlink_to('libggml-cuda.so.0.25.1')
x=subprocess.run(['ldd','-r',str(B/'bin/llama-server')],env={**os.environ,'LD_LIBRARY_PATH':str(B/'bin')},capture_output=True,text=True);(B/'ldd.txt').write_text(x.stdout+x.stderr);assert 'undefined symbol' not in x.stdout+x.stderr and 'not found' not in x.stdout+x.stderr
(B/'build.json').write_text(json.dumps(dict(commands=commands,source=pin(B/'w1a1.cu'),patch=pin(C/'native.patch'),library=pin(B/'bin/libggml-cuda.so.0.25.1'),reused_objects=[pin(Path(v)) for v in objects if v!=str(B/'w1a1.o')]),indent=2));print('BUILD_PASS')
