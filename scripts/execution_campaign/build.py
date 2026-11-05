"""Isolated execution overlay: frozen runtime04 objects, drafter-only changes."""
import argparse,difflib,hashlib,json,subprocess,sys
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');F=P/'runs/precision-native-20261104-04';B=P/'runs/w1a8-modern-packed-runtime-20261103-01/build02'
ROOT=Path(__file__).resolve().parents[2]
def pin(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
def build(r,copy=False,bound=False,dot='original',threads=256,extra=()):
 r.mkdir(exist_ok=False);(r/'bin').mkdir();(r/'ggml-impl.h').symlink_to(F/'ggml/src/ggml-impl.h');manifest=json.load(open(F/'build.json'));commands=[];objects={}
 sources={'w1a1.cu':F/'ggml/src/ggml-cuda/w1a1.cu','dflash.cpp':F/'src/models/dflash.cpp'}
 for name,src in sources.items():
  s=src.read_text();original=s
  if name=='dflash.cpp' and copy:s=s.replace('ggml_w1ax_pack(ctx0,ggml_cont(ctx0,input),bits)','ggml_w1ax_pack(ctx0,input,bits)')
  if name=='w1a1.cu':
   s=s.replace('// Four output rows per block', (ROOT/'scripts/execution_campaign/kernels.cuh').read_text()+'\n// Four output rows per block')
   if bound:
    needle='    w1ax_quantize<<<dim3((unsigned) n), 256, 0, ctx.stream()>>>('
    replacement=f'''    if (ggml_get_op_params_i32(dst,1)) {{
        exec_known_bound<{threads}><<<dim3((unsigned)n), {threads}, 0, ctx.stream()>>>((const float*)acts->data,k,codes,scales,scales+n,ggml_get_op_params_i32(dst,0),ggml_get_op_params_f32(dst,2),dst->src[1]?(const float*)dst->src[1]->data:nullptr);
    }} else w1ax_quantize<<<dim3((unsigned) n), 256, 0, ctx.stream()>>>('''
    s=s.replace(needle,replacement)
   if dot!='original':
    needle='    w1ax_integer_dot<<<dim3((unsigned) ((m+3)/4), (unsigned) n), dim3(32, 4), 0, ctx.stream()>>>('
    if dot=='cross':replacement='    if (n<=4) exec_cross_dot<4><<<dim3((unsigned)((m+3)/4)),dim3(32,4),0,ctx.stream()>>>('
    elif dot=='mma':replacement='    if (n<=8 && k%32==0) exec_mma_dot<<<dim3((unsigned)((m+63)/64)),dim3(32,4),0,ctx.stream()>>>('
    s=s.replace(needle,replacement)
    # New kernels accept n following sums; original fallback retains signature.
    start=s.index('    if (n<=',s.index('void ggml_cuda_w1a1_mul_mat'))
    end=s.index('    CUDA_CHECK(cudaGetLastError());',start)
    call=s[start:end];call=call.replace('layout.scale_offset+n));','layout.scale_offset+n), n);')
    orig=original[original.index(needle):original.index('    CUDA_CHECK(cudaGetLastError());',original.index(needle))]
    s=s[:start]+call+'    else '+orig.strip()+'\n'+s[end:]
  if s==original:continue
  dest=r/name;dest.write_text(s);(r/(name+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/'+str(src.relative_to(F)),tofile='b/'+str(src.relative_to(F)))))
  cmd=next(c.copy() for c in manifest['commands'] if '-c' in c and str(F/(name+'.o')) in c)
  old_obj=str(F/(name+'.o'));obj=r/(name+'.o');objects[old_obj]=str(obj)
  cmd=[str(r) if x==str(F) else '-I'+str(r) if x=='-I'+str(F) else x for x in cmd]
  cmd[cmd.index('-o')+1]=str(obj)
  if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
  if name=='dflash.cpp':
   unity=r/'unity.cxx';unity.write_text((F/'unity.cxx').read_text().replace(str(src),str(dest)));cmd[cmd.index('-c')+1]=str(unity)
  else:cmd[cmd.index('-c')+1]=str(dest)
  commands.append(cmd)
  with (r/'build.log').open('a') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 for lib,objectname in [('libggml-cuda.so.0.28.0','w1a1.cu.o'),('libllama.so.0.8.0','dflash.cpp.o')]:
  if str(F/objectname) not in objects:continue
  cmd=next(c.copy() for c in manifest['commands'] if '-shared' in c and str(F/'bin'/lib) in c)
  cmd=[objects.get(x,x) for x in cmd];cmd[cmd.index('-o')+1]=str(r/'bin'/lib);commands.append(cmd)
  with (r/'build.log').open('a') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 for src in (F/'bin').iterdir():
  dest=r/'bin'/src.name
  if dest.exists():continue
  own=r/'bin'/src.resolve().name;dest.symlink_to(own if own.exists() else src.resolve())
 (r/'build.json').write_text(json.dumps(dict(frozen_base=pin(F/'build.json'),variant=dict(copy=copy,bound=bound,dot=dot,threads=threads),commands=commands,files=[pin(x) for x in r.iterdir() if x.suffix in ('.cu','.cpp','.patch')],libraries=[pin(x) for x in (r/'bin').iterdir() if not x.is_symlink()]),indent=2)+'\n')
 return r
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('out',type=Path);a.add_argument('--copy',action='store_true');a.add_argument('--bound',action='store_true');a.add_argument('--dot',choices=['original','cross','mma'],default='original');a.add_argument('--threads',type=int,default=256);v=a.parse_args();build(v.out,v.copy,v.bound,v.dot,v.threads)
