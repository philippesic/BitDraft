from pathlib import Path
import subprocess,shlex,hashlib,json,os,time,difflib
P=Path('/home/philip/binary-eagle-decoding');A=P/'runs/w1a8-modern-packed-runtime-20261103-01';B=A/'build02';R=P/'runs/w1a8-modern-headskip-20261103-01';S=R/'source';S.mkdir();old=A/'source/src/models/eagle3.cpp';original=old.read_text();anchor='''    res->t_h_nextn = cur;

    cur = build_norm(cur,''';assert original.count(anchor)==1
new=original.replace(anchor,'''    res->t_h_nextn = cur;

    // Cache-only catch-up still needs its original prenorm/KV computation,
    // but has no logits to expose or sample. Do not execute the unused head.
    if (n_outputs == 0) {
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur,''');source=S/'eagle3.cpp';source.write_text(new);(R/'headskip.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='a/src/models/eagle3.cpp',tofile='b/src/models/eagle3.cpp')))
oldunity=B/'src/CMakeFiles/llama.dir/Unity/unity_2_cxx.cxx';u=oldunity.read_text();assert str(old) in u;unity=S/'unity_2_cxx.cxx';unity.write_text(u.replace(str(old),str(source)))
obj='src/CMakeFiles/llama.dir/Unity/unity_2_cxx.cxx.o';lib='bin/libllama.so.0.8.0'
def command(target):return subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()[-1]
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
compile=shlex.split(command(obj));compile=[str(unity) if x==str(oldunity) else str(R/'unity_2.o') if x==obj else str(R/'unity_2.o.d') if x==obj+'.d' else x for x in compile];compile[1:1]=['-I',str(A/'source/src/models')]
link=shlex.split(command(lib));assert link[:2]==[':','&&'] and link[-2:]==['&&',':'];link=link[2:-2];link=[str(R/'unity_2.o') if x==obj else str(R/'bin/libllama.so.0.8.0') if x==lib else '-Wl,--dependency-file='+str(R/'link.d') if x.startswith('-Wl,--dependency-file=') else x for x in link];(R/'bin').mkdir();env={**os.environ,'CUDA_VISIBLE_DEVICES':''};phases=[];t=time.monotonic()
for name,argv in [('compile',compile),('link',link)]:
 start=time.monotonic()
 with (R/(name+'.log')).open('x') as out:subprocess.run(argv,cwd=B,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
 phases.append({'phase':name,'wall_seconds':time.monotonic()-start,'argv':argv})
for p in (B/'bin').iterdir():
 if p.name.startswith('libllama.so'):continue
 (R/'bin'/p.name).symlink_to(p.resolve())
for name in ['libllama.so','libllama.so.0']:(R/'bin'/name).symlink_to('libllama.so.0.8.0')
inputs=[pin(old),pin(oldunity),pin(__file__)]+[pin(B/x) for x in shlex.split(command(lib)) if x.endswith(('.o','.a'))]
r={'status':'HEADSKIP_COMPILED_NOT_GPU_QUALIFIED','source':pin(source),'unity':pin(unity),'original':pin(old),'patch':pin(R/'headskip.patch'),'library':pin(R/'bin/libllama.so.0.8.0'),'inputs':inputs,'phases':phases,'wall_seconds':time.monotonic()-t,'LD_LIBRARY_PATH':str(R/'bin')+':'+str(B/'bin'),'scope':'Only skipzero-outputhead;prenorm/KVunchanged;bridgefixnotcombined;frozenruntimesuntouched'};(R/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'wall_seconds':r['wall_seconds']}))
