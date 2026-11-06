from pathlib import Path
import subprocess,shlex,hashlib,json,os,time,difflib
P=Path('/home/philip/binary-eagle-decoding');A=P/'runs/w1a8-modern-packed-runtime-20261103-01';B=A/'build02';R=P/'runs/w1a8-modern-bridge-fix-20261103-01';S=R/'source/common';S.mkdir(parents=True);(S.parent/'src').symlink_to(A/'source/src',target_is_directory=True)
def pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':d,'bytes':p.stat().st_size}
old=A/'source/common/speculative.cpp';original=old.read_text();anchor='''        batch.clear();

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];''';assert original.count(anchor)==1
new=original.replace(anchor,anchor.replace('        batch.clear();','''        batch.clear();
        // common_batch retains non-owning spans until batch.get(). Keep the
        // previous boundary alive while pending_g_last is refreshed below.
        std::vector<std::vector<float>> bridge_g(n_seq);'''))
a='''                    batch.set_embd(idx, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec });''';assert new.count(a)==1
new=new.replace(a,'''                    bridge_g[seq_id] = pending_g_last[seq_id];
                    batch.set_embd(idx, { bridge_g[seq_id].data(), 1, (size_t) n_embd_dec });''');source=S/'speculative.cpp';source.write_text(new);(R/'bridge.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='a/common/speculative.cpp',tofile='b/common/speculative.cpp')))
obj='common/CMakeFiles/llama-common.dir/speculative.cpp.o';lib='bin/libllama-common.so.0.8.0'
def command(target):return subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()[-1]
compile=shlex.split(command(obj));compile=[str(source) if x==str(old) else str(R/'speculative.cpp.o') if x==obj else str(R/'speculative.cpp.o.d') if x==obj+'.d' else x for x in compile]
link=shlex.split(command(lib));assert link[:2]==[':','&&'] and link[-2:]==['&&',':'];link=link[2:-2];link=[str(R/'speculative.cpp.o') if x==obj else str(R/'bin/libllama-common.so.0.8.0') if x==lib else '-Wl,--dependency-file='+str(R/'link.d') if x.startswith('-Wl,--dependency-file=') else x for x in link]
(R/'bin').mkdir();env={**os.environ,'CUDA_VISIBLE_DEVICES':''};phases=[];t=time.monotonic()
for name,argv in [('compile',compile),('link',link)]:
 started=time.monotonic()
 with (R/(name+'.log')).open('x') as out:subprocess.run(argv,cwd=B,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
 phases.append({'phase':name,'wall_seconds':time.monotonic()-started,'argv':argv})
for path in (B/'bin').iterdir():
 if path.name.startswith('libllama-common.so'):continue
 (R/'bin'/path.name).symlink_to(path.resolve())
for name in ['libllama-common.so','libllama-common.so.0']:(R/'bin'/name).symlink_to('libllama-common.so.0.8.0')
inputs=[pin(old),pin(__file__)]+[pin(B/x) for x in shlex.split(command(lib)) if x.endswith(('.o','.a'))]
result={'status':'ISOLATED_BRIDGE_LIBRARY_COMPILED_NOT_GPU_TESTED','source':pin(source),'original':pin(old),'patch':pin(R/'bridge.patch'),'library':pin(R/'bin/libllama-common.so.0.8.0'),'inputs':inputs,'phases':phases,'wall_seconds':time.monotonic()-t,'LD_LIBRARY_PATH':str(R/'bin')+':'+str(B/'bin'),'original_runtime_unchanged':True};(R/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds']}))
