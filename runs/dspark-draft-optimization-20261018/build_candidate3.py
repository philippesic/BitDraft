from pathlib import Path
import subprocess,shlex,json,hashlib,shutil,time
R=Path('/home/philip/binary-eagle-decoding');O=R/'runs/dspark-draft-optimization-20261018';S=R/'runs/checkouts/native-seven-5eb221-20261006';B=R/'runs/build/native-seven-5eb221-sm120-20261006';C=O/'native-dp4a-03';D=O/'build-dp4a-03'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/dspark-draft-optimization-20261018/candidate3-w1a1.cu');assert sha(source)=='6598d96b6d091a2c416662fd6713958a810d878d9f9fee799c16d6c6d23e11bd'
assert not C.exists() and not D.exists();assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
subprocess.run(['git','-C',str(S),'worktree','add','--detach',str(C),'5eb221ec53cb4317aef963e3bdec3cc69401eca4'],check=True)
shutil.copyfile(source,C/'ggml/src/ggml-cuda/w1a1.cu');D.mkdir();shutil.copytree(B/'bin',D/'bin',symlinks=True)
entry=next(x for x in json.loads((B/'compile_commands.json').read_text()) if x['file']==str(S/'ggml/src/ggml-cuda/w1a1.cu'))
args=shlex.split(entry['command']);oldobj=args[args.index('-o')+1];newobj=D/'w1a1.cu.o';args=[x.replace(str(S/'ggml/src/ggml-cuda/w1a1.cu'),str(C/'ggml/src/ggml-cuda/w1a1.cu')).replace(oldobj,str(newobj)) for x in args]
commands=subprocess.check_output([str(R/'.venv/bin/ninja'),'-C',str(B),'-t','commands','bin/libggml-cuda.so.0.25.1'],text=True).splitlines();link=commands[-1]
assert oldobj in link and ' -shared ' in link
link=link.replace(oldobj,str(newobj)).replace('bin/libggml-cuda.so',str(D/'bin/libggml-cuda.so'))
receipt=dict(source_sha256=sha(source),base_revision='5eb221ec53cb4317aef963e3bdec3cc69401eca4',compile_argv=args,compile_cwd=entry['directory'],link_command=link,link_cwd=str(B),original_cuda_sha256=sha(B/'bin/libggml-cuda.so.0.25.1'),started=time.time())
(D/'build-plan.json').write_text(json.dumps(receipt,indent=2));print('COMPILE',flush=True);subprocess.run(args,cwd=entry['directory'],check=True,timeout=240);print('LINK',flush=True);subprocess.run(link,cwd=B,shell=True,check=True,timeout=120)
receipt.update(ended=time.time(),candidate_cuda_sha256=sha(D/'bin/libggml-cuda.so.0.25.1'),candidate_object_sha256=sha(newobj),status='BUILT')
assert sha(B/'bin/libggml-cuda.so.0.25.1')==receipt['original_cuda_sha256']
(D/'build-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ['compile_argv','link_command']}),flush=True)
