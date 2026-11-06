import hashlib,json,subprocess
from pathlib import Path
R=Path('/home/philip/binary-eagle-decoding');S=R/'runs/checkouts/native-seven-5eb221-20261006';B=R/'runs/build/native-seven-5eb221-sm120-20261006'
P=R/'results/dspark-launch-20261006/native-seven-build-provenance.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(S),*args],text=True).strip()
assert sha(P)=='5efc1d58619fe41bc4821cecb91743934887cd6b4195b4b1df003926fe112e43'
v=json.loads(P.read_text());assert v['native_commit']=='5eb221ec53cb4317aef963e3bdec3cc69401eca4'
head=git('rev-parse','HEAD');status=git('status','--porcelain');diff=git('diff','--binary','HEAD')
files=[]
for item in v['files']:
 p=Path(item['path'])
 if p.name in {'llama-server','CMakeCache.txt','compile_commands.json'} or '.so' in p.name:
  observed=sha(p);assert observed==item['sha256'],str(p);files.append(dict(path=str(p),sha256=observed,bytes=p.stat().st_size))
assert sha(B/'bin/llama-server')=='cfde0aa023e88a57c002e9579b402a59def5cefb258c450b803fc932682d29a8'
value=dict(original_receipt=dict(path=str(P),sha256=sha(P)),current_source=dict(path=str(S),head=head,status=status,diff=diff),files=files,configure=v['configure'],build=v['build'],source_files={name:sha(S/name) for name in ['common/speculative.cpp','common/sampling.cpp','src/models/dflash.cpp']})
assert head==v['native_commit'] and not status and not diff
print(json.dumps(value,indent=2))
