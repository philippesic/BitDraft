"""Build the retained patch from runtime04 + frozen baseline objects in a new directory."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shlex,shutil
P=Path('/home/philip/binary-eagle-decoding');F=P/'runs/precision-native-20261104-04';S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source';B=S.parent/'build02';ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments/execution-campaign-20261105/runtime'
def pin(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);a=ap.parse_args();r=a.out.resolve();r.mkdir(exist_ok=False);(r/'bin').mkdir();pins=json.load(open(E/'source-pins.json'));manifest=json.load(open(F/'build.json'));commands=[];objects={}
for entry in pins['files']:
 src=Path(entry['base']['path']);assert pin(src)['sha256']==entry['base']['sha256'];dest=r/entry['relative'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
(r/'ggml-impl.h').symlink_to(F/'ggml/src/ggml-impl.h')
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(E/'optimized.patch')],cwd=r,check=True)
for entry in pins['files']:assert pin(r/entry['relative'])['sha256']==entry['optimized']['sha256']
def ninja(target):return shlex.split(subprocess.check_output(['ninja','-C',str(B),'-t','commands',target],text=True).splitlines()[-1])
with (r/'build.log').open('w') as log:
 for entry in pins['files']:
  name=entry['relative'];short=Path(name).name;src=r/name;obj=r/(short+'.o')
  if name.startswith('common/'):
   target='common/CMakeFiles/llama-common.dir/speculative.cpp.o';cmd=ninja(target);objects[target]=str(obj);cmd+=['-I'+str((S/name).parent)]
  else:
   original=str(F/(short+'.o'));cmd=next(c.copy() for c in manifest['commands'] if '-c' in c and original in c);objects[original]=str(obj);cmd=['-I'+str(r) if x=='-I'+str(F) else x for x in cmd]
  if short=='dflash.cpp':
   unity=r/'unity.cxx';unity.write_text((F/'unity.cxx').read_text().replace(str(F/name),str(src)));src=unity
  cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj)
  if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
  commands.append(cmd);subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 for lib in ['libggml-cuda.so.0.28.0','libllama.so.0.8.0','libllama-common.so.0.8.0']:
  cmd=ninja('bin/'+lib)[2:-2] if 'common' in lib else next(c.copy() for c in manifest['commands'] if '-shared' in c and str(F/'bin'/lib) in c)
  cmd=[objects.get(x,x) for x in cmd if not x.startswith('-Wl,--dependency-file=')];cmd[cmd.index('-o')+1]=str(r/'bin'/lib);commands.append(cmd);subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
for src in (F/'bin').iterdir():
 dest=r/'bin'/src.name
 if dest.exists():continue
 own=r/'bin'/src.resolve().name;dest.symlink_to(own if own.exists() else src.resolve())
(r/'build.json').write_text(json.dumps(dict(frozen_base=pin(F/'build.json'),patch=pin(E/'optimized.patch'),commands=commands,source_files=[pin(r/x['relative']) for x in pins['files']],libraries=[pin(x) for x in (r/'bin').iterdir() if not x.is_symlink()]),indent=2)+'\n')
print(r)
