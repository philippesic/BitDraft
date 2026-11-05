"""Recompile selected frozen-overlay objects without touching shared base outputs."""
from pathlib import Path
import json,subprocess,difflib
from scripts.execution_campaign.build import F,B,pin

def relink(r,changes,more_objects=None):
 old=json.load(open(r/'build.json'));frozen=json.load(open(F/'build.json'));objects={};commands=[]
 for name,content in changes.items():
  dest=r/name;before=dest.read_text() if dest.exists() else (F/name).read_text();dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.is_symlink():dest.unlink()
  dest.write_text(content);(r/(Path(name).name+'.extended.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),content.splitlines(True),fromfile='a/'+name,tofile='b/'+name)))
  if Path(name).suffix=='.h':continue
  short=Path(name).name
  cmd=next(c.copy() for c in frozen['commands'] if '-c' in c and str(F/(short+'.o')) in c)
  cmd=['-I'+str(r) if x=='-I'+str(F) else x for x in cmd];obj=r/(short+'.o');objects[str(F/(short+'.o'))]=str(obj)
  cmd[cmd.index('-o')+1]=str(obj)
  if '-MF' in cmd:cmd[cmd.index('-MF')+1]=str(obj)+'.d'
  if short=='dflash.cpp':
   unity=r/'unity.cxx';unity.write_text((F/'unity.cxx').read_text().replace(str(F/'src/models/dflash.cpp'),str(dest)));cmd[cmd.index('-c')+1]=str(unity)
  else:cmd[cmd.index('-c')+1]=str(dest)
  commands.append(cmd)
  with (r/'extended-build.log').open('a') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 # Include the existing base overlay objects when relinking.
 for short in ['w1a1.cu.o','dflash.cpp.o','ggml-cuda.cu.o']:
  if (r/short).exists():objects[str(F/short)]=str(r/short)
 for lib in ['libggml-cuda.so.0.28.0','libllama.so.0.8.0']:
  cmd=next(c.copy() for c in frozen['commands'] if '-shared' in c and str(F/'bin'/lib) in c);cmd=[objects.get(x,x) for x in cmd];cmd[cmd.index('-o')+1]=str(r/'bin'/lib)
  dest=r/'bin'/lib
  if dest.is_symlink():dest.unlink()
  commands.append(cmd)
  with (r/'extended-build.log').open('a') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 for src in (r/'bin').iterdir():
  if src.is_symlink() and src.name in ('libllama.so','libllama.so.0','libggml-cuda.so','libggml-cuda.so.0'):
   name='libllama.so.0.8.0' if 'llama' in src.name else 'libggml-cuda.so.0.28.0';src.unlink();src.symlink_to(r/'bin'/name)
 old['extended_commands']=commands;old['extended_files']=[pin(r/name) for name in changes];old['libraries']=[pin(p) for p in (r/'bin').iterdir() if not p.is_symlink()];(r/'build.json').write_text(json.dumps(old,indent=2)+'\n')
