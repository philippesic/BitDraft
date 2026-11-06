"""Bounded quiet inventory of this assignment; no deletion or cache retirement."""
from pathlib import Path
import os,json,time,shutil,subprocess
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
for name in ['fresh-finish04','request-final02']:
 s=json.loads((W/'supervisors'/name/'state.json').read_text());assert s['status']=='finished' and s['exit_code']==0
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
k=json.loads((R/'interop-keeper03.json').read_text());os.environ['WSL_INTEROP']=k['interop']
def sizes(root):
 total=0;files=0;links=0;seen=set();unique=0
 for folder,dirs,names in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
  for name in names:
   p=Path(folder)/name
   if p.is_symlink():links+=1;continue
   if not p.is_file():continue
   s=p.stat();total+=s.st_size;files+=1;key=(s.st_dev,s.st_ino)
   if key not in seen:seen.add(key);unique+=s.st_size
 return {'logical_bytes':total,'unique_inode_payload_bytes':unique,'files':files,'symlinks_skipped':links}
def reason(name):
 if 'checkpoint' in name:return 'Protected paid or qualification states, retained under existing lifecycle policy.'
 if name=='data' or 'teacher' in name or 'capture' in name:return 'Valuable selected inputs/capture evidence; raw quality logits archived exactly onMac with restore instructions.'
 if 'source' in name or 'controller' in name:return 'Qualified reusable source, state, models and negative evidence; keep.'
 if 'qualification' in name:return 'Required continuation/native/memory proofs and reusable CUDA cache; keep.'
 return 'Useful experiment, source or process-closure evidence; bounded metadata and retained results.'
entries=[]
for root in [W,R]:
 for p in root.iterdir():
  if p.is_symlink():continue
  if p.is_dir():entries.append({'path':str(p),**sizes(p),'owner':'current assignment','retention':reason(p.name)})
entries.sort(key=lambda x:-x['logical_bytes'])
ps='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe';native=int(subprocess.check_output([ps,'-NoProfile','-Command','(Get-Volume -DriveLetter C).SizeRemaining'],text=True).strip())
out={'observed_unix':time.time(),'scope':[str(W),str(R)],'entries':entries,'free_bytes':{'nativeC':native,'WSL':shutil.disk_usage(R).free},'size_scope':'Logical bytes and per-directory inode deduplication; not physical reclaim estimates. Windows/WSL free measured separately. Other actors changedWSL free since earlier session; no claim of our reclaim.','action':'Preserve valuable results, archives, inputs, checkpoints and reusable caches. Exact closed qualification raw duplication consolidated only under separate receipt; no broad cleanup.'}
with (R/'storage-final-inventory01.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps({'free_bytes':out['free_bytes'],'largest_retained':[{k:e[k] for k in ['path','logical_bytes','retention']} for e in entries[:8]]}),flush=True)
