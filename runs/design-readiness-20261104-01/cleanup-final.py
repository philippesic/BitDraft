from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,subprocess
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';C=Path(json.loads((R/'freeze.json').read_text())['output']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def free():return {str(p):shutil.disk_usage(p).free for p in [P,Path('/mnt/c')]}
identities=[];states=[]
for path in sorted((P/'runs').glob('local-design-readiness-*/state.json')):
 state=read(path);assert state['status']=='finished',(path,state['status']);states.append(dict(path=str(path),status=state['status'],exit_code=state.get('exit_code')))
for path in R.glob('*identity.json'):
 row=read(path)
 if 'pid' in row:identities.append(row)
for path in R.glob('*ownership.json'):
 row=read(path)
 for x in row['identities']:identities.append({**x,'boot_id':row['boot_id']})
releases=[]
for path in R.rglob('release.json'):
 row=read(path);assert row['owned_process_reaped'] and row['owned_process_group_absent'] and row['returncode']==0;identities.append(row['identity']);releases.append(str(path))
groups=set()
for row in identities:
 if row.get('boot_id')!=boot:continue
 groups.add(row['pgid']);proc=Path('/proc')/str(row['pid'])
 if proc.exists():assert int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])!=row['start_ticks'],row
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:group=int((proc/'stat').read_text().rsplit(')',1)[1].split()[2])
 except (OSError,ValueError,IndexError):continue
 assert group not in groups,group
before=free();models=collections.defaultdict(list)
for path in sorted(C.rglob('model.gguf')):
 assert path.is_file() and not path.is_symlink();models[digest(path)].append(path)
actions=[]
for sha,paths in models.items():
 source=paths[0]
 for dest in paths[1:]:
  if source.stat().st_ino==dest.stat().st_ino and source.stat().st_dev==dest.stat().st_dev:continue
  assert digest(source)==digest(dest)==sha
  temporary=dest.with_name(dest.name+'.__readiness_hardlink__');assert not temporary.exists()
  try:
   os.link(source,temporary);assert digest(dest)==sha;os.replace(temporary,dest);assert digest(dest)==sha
   actions.append(dict(action='lossless_hardlink_dedup',source=str(source),destination=str(dest),sha256=sha,logical_bytes=dest.stat().st_size))
  except OSError as error:actions.append(dict(action='retained_duplicate_link_unavailable',source=str(source),destination=str(dest),reason=str(error)))
  finally:temporary.unlink(missing_ok=True)
removed=[]
for directory in [P/'scripts/design_readiness/__pycache__',R/'__pycache__']:
 if directory.exists():
  assert directory.is_dir() and not directory.is_symlink();size=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file());shutil.rmtree(directory);removed.append(dict(path=str(directory),bytes=size))
assert not (C/'synthetic-full-state-scratch.pt').exists()
inventory=[]
for root,kind in [(R,'raw evidence, source snapshots and recovery commands'),(C,'paid checkpoints, deployed exports and resumable states')]:
 files=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()];physical={}
 for p in files:
  s=p.stat();physical[(s.st_dev,s.st_ino)]=s.st_size
 inventory.append(dict(path=str(root),files=len(files),logical_bytes=sum(p.stat().st_size for p in files),unique_inode_bytes=sum(physical.values()),retention=kind))
result=dict(status='NO_LIVE_OWNED_JOBS_AND_BOUNDED_STORAGE_REVIEW_PASS',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,supervisors=states,identities_checked=len(identities),released_native_groups=len(releases),nvidia_compute=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True),free_before=before,free_after=free(),dedup=actions,removed_disposable=removed,retained_inventory=inventory,notes=['All paid checkpoint files retained. Hardlinks preserve every export path and byte; no model precision/value changed.','Only task-local bytecode and verified duplicate export storage reclaimed. Shared sources, caches and unrelated work untouched.','Synthetic2.8GB scratch was already removed after hash-verified restore.','Available-byte deltas can reflect concurrent Windows activity; WSL filesystem and C: reported separately. No VHDX compaction.'])
(R/'cleanup-final.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
