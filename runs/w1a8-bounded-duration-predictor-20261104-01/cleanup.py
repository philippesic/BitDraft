from pathlib import Path
import json,os,datetime,subprocess,shutil
P=Path('/home/philip/binary-eagle-decoding');N=P/'runs/w1a8-bounded-duration-predictor-20261104-01';boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def read(p):return json.loads(Path(p).read_text())
runids=['local-w1a8-bounded-old5760-20261104-01','local-w1a8-bounded-old5760-20261104-02','local-w1a8-bounded-serving-20261104-01'];states={r:read(P/'runs'/r/'state.json') for r in runids};assert all(s['status']=='finished' for s in states.values());assert states[runids[1]]['exit_code']==states[runids[2]]['exit_code']==0
ids=[];groups=set()
for f in ['ownership.json','serving-ownership.json']:
 r=read(N/f);assert r['boot_id']==boot;ids+=r['identities']+r.get('attempt01',[])
ids.append(read(N/'trainer-identity.json'))
for s in states.values():groups.add(s['pgid'])
releases=[]
for d in [N/'old5760-http-qualification01',N/'repeated-serving01']:
 for f in d.glob('rep-*/release.json'):
  r=read(f);assert r['owned_process_reaped'] and r['owned_process_group_absent'] and r['returncode']==0;groups.add(r['identity']['pgid']);ids.append(r['identity']);releases.append(str(f))
assert len(releases)==36
live=[]
for r in ids:
 q=Path('/proc')/str(r['pid'])
 if q.exists():
  tick=int((q/'stat').read_text().rsplit(')',1)[1].split()[19])
  if tick==r['start_ticks']:live.append(r)
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  fields=(q/'stat').read_text().rsplit(')',1)[1].split()
  if int(fields[2]) in groups:live.append(dict(pid=int(q.name),pgid=int(fields[2])))
 except (OSError,ValueError,IndexError):pass
assert not live,live
before={p:shutil.disk_usage(p).free for p in [str(P),'/mnt/c']};removed=[];cache=N/'__pycache__'
if cache.exists():
 for f in cache.iterdir():
  assert f.is_file() and f.suffix=='.pyc';removed.append(dict(path=str(f),bytes=f.stat().st_size));f.unlink()
 cache.rmdir()
after={p:shutil.disk_usage(p).free for p in [str(P),'/mnt/c']}
receipt=dict(status='ALL_OWNED_JOBS_TERMINAL_AND_RELEASED',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_owned_processes_and_groups_absent=True,boot_id=boot,states=states,checked_identity_count=len(ids),checked_groups=sorted(groups),native_release_receipts=releases,hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),removed_disposable=removed,free_bytes_before=before,free_bytes_after=after)
(N/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
paths=[N,Path(read(N/'plan.json')['output'])]+[P/'runs'/r for r in runids];inventory=[]
for p in paths:
 files=[f for f in p.rglob('*') if f.is_file()];inventory.append(dict(path=str(p),files=len(files),logical_bytes=sum(f.stat().st_size for f in files),category='protected training checkpoints/export and evidence' if p==paths[1] else 'evidence and reusable supervisor caches',owner='bounded duration/predictor soleGPU operator',retention='Preserve new paid checkpoints, export and sufficient raw outputs/config/identity/release evidence; no pre-existing outputs removed.',next_review='Only after durable report audit and explicit retention review.'))
(N/'retained-inventory.json').write_text(json.dumps(dict(at_utc=receipt['at_utc'],paths=inventory,free_bytes=after,removed=removed,source_worktrees_removed=False),indent=2)+'\n');print(json.dumps(dict(released_groups=len(releases),checked_identities=len(ids),retained=inventory,free_bytes=after)))
