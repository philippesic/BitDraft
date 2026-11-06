from pathlib import Path
import collections, datetime, hashlib, json, os, stat, time

P=Path('/home/philip/binary-eagle-decoding')
OUT=P/'runs/storage-purge-20261104-01'
def free():return {str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in [P,Path('/mnt/c')]}
def identity(path):
    s=path.lstat()
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,mode=stat.S_IFMT(s.st_mode),uid=s.st_uid)
def unchanged(path,expected):
    return identity(path)=={k:expected[k] for k in ['dev','ino','size','mtime_ns','mode','uid']}
def pin(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
manifest_path=OUT/'manifest.json'
with manifest_path.open('rb') as stream:manifest_sha=hashlib.file_digest(stream,'sha256').hexdigest()
m=json.loads(manifest_path.read_text());roots=[Path(x) for x in m['roots']]
assert m['schema']=='explicit_authorized_obsolete_artifact_purge_manifest_v1'
assert not (OUT/'actions.jsonl').exists(),'one-shot purge already started'
assert all(root.resolve()==root and root.is_dir() for root in roots)
files=m['files'];selected={x['path'] for x in files}
assert len(selected)==len(files) and len(selected)>1000
for row in files:
    path=Path(row['path'])
    assert any(path.is_relative_to(root) for root in roots) and path.resolve()==path
    assert unchanged(path,row['identity']) and row['identity']['mode']==stat.S_IFREG
    assert row['identity']['uid']==os.getuid()
    assert row['path'] not in m['protected_files']
for name,expected in m['protected_files'].items():assert unchanged(Path(name),expected),name

# Check actual open descriptors/mappings, not historical state labels.
open_conflicts=[]
for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:
        for fd in (process/'fd').iterdir():
            try:
                target=os.readlink(fd)
                if target in selected:open_conflicts.append(dict(pid=int(process.name),path=target,kind='fd'))
            except (OSError,FileNotFoundError):pass
        with (process/'maps').open() as stream:
            for line in stream:
                fields=line.rstrip().split(maxsplit=5)
                if len(fields)==6 and fields[5] in selected:open_conflicts.append(dict(pid=int(process.name),path=fields[5],kind='mapping'))
    except (OSError,FileNotFoundError):pass
assert not open_conflicts,open_conflicts
before=free();started=time.monotonic();counts=collections.Counter();logical=collections.Counter()
modern=[P/'runs/w1a8-modern-packed-init-20261103-01/model.gguf',Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/model.gguf'),Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt')]
modern_before=[pin(path) for path in modern]
(OUT/'execution-start.json').write_text(json.dumps(dict(before_free_bytes=before,manifest_sha256=manifest_sha,no_open_consumer_conflicts=True,modern_protected=modern_before,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
with (OUT/'actions.jsonl').open('x') as log:
    for index,row in enumerate(files):
        path=Path(row['path']);assert unchanged(path,row['identity']),str(path)
        path.unlink();counts[row['category']]+=1;logical[row['category']]+=row['identity']['size']
        log.write(json.dumps(dict(path=str(path),category=row['category'],action='purged_obsolete_payload',bytes=row['identity']['size']))+'\n')
        if index%100==0:
            log.flush();os.fsync(log.fileno());print(json.dumps(dict(stage='purge_progress',removed=index+1,total=len(files))),flush=True)
    for alias in m['aliases']:
        path=Path(alias['path']);assert path.is_symlink() and os.readlink(path)==alias['target']
        assert not path.exists(),'only aliases dangling after selected payload deletion may be retired'
        path.unlink();log.write(json.dumps(dict(path=str(path),action='removed_dangling_payload_alias',target=alias['target']))+'\n')
    log.flush();os.fsync(log.fileno())
assert all(not Path(row['path']).exists() for row in files)
for name,expected in m['protected_files'].items():assert unchanged(Path(name),expected),name
modern_after=[pin(path) for path in modern];assert modern_before==modern_after
after=free()
report=dict(schema='authorized_storage_purge_result_v1',status='PURGE_COMPLETE_RETAINED_FILES_VERIFIED',manifest_sha256=manifest_sha,removed_files=len(files),removed_dangling_aliases=len(m['aliases']),category_removed_files=dict(counts),category_logical_bytes=dict(logical),before_free_bytes=before,after_free_bytes=after,observed_free_byte_delta={name:after[name]-before[name] for name in before},retained_fingerprints_verified=len(m['protected_files']),modern_hashes_unchanged=modern_after,no_live_consumers=True,wall_seconds=time.monotonic()-started,scope='Historical obsolete intermediates/raw arrays permanently purged under explicit user discretion. Final/best historical checkpoints, GGUFs, compact evidence/configs/source and protected current dependencies retained. No VHDX compaction or experiment restart. WSL free-space gain is internal filesystem space, not a Windows-volume reclamation claim.')
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
