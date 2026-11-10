"""Preserve historical artifacts while freeing local space for training data.

Remote checksum work runs on the storage host, not through SSHFS. A file is
replaced by a cold-storage symlink only after both copies have equal SHA-256.
This one-time migration verification is deliberately outside training updates.
"""
import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def remote_hash(plan_path):
    plan=json.load(open(plan_path));root=Path(plan['remote'])
    receipt=root/'checksum-progress.json'
    records=json.load(open(receipt)) if receipt.exists() else {}
    changed=[];nbytes=0
    for row in plan['files']:
        relative=row['relative']
        if relative in records:continue
        path=root/relative
        try:before=path.stat()
        except FileNotFoundError:continue
        if not stat.S_ISREG(before.st_mode) or before.st_size!=row['bytes']:continue
        digest=sha(path);after=path.stat()
        assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
        record=dict(relative=relative,bytes=before.st_size,sha256=digest,checked_unix=time.time())
        records[relative]=record;changed.append(record);nbytes+=before.st_size
        if nbytes>=16*2**30:break
    atomic(receipt,records)
    print(json.dumps(dict(records=changed,total_verified=len(records))),flush=True)


def retire():
    from .common import P,R,save
    work=R/'pause-fix-20261110';plan=json.load(open(work/'cold-storage-plan.json'))
    mount=R/'cold-storage-mount'
    assert mount.is_mount()
    journal=work/'cold-retirement.jsonl'
    completed={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);completed[row['relative']]=row
    rows={row['relative']:row for row in plan['files']}
    while len(completed)<len(rows):
        command=['ssh','pstack-2080','python3',plan['remote']+'/verifier.py',
                 '--remote-hash',plan['remote']+'/plan.json']
        result=json.loads(subprocess.check_output(command,text=True))
        # The remote hash cache can outlive an interrupted local retire process.
        if not result['records']:
            result['records']=json.loads(subprocess.check_output([
                'ssh','pstack-2080','cat',plan['remote']+'/checksum-progress.json'],text=True)).values()
        for evidence in result['records']:
            relative=evidence['relative']
            if relative in completed:continue
            row=rows[relative];source=Path(row['path']);destination=mount/relative
            assert source.is_relative_to(P/'runs') and not source.is_symlink()
            before=source.stat()
            assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_nlink)==(
                row['device'],row['inode'],row['bytes'],row['mtime_ns'],1),source
            assert sha(source)==evidence['sha256'],source
            assert destination.is_file() and destination.stat().st_size==before.st_size
            after=source.stat()
            assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
            temp=source.with_name(source.name+'.cold-link-'+str(os.getpid()))
            temp.symlink_to(destination);os.replace(temp,source)
            record=dict(**evidence,original=str(source),preserved_at=plan['remote']+'/'+relative,
                symlink=str(destination),freed_allocated_bytes=row['blocks'],retired_unix=time.time())
            with journal.open('a') as out:
                out.write(json.dumps(record)+'\n');out.flush();os.fsync(out.fileno())
            completed[relative]=record
        save(work/'cold-retirement-progress.json',dict(checked_unix=time.time(),files=len(completed),
            total_files=len(rows),freed_bytes=sum(x['freed_allocated_bytes'] for x in completed.values()),
            free_bytes=__import__('shutil').disk_usage(P).free))
        if len(completed)==len(rows):break
        supervisor=json.load(open(P/'runs/he4-localize-offload-20261110/state.json'))
        if supervisor['status'] in ('finished','failed','interrupted'):
            assert supervisor['exit_code']==0,supervisor
            if not result['records']:raise RuntimeError('Offload ended with unverified files')
        time.sleep(5)
    print('PRESERVED_AND_RETIRED',len(completed),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--remote-hash',type=Path);args=ap.parse_args()
    if args.remote_hash:remote_hash(args.remote_hash)
    else:retire()
