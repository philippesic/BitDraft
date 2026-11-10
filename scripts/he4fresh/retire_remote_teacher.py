"""Retire the exact migrated source only after local admission and matched smoke."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from .common import R,MANIFEST_HASH,pin,save


def main(smoke):
    local=json.load(open(R/'local-data.json'))
    result=json.load(open(smoke/'result.json'))
    assert local['status']=='VERIFIED_LOCAL_READY'
    assert local['manifest_sha256']==MANIFEST_HASH
    assert result['status']=='MATCHED_LOCAL_LOADER_SMOKE_PASS'
    assert json.load(open(smoke/'gpu-release.json'))['clients'].strip()==''
    assert not (R/'teacher-remote-comparator').is_mount()
    teacher=Path(local['teacher']);assert not teacher.is_mount()
    assert pin(teacher/'supplemental/combined-manifest.json')['sha256']==MANIFEST_HASH
    verified=json.load(open(R/'pause-fix-20261110/dataset-verified.json'))
    assert verified['dataset_transfer']['exit_code']==0 and verified['sequences']==15647
    receipt=R/'pause-fix-20261110/remote-teacher-retirement.json'
    if receipt.exists():
        prior=json.load(open(receipt))
        assert prior['status']=='RETIREMENT_FAILED_REQUIRES_INSPECTION'
        number=1
        while receipt.with_name(f'remote-teacher-retirement-attempt-{number:02d}.json').exists():number+=1
        receipt.rename(receipt.with_name(f'remote-teacher-retirement-attempt-{number:02d}.json'))
    save(receipt,dict(status='AUTHORIZED_VERIFIED_RETIREMENT_START',local=pin(R/'local-data.json'),
        smoke=pin(smoke/'result.json'),source='pstack-2080:/home/philip/he4-capture-2080ti',
        preserved_checkpoint_backup='/home/philip/he4broad-training-20261110-01',started_unix=time.time()))
    code=r'''
import hashlib,json,os,pathlib,shutil,time
root=pathlib.Path('/home/philip/he4-capture-2080ti')
assert root.is_dir() and not root.is_symlink() and root.resolve()==root
before=root.stat();assert before.st_uid==os.getuid()
manifest=root/'supplemental/combined-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='d56b7ee0572fec4333ec9c271a69befe1b0fb51e6507ef878f37ad90f1901592'
prefix=str(root);consumers=[];denied=[];protected_services=[];ancestors=set();parent=os.getpid()
while parent>1 and parent not in ancestors:
 ancestors.add(parent)
 parent=int(pathlib.Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()[1])
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  links=[proc/'cwd',proc/'root']+list((proc/'fd').iterdir())
  for link in links:
   try:target=os.readlink(link)
   except FileNotFoundError:continue
   if target==prefix or target.startswith(prefix+'/'):
    consumers.append(dict(pid=int(proc.name),link=str(link),target=target))
 except FileNotFoundError:continue
 except PermissionError:
  if not proc.exists():continue
  command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
  # These nondumpable login/key-management services cannot expose their fd trees.
  # Exact commands were inspected after the first guarded attempt aborted.
  if command in ('/usr/lib/systemd/systemd --user','(sd-pam)','/usr/bin/gpg-agent --supervised') or (int(proc.name) in ancestors and command=='sshd: philip@notty'):
   protected_services.append(dict(pid=int(proc.name),command=command,stat=(proc/'stat').read_text()))
  else:denied.append(dict(pid=int(proc.name),command=command))
assert not consumers,consumers
assert not denied,denied
assert root.stat().st_ino==before.st_ino and root.stat().st_dev==before.st_dev
free_before=shutil.disk_usage(root.parent).free
shutil.rmtree(root)
assert not root.exists()
assert pathlib.Path('/home/philip/he4broad-training-20261110-01/candidate/recovery.json').is_file()
print(json.dumps(dict(status='REMOTE_DATASET_REMOVED_AFTER_VERIFIED_LOCAL_COPY_AND_SMOKE',source=str(root),
 consumers=consumers,same_user_proc_scan_denied=denied,protected_login_services=protected_services,inode=before.st_ino,device=before.st_dev,
 free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(root.parent).free,completed_unix=time.time(),
 checkpoint_backup_preserved=True)))
'''
    completed=subprocess.run(['ssh','pstack-2080','python3','-'],input=code,text=True,capture_output=True)
    prior=json.load(open(receipt));prior['stdout']=completed.stdout;prior['stderr']=completed.stderr;prior['exit_code']=completed.returncode
    if completed.returncode==0:prior.update(json.loads(completed.stdout))
    else:prior['status']='RETIREMENT_FAILED_REQUIRES_INSPECTION'
    save(receipt,prior);assert completed.returncode==0,completed.stderr
    print(prior['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('smoke',type=Path)
    main(parser.parse_args().smoke)
