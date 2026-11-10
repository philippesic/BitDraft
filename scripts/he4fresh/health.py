"""CPU-only owned-process/progress snapshot for the dedicated health event."""
import json
import subprocess
import time
from .common import *


def alive(owner):
    try:
        stat = Path(f'/proc/{owner["pid"]}/stat').read_text()
        return (Path('/proc/sys/kernel/random/boot_id').read_text()==owner['boot_id'] and
                stat.rsplit(')',1)[1].split()[19]==owner['stat'].rsplit(')',1)[1].split()[19] and
                stat.rsplit(')',1)[1].split()[0] not in ('Z','X'))
    except (OSError,KeyError,IndexError): return False


def read(path): return json.load(open(path)) if path.exists() else None


def main():
    from .audit import audit
    owner=read(R/'gpu-owner.json');stage=read(R/'stage.json');live=read(OUT/'live.json')
    children=[]
    if owner and alive(owner['owner']):
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal(): continue
            try:
                stat=(entry/'stat').read_text()
                if int(stat.rsplit(')',1)[1].split()[1])==owner['owner']['pid']:
                    children.append(dict(pid=int(entry.name),stat=stat,
                        command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
                        io=(entry/'io').read_text()))
            except (OSError,ValueError,UnicodeError): pass
    result=dict(checked_unix=time.time(),owner_alive=alive(owner['owner']) if owner else False,
        mounts=dict(teacher=TEACHER.is_mount(),archive=DEST.is_mount()),
        storage=dict(mode='local' if (R/'local-data.json').exists() else 'original-remote',
            teacher_exists=TEACHER.is_dir(),archive_exists=DEST.is_dir(),
            local_device=(TEACHER.stat().st_dev==DEST.stat().st_dev==P.stat().st_dev)
                if TEACHER.exists() and DEST.exists() else False),
        stage=stage,live=live,workflow=read(R/'workflow-result.json'),
        supervisor=read(Path(owner['supervisor_state'])) if owner and owner.get('supervisor_state') else read(P/'runs/he4fresh-supervisor-20261110-01/state.json'),
        gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),
        proc_locks=Path('/proc/locks').read_text(),
        disk=subprocess.check_output(['df','-h',str(P),'/mnt/c',str(DEST)],text=True),
        children=children,exposure=audit())
    save(R/'health/latest.json',result)
    brief = {key:live.get(key) for key in ('status','updates','additional_update_seconds','training_seconds','last_loss')} if live else None
    print(json.dumps(dict(owner_alive=result['owner_alive'],stage=stage['stage'] if stage else None,
        stage_seconds=time.time()-stage['started_unix'] if stage else None,live=brief,
        mounts=result['mounts'],workflow=result['workflow']),indent=2))


if __name__=='__main__': main()
