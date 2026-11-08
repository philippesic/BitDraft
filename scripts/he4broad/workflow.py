"""Single locked, supervised GPU owner; serialize training and existing native panels."""
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from .common import *

stopped = False


def stop(*_):
    global stopped
    stopped = True


def run(module, args, label):
    pause()
    if stopped: raise InterruptedError('Stop requested between complete stages')
    command = [sys.executable,'-m',module,*map(str,args)]
    save(R/(label+'-command.json'),dict(argv=command,cwd=str(ROOT),owner=identity()))
    with (R/(label+'.log')).open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def reserve():
    DEST.mkdir(exist_ok=False)
    before = shutil.disk_usage(DEST).free
    slots = [str(x) for x in POINTS]+['paused']
    assert before > len(slots)*SLOT_BYTES + 8*2**30, before
    assert shutil.disk_usage(R).free > 10*2**30
    records = []
    for label in slots:
        path = DEST/('reserve-'+label)
        fd = os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        try:
            os.posix_fallocate(fd,0,SLOT_BYTES)
            os.fsync(fd)
        except BaseException:
            path.unlink()
            raise
        finally: os.close(fd)
        stat = path.stat()
        assert stat.st_size == SLOT_BYTES and stat.st_blocks*512 >= SLOT_BYTES
        records.append(dict(path=str(path),bytes=stat.st_size,allocated_bytes=stat.st_blocks*512))
        save(R/'storage-reservation.json',dict(status='RESERVING',before_free_bytes=before,slots=records))
    save(R/'storage-reservation.json',dict(status='PASS',before_free_bytes=before,
         remaining_free_bytes=shutil.disk_usage(DEST).free,slots=records,
         policy='Physical allocation released one slot at a time immediately before checkpoint; all checkpoints retained.'))


def panel(point, formal):
    label = ('formal-' if formal else 'smoke-')+str(point)
    arms = dict(target=None,q4=str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),
                HE4broad=str(OUT/f'model-{point}.gguf'))
    save(R/'panels'/(label+'-arms.json'),arms)
    save(R/'panels'/(label+'-cache.json'),dict(q4=['f16','f16'],HE4broad=['f16','f16']))
    args = ['--arms',R/'panels'/(label+'-arms.json'),'--draft-cache-map',R/'panels'/(label+'-cache.json'),
        '--runtime',RUNTIME,'--target',P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf',
        '--depth',2,'--cohort','capture3','--cohort-label','consumed-DEV3-HE4broad-'+label,
        '--out',R/label,'--reps',5 if formal else 1,'--warmups',2 if formal else 0]
    if not formal: args += ['--output-cap',64]
    run('scripts.component_campaign.native',args,label)


def release_reservations():
    receipt = R/'storage-reservation.json'
    if not receipt.exists(): return
    for row in json.load(open(receipt))['slots']:
        path = __import__('pathlib').Path(row['path'])
        if path.exists():
            assert path.parent == DEST and not path.is_symlink()
            assert path.stat().st_size == row['bytes']
            path.unlink()


def main():
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    pause()
    lock = (P/'runs/contender-gpu.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(R/'gpu-owner.json',dict(owner=identity(),lock=str(P/'runs/contender-gpu.lock'),
         tmux_socket='binary-eagle-runtime',tmux_session='he4broad-20261108-01'))
    try:
        # Under the lock, repeat the host check before any CUDA context is made.
        gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        assert not gpu, 'Another compute job owns GPU: '+gpu
        save(R/'gpu-preflight.json',dict(compute_processes=gpu,
             nvidia_smi=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),owner=identity()))
        reserve()
        for point in POINTS:
            run('scripts.he4broad.train',['--until',point],'train-'+str(point))
            if stopped or json.load(open(OUT/'result.json'))['status'] == 'PAUSED': break
            run('scripts.he4broad.publish',[],'publish-'+str(point))
            if point in (0,14400,BUDGET): panel(point,False)
            if point in (0,BUDGET): panel(point,True)
        if not stopped and json.load(open(OUT/'result.json'))['status'] == 'COMPLETE':
            run('scripts.he4broad.verify',[],'final-recovery')
        # Publishing is CPU only and also records partial or stopped evidence.
        from .publish import main as publish
        save(R/'workflow-result.json',dict(status='PAUSED' if stopped else json.load(open(OUT/'result.json'))['status'],owner=identity()))
        publish()
    except BaseException as error:
        save(R/'workflow-result.json',dict(status='PAUSED' if stopped else 'STOPPED_WITH_ERROR',error=repr(error),owner=identity()))
        from .publish import main as publish
        publish()
        raise
    finally:
        release_reservations()
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
        save(R/'gpu-release.json',dict(lock_released=True,owner=identity(),
             nvidia_smi=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True)))


if __name__ == '__main__': main()
