"""One supervised GPU owner; resumable serialized training/native stages."""
import contextlib
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


def run(module,args,label):
    pause()
    if stopped: raise InterruptedError('Stop requested between complete stages')
    stamp = str(time.time_ns())
    command = [sys.executable,'-m',module,*map(str,args)]
    save(R/'stage.json',dict(stage=label,started_unix=time.time(),owner=identity(),argv=command))
    save(R/'commands'/(label+'-'+stamp+'.json'),dict(argv=command,cwd=str(ROOT),owner=identity()))
    with (R/(label+'-'+stamp+'.log')).open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def panel(point):
    formal = point != 60
    label = ('formal-' if formal else 'smoke-')+str(point)
    out = R/label
    if (out/'result.json').exists():
        data = json.load(open(out/'result.json'))
        if data['status']=='COMPLETE' and all(x['correct'] for x in data['rows']): return
    if out.exists(): out.rename(R/(label+'-failed-'+str(time.time_ns())))
    # Only one recoverable current-model cache on local disk; native loading and
    # request timing are local, so network latency is outside measured serving.
    cache = R/'native-cache'; cache.mkdir(exist_ok=True)
    model = cache/'model.gguf'
    source = OUT/f'model-{point}.gguf'
    if model.exists(): model.unlink()
    assert shutil.disk_usage(cache).free > source.stat().st_size + 512*2**20
    shutil.copyfile(source,model)
    assert pin(model)['sha256'] == pin(source)['sha256']
    arms = dict(target=None,q4=str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),HE4broad=str(model))
    save(R/'panels'/(label+'-arms.json'),arms)
    save(R/'panels'/(label+'-cache.json'),dict(q4=['f16','f16'],HE4broad=['f16','f16']))
    args = ['--arms',R/'panels'/(label+'-arms.json'),'--draft-cache-map',R/'panels'/(label+'-cache.json'),
        '--runtime',RUNTIME,'--target',P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf',
        '--depth',2,'--cohort','capture3','--cohort-label','consumed-DEV3-HE4fresh-'+label,
        '--out',out,'--reps',5 if formal else 1,'--warmups',2 if formal else 0]
    if not formal: args += ['--output-cap',64]
    try: run('scripts.component_campaign.native',args,label)
    finally: model.unlink(missing_ok=True)


def main():
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    pause()
    assert TEACHER.is_mount() and DEST.is_mount()
    assert shutil.disk_usage(DEST).free > 80*2**30
    locks = []
    try:
        for path in (P/'runs/contender-gpu.lock',Path('/home/philip/.cache/semantic-offload/gpu-0.lock'),Path('/tmp/vllm-semantic-cache-gpu.lock')):
            lock = path.open('a+'); locks.append(lock)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        assert not gpu, 'Other GPU job: '+gpu
        save(R/'gpu-owner.json',dict(owner=identity(),locks=[str(x.name) for x in locks],
            tmux_socket='binary-eagle-runtime',tmux_session='he4fresh-20261110-01',source=pin(R/'source/source-manifest.json')))
        for point in POINTS:
            readouts = json.load(open(OUT/'readouts.json')) if (OUT/'readouts.json').exists() else []
            if str(point) not in {row['label'] for row in readouts}:
                run('scripts.he4fresh.train',['--until',point],'train-'+str(point))
            result = json.load(open(OUT/'result.json'))
            if stopped or result['status']=='PAUSED': break
            panel(point)
            run('scripts.he4fresh.publish',[],'publish-'+str(point))
        result = json.load(open(OUT/'result.json'))
        if not stopped and result['status']=='COMPLETE' and not (R/'final-recovery.json').exists():
            run('scripts.he4fresh.verify',[],'final-recovery')
        save(R/'workflow-result.json',dict(status='PAUSED' if stopped else result['status'],owner=identity()))
        from .publish import main as publish
        publish()
    except BaseException as error:
        save(R/'workflow-result.json',dict(status='PAUSED' if stopped else 'STOPPED_WITH_ERROR',error=repr(error),owner=identity()))
        with contextlib.suppress(Exception):
            from .publish import main as publish
            publish()
        raise
    finally:
        for lock in reversed(locks):
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        save(R/'gpu-release.json',dict(lock_released=True,owner=identity(),
            nvidia_smi=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True)))


if __name__=='__main__': main()
