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
from .clock import UPDATE_POINTS, POLICY, label as update_label, panel_label

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


def panel(point, update_clock=False):
    formal = update_clock or point != 60
    label = panel_label(point) if update_clock else ('formal-' if formal else 'smoke-')+str(point)
    out = R/label
    if (out/'result.json').exists():
        data = json.load(open(out/'result.json'))
        if data['status']=='COMPLETE' and all(x['correct'] for x in data['rows']): return
    if out.exists(): out.rename(R/(label+'-failed-'+str(time.time_ns())))
    # Only one recoverable current-model cache on local disk; native loading and
    # request timing are local, so network latency is outside measured serving.
    cache = R/'native-cache'; cache.mkdir(exist_ok=True)
    model = cache/'model.gguf'
    source = OUT/f'model-{update_label(point) if update_clock else point}.gguf'
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
    from .data import local_admission
    local_admission(json.load(open(R/'recipe.json'))['config'])
    # Reserve the remaining checkpoints/exports plus recovery/export scratch.
    readouts=json.load(open(OUT/'readouts.json'))
    complete={row['label'] for row in readouts}
    remaining=sum(update_label(point) not in complete for point in UPDATE_POINTS)
    state_bytes=json.load(open(OUT/'recovery.json'))['archive_bytes']
    native_bytes=max(row['model']['bytes'] for row in readouts)
    required=remaining*(state_bytes+native_bytes+64*2**20)+8*2**30
    save(R/'headroom.json',dict(remaining_checkpoints=remaining,required_bytes=required,free_bytes=shutil.disk_usage(DEST).free))
    assert shutil.disk_usage(DEST).free>required, 'Insufficient local checkpoint headroom'
    locks = []
    try:
        for path in (P/'runs/contender-gpu.lock',Path('/home/philip/.cache/semantic-offload/gpu-0.lock'),Path('/tmp/vllm-semantic-cache-gpu.lock')):
            lock = path.open('a+'); locks.append(lock)
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        assert not gpu, 'Other GPU job: '+gpu
        parent_args=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().split('\0')
        supervisor_index=next(i for i,x in enumerate(parent_args) if Path(x).name=='remote_job.py')
        supervisor_state=P/'runs'/parent_args[supervisor_index+1]/'state.json'
        save(R/'gpu-owner.json',dict(owner=identity(),locks=[str(x.name) for x in locks],
            tmux_socket='binary-eagle-runtime',tmux_session=os.environ.get('HE4_TMUX_SESSION','he4fresh-20261110-01'),
            source=pin(ROOT/'source-manifest.json'),supervisor_state=str(supervisor_state),training_policy=POLICY))
        save(R/'workflow-result.json',dict(status='RUNNING',owner=identity(),training_policy=POLICY))
        # Preserve/complete already-earned historical wall-hour panels before new update-hour panels.
        for row in readouts:
            if str(row['label']).isdecimal(): panel(int(row['label']))
        for point in UPDATE_POINTS:
            readouts = json.load(open(OUT/'readouts.json')) if (OUT/'readouts.json').exists() else []
            new_stage=update_label(point) not in {row['label'] for row in readouts}
            if new_stage:
                run('scripts.he4fresh.train',['--until',point],'train-'+update_label(point))
            result = json.load(open(OUT/'result.json'))
            if stopped or (new_stage and result['status']=='PAUSED'): break
            panel(point, update_clock=True)
            run('scripts.he4fresh.publish',[],'publish-'+update_label(point))
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
