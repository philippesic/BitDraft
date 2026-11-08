"""Independent CPU timer: observe HE4broad, persist alarms, never signal/restart it."""
import argparse
import datetime as dt
import fcntl
import json
import os
import re
import signal
import threading
import time
from pathlib import Path

from .common import P, R, OUT, REPORT, DEST, BUDGET, identity, pin, save
from scripts.component_campaign.common import sync_directory

W = R / 'watchdog'
SUPERVISOR = P / 'runs/he4broad-supervisor-20261108-02/state.json'
INTERVAL = 600
STALL = 1200
BEGIN = '<!-- HE4BROAD_WATCHDOG_BEGIN -->'
END = '<!-- HE4BROAD_WATCHDOG_END -->'
OOM = re.compile(r'OutOfMemoryError|CUDA (?:error: )?out of memory', re.I)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def process(pid):
    try:
        path = Path('/proc') / str(pid)
        raw = (path / 'stat').read_text()
        fields = raw.rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return dict(pid=int(pid), stat=raw, start=fields[19], ppid=int(fields[1]),
                    pgid=int(fields[2]), argv=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def alive(expected, boot):
    actual = process(expected['pid'])
    return bool(actual and boot == expected['boot_id'] and
                actual['start'] == expected['stat'].rsplit(')', 1)[1].split()[19])


def lock_holders(path, kernel_locks):
    stat = path.stat()
    result = []
    for row in re.finditer(r'^\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s+(\d+)\s+([\da-f]+):([\da-f]+):(\d+)\s', kernel_locks, re.M | re.I):
        if (int(row[2],16),int(row[3],16),int(row[4])) == (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
            result.append(int(row[1]))
    return result


def snapshot(config):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text()
    live = read(OUT/'live.json')
    owner = config['workflow']
    children = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        current = process(path.name)
        if current and current['ppid'] == owner['pid'] and current['pgid'] == owner['pgid']:
            argv = current['argv']
            if '-m' in argv:
                current['module'] = argv[argv.index('-m')+1]
                children.append(current)
    trainers = [p for p in children if p['module'] == 'scripts.he4broad.train']
    oom = []
    for path in sorted(R.glob('train-*.log')) + [OUT/'update-failure.json']:
        if not path.exists():
            continue
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536))
            tail = stream.read().decode(errors='replace')
        if OOM.search(tail):
            oom.append(str(path))
    holders = lock_holders(P/'runs/contender-gpu.lock',Path('/proc/locks').read_text())
    return dict(boot_id=boot,live=live,live_mtime=(OUT/'live.json').stat().st_mtime,
        workflow_alive=alive(owner,boot),supervisor_alive=alive(config['supervisor'],boot),
        trainer_alive=bool(trainers),trainers=trainers,children=children,
        live_owner_alive=alive(live['owner'],boot) if live and 'owner' in live else False,
        lock_holders=holders,lock_owned=holders == [owner['pid']],oom_logs=oom,
        supervisor=read(SUPERVISOR,{}),workflow_result=read(R/'workflow-result.json',{}),
        training_result=read(OUT/'result.json',{}),completion=read(R/'completion-callback.json',{}),
        final_recovery=read(R/'final-recovery.json',{}))


def evaluate(sample, previous, now):
    """Pure state transition, including expected hourly/native process changes."""
    live = sample.get('live') or {}
    progress = [live.get('updates'),live.get('additional_update_seconds')]
    prior = previous.get('progress')
    advanced = bool(prior and all(a is not None and b is not None and a > b for a,b in zip(progress,prior)))
    same_boot = previous.get('boot_id') == sample['boot_id']
    last_progress = previous.get('last_progress_at',sample['live_mtime']) if same_boot else sample['live_mtime']
    if advanced:
        last_progress = sample['live_mtime']
    age = max(0,now-last_progress)
    record = dict(checked_at=utc(now),checked_at_epoch=now,boot_id=sample['boot_id'],progress=progress,
        advanced=advanced,last_progress_at=last_progress,no_progress_seconds=age,
        next_check_at=utc(now+INTERVAL),status='HEALTHY',failures=[],resume_attempts=0,
        resume_policy='UNSUPPORTED: existing supervisor has no safe post-failure resume mode; no restart attempted.')
    workflow = sample.get('workflow_result',{})
    training = sample.get('training_result',{})
    completion = sample.get('completion',{})
    clean = (workflow.get('status') == 'COMPLETE' and training.get('actual_update_seconds',0) >= BUDGET
             and sample.get('final_recovery',{}).get('status') == 'FULL_STATE_GPU_RESTORE_AND_NATIVE_BYTE_IDENTITY_PASS')
    requested_stop = (workflow.get('status') == 'PAUSED' or
                      bool(sample.get('supervisor',{}).get('received_signal')))
    if clean:
        if completion.get('status') == 'COMPLETE':
            record['status'] = 'COMPLETE'
        elif completion.get('status') in ('FINALIZATION_BLOCKED','PUSH_BLOCKED_LOCAL_REPORT_RETAINED'):
            record['failures'].append('COMPLETION_CALLBACK_FAILED')
        elif age > STALL:
            record['failures'].append('COMPLETION_CALLBACK_STALLED')
        else:
            record['status'] = 'FINISHING'
        # Lock release and trainer exit are expected after the full budget.
        return classified(record)
    if requested_stop:
        record['status'] = 'PAUSED' if not sample['workflow_alive'] else 'STOPPING'
        return classified(record)
    if not sample['workflow_alive']:
        record['failures'].append('WORKFLOW_DIED')
    if not sample['supervisor_alive']:
        record['failures'].append('SUPERVISOR_DIED')
    if live.get('status') == 'TRAINING' and not sample['live_owner_alive'] and not sample['trainer_alive']:
        record['failures'].append('TRAINER_DIED')
    if not sample['lock_owned']:
        record['failures'].append('GPU_LOCK_LOST')
    if sample['oom_logs']:
        record['failures'].append('OOM_REPORTED')
    if workflow.get('status') == 'STOPPED_WITH_ERROR':
        record['failures'].append('WORKFLOW_FAILED')
    if len(sample['trainers']) > 1:
        record['failures'].append('MULTIPLE_OWNED_TRAINERS')
    if prior and progress != prior and not advanced:
        record['failures'].append('PROGRESS_REGRESSED_OR_INCONSISTENT')
    if sample['workflow_alive'] and age >= STALL:
        record['failures'].append('NO_PROGRESS_20_MINUTES')
    # A missing old live.json PID is normal while a new hourly trainer starts,
    # or while the workflow runs a native panel/publisher/final verification.
    if not sample['trainer_alive'] and sample['workflow_alive'] and not sample['children']:
        record['status'] = 'BETWEEN_STAGES'
    elif not sample['trainer_alive'] and sample['workflow_alive']:
        record['status'] = 'EVALUATION_OR_FINALIZATION'
    return classified(record)


def classified(record):
    if record['failures']:
        record['status'] = 'FAILURE_NO_AUTOMATIC_RESUME'
        record['next_check_at'] = utc(record['checked_at_epoch']+30)
    return record


def failure_note(incident, current):
    """Keep a visible alarm in the report without changing training state."""
    path = REPORT/'report.md'
    text = (f'{BEGIN}\n\n**WATCHDOG FAILURE: {", ".join(incident["failures"])}.** '
        f'Detected {incident["checked_at"]}; observed {incident["progress"][0]} updates / '
        f'{incident["progress"][1]} update seconds. Current watchdog status: **{current["status"]}**. '
        'The watchdog does not stop or restart the trainer. Automatic resume attempts: **0**; '
        'the existing supervisor has no safe post-failure resume mode. The last good HE4broad '
        'checkpoint remains recorded in `candidate/recovery.json`; no B0, HE4 or random fallback is permitted. '
        '[Durable failure evidence](../../runs/he4broad-20261108-01/watchdog/failure.json) · '
        '[Watchdog heartbeat](../../runs/he4broad-20261108-01/watchdog/state.json).\n\n'+END+'\n\n')
    for _ in range(3):
        before = path.stat() if path.exists() else None
        original = path.read_text() if before is not None else '# HE4broad\n\n'
        content = original
        if BEGIN in content and END in content:
            start = content.index(BEGIN); end = content.index(END,start)+len(END)
            content = content[:start]+content[end:].lstrip('\n')
        title,sep,rest = content.partition('\n')
        output = title+'\n\n'+text+rest.lstrip('\n')
        if original == output:
            return
        current_stat = path.stat() if path.exists() else None
        if before != current_stat:
            continue
        temp = path.with_name('.watchdog-report-'+str(os.getpid())+'.tmp')
        try:
            with temp.open('x') as stream:
                stream.write(output); stream.flush(); os.fsync(stream.fileno())
            if (path.stat() if path.exists() else None) != before:
                continue
            temp.replace(path)
            sync_directory(path.parent)
            return
        finally:
            temp.unlink(missing_ok=True)
    raise RuntimeError('Report changed concurrently; durable failure.json retained; retry next tick')


def check(config):
    previous = read(W/'state.json',{})
    now = time.time()
    sample = snapshot(config)
    record = evaluate(sample,previous,now)
    record['observation'] = sample
    incident = read(W/'failure.json')
    if record['failures']:
        if incident is None:
            incident = {k:v for k,v in record.items() if k != 'observation'}
            incident['observation'] = sample
            incident['last_good_checkpoint_receipt'] = read(OUT/'recovery.json')
            incident['source'] = pin(__file__)
            save(W/'failure.json',incident)
        save(W/'latest-failure.json',record)
    elif incident and record['advanced']:
        record['incident_followup'] = 'Progress recovered without watchdog intervention; original failure retained.'
    save(W/'state.json',record)
    with (W/'checks.jsonl').open('a') as log:
        log.write(json.dumps({k:v for k,v in record.items() if k != 'observation'})+'\n')
        log.flush(); os.fsync(log.fileno())
    if incident:
        failure_note(incident,record)
    print(json.dumps({k:record[k] for k in ('checked_at','status','progress','advanced','no_progress_seconds','failures')}),flush=True)
    return record


def configure():
    owner = read(R/'gpu-owner.json')['owner']
    supervisor = read(SUPERVISOR)
    current = process(supervisor['supervisor_pid'])
    assert current and process(owner['pid'])['ppid'] == current['pid']
    expected = dict(pid=current['pid'],stat=current['stat'],pgid=current['pgid'],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text())
    config = dict(workflow=owner,supervisor=expected,interval_seconds=INTERVAL,stall_seconds=STALL,
        training_source_untouched=True,source=pin(__file__),recovery_attempt_limit=1,
        safe_resume_supported=False,automatic_resume_attempts=0,clean_exit_hook_preserved=True)
    save(W/'config.json',config)
    return config


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--once',action='store_true')
    args = ap.parse_args()
    W.mkdir(exist_ok=True)
    with (W/'monitor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = read(W/'config.json') or configure()
        save(W/'owner.json',identity())
        done = threading.Event()
        signal.signal(signal.SIGTERM,lambda *_:done.set())
        signal.signal(signal.SIGINT,lambda *_:done.set())
        while not done.is_set():
            try:
                record = check(config)
                if args.once or record['status'] in ('COMPLETE','PAUSED'):
                    break
                interval = 30 if (W/'failure.json').exists() else INTERVAL
            except Exception as error:
                prior = read(W/'state.json',{})
                record = dict(prior,checked_at=utc(time.time()),status='MONITOR_ERROR',failures=['MONITOR_ERROR'],
                    error=repr(error),progress=prior.get('progress',[None,None]),resume_attempts=0,
                    next_check_at=utc(time.time()+30))
                save(W/'state.json',record)
                save(W/'monitor-error.json',record)
                if not (W/'failure.json').exists():save(W/'failure.json',record)
                try:failure_note(read(W/'failure.json'),record)
                except Exception:pass
                print('MONITOR_ERROR',repr(error),flush=True)
                if args.once:raise
                interval = 30
            done.wait(interval)
        save(W/'exit.json',dict(status=record['status'],stopped=done.is_set(),owner=identity(),
                               training_signals_sent=0,training_restarts=0))


if __name__ == '__main__': main()
