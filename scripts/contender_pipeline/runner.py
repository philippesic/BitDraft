"""Sequential, budgeted execution of trusted candidate adapters; no shell commands.

Adapters own scientific semantics. A zero exit is insufficient: declared JSON
checks and artifact pins must pass. Plans are immutable across stage resumes.
Failed jobs require an explicit recovery plan; never overwrite their outputs.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from log_stream import LogStream


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def pin(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(plan):
    require(plan['schema'] == 'contender_plan_v1', 'unknown plan schema')
    require(positive(plan['budget_seconds']), 'invalid total budget')
    require(positive(plan['grace_seconds']), 'invalid cleanup grace')
    require(plan['jobs'], 'empty plan')
    seen = set()
    previous_stage = -1
    for job in plan['jobs']:
        require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', job['id']) and job['id'] not in seen, 'invalid/duplicate job id')
        require(type(job['stage']) is int and job['stage'] >= previous_stage, 'stages must be ordered')
        previous_stage = job['stage']
        require(set(job['depends_on']) <= seen, 'dependency must precede job')
        seen.add(job['id'])
        require(positive(job['timeout_seconds']), 'invalid job timeout')
        require(job['argv'] and all(type(x) is str and x for x in job['argv']), 'invalid argv')
        require(Path(job['argv'][0]).is_absolute() and Path(job['cwd']).is_absolute(), 'use absolute executable/cwd')
        require(job['checks'], 'success requires evidence checks')
        require(type(job['candidate']) is str and job['candidate'], 'candidate required for accounting')
    require(all(positive(v) for v in plan['candidate_budgets'].values()), 'invalid candidate budget')
    require(all(j['candidate'] in plan['candidate_budgets'] for j in plan['jobs']), 'missing candidate budget')
    for ref in plan['inputs']:
        require(pin(ref['path']) == ref, 'input changed: ' + ref['path'])


def check_results(job):
    evidence = []
    for check in job['checks']:
        value = read(check['path'])
        for key in check['keys']:
            value = value[key]
        require(type(value) is type(check['equals']) and value == check['equals'], 'result check failed: ' + str(check))
        evidence.append(pin(check['path']))
    for path in job.get('artifacts', []):
        evidence.append(pin(path))
    return evidence


def identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]),
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def group_live(pgid):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            pass
    return False


def stop_group(process, grace):
    for sig, limit in [(signal.SIGTERM, grace), (signal.SIGKILL, 5)]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + limit
        while group_live(process.pid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(.05)
        if not group_live(process.pid):
            break
    process.wait(timeout=5)
    require(not group_live(process.pid), 'job group remains live')


def execute(plan_path, output, through, allow_gpu=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = read(plan_path)
    validate(plan)
    if plan.get('gpu'):
        require(allow_gpu, 'explicit GPU flag required')
        control = read(plan['gpu']['control'])['rtx5080']
        require(not control.get('pause_requested') and not control.get('stop_all_requested'), 'GPU paused')
    with contextlib.ExitStack() as stack:
        stack.enter_context(locked(output / 'run.lock'))
        if plan.get('gpu'):
            stack.enter_context(locked(plan['gpu']['lock']))
        state_path = output / 'state.json'
        frozen = pin(plan_path)
        if state_path.exists():
            state = read(state_path)
        else:
            carried = {}
            prior = plan.get('prior_state')
            if prior:
                require(pin(prior['path']) == prior, 'prior accounting changed')
                previous = read(prior['path'])
                require(all(r['status'] != 'RUNNING' and r.get('group_released', False) for r in previous['jobs'].values()), 'prior processes not released')
                carried = previous.get('candidate_charged_seconds', {})
                if not carried:
                    for r in previous['jobs'].values():
                        carried[r['candidate']] = carried.get(r['candidate'], 0.) + r['charged_seconds']
                require(abs(sum(carried.values()) - previous['charged_seconds']) < .001, 'prior accounting mismatch')
                require(set(carried) <= set(plan['candidate_budgets']), 'prior candidate missing')
            state = dict(plan=frozen, jobs={}, charged_seconds=sum(carried.values()), candidate_charged_seconds=carried)
        if 'candidate_charged_seconds' not in state:
            state['candidate_charged_seconds'] = {}
            for r in state['jobs'].values():
                key = r['candidate']
                state['candidate_charged_seconds'][key] = state['candidate_charged_seconds'].get(key, 0.) + r['charged_seconds']
        require(state['plan'] == frozen, 'plan changed; use a new explicit plan')
        for job_id, row in state['jobs'].items():
            require(row['status'] == 'PASS', 'prior attempt incomplete/failed: ' + job_id + '; inspect processes and use explicit recovery plan')
            for ref in row['evidence']:
                require(pin(ref['path']) == ref, 'completed evidence changed')
        stopping = []
        handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, lambda signum, frame: stopping.append(signum))
        try:
            for job in plan['jobs']:
                if job['stage'] > through or job['id'] in state['jobs']:
                    continue
                require(all(state['jobs'].get(d, {}).get('status') == 'PASS' for d in job['depends_on']), 'dependency incomplete')
                require(not stopping, 'stop requested')
                for ref in plan['inputs']:
                    require(pin(ref['path']) == ref, 'input changed before job')
                if plan.get('gpu'):
                    control = read(plan['gpu']['control'])['rtx5080']
                    require(not control.get('pause_requested') and not control.get('stop_all_requested'), 'GPU paused')
                spent = state['candidate_charged_seconds'].get(job['candidate'], 0.)
                remaining = min(plan['budget_seconds'] - state['charged_seconds'], plan['candidate_budgets'][job['candidate']] - spent)
                # Reserve full timeout plus cleanup before launching. No optimistic oversubscription.
                reservation = job['timeout_seconds'] + plan['grace_seconds'] + 5
                require(remaining >= reservation, 'insufficient reserved budget for ' + job['id'])
                for check in job['checks']:
                    require(not Path(check['path']).exists(), 'stale output exists: ' + check['path'])
                row = dict(status='RUNNING', candidate=job['candidate'], charged_seconds=reservation,
                           argv=job['argv'], evidence=[], started_unix=time.time())
                state['jobs'][job['id']] = row
                state['charged_seconds'] += reservation
                state['candidate_charged_seconds'][job['candidate']] = spent + reservation
                atomic(state_path, state)
                started = time.monotonic()
                process = None
                relay = None
                try:
                    with contextlib.nullcontext():
                        process = subprocess.Popen(job['argv'], cwd=job['cwd'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                        relay = LogStream(process.stdout, output / (job['id'] + '.log'))
                        row['identity'] = identity(process.pid)
                        atomic(state_path, state)
                        reason = None
                        while process.poll() is None:
                            if relay.errors:
                                reason = 'LOG_FAILURE'
                                break
                            if stopping:
                                reason = 'INTERRUPTED'
                                break
                            if time.monotonic() - started >= job['timeout_seconds']:
                                reason = 'TIMEOUT'
                                break
                            if plan.get('gpu'):
                                control = read(plan['gpu']['control'])['rtx5080']
                                if control.get('pause_requested') or control.get('stop_all_requested'):
                                    reason = 'PAUSED'
                                    break
                            time.sleep(.1)
                        stop_group(process, plan['grace_seconds'])
                        relay.finish()
                        row['returncode'] = process.returncode
                        require(reason is None, reason or '')
                        require(process.returncode == 0, 'adapter exit ' + str(process.returncode))
                        for ref in plan['inputs']:
                            require(pin(ref['path']) == ref, 'input changed during job')
                        row['evidence'] = check_results(job)
                        row['status'] = 'PASS'
                except BaseException as exc:
                    row['status'] = 'FAILED'
                    row['error'] = str(exc)
                    raise
                finally:
                    if process is not None:
                        stop_group(process, plan['grace_seconds'])
                        row['group_released'] = not group_live(process.pid)
                    if relay is not None:
                        try:
                            relay.finish()
                        except RuntimeError as exc:
                            row.update(status='FAILED', log_error=str(exc))
                    elapsed = time.monotonic() - started
                    state['charged_seconds'] += elapsed - reservation
                    state['candidate_charged_seconds'][job['candidate']] += elapsed - reservation
                    row['charged_seconds'] = elapsed
                    row['ended_unix'] = time.time()
                    atomic(state_path, state)
            state['status'] = 'COMPLETE' if len(state['jobs']) == len(plan['jobs']) else 'STAGE_COMPLETE'
            atomic(state_path, state)
            return state
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--allow-gpu', action='store_true')
    args = parser.parse_args()
    result = execute(args.plan, args.output, args.through, args.allow_gpu)
    print(json.dumps(dict(status=result['status'], charged_seconds=result['charged_seconds'])))


if __name__ == '__main__':
    main()
