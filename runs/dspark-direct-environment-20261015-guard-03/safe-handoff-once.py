#!/usr/bin/env python3
"""One bounded exact-OP04 gap guard. Only the sole operator may execute it."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import time

BOOT = '7f6f4496-6b72-494a-8331-8130e942b4f0'
SUPERVISOR, SUPERVISOR_BIRTH = 95097, 23241745
CONTROLLER, CONTROLLER_BIRTH = 95107, 23241771
LANE_SHA = '478f3292f8fe2a7b31f9960937816be4a565658cd39dbb72adf0a23a9cd0a155'
HELPER_SHA = '6c59944494a987bb5c7b7d5df7f1fdfd7a26da47432efe3462fb20cbdc4ed593'
PIPELINE_SHA = 'a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af'
GPU = 'GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e'
MINIMUM_UPDATES = 15566
HERE = Path(__file__).resolve().parent
RUN = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane')
SUPERVISOR_STATE = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-op04-shard-resume-operation-20261015-11/supervisor/runs/dspark-op04-shard-resume-20261015-11/state.json')


class Waiting(Exception):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    prior = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prior
    return module


# Exact current11 producer specialization. The original accepted helper remains
# byte-identical; its unchanged two-omission quality reader receives previous09.
CURRENT11_ATTEMPT = '/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-training-20261013/dspark-fresh-scale-only-12h-20261015-04/lane/attempts/9567ebe18b6d457d9e2e07f20028a0bc'
PREVIOUS09_FIXED_SHA = '09a6f2c562b6e8c6d5dfb32481b25b7c52bba7038fd1979babbb6fde9119fb6d'
SHARD_PHASE_PREFIX_COUNT = 35
SHARD_PHASE_PREFIX_SHA = 'e10af6aa960be98d78d77ae02a0f5782a535f9042161350384d3b9b398b91037'


def install_current11_quality_reader(helper):
    original = helper.op04_current09_shard_quality_state
    def current11_quality_state(bindings, state, metadata, lane):
        summaries = ('evaluation_status', 'measurement_status',
                     'strict_quality_status', 'continuation_status')
        previous = state.get('previous_attempt')
        if (not isinstance(previous, dict)
                or state.get('schema') != 'nine_model_lane_state_v1'
                or state.get('bundle_sha256') != LANE_SHA
                or state.get('candidate') != 'dspark_a8'
                or state.get('attempt') != CURRENT11_ATTEMPT
                or state.get('status') not in ('training', 'restoring_shards')
                or any(key in state for key in summaries)
                or metadata['evaluation_receipt']['sha256'] != '8ad3fb4c868ac44aee5b43a6c57403341970a505e7af77ac422b8ee0b9e1156c'
                or state.get('evaluation_receipt') != metadata['evaluation_receipt']
                or state.get('evaluations') != [metadata['evaluation_receipt']]
                or state.get('evaluation_receipt') != previous.get('evaluation_receipt')
                or state.get('evaluations') != previous.get('evaluations')):
            raise ValueError('exact current11 retained quality/history differs')
        fixed = {key: value for key, value in previous.items() if key != 'shard_phases'}
        if sha(json.dumps(fixed, sort_keys=True, separators=(',', ':')).encode()) != PREVIOUS09_FIXED_SHA:
            raise ValueError('exact previous09 immutable subtree differs')
        phases = state.get('shard_phases')
        if (not isinstance(phases, list) or phases != previous.get('shard_phases')
                or len(phases) < SHARD_PHASE_PREFIX_COUNT
                or sha(json.dumps(phases[:SHARD_PHASE_PREFIX_COUNT], sort_keys=True,
                                  separators=(',', ':')).encode()) != SHARD_PHASE_PREFIX_SHA):
            raise ValueError('original5d shared append-only shard phase prefix differs')
        return original(bindings, previous, metadata, lane)
    helper.op04_current09_shard_quality_state = current11_quality_state
    return original


def read_small(path):
    path = Path(path)
    before = path.lstat()
    if (not path.is_absolute() or path != path.resolve() or not stat.S_ISREG(before.st_mode)
            or before.st_size > 8 * 1024**2):
        raise ValueError('canonical bounded regular metadata required: ' + str(path))
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after):
        raise Waiting('metadata changed while reading')
    return raw, {'path': str(path), 'sha256': sha(raw)}, before.st_mtime_ns


def identities_and_children():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('boot changed')
    for pid, birth in ((SUPERVISOR, SUPERVISOR_BIRTH), (CONTROLLER, CONTROLLER_BIRTH)):
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text.rsplit(') ', 1)[1].split()
        if int(fields[19]) != birth or fields[0] in ('Z', 'X'):
            raise ValueError('exact owned PID/birth is no longer live')
        if pid == CONTROLLER and int(fields[1]) != SUPERVISOR:
            raise ValueError('controller parent changed')
    return (Path('/proc') / str(CONTROLLER) / 'task' / str(CONTROLLER) / 'children').read_text().split()


def authorization(lane):
    raw, record, _ = read_small(lane['gpu_control_path'])
    control = json.loads(raw)
    if (control.get('rtx5080', {}).get('pause_requested') is not False
            or control.get('rtx5080', {}).get('stop_all_requested') is not False
            or control.get('rtx2080ti', {}).get('pause_requested') is not True or (RUN / 'STOP').exists()):
        raise InterruptedError('pause/STOP/authorization changed')
    raw, supervisor_record, _ = read_small(SUPERVISOR_STATE)
    supervisor = json.loads(raw)
    if (supervisor.get('status') != 'running' or supervisor.get('pid') != CONTROLLER
            or supervisor.get('supervisor_pid') != SUPERVISOR
            or supervisor.get('received_signal') is not None or supervisor.get('stop_grace_seconds', 0) < 90):
        raise InterruptedError('original supervisor authorization changed')
    return [record, supervisor_record]


def publication_wait(bindings, records, mtimes, resource_progress):
    """Only two actual5d write-prefix shapes; never suppress helper failures."""
    state, endpoint, latest, status, ledger, admission = [json.loads(records[k]) for k in
        ('state','endpoint','latest','training_status','ledger','admission')]
    old, new = endpoint.get('counters', {}), latest.get('cursor', {})
    request = status.get('shard_request', {})
    reservation = request.get('reservation', {})
    checkpoint = {k: latest.get(k) for k in ('path','sha256')}
    expected_source = sha(json.dumps({**admission.get('source', {}), 'bundle_sha256': LANE_SHA},
                                    sort_keys=True,separators=(',',':')).encode())
    # These are original committed shard publication joins, without model reads.
    # Incomplete/corrupt/unrelated data falls through to the unchanged validator.
    matched_new = (
        sha(records['admission']) == '8895c1f04c7518eea79e9a1a6081d6997a1742c6c041453f073c3beeefb7bcd8'
        and state.get('bundle_sha256') == LANE_SHA and state.get('pending_shard_resume') is True
        and state.get('train_receipt') == bindings['metadata']['endpoint']
        and latest.get('schema') == 'block_qat_checkpoint_v1' and latest.get('committed') is True
        and latest.get('source_sha256') == expected_source
        and type(new.get('step')) is int and new['step'] > 0
        and status.get('status') == 'shard_required' and status.get('cursor') == new
        and status.get('checkpoint') == latest and ledger.get('active_attempt') is None
        and ledger.get('schema') == 'continuous_training_budget_v1'
        and ledger.get('source_sha256') == LANE_SHA and ledger.get('max_seconds') == 43200
        and ledger.get('training_seconds') == new.get('elapsed_seconds')
        and new.get('batch_reservation') == reservation and new.get('data_cursor') == reservation.get('cursor')
        and request.get('plan_sha256') == reservation.get('plan_sha256')
        and request.get('shard_ids') == reservation.get('shard_ids')
        and endpoint.get('schema') == 'nine_model_stage_receipt_v1'
        and endpoint.get('bundle_sha256') == LANE_SHA and endpoint.get('status') == 'PASS'
        and endpoint.get('committed') is True and endpoint.get('completion_reason') == 'shard_required')
    if not matched_new:
        return
    if (state.get('status') == 'awaiting_evaluation'
            and state.get('pending_training') == bindings['metadata']['endpoint']
            and old == new and endpoint.get('checkpoint') == checkpoint
            and endpoint.get('shard_request') == request):
        raise Waiting('original5d matched shard receipt published before restoring_shards stage')
    event = resource_progress.get('observations', [{}])[-1]
    if (state.get('status') in ('training','restoring_shards')
            and mtimes['state'] < min(mtimes['latest'],mtimes['ledger'],mtimes['training_status'])
            and mtimes['endpoint'] < min(mtimes['latest'],mtimes['ledger'],mtimes['training_status'])
            and type(old.get('step')) is int and old['step'] < new['step']
            and type(old.get('elapsed_seconds')) in (int,float)
            and old['elapsed_seconds'] < new['elapsed_seconds']
            and endpoint.get('checkpoint') != checkpoint
            and event.get('observed_unix', 0)*1e9 < max(mtimes['latest'],mtimes['ledger'],mtimes['training_status'])):
        raise Waiting('original5d prior authenticated shard receipt while newer settled shard publication awaits state/resource commit')


def failure_input_context(bindings, records, mtimes, resource_progress):
    """Already read hashes/scalars only. Never performs an extra runtime probe."""
    values={k:json.loads(records[k]) for k in ('state','endpoint','latest','training_status','ledger')}
    def cursor(v):
        c=v.get('counters',v.get('cursor',{}))
        return {'step':c.get('step'),'elapsed_seconds':c.get('elapsed_seconds'),
                'stage':c.get('stage'),'cursor_sha256':sha(json.dumps(c,sort_keys=True,separators=(',',':')).encode())}
    return {'metadata':{k:{**bindings['metadata'][k], 'observed_sha256':sha(v), 'bytes':len(v),
                          'mtime_ns':mtimes[k]} for k,v in records.items()},
      'state':{k:values['state'].get(k) for k in ('status','pending_shard_resume','pending_training',
          'train_receipt','bundle_sha256','admission','campaign_complete','evaluation_receipt',
          'evaluation_status','measurement_status','strict_quality_status','continuation_status')},
      'endpoint':{**cursor(values['endpoint']),**{k:values['endpoint'].get(k) for k in
          ('schema','status','committed','completion_reason','checkpoint','bundle_sha256','config_sha256')}},
      'latest':{**cursor(values['latest']),**{k:values['latest'].get(k) for k in
          ('schema','committed','path','sha256','source_sha256')}},
      'training_status':{**cursor(values['training_status']),'status':values['training_status'].get('status')},
      'ledger':values['ledger'],
      'resource_progress':{k:resource_progress.get('observations',[{}])[-1].get(k) for k in
          ('status','observed_unix','training_budget_active')}}


FAILURE_INPUT_ROLES = frozenset(('admission','endpoint','evaluation_receipt','evaluation_report',
    'lane','latest','ledger','request','state','timed_state','training_status'))

def snapshot_failure_inputs(directory, bindings, records, mtimes, sidecar_record, sidecar_raw, sidecar_mtime):
    """Failure observer only: copy observed buffers, never reopen their sources."""
    directory=Path(directory)
    if (directory != HERE/'failure-inputs-01' or not directory.is_absolute()
            or HERE != HERE.resolve() or directory == RUN or RUN in directory.parents
            or set(records) != FAILURE_INPUT_ROLES):
        raise ValueError('exact outside-RUN failure snapshot destination/roles required')
    copies={role:(records[role],bindings['metadata'][role],mtimes[role]) for role in sorted(records)}
    copies['resource_handoff_progress']=(sidecar_raw,sidecar_record,sidecar_mtime)
    if (any(not isinstance(raw,bytes) or len(raw)>8*1024**2 for raw,_,_ in copies.values())
            or sum(len(raw) for raw,_,_ in copies.values())>32*1024**2):
        raise ValueError('failure snapshot exceeds 8MiB per record or32MiB total')
    for raw,record,mtime in copies.values():
        path=Path(record['path'])
        if (not path.is_absolute() or str(path)!=os.path.normpath(str(path))
                or type(mtime) is not int or mtime<0 or sha(raw)!=record['sha256']):
            raise ValueError('observed failure input locator/hash/mtime differs')
    directory.mkdir(mode=0o700,exist_ok=False)
    members={}
    for role,(raw,record,mtime) in copies.items():
        destination=directory/(role+'.json')
        with destination.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        snapshot=destination.read_bytes()
        if snapshot!=raw:
            raise ValueError('failure snapshot copy bytes differ')
        members[role]={'source_path':record['path'],'source_sha256':record['sha256'],
            'source_mtime_ns':mtime,'source_bytes':len(raw),'snapshot_file':destination.name,
            'snapshot_sha256':sha(snapshot),'snapshot_bytes':len(snapshot)}
    manifest={'schema':'exact_guard_failure_observed_buffers_v1','complete':True,
              'scope':'already-read metadata_candidate failure inputs; no new source/runtime probes',
              'total_bytes':sum(len(raw) for raw,_,_ in copies.values()),'members':members}
    manifest_path=directory/'manifest.json'
    manifest_text=json.dumps(manifest,indent=2)+'\n'
    if manifest['total_bytes']+len(manifest_text.encode())>32*1024**2:
        raise ValueError('failure snapshot with manifest exceeds32MiB total')
    with manifest_path.open('x') as stream:
        stream.write(manifest_text);stream.flush();os.fsync(stream.fileno())
    return {'directory':str(directory),'manifest_sha256':sha(manifest_path.read_bytes()),
            'records':len(members),'bytes':manifest['total_bytes'],'snapshot_sealed':True}

def observe_metadata_failure(persist, bindings, records, mtimes, sidecar_record, sidecar_raw, sidecar_mtime):
    """Observer errors/partial files can never replace the original exception."""
    try:
        snapshot=snapshot_failure_inputs(HERE/'failure-inputs-01',bindings,records,mtimes,
                                         sidecar_record,sidecar_raw,sidecar_mtime)
        persist('NO_SIGNAL_METADATA_FAILURE_INPUTS_SNAPSHOT_SEALED',failure_inputs=snapshot,
                input_context=failure_input_context(bindings,records,mtimes,json.loads(sidecar_raw)))
    except BaseException as error:
        try:
            persist('NO_SIGNAL_METADATA_FAILURE_INPUTS_CAPTURE_FAILED',snapshot_sealed=False,
                    directory=str(HERE/'failure-inputs-01'),capture_error_type=type(error).__name__,
                    capture_error=str(error))
        except BaseException:
            pass


def metadata_candidate(helper, bindings, records, mtimes, resource_progress, *, minimum_updates=MINIMUM_UPDATES):
    if sha(records['endpoint']) != bindings['metadata']['endpoint']['sha256']:
        raise ValueError('endpoint bytes differ from authenticated state locator')
    endpoint = json.loads(records['endpoint'])
    step = endpoint.get('counters', {}).get('step')
    if type(step) is not int:
        raise ValueError('authenticated endpoint has no completed-update counter')
    # Controller state can still name the prior receipt while latest/status and
    # settled budget already describe the new child exit. An old authenticated
    # receipt is a WAIT, before any cross-snapshot joins can reject that overlap.
    if step <= minimum_updates:
        raise Waiting('waiting for a newly committed shard beyond runtime update floor')
    publication_wait(bindings, records, mtimes, resource_progress)
    lane, state = helper.validate_op04_metadata(bindings, records)
    ledger = json.loads(records['ledger'])
    if state.get('status') not in ('restoring_shards', 'training'):
        raise ValueError('controller stage failed/stopped')
    event = resource_progress['observations'][-1]
    if (event.get('status') != 'PASS' or event.get('training_budget_active') is not False
            or event.get('observed_unix', 0) * 1e9 < max(mtimes['endpoint'], mtimes['ledger'], mtimes['latest'])
            or event.get('baseline') != state['resource_baseline'] or event.get('policy') != lane['resource_policy']
            or event.get('owned_release', {}).get('owned_process_groups_absent') is not True
            or event.get('owned_release', {}).get('owned_cuda_pids_absent') is not True
            or event.get('owned_release', {}).get('other_context_pids') != []
            or event.get('resource_return', {}).get('dxg_holders') != []):
        raise Waiting('new shard has no fresh original resource PASS yet')
    return lane, state, endpoint, ledger, event


def final_signal(candidate, recheck, send):
    # No model/user ACK/tool roundtrip lies between this final fresh check and
    # the only signal call. pidfd binds the exact supervisor rather than a PID.
    recheck(candidate)
    send()


def send_before_deadline(deadline, send, now=time.monotonic):
    if now() >= deadline:
        raise Waiting('bounded guard deadline expired; no signal')
    send()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-file', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 1800 or os.getuid() != 1000 or os.geteuid() != 1000:
        raise ValueError('operator UID1000 and bounded <=1800 seconds required')
    output = args.decision_file
    if not output.is_absolute() or output.parent != HERE or RUN == output or RUN in output.parents:
        raise ValueError('fresh canonical decision path outside sealed lane required')
    with output.open('x') as stream:
        stream.write(json.dumps({'status': 'ARMED_NO_SIGNAL', 'guard_pid': os.getpid(), 'started_unix': time.time(), 'maximum_seconds': args.max_seconds}) + '\n')
    if sha((HERE / 'accepted-helper.py').read_bytes()) != HELPER_SHA:
        raise ValueError('accepted helper source changed')
    helper = load(HERE / 'accepted-helper.py', 'accepted_op04_guard_helper')
    install_current11_quality_reader(helper)
    bindings = json.loads((HERE / 'selection.json').read_text())
    pipeline_path = Path(bindings['original_pipeline']['path'])
    if sha(pipeline_path.read_bytes()) != PIPELINE_SHA:
        raise ValueError('original resource consumer source changed')
    pipeline = load(pipeline_path, 'original_op04_guard_resources')
    lane = json.loads(read_small(bindings['metadata']['lane']['path'])[0])
    if sha(read_small(bindings['metadata']['lane']['path'])[0]) != LANE_SHA:
        raise ValueError('exact lane changed')
    bootstrap_sources = [read_small(path)[1] for path in (Path(__file__).resolve(), HERE / 'accepted-helper.py', HERE / 'selection.json', pipeline_path)]
    events = []
    def persist(status, **extra):
        events.append(dict(status=status, observed_unix=time.time(), **extra))
        pipeline.atomic_json(output, {'schema': 'dspark_exact_safe_handoff_once_v1', 'events': events})
    def interrupted(signum, _frame):
        raise InterruptedError('guard itself received signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    deadline = started + args.max_seconds
    pidfd = None
    sent = False
    try:
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise ValueError('exact-supervisor pidfd signaling required')
        while time.monotonic() - started < args.max_seconds:
            children = identities_and_children()
            controls = authorization(lane)
            if children:
                time.sleep(0.5)
                continue
            try:
                raw, state_record, _ = read_small(RUN / 'state.json')
                state = json.loads(raw)
                if state.get('status') in ('failed', 'stopped', 'training_complete'):
                    raise ValueError('original controller stage is terminal')
                if not state.get('pending_shard_resume') or state.get('train_receipt') is None:
                    time.sleep(0.5)
                    continue
                current = json.loads((HERE / 'selection.json').read_text())
                current['metadata'].update(state=state_record, endpoint=state['train_receipt'], evaluation_receipt=state['evaluation_receipt'])
                result = json.loads(read_small(state['evaluation_receipt']['path'])[0])
                current['metadata'].update(evaluation_report=result['evaluation'], request=result['request'])
                records, mtimes = {}, {}
                for role, locator in current['metadata'].items():
                    raw, observed, mtime = read_small(locator['path'])
                    if role in ('state', 'endpoint', 'evaluation_receipt', 'evaluation_report', 'request', 'lane', 'admission') and observed != locator:
                        raise Waiting('sealed source changed during gap observation')
                    current['metadata'][role] = observed
                    records[role], mtimes[role] = raw, mtime
                raw, sidecar_record, sidecar_mtime = read_small(RUN / 'resource-handoff-progress.json')
                try:
                    candidate = metadata_candidate(helper, current, records, mtimes, json.loads(raw))
                except Waiting:
                    time.sleep(0.5)
                    continue
                except (ValueError, KeyError):
                    observe_metadata_failure(persist, current, records, mtimes, sidecar_record, raw, sidecar_mtime)
                    raise
                _, state, endpoint, ledger, event = candidate
                observer = pipeline.LinuxResources(GPU)
                release = observer.require_released([], ())
                observed = observer.snapshot()
                pipeline.resource_gate(observed, state['resource_baseline'], lane['resource_policy'])
                pipeline.resource_gate(event['resource_return'], event['baseline'], event['policy'])
                if release.get('other_context_pids') != [] or observed.get('dxg_holders') != [] or observed.get('boot_id') != BOOT:
                    raise Waiting('GPU/context release is not empty')
                seals = [*current['metadata'].values(), sidecar_record, *controls, *bootstrap_sources]
                decision = dict(completed_updates=endpoint['counters']['step'], trainer_seconds=ledger['training_seconds'],
                                source_hashes=seals, release=release, observed=observed,
                                supervisor={'pid': SUPERVISOR, 'birth_ticks': SUPERVISOR_BIRTH},
                                controller={'pid': CONTROLLER, 'birth_ticks': CONTROLLER_BIRTH})
                persist('ELIGIBLE_PRE_FINAL_CHECK_NO_SIGNAL', **decision)
                pidfd = os.pidfd_open(SUPERVISOR)
                def recheck(_):
                    authorization(lane)
                    for record in seals:
                        if read_small(record['path'])[1] != record:
                            raise Waiting('final metadata/source hash changed; no signal')
                    if identities_and_children():
                        raise Waiting('new trainer/capture child appeared; no signal')
                def send():
                    nonlocal sent
                    send_before_deadline(deadline, lambda: signal.pidfd_send_signal(pidfd, signal.SIGTERM))
                    sent = True
                final_signal(decision, recheck, send)
                persist('SIGTERM_SENT_ONLY_TO_EXACT_SUPERVISOR', **decision)
                return
            except (ValueError, KeyError) as error:
                if sent:
                    raise
                # This loop began with no child. Retry only if a fresh actual
                # child is now present in the same verified controller lineage;
                # unchanged invalid metadata/resource contracts still terminate.
                if identities_and_children():
                    if pidfd is not None:
                        os.close(pidfd)
                        pidfd = None
                    persist('NO_SIGNAL_FRESH_BUSY_CHILD_RETRY', reason=str(error))
                    time.sleep(0.5)
                    continue
                raise
            except Waiting as error:
                if pidfd is not None:
                    os.close(pidfd)
                    pidfd = None
                persist('NO_SIGNAL_RETRY_NEXT_GAP', reason=str(error))
                time.sleep(0.5)
                continue
        persist('NO_SIGNAL_BOUNDED_WAIT_EXPIRED')
    except Waiting as error:
        persist('NO_SIGNAL_FINAL_GUARD_CHANGED', reason=str(error))
    except BaseException as error:
        persist('SIGNAL_SENT_THEN_GUARD_ERROR' if sent else 'NO_SIGNAL_FAIL_CLOSED', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        if pidfd is not None:
            os.close(pidfd)


if __name__ == '__main__':
    main()
