"""Finite single-arm fresh EAL curve after actual independent qualification.

The lane, numeric source, original guards and actual native verifier are unchanged.
No retry of failed or guard-stopped training. This is a new authorized fork.
"""
from pathlib import Path
import datetime
import os
import subprocess
import sys
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
S = R / 'source03'
sys.path[:0] = [str(S / 'scripts'), str(P / 'scripts')]
os.environ['PYTHONPATH'] = str(S / 'scripts') + ':' + str(P / 'scripts')
from training_system.io import read, pin, authenticate, atomic_json, digest
from training_system.device_proofs import _proofs
from training_system.train import continuation_contract
from training_system.lane import admit, dispatch, extend
from training_system.promotion import verified_metrics, decide
from training_system.resources import admit_capacity
import shutil

plan = read(R / 'fresh-restart-plan03.json')
assert plan['milestones'] == [480, 4800, 14400]
assert plan['training_deadline_unix'] == 1793507400
qstate = read(W / 'supervisors/fresh-qualify01/state.json')
assert qstate['status'] == 'finished' and qstate['exit_code'] == 0
qualification = pin(W / 'fresh-qualification01/result.json')
record, hardware, proofs = _proofs(qualification, 'control')
keeper = read(R / 'interop-keeper02.json')
assert Path(keeper['interop']).exists()
os.environ['WSL_INTEROP'] = keeper['interop']
os.environ['CUDA_CACHE_PATH'] = str(W / 'fresh-qualification01/cuda')
root = W / 'fresh-training01'
root.mkdir(exist_ok=False)
spec = read(W / 'fresh-base-spec01.json')
spec.update(target_updates=480, max_invocation_train_seconds=840,
            checkpoint_root=str(W / 'fresh-training-checkpoints01'),
            retention_policy='latest_verified_and_previous',
            job_tag='genuine-fresh-EAL-curve01')
assert digest(continuation_contract(spec)) == record['adapter_state_key']
guard = read(W / 'fresh-qualification-guard01.json')
guard.update(max_train_seconds=840, wall_seconds=2100,
             startup_seconds=600, stall_seconds=600)
guard['env'] = {**guard['env'], 'CUDA_CACHE_PATH': os.environ['CUDA_CACHE_PATH'],
                'WSL_INTEROP': keeper['interop']}
guard['inputs'] += [pin(__file__), pin(R / 'fresh-restart-plan03.json'), qualification]
gp = root / 'guard.json'
atomic_json(gp, guard)

def capacity(reservation):
    return admit_capacity({'nativeC': {'free_bytes': shutil.disk_usage(W).free,
                                      'floor_bytes': 10 * 1024**3,
                                      'reservation_bytes': reservation},
                           'WSL': {'free_bytes': shutil.disk_usage(R).free,
                                   'floor_bytes': 2 * 1024**3,
                                   'reservation_bytes': 256 * 1024**2}})

capacity(12 * 1024**3)
jobroot, job = admit(spec, root / 'jobs')
state = {'schema': 'finite_genuine_fresh_curve_v1', 'status': 'RUNNING',
         'job': str(jobroot), 'qualification': qualification,
         'plan': pin(R / 'fresh-restart-plan03.json'), 'native': {}, 'events': []}

def save(event):
    state['events'].append({'unix': time.time(), **event})
    atomic_json(root / 'state.json', state)

save({'action': 'qualified_single_arm_admitted', 'hardware': hardware})
parent = pin(W / 'parent-native01/result.json')
for target in plan['milestones']:
    if time.time() + 900 >= 1793507400:
        state['status'] = 'FINITE_BOUNDARY'
        save({'action': 'insufficient_remaining_time_before_stage', 'target': target})
        break
    job = read(jobroot / 'job.json')
    if target > job['spec']['target_updates']:
        previous = job['spec']['target_updates']
        finish = read(authenticate(job['finish']))
        native = state['native'][str(previous)]
        approval = {'schema': 'training_stage_approval_v2', 'job_key': job['key'],
                    'from_updates': previous, 'to_updates': target,
                    'kind': 'protected_exploration', 'model': finish['model'],
                    'native_evaluation': native, 'control_evaluation': parent,
                    'computed': decide(native, parent, model=finish['model'], manifest=spec['manifest']),
                    'plan': pin(R / 'fresh-restart-plan03.json')}
        ap = root / ('approval-%d.json' % target)
        atomic_json(ap, approval)
        job = extend(jobroot, target, pin(ap))
    while job['status'] != 'COMPLETE':
        if job['status'] not in ['ADMITTED', 'PAUSED']:
            state['status'] = job['status']
            save({'action': 'terminal_training_result_no_retry', 'job_status': job['status']})
            print('FRESH_TERMINAL_NO_RETRY', job['status'], flush=True)
            raise SystemExit(0)
        # Reserve a complete next invocation plus export/native/closure time.
        remaining = target - (read(authenticate(job['finish']))['local_updates'] if job.get('finish') else 0)
        wall_reserve = min(1800, max(1200, remaining * .24 + 900))
        if time.time() + wall_reserve > 1793507400:
            state['status'] = 'FINITE_BOUNDARY'
            save({'action': 'insufficient_remaining_time_before_invocation', 'target': target})
            raise SystemExit(0)
        admission = capacity(12 * 1024**3)
        save({'action': 'dispatch', 'target': target, 'capacity': admission})
        job = dispatch(jobroot, guard, resume=job['status'] == 'PAUSED')
        save({'action': 'slice_return', 'job_status': job['status'], 'finish': job.get('finish')})
    finish = read(authenticate(job['finish']))
    assert finish['status'] == 'EXPOSURE_COMPLETE' and finish['local_updates'] == target
    capacity(6 * 1024**3)
    native_root = root / ('native-%d' % target)
    config = {**read(W / 'fresh-qualification-config01.json')['native'],
              'model': finish['model'], 'manifest': spec['manifest']}
    cp = root / ('native-config-%d.json' % target)
    atomic_json(cp, config)
    ng = {**guard, 'inputs': list(guard['inputs']) + [pin(cp), finish['model'], config['helper'], job['finish']]}
    ng.update(output=str(root / ('native-guard-%d' % target)), max_train_seconds=0,
              wall_seconds=1200, progress_paths=[str(native_root / 'native-0000/rows.jsonl')])
    ngp = root / ('native-guard-config-%d.json' % target)
    atomic_json(ngp, ng)
    subprocess.run([sys.executable, '-B', str(S / 'scripts/shortlist_pilots/guard.py'),
                    '--config', str(ngp), '--', sys.executable, '-B',
                    str(S / 'scripts/training_system/native.py'), '--config', str(cp),
                    '--output', str(native_root), '--batch-cases', '144'], check=True)
    nr = pin(native_root / 'result.json')
    metrics, proof = verified_metrics(nr, model=finish['model'], manifest=spec['manifest'])
    state['native'][str(target)] = nr
    save({'action': 'native_complete_independently_verified', 'target': target,
          'metrics': metrics, 'proof': proof})
    print('FRESH_NATIVE_COMPLETE', target, metrics['development']['prefix5_sum'] / metrics['development']['windows'], flush=True)
else:
    state['status'] = 'COMPLETE'
    save({'action': 'all_planned_milestones_complete'})
print('FINITE_FRESH_CURVE_RETURN', state['status'], flush=True)
