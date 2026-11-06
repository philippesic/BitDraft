"""One predeclared valid fresh pre-guard endpoint; zero optimizer updates."""
from pathlib import Path
import os
import sys
import subprocess

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
S = R / 'source03'
sys.path[:0] = [str(S / 'scripts'), str(P / 'scripts')]
from training_system.io import read, pin, authenticate, atomic_json, digest
from training_system.train import continuation_contract
from training_system.promotion import verified_metrics
from training_system.resources import admit_capacity
import shutil

supervisor = read(W / 'supervisors/fresh-training01/state.json')
assert supervisor['status'] == 'finished' and supervisor['exit_code'] == 0
curve = read(W / 'fresh-training01/state.json')
jobroot = Path(curve['job'])
job = read(jobroot / 'job.json')
assert job['status'] == 'GUARD_STOPPED' and job['owner'] is None
plan = R / 'fresh-restart-plan04.json'
authenticate(pin(plan))
key = digest(continuation_contract(job['spec']))
closed = []
for attempt in job['attempts']:
    f = Path(attempt['output']) / 'finish.json'
    finish = read(f)
    assert finish['contract_sha256'] == key
    closed.append((attempt, finish, pin(f)))
stopped = next(x for x in closed if x[1]['status'] == 'ORIGINAL_GUARD_STOP')
eligible = [x for x in closed if x[1]['status'] in ['PAUSED', 'EXPOSURE_COMPLETE']
            and 0 < x[1]['local_updates'] < stopped[1]['local_updates']
            and Path(x[1]['checkpoint']['path']).exists()]
if not eligible:
    atomic_json(R / 'fresh-valid-prestop-none01.json',
                {'reason': 'No retained positive valid long endpoint; qualified8 reference requires its separately predeclared condition.',
                 'guard_stop': stopped[2], 'qualified_reference_plan': pin(R / 'fresh-prestop-qualified-reference01.json')})
    raise SystemExit(0)
attempt, selected, selected_ref = max(eligible, key=lambda x: x[1]['local_updates'])
for native_ref in curve['native'].values():
    if read(authenticate(native_ref))['config']['model'] == selected.get('model'):
        atomic_json(R / 'fresh-valid-prestop-already-evaluated01.json',
                    {'selected_finish': selected_ref, 'native_result': native_ref, 'additional_updates': 0})
        print('HIGHEST_VALID_FRESH_PRESTOP_ALREADY_EVALUATED', flush=True)
        raise SystemExit(0)
checkpoint = selected['checkpoint']
authenticate(checkpoint)
guard_receipt = read(Path(attempt['guard_output']) / 'result.json')
assert guard_receipt['returncode'] == 0 and not guard_receipt.get('error')
keeper = read(R / 'interop-keeper02.json')
assert Path(keeper['interop']).exists()
os.environ['WSL_INTEROP'] = keeper['interop']
os.environ['PYTHONPATH'] = str(S / 'scripts') + ':' + str(P / 'scripts')
os.environ['CUDA_CACHE_PATH'] = str(W / 'fresh-qualification01/cuda')
root = W / 'fresh-valid-prestop01'
root.mkdir(exist_ok=False)
spec = {**job['spec'], 'checkpoint_root': str(root / 'checkpoints'),
        'export_policy': 'every_invocation', 'stop_path': str(root / 'STOP'),
        'deadline_unix': 1793510100, 'job_tag': 'fresh-valid-prestop-zero-export'}
# A zero-update export writes one <=4GiB checkpoint and <=2GiB export scratch.
# Retain the original8GiB attempt cap,12GiB write cap and10GiB floor. This
# distinct operational reservation covers the full attempt; numeric state stays
# bound to the original qualified continuation contract.
spec['storage_budget'] = {**spec['storage_budget'], 'volumes': {
    name: {**volume, 'reservation_bytes': 8 * 1024**3}
    for name, volume in spec['storage_budget']['volumes'].items()}}
assert digest(continuation_contract(spec)) == key
admission = admit_capacity({'nativeC': {'free_bytes': shutil.disk_usage(W).free,
                                       'floor_bytes': 10 * 1024**3,
                                       'reservation_bytes': 8 * 1024**3}})
sp = root / 'spec.json'
atomic_json(sp, spec)
atomic_json(root / 'zero-update-capacity.json', admission)
base_guard = read(W / 'fresh-qualification-guard01.json')

def guarded(label, argv, inputs, progress):
    guard = {**base_guard, 'inputs': list(base_guard['inputs']) + inputs + [pin(__file__), pin(plan)]}
    guard.update(output=str(root / (label + '-guard')), max_train_seconds=0,
                 wall_seconds=1200, deadline_unix=1793510100,
                 startup_seconds=600, stall_seconds=600, progress_paths=progress)
    gp = root / (label + '-guard-config.json')
    atomic_json(gp, guard)
    subprocess.run([sys.executable, '-B', str(S / 'scripts/shortlist_pilots/guard.py'),
                    '--config', str(gp), '--', *argv], check=True)
    receipt = read(root / (label + '-guard/result.json'))
    assert receipt['returncode'] == 0 and not receipt.get('error')

training = root / 'zero-export'
guarded('export', [sys.executable, '-B', str(S / 'scripts/training_system/train.py'),
                   '--spec', str(sp), '--output', str(training), '--allow-cuda',
                   '--resume', checkpoint['path'], '--stop-after', '0'],
        [pin(sp), checkpoint, selected_ref, stopped[2]],
        [str(training / 'progress.json'), str(training / 'latest.json')])
finish = read(training / 'finish.json')
assert finish['status'] == 'PAUSED' and finish['local_updates'] == selected['local_updates']
assert finish['invocation_updates'] == finish['invocation_TRAIN_seconds'] == 0
from training_system.checkpoints import load_private
from shortlist_pilots.binary.parent_loader import equal
left, right = load_private(checkpoint), load_private(finish['checkpoint'])
assert equal(left, right)
del left, right
cfg = {**read(W / 'fresh-qualification-config01.json')['native'],
       'model': finish['model'], 'manifest': spec['manifest']}
cp = root / 'native-config.json'
atomic_json(cp, cfg)
native = root / 'native'
guarded('native', [sys.executable, '-B', str(S / 'scripts/training_system/native.py'),
                   '--config', str(cp), '--output', str(native), '--batch-cases', '144'],
        [pin(cp), finish['model'], cfg['helper'], pin(training / 'finish.json')],
        [str(native / 'native-0000/rows.jsonl')])
nr = pin(native / 'result.json')
metrics, proof = verified_metrics(nr, model=finish['model'], manifest=spec['manifest'])
receipt = {'scope': 'One highest retained valid fresh checkpoint preceding original guard, predeclared before long results.',
           'original_checkpoint': checkpoint, 'selected_original_finish': selected_ref,
           'original_guard_stop_finish': stopped[2], 'zero_export_finish': pin(training / 'finish.json'),
           'model': finish['model'], 'native_result': nr, 'local_updates': finish['local_updates'],
           'optimizer_updates': 0, 'TRAIN_seconds': 0, 'complete_state_unchanged': True,
           'guard_stopped_checkpoint_loaded': False, 'native_proof': proof}
atomic_json(R / 'fresh-valid-prestop-receipt01.json', receipt)
print('FRESH_VALID_PRESTOP_NATIVE_COMPLETE', finish['local_updates'], flush=True)
