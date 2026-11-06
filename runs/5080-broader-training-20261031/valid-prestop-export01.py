"""One predeclared valid checkpoint endpoint, only after main07 quiesces.

Use the unchanged qualified trainer with stop-after zero. Never load the
guard-stopped checkpoint, update an optimizer, or mutate the original job.
"""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
S = R / 'source02'
sys.path[:0] = [str(S / 'scripts'), str(P / 'scripts')]
from training_system.io import read, pin, authenticate, atomic_json, digest
from training_system.train import continuation_contract

assert len(sys.argv) == 2 and sys.argv[1] in ['control', 'dpal']
name = sys.argv[1]
supervisor = read(W / 'supervisors/main07/state.json')
assert supervisor['status'] == 'finished' and supervisor['exit_code'] == 0
controller = read(W / 'main-controller02/controller.json')
assert controller['status'] in ['PAUSED', 'COMPLETE', 'BLOCKED'] and controller['owner'] is None
plan = read(R / 'native-analysis-addendum02.json')
assert plan['schema'] == 'broader_native_analysis_contingency_v1'
jobroot = Path(controller['candidates'][name]['job'])
assert jobroot.parent in [W / 'main-controller01/jobs', W / 'main-controller02/jobs']
job = read(jobroot / 'job.json')
assert job['status'] == 'GUARD_STOPPED' and job['owner'] is None
key = digest(continuation_contract(job['spec']))
closed = []
for attempt in job['attempts']:
    out = Path(attempt['output'])
    assert out.parent == jobroot
    finish = read(out / 'finish.json')
    assert finish['contract_sha256'] == key
    closed.append((attempt, finish))
first_stop = next((a, f) for a, f in closed if f['status'] == 'ORIGINAL_GUARD_STOP')
eligible = [(a, f) for a, f in closed
            if f['status'] in ['PAUSED', 'EXPOSURE_COMPLETE']
            and 0 < f['local_updates'] < first_stop[1]['local_updates']
            and Path(f['checkpoint']['path']).exists()]
assert eligible, 'no retained valid positive-update pre-stop checkpoint'
attempt, selected = max(eligible, key=lambda row: row[1]['local_updates'])
guard_result = read(Path(attempt['guard_output']) / 'result.json')
assert guard_result['returncode'] == 0 and not guard_result.get('error')
checkpoint = selected['checkpoint']
authenticate(checkpoint)
for ref in controller['candidates'][name]['native'].values():
    native = read(authenticate(ref))
    if selected.get('model') and native['config']['model'] == selected['model']:
        atomic_json(R / ('valid-prestop-' + name + '-already-evaluated.json'),
                    {'selected_finish': pin(Path(attempt['output']) / 'finish.json'),
                     'native_result': ref, 'reason': 'highest valid endpoint already evaluated; no extra endpoint'})
        print('HIGHEST_VALID_ENDPOINT_ALREADY_EVALUATED', flush=True)
        raise SystemExit(0)

keeper = read(R / 'interop-keeper02.json')
assert Path(keeper['interop']).exists()
os.environ['WSL_INTEROP'] = keeper['interop']
root = W / ('valid-prestop-' + name + '01')
root.mkdir(exist_ok=False)
spec = {**job['spec'], 'checkpoint_root': str(root / 'checkpoints'),
        'export_policy': 'every_invocation', 'job_tag': 'valid-prestop-zero-export',
        'stop_path': str(root / 'STOP'), 'deadline_unix': 1793510100}
assert digest(continuation_contract(spec)) == key
sp = root / 'spec.json'
atomic_json(sp, spec)
base_guard = read(W / 'guard04.json')


def guarded(label, command, *, inputs, progress):
    guard = {**base_guard, 'env': {**base_guard.get('env', {}), 'WSL_INTEROP': keeper['interop']}}
    guard.update(output=str(root / (label + '-guard')), max_train_seconds=0,
                 wall_seconds=1200, deadline_unix=1793510100,
                 startup_seconds=600, stall_seconds=600, progress_paths=progress)
    guard['inputs'] = list(guard['inputs']) + inputs + [pin(__file__), pin(R / 'controller02/guard.py')]
    gp = root / (label + '-guard-config.json')
    atomic_json(gp, guard)
    subprocess.run([sys.executable, '-B', str(R / 'controller02/guard.py'),
                    '--config', str(gp), '--', *command], check=True)
    receipt = read(root / (label + '-guard/result.json'))
    assert receipt['returncode'] == 0 and not receipt.get('error')


training = root / 'zero-export'
guarded('export', [sys.executable, '-B', str(S / 'scripts/training_system/train.py'),
                  '--spec', str(sp), '--output', str(training), '--allow-cuda',
                  '--resume', checkpoint['path'], '--stop-after', '0'],
        inputs=[pin(sp), checkpoint, pin(Path(attempt['output']) / 'finish.json'),
                pin(Path(first_stop[0]['output']) / 'finish.json'), pin(R / 'native-analysis-addendum02.json')],
        progress=[str(training / 'progress.json'), str(training / 'latest.json')])
finish = read(training / 'finish.json')
assert finish['status'] == 'PAUSED' and finish['local_updates'] == selected['local_updates']
assert finish['invocation_updates'] == finish['invocation_TRAIN_seconds'] == 0
assert finish['contract_sha256'] == key
authenticate(finish['checkpoint'])
authenticate(finish['model'])
# Exact state equality is independent of the producer's zero-update counter.
from training_system.checkpoints import load_private
from shortlist_pilots.binary.parent_loader import equal
left = load_private(checkpoint)
right = load_private(finish['checkpoint'])
assert equal(left, right), 'zero-update exporter changed complete checkpoint state'
del left, right

qc = read(W / 'qualification-config04.json')
cfg = {**qc['native'], 'model': finish['model'], 'manifest': spec['manifest']}
cp = root / 'native-config.json'
atomic_json(cp, cfg)
native = root / 'native'
guarded('native', [sys.executable, '-B', str(S / 'scripts/training_system/native.py'),
                  '--config', str(cp), '--output', str(native), '--batch-cases', '144'],
        inputs=[pin(cp), finish['model'], cfg['helper'], pin(training / 'finish.json')],
        progress=[str(native / 'native-0000/rows.jsonl')])
result = read(native / 'result.json')
assert result['config']['model'] == finish['model'] and result['config']['manifest'] == spec['manifest']
receipt = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope': 'One highest retained valid pre-stop checkpoint, predeclared addendum02; no search or training retry.',
           'objective': name, 'local_updates': finish['local_updates'], 'optimizer_updates': 0,
           'TRAIN_seconds': 0., 'complete_state_unchanged': True,
           'selected_original_finish': pin(Path(attempt['output']) / 'finish.json'),
           'original_checkpoint': checkpoint, 'original_guard_stop_finish': pin(Path(first_stop[0]['output']) / 'finish.json'),
           'zero_export_finish': pin(training / 'finish.json'),
           'model': finish['model'], 'native_result': pin(native / 'result.json'),
           'plan': pin(R / 'native-analysis-addendum02.json'),
           'guard_stopped_checkpoint_loaded': False}
atomic_json(R / ('valid-prestop-' + name + '-receipt01.json'), receipt)
print(json.dumps(receipt), flush=True)
