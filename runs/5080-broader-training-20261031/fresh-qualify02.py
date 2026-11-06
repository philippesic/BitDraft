"""Conditional source03 bootstrap and actual qualification, serial storage review02; no launch on import.

Requires the independently recorded serving-gate decision. Existing paid
trajectories are read-only. This does not reuse their qualification proof.
"""
from pathlib import Path
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
S = R / 'source03'
sys.path[:0] = [str(S / 'scripts'), str(P / 'scripts')]
os.environ['PYTHONPATH'] = str(S / 'scripts') + ':' + str(P / 'scripts')
from training_system.io import read, pin, authenticate, atomic_json
from training_system.lane import locked
from training_system.resources import qualification_budget, qualification_layout, qualification_usage, qualification_capacity, admit_capacity

decision = read(R / 'fresh-allocation-decision02.json')
assert decision['action'] == 'qualify_genuine_fresh_full48'
assert decision['training_deadline_unix'] == 1793507400
assert decision['request_summary'] == pin(W / 'request-final01/summary.json')
assert shutil.disk_usage(W).free >= decision['whole_operation_required_nativeC_bytes'] + 2 * 1024**3  # extra long-run JIT allowance
assert time.time() + 2.5 * 3600 <= 1793507400
for name in ['main07', 'request-final01']:
    state = read(W / 'supervisors' / name / 'state.json')
    assert state['status'] == 'finished' and state['exit_code'] == 0
assert read(W / 'main-controller02/controller.json')['owner'] is None
keeper = read(R / 'interop-keeper02.json')
assert Path(keeper['interop']).exists()
os.environ['WSL_INTEROP'] = keeper['interop']
prepared = read(R / 'source03-preparation.json')
assert prepared['source'] == pin(S / 'manifest.json')
manifest = read(authenticate(prepared['source']))
for item in manifest['files']:
    assert pin(S / item['path'])['sha256'] == item['sha256']

base = read(W / 'base-spec04.json')
def rebase(value):
    if isinstance(value, dict):
        if set(value) >= {'path', 'sha256'} and value['path'].startswith(str(R / 'source02') + '/'):
            return {**value, **pin(S / Path(value['path']).relative_to(R / 'source02'))}
        return {k: rebase(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase(v) for v in value]
    return value
spec = rebase(base)
for key in ['model_cache', 'model_cache_producer']:
    spec.pop(key, None)
spec['sources'][str(S / 'scripts/training_system/fresh_base.py')] = pin(S / 'scripts/training_system/fresh_base.py')
GiB = 1024 ** 3
spec.update(objective='control', target_updates=8, deadline_unix=1793507400,
            max_invocation_train_seconds=60, checkpoint_interval=0,
            checkpoint_root=str(W / 'fresh-qualification-checkpoints01'),
            job_tag='fresh-full48-qualification01', stop_path=str(W / 'STOP'),
            retention_policy='latest_verified', retention_references_complete=True,
            retention_references=[], export_policy='milestones')
spec['storage_budget']['volumes']['nativeC']['reservation_bytes'] = 12 * GiB
qc = read(W / 'qualification-config04.json')
budget = qualification_budget(spec, windows=len(read(authenticate(spec['manifest']))['windows']),
                              model_bytes=Path(spec['parent']['model']['path']).stat().st_size,
                              objectives=1, crash_recovery=True,
                              memory_processes_per_objective=4,
                              additional_training_processes=0,
                              split_before_continuous=True)
layout = qualification_layout(budget, execution_path=W / 'fresh-qualification-jobs01',
                              host_path=Path('/mnt/c'),
                              checkpoint_path=Path(spec['checkpoint_root']),
                              execution_backed_by_host=True, same_volume=True)
# Bootstrap is serial and creates one GGUF; it fits inside the qualifier's
# existing three-export bound. Preserve all original checkpoint/native/JIT bounds.
# Reuse the closed cache by moving its actual bytes into this owned ledger.
output = W / 'fresh-qualification01'
assert not output.exists()
cache = W / 'supervisors/main07/cuda'
cache_records = []
for path in cache.rglob('*'):
    if path.is_file():
        cache_records.append({'relative': str(path.relative_to(cache)), **pin(path)})
cache_bytes = sum(Path(x['path']).stat().st_size for x in cache_records)
assert cache_bytes >= 900_000_000
admit_capacity({'nativeC': {'free_bytes': shutil.disk_usage(W).free,
                           'floor_bytes': 10 * GiB,
                           'reservation_bytes': layout['volumes']['host']['reservation_bytes'] - cache_bytes},
                'WSL': {'free_bytes': shutil.disk_usage(R).free,
                        'floor_bytes': 2 * GiB, 'reservation_bytes': GiB // 4}})
output.mkdir(exist_ok=False)
cache.rename(output / 'cuda')
for item in cache_records:
    assert pin(output / 'cuda' / item['relative'])['sha256'] == item['sha256']
os.environ['CUDA_CACHE_PATH'] = str(output / 'cuda')
root = output / 'bootstrap'
root.mkdir(exist_ok=False)
usage = qualification_usage(layout, output, [], [])
capacity = qualification_capacity(layout, usage, free_bytes={'host': shutil.disk_usage(W).free})
atomic_json(root / 'preflight-capacity.json', {
    'qualification_layout': layout, 'admission': capacity,
    'serial_phase_reason': 'Bootstrap creates one GGUF, within existing three-export bound. No concurrent training or native outputs.',
    'cache_reuse': {'from': str(cache), 'to': str(output / 'cuda'), 'bytes': cache_bytes, 'files': cache_records},
    'no_qualification_gate_or_bound_reduced': True,
    'whole_operation_required_bytes': decision['whole_operation_required_nativeC_bytes'],
    'additional_long_JIT_allowance_bytes': 2 * GiB})
controller = read(W / 'main-controller02/controller.json')
job = read(Path(controller['candidates']['control']['job']) / 'job.json')
contract = Path(job['attempts'][-1]['output']) / 'runtime-contract.json'
binding = {'schema': 'released_bf16_full48_serving_restart_v1', 'seed': 42,
           'released_model': spec['source_gguf'], 'ordinary_donor': spec['parent']['model'],
           'architecture_contract': pin(contract), 'config': read(contract)['representation']['config']}
with locked(Path.home() / '.config/binary-eagle-decoding/rtx5080-campaign.lock'):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    from training_system.fresh_base import build_cpu
    model, baseline, metadata = build_cpu(binding, root, bootstrap=True)
    del model, baseline, metadata
    import gc
    gc.collect()
binding['native_model'] = pin(root / 'fresh-initial.gguf')
bp = root / 'binding.json'
atomic_json(bp, binding)
spec['fresh_initializer'] = pin(bp)
from training_system.train import validate_spec
validate_spec(spec)
sp = W / 'fresh-base-spec01.json'
atomic_json(sp, spec)
guard = rebase(read(W / 'guard04.json'))
guard.update(deadline_unix=1793507400, wall_seconds=1800,
             startup_seconds=600, stall_seconds=600,
             stop_paths=[str(W / 'STOP')], max_train_seconds=840)
guard['env'] = {**guard['env'], 'PYTHONPATH': os.environ['PYTHONPATH'], 'WSL_INTEROP': keeper['interop'], 'CUDA_CACHE_PATH': str(output / 'cuda')}
guard['inputs'] += [pin(bp), binding['native_model'], pin(__file__), pin(R / 'fresh-allocation-decision02.json'), prepared['source']]
guard['training_package_manifest'] = prepared['source']
guard['dispatcher_source'] = pin(S / 'scripts/training_system/lane.py')
guard['guard_launcher'] = pin(S / 'scripts/shortlist_pilots/guard.py')
gp = W / 'fresh-qualification-guard01.json'
atomic_json(gp, guard)
config = {**qc, 'cli': str(S / 'scripts/training_system/cli.py'), 'spec': spec,
          'guard_template': str(gp), 'run_root': str(W / 'fresh-qualification-jobs01'),
          'objectives': ['control'], 'crash_recovery': True,
          'complete_pair_memory_probe': True, 'max_memory_probe_processes_per_objective': 4,
          'maximum_pause_continuations': 0}
cp = W / 'fresh-qualification-config01.json'
atomic_json(cp, config)
atomic_json(root / 'qualification-launch.json',
            {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source': prepared['source'], 'binding': pin(bp), 'config': pin(cp),
             'decision': pin(R / 'fresh-allocation-decision02.json'),
             'qualification_reused': False, 'long_training_started': False})
# Seal preparation in the ordinary qualifier format. --resume enters the
# unchanged run() with no prior progress/results/proofs, so all actual tests run.
assert not (output / 'progress.json').exists() and not (output / 'result.json').exists()
atomic_json(output / 'qualification-config.json', {'config': config, 'started_unix': time.time()})
qualification_capacity(layout, qualification_usage(layout, output, [], []),
                       free_bytes={'host': shutil.disk_usage(W).free})
print('GENUINE_FRESH_BOOTSTRAP_COMPLETE; ACTUAL_NEW_QUALIFICATION_STARTING', flush=True)
subprocess.run([sys.executable, '-B', str(S / 'scripts/training_system/qualify.py'),
                '--config', str(cp), '--output', str(output), '--resume'], check=True)
from training_system.device_proofs import _proofs
_proofs(pin(W / 'fresh-qualification01/result.json'), 'control')
print('GENUINE_FRESH_ACTUAL_QUALIFICATION_COMPLETE', flush=True)
