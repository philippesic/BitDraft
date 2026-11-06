"""Seal completed fresh qualification, never live training outputs."""
from pathlib import Path
import hashlib
import json
import shutil
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
Q = W / 'fresh-qualification01'
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
state = read(W / 'supervisors/fresh-qualify01/state.json')
assert state['status'] == 'finished' and state['exit_code'] == 0
refs = {
    'fresh-qualification-result.json': Q / 'result.json',
    'fresh-qualification-supervisor.json': W / 'supervisors/fresh-qualify01/state.json',
    'fresh-qualification-config.json': W / 'fresh-qualification-config01.json',
    'fresh-qualification-guard.json': W / 'fresh-qualification-guard01.json',
    'fresh-base-spec.json': W / 'fresh-base-spec01.json',
    'fresh-bootstrap-binding.json': Q / 'bootstrap/binding.json',
    'fresh-initialization.json': Q / 'bootstrap/fresh-initialization.json',
    'fresh-preflight-capacity.json': Q / 'bootstrap/preflight-capacity.json',
    'fresh-state-parity.json': Q / 'control-state-parity.json',
    'fresh-native-parity.json': Q / 'control-native-parity.json',
    'fresh-memory-coverage.json': Q / 'control-memory-coverage.json',
    'fresh-continuous-native.json': Q / 'control-continuous-native/result.json',
    'fresh-resumed-native.json': Q / 'control-resumed-native/result.json',
    'fresh-restart-plan04.json': R / 'fresh-restart-plan04.json',
    'fresh-prestop-qualified-reference01.json': R / 'fresh-prestop-qualified-reference01.json',
    'fresh-qualify02.py': R / 'fresh-qualify02.py',
    'train-fresh02.py': R / 'train-fresh02.py',
    'freeze-fresh-qualification12.py': Path(__file__),
}
for folder in [Q / ('control-memory-%03d' % i) for i in range(4)]:
    assert (folder / 'memory-probe.json').exists()
    refs[folder.name + '-guard.json'] = Q / (folder.name + '-guard/result.json')
    for name in ['memory-probe.json', 'progress.json']:
        refs[folder.name + '-' + name] = folder / name
for number, folder in enumerate(sorted(p.parent for p in (W / 'fresh-qualification-jobs01/jobs').glob('*/job.json'))):
    job = read(folder / 'job.json')
    assert job['status'] in ['COMPLETE', 'RETIRED_QUALIFICATION'] and job['owner'] is None
    for attempt in job['attempts']:
        stem = 'fresh-qjob%d-attempt%03d' % (number, attempt['number'])
        for name in ['finish.json', 'runtime-contract.json']:
            path = Path(attempt['output']) / name
            if path.exists():
                refs[stem + '-' + name] = path
        refs[stem + '-guard.json'] = Path(attempt['guard_output']) / 'result.json'
root = R / 'frozen-receipts12'
assert not root.exists() and not (R / 'receipt-source12.json').exists()
assert len(refs) <= 64
root.mkdir()
inventory = {'observed_unix': time.time(), 'hashes': {}, 'files': {}}
for name, source in refs.items():
    assert source.is_file() and not source.is_symlink() and source.stat().st_size <= 8 * 1024**2
    value = sha(source)
    target = root / name
    shutil.copyfile(source, target)
    assert sha(target) == sha(source) == value
    inventory['hashes'][name] = value
    inventory['files'][name] = {'source': str(source), 'staged': str(target),
                              'bytes': target.stat().st_size, 'sha256': value}
assert sum(v['bytes'] for v in inventory['files'].values()) <= 64 * 1024**2
with (R / 'receipt-source12.json').open('x') as f:
    json.dump(inventory, f, indent=2)
print({'packet': '12', 'files': len(refs),
       'bytes': sum(v['bytes'] for v in inventory['files'].values())}, flush=True)
