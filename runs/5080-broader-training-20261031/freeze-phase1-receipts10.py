"""Seal closed phase1 native and HTTP evidence for independent Mac receipt."""
from pathlib import Path
import collections
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def pin(p):
    return {'path': str(p), 'sha256': sha(p)}
state = read(W / 'supervisors/request-final01/state.json')
assert state['status'] == 'finished' and state['exit_code'] == 0
summary = read(W / 'request-final01/summary.json')
assert summary['diagnostic'] is False
assert all(c['requests'] == 120 for c in summary['panels']['development']['cells'].values())
config = read(W / 'request-final-config01.json')
root = R / 'frozen-receipts10'
assert not root.exists() and not (R / 'receipt-source10.json').exists()
rows = [json.loads(line) for line in (W / 'request-final01/requests.jsonl').read_text().splitlines()]
assert len(rows) == 720
root.mkdir()
archive = root / 'phase1-http.tar.gz'
index = []
counts = collections.Counter()
with archive.open('xb') as output:
    with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for row in rows:
                cell = (row['panel'], row['arm'], row['repetition'])
                assert row['panel'] == 'development' and row['arm'] in config['models']
                assert row['warmup_excluded'] and 0 <= row['repetition'] < 5
                ordinal = counts[cell]
                counts[cell] += 1
                folder = Path(cell[0]) / ('rep-%02d' % cell[2]) / cell[1] / ('prompt-%04d' % ordinal)
                for name in ['request.json', 'response.json', 'metrics-before.prom', 'metrics-after.prom']:
                    relative = folder / name
                    source = W / 'request-final01' / relative
                    before = pin(source)
                    data = source.read_bytes()
                    assert len(data) < 1024 ** 2 and hashlib.sha256(data).hexdigest() == before['sha256']
                    item = tarfile.TarInfo(str(relative))
                    item.size = len(data)
                    item.mode = 0o600
                    item.mtime = 0
                    tar.addfile(item, io.BytesIO(data))
                    assert pin(source) == before
                    index.append({'relative': str(relative), 'source': before, 'bytes': len(data)})
assert len(counts) == 30 and set(counts.values()) == {24}
(root / 'phase1-http-index.json').write_text(json.dumps({'files': index, 'archive': pin(archive)}, indent=2) + '\n')
pre = read(R / 'valid-prestop-dpal-receipt01.json')
refs = {
    'dpal8629-native-result.json': Path(pre['native_result']['path']),
    'dpal8629-valid-receipt.json': R / 'valid-prestop-dpal-receipt01.json',
    'dpal8629-zero-export-finish.json': Path(pre['zero_export_finish']['path']),
    'dpal8629-zero-export-guard.json': W / 'valid-prestop-dpal01/export-guard/result.json',
    'dpal8629-native-guard.json': W / 'valid-prestop-dpal01/native-guard/result.json',
    'dpal8629-runtime-contract.json': W / 'valid-prestop-dpal01/zero-export/runtime-contract.json',
    'narrow32400-native-result.json': W / 'narrow32400-native01/result.json',
    'narrow32400-native-guard.json': W / 'narrow32400-native-guard01/result.json',
    'main07-supervisor.json': W / 'supervisors/main07/state.json',
    'main07-paused-controller.json': W / 'main-controller02/controller.json',
    'serving-gate-controller-pause04.json': R / 'serving-gate-controller-pause04.json',
    'native-dedup02.json': R / 'native-dedup02.json',
    'phase1-selection.json': R / 'final-selection.json',
    'phase1-request-config.json': W / 'request-final-config01.json',
    'phase1-request-summary.json': W / 'request-final01/summary.json',
    'phase1-requests.jsonl': W / 'request-final01/requests.jsonl',
    'phase1-request-guard.json': W / 'request-final-guard01/result.json',
    'phase1-request-supervisor.json': W / 'supervisors/request-final01/state.json',
    'request-evaluator01.py': R / 'request-evaluator01.py',
    'request-final03.py': R / 'request-final03.py',
    'freeze-serving-selection01.py': R / 'freeze-serving-selection01.py',
    'freeze-phase1-receipts10.py': Path(__file__),
}
panel = config['panels']['development']
for key in ['protocol', 'prompts', 'target_reference']:
    ref = panel[key]
    assert sha(ref['path']) == ref['sha256']
    refs['phase1-' + key + '.json'] = Path(ref['path'])
controller = read(W / 'main-controller02/controller.json')
job = read(Path(controller['candidates']['dpal']['job']) / 'job.json')
for attempt in job['attempts']:
    stem = 'dpal-attempt%03d' % attempt['number']
    refs[stem + '-finish.json'] = Path(attempt['output']) / 'finish.json'
    refs[stem + '-guard.json'] = Path(attempt['guard_output']) / 'result.json'
inventory = {'observed_unix': time.time(), 'hashes': {}, 'files': {}}
for name, source in refs.items():
    assert source.is_file() and not source.is_symlink() and source.stat().st_size <= 8 * 1024 ** 2
    before = sha(source)
    dest = root / name
    assert not dest.exists()
    shutil.copyfile(source, dest)
    assert sha(dest) == sha(source) == before
    inventory['hashes'][name] = before
    inventory['files'][name] = {'source': str(source), 'staged': str(dest), 'bytes': dest.stat().st_size, 'sha256': before}
for name in ['phase1-http.tar.gz', 'phase1-http-index.json']:
    dest = root / name
    assert dest.stat().st_size <= 8 * 1024 ** 2
    value = sha(dest)
    inventory['hashes'][name] = value
    inventory['files'][name] = {'source': str(dest), 'staged': str(dest), 'bytes': dest.stat().st_size, 'sha256': value}
assert len(inventory['files']) <= 64 and sum(v['bytes'] for v in inventory['files'].values()) <= 64 * 1024 ** 2
with (R / 'receipt-source10.json').open('x') as f:
    json.dump(inventory, f, indent=2)
print({'packet': '10', 'files': len(inventory['files']), 'bytes': sum(v['bytes'] for v in inventory['files'].values()), 'HTTP_files': len(index)}, flush=True)
