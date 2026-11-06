"""Seal closed phase2 native and HTTP evidence for independent Mac receipt."""
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
state = read(W / 'supervisors/request-final02/state.json')
assert state['status'] == 'finished' and state['exit_code'] == 0
summary = read(W / 'request-final02/summary.json')
assert summary['diagnostic'] is False
assert all(c['requests'] == 120 for c in summary['panels']['development']['cells'].values())
config = read(W / 'request-final-config02.json')
root = R / 'frozen-receipts13'
assert not root.exists() and not (R / 'receipt-source13.json').exists()
rows = [json.loads(line) for line in (W / 'request-final02/requests.jsonl').read_text().splitlines()]
assert len(rows) == 720
root.mkdir()
archive = root / 'phase2-http.tar.gz'
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
                    source = W / 'request-final02' / relative
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
(root / 'phase2-http-index.json').write_text(json.dumps({'files': index, 'archive': pin(archive)}, indent=2) + '\n')
refs = {
    'phase2-selection.json': R / 'phase2-selection06.json',
    'phase2-serving-design02.json': R / 'phase2-serving-design02.json',
    'evaluation-guard-transition01.json': R / 'evaluation-guard-transition01.json',
    'evaluation-guard-negative01.json': R / 'evaluation-guard-negative01.json',
    'failed-native02-supervisor.json': W / 'supervisors/fresh-finish02/state.json',
    'failed-native02-log.txt': W / 'supervisors/fresh-finish02/stdout.log',
    'failed-native03-supervisor.json': W / 'supervisors/fresh-finish03/state.json',
    'failed-native03-log.txt': W / 'supervisors/fresh-finish03/stdout.log',
    'evaluation-guard01.py': R / 'evaluation-guard01.py',
    'evaluation-renewed-allocation01.json': R / 'evaluation-renewed-allocation01.json',
    'prior-native-completion-interrupted.json': W / 'supervisors/fresh-finish01/state.json',
    'fresh-training-launch.json': R / 'fresh-training-supervisor-launch01.json',
    'phase2-request-config.json': W / 'request-final-config02.json',
    'phase2-request-summary.json': W / 'request-final02/summary.json',
    'phase2-requests.jsonl': W / 'request-final02/requests.jsonl',
    'phase2-request-guard.json': W / 'request-final-guard02/result.json',
    'phase2-request-supervisor.json': W / 'supervisors/request-final02/state.json',
    'fresh-training-state.json': W / 'fresh-training01/state.json',
    'fresh-native-only-completion.json': R / 'fresh-training-completion-recovery04.json',
    'fresh-native-only-supervisor.json': W / 'supervisors/fresh-finish04/state.json',
    'fresh-native-interrupted-state.json': R / 'fresh-training-interrupted-state04.json',
    'finish-fresh-native04.py': R / 'finish-fresh-native04.py',
    'fresh-native-recovery-guard.json': W / 'fresh-training01/native-guard-14400-recovery04/result.json',
    'fresh-training-supervisor.json': W / 'supervisors/fresh-training01/state.json',
    'request-evaluator01.py': R / 'request-evaluator01.py',
    'request-final09.py': R / 'request-final09.py',
    'freeze-phase2-selection06.py': R / 'freeze-phase2-selection06.py',
    'freeze-phase2-receipts13g.py': Path(__file__),
}
curve = read(W / 'fresh-training01/state.json')
assert read(W / 'supervisors/fresh-finish04/state.json')['exit_code'] == 0
for target, ref in curve['native'].items():
    assert sha(ref['path']) == ref['sha256']
    refs['fresh-native-' + target + '.json'] = Path(ref['path'])
pre = R / 'fresh-valid-prestop-receipt01.json'
if pre.exists():
    receipt = read(pre)
    refs['fresh-prestop-receipt.json'] = pre
    refs['fresh-prestop-native.json'] = Path(receipt['native_result']['path'])
    refs['fresh-prestop-finish.json'] = Path(receipt['zero_export_finish']['path'])
panel = config['panels']['development']
for key in ['protocol', 'prompts', 'target_reference']:
    ref = panel[key]
    assert sha(ref['path']) == ref['sha256']
    refs['phase2-' + key + '.json'] = Path(ref['path'])
job = read(Path(curve['job']) / 'job.json')
assert job['owner'] is None and job['status'] != 'RUNNING'
refs['fresh-closed-job.json'] = Path(curve['job']) / 'job.json'
for attempt in job['attempts']:
    stem = 'fresh-attempt%03d' % attempt['number']
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
for name in ['phase2-http.tar.gz', 'phase2-http-index.json']:
    dest = root / name
    assert dest.stat().st_size <= 8 * 1024 ** 2
    value = sha(dest)
    inventory['hashes'][name] = value
    inventory['files'][name] = {'source': str(dest), 'staged': str(dest), 'bytes': dest.stat().st_size, 'sha256': value}
assert len(inventory['files']) <= 64 and sum(v['bytes'] for v in inventory['files'].values()) <= 64 * 1024 ** 2
with (R / 'receipt-source13.json').open('x') as f:
    json.dump(inventory, f, indent=2)
print({'packet': '13', 'files': len(inventory['files']), 'bytes': sum(v['bytes'] for v in inventory['files'].values()), 'HTTP_files': len(index)}, flush=True)
