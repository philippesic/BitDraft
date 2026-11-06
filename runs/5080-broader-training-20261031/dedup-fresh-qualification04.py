"""Share only byte-identical closed fresh qualification raw arrays."""
from pathlib import Path
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import time

R = Path('/home/philip/binary-eagle-decoding/runs/5080-broader-training-20261031')
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
Q = W / 'fresh-qualification01'
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def identity(p):
    s = p.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink)
def free():
    ps = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    native = int(subprocess.check_output([ps, '-NoProfile', '-Command',
                 '(Get-Volume -DriveLetter C).SizeRemaining'], text=True).strip())
    return {'nativeC': native, 'WSL': shutil.disk_usage(R).free}

os.environ['WSL_INTEROP'] = read(R / 'interop-keeper03.json')['interop']
state = read(W / 'supervisors/fresh-qualify01/state.json')
assert state['status'] == 'finished' and state['exit_code'] == 0
curve = read(W / 'fresh-training01/state.json')
assert '4800' in curve['native']
job = read(Path(curve['job']) / 'job.json')
training = read(W / 'supervisors/fresh-finish04/state.json')
assert training['status'] == 'finished' and training['exit_code'] == 0
http = read(W / 'supervisors/request-final02/state.json')
assert http['status'] == 'finished' and http['exit_code'] == 0
assert job['owner'] is None and job['status'] != 'RUNNING'
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
guard = read(W / 'fresh-training01/guard.json')
assert not any('/fresh-qualification01/control-' in ref['path'] for ref in guard['inputs'])
assert not (R / 'native-dedup04.json').exists()
assert not (R / 'native-dedup04-start.json').exists()
proof_path = Q / 'control-native-parity.json'
proof_hash = sha(proof_path)
proof = read(proof_path)
assert proof['exact'] and proof['windows'] == 288
assert len(proof['left_raw']) == len(proof['right_raw']) == 288
records = []
for left, right in zip(proof['left_raw'], proof['right_raw']):
    a, b = Path(left['path']), Path(right['path'])
    assert a.is_relative_to(Q / 'control-continuous-native')
    assert b.is_relative_to(Q / 'control-resumed-native')
    assert not a.is_symlink() and not b.is_symlink()
    assert left['sha256'] == right['sha256'] == sha(a) == sha(b)
    assert a.read_bytes() == b.read_bytes()
    sa, sb = identity(a), identity(b)
    assert sa[0] == sb[0] and sa[2] == sb[2]
    records.append({'left': left, 'right': right, 'left_identity': sa,
                    'right_identity': sb, 'bytes': sa[2]})
before = free()
with (R / 'native-dedup04-start.json').open('x') as f:
    json.dump({'records': records, 'before': before, 'unix': time.time(),
               'pid': os.getpid(), 'birth': Path('/proc/self/stat').read_text().split()[21],
               'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}, f, indent=2)
for row in records:
    a, b = Path(row['left']['path']), Path(row['right']['path'])
    assert identity(a) == tuple(row['left_identity'])
    assert identity(b) == tuple(row['right_identity'])
    if a.stat().st_ino == b.stat().st_ino:
        row['already_shared'] = True
        continue
    assert a.stat().st_nlink == b.stat().st_nlink == 1
    temporary = b.with_name(b.name + '.owned-dedup04-link')
    assert not temporary.exists()
    os.link(a, temporary)
    try:
        assert sha(temporary) == row['right']['sha256']
        os.replace(temporary, b)
    finally:
        if temporary.exists():
            temporary.unlink()
    assert a.stat().st_ino == b.stat().st_ino and a.stat().st_nlink == b.stat().st_nlink == 2
    assert sha(a) == sha(b) == row['left']['sha256']
    with b.open('rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            assert hashlib.sha256(data).hexdigest() == row['right']['sha256']
    row['both_paths_hash_and_mmap_exact'] = True
after = free()
assert sha(proof_path) == proof_hash
receipt = {'schema': 'owned_closed_raw_exact_hardlink_dedup_v1', 'observed_unix': time.time(),
           'scope': 'Only fresh qualification continuous/resumed raw arrays; both paths and contents preserved.',
           'ownership': 'Current finite assignment; qualification supervisor closed0 before operation.',
           'files_consolidated': sum(not r.get('already_shared', False) for r in records),
           'logical_duplicate_payload_bytes': sum(r['bytes'] for r in records if not r.get('already_shared', False)),
           'free_bytes_before': before, 'free_bytes_after': after,
           'nativeC_observed_delta': after['nativeC'] - before['nativeC'],
           'concurrency': 'GPU quiet and fresh training closed; filesystem deltas may still include unrelated OS activity.',
           'records': records, 'no_unique_payload_or_checkpoint_deleted': True}
with (R / 'native-dedup04.json').open('x') as f:
    json.dump(receipt, f, indent=2)
print(json.dumps({k: v for k, v in receipt.items() if k != 'records'}), flush=True)
