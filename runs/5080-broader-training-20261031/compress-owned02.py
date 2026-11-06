"""Finite byte-preserving per-file compression of closed, owned evidence.

No path deletion, directory inheritance, OS/volume/VHDX operation or model edit.
"""
from pathlib import Path
import collections
import hashlib
import json
import mmap
import os
import shutil
import subprocess
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
os.environ['WSL_INTEROP'] = json.loads((R / 'interop-keeper02.json').read_text())['interop']
for name in ['main07', 'request-final01', 'zero-export01', 'narrow-reference01']:
    state = json.loads((W / 'supervisors' / name / 'state.json').read_text())
    assert state['status'] == 'finished' and state['exit_code'] == 0
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()

def free():
    ps = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    native = int(subprocess.check_output([ps, '-NoProfile', '-Command', '(Get-Volume -DriveLetter C).SizeRemaining'], text=True).strip())
    return {'nativeC': native, 'WSL': shutil.disk_usage(R).free}

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()

receipt = R / 'owned-file-compression02.jsonl'
assert not receipt.exists()
prior = json.loads((R / 'owned-file-compression01.jsonl').read_text().splitlines()[0])
inventory = prior['files']
# Process the single18GB capture last, after smaller files establish its scratch
# allowance. The original failed admission changed no file.
inventory.sort(key=lambda x: (x['bytes'] > 8 * 1024**3, -x['bytes'], x['path']))
with receipt.open('x') as f:
    f.write(json.dumps({'event': 'inventory', 'unix': time.time(), 'files': inventory, 'before': free(),
                       'prior_admission_failure': str(R / 'owned-file-compression01.jsonl')}) + '\n')
for item in inventory:
    if time.time() > 1793496600:  # 01:30 UTC: finite storage operation, then review.
        break
    p = Path(item['path'])
    st = p.stat()
    assert st.st_ino == item['inode'] and st.st_size == item['bytes'] and st.st_mtime_ns == item['mtime_ns'] and st.st_nlink == 1
    before = free()
    # Compression may require a temporary representation of this one file.
    if before['nativeC'] <= 10 * 1024**3 + st.st_size + 1024**3:
        continue
    digest = sha(p)
    win = subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
    started = time.time()
    command = ['/mnt/c/Windows/System32/compact.exe', '/C', '/Q', '/EXE:LZX', win]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with receipt.open('a') as f:
        f.write(json.dumps({'event': 'child_started', 'path': str(p), 'sha256': digest, 'pid': child.pid, 'unix': started}) + '\n')
    try:
        stdout, stderr = child.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        child.terminate()
        stdout, stderr = child.communicate(timeout=30)
        raise RuntimeError('Finite per-file compression timeout; inspect recorded child and file')
    assert child.returncode == 0, (child.returncode, stdout, stderr)
    assert p.stat().st_ino == item['inode'] and p.stat().st_size == item['bytes']
    assert sha(p) == digest
    with p.open('rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            assert hashlib.sha256(m).hexdigest() == digest
    after = free()
    row = {'event': 'compressed_verified', **item, 'sha256': digest, 'before': before, 'after': after,
           'observed_free_delta': {k: after[k] - before[k] for k in before}, 'seconds': time.time() - started,
           'returncode': child.returncode, 'stdout': stdout, 'stderr': stderr, 'hash_and_mmap_exact': True}
    with receipt.open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps({k: row[k] for k in ['path', 'bytes', 'observed_free_delta', 'seconds']}), flush=True)
    # Enough for unchanged qualification reservation after existing cache reuse,
    # plus 0.5GiB extra assistant margin; no need to mutate further files.
    if after['nativeC'] >= 32_285_208_452 - 900_652_732 + 512 * 1024**2:
        break
with receipt.open('a') as f:
    f.write(json.dumps({'event': 'finished', 'unix': time.time(), 'after': free(), 'contents_deleted': False}) + '\n')
