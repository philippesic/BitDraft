"""Retire one redundant PC raw file only after exact Mac preservation.

The selected fit/teacher/native inputs, original capture receipts, and all model
states stay on PC. Restore raw capture from the recorded Mac path before reuse.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
receipt = json.loads(sys.argv[1])
source = receipt['source']
p = W / 'capture05/capture.quality_logits.f32'
assert source['path'] == str(p) and receipt['byte_exact'] is True
assert receipt['destination_sha256'] == source['sha256']
assert receipt['destination_bytes'] == source['bytes']
assert receipt['destination'] == '/Users/pippo/github/binary-eagle-decoding/runs/5080-broader-training-20261031/preserved-pc-capture01/capture.quality_logits.f32'
assert not p.is_symlink() and p.stat().st_nlink == 1
complete = json.loads((p.parent / 'complete.json').read_text())
release = json.loads((p.parent / 'release.json').read_text())
assert complete['status'] == 'SERVING_CAPTURE_COMPLETE' and len(complete['records']) == 288
assert release == {'reaped': True, 'returncode': 0}
assert complete['files'][p.name] == {'path': str(p), 'sha256': source['sha256']}
assert p.stat().st_size == source['bytes'] and p.stat().st_mtime_ns == source['mtime_ns']
assert hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() == source['sha256']
for name in ['guard04.json', 'base-spec04.json', 'qualification-config04.json']:
    assert str(p) not in (W / name).read_text()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        for fd in (proc / 'fd').iterdir():
            try:
                assert os.readlink(fd) != str(p), 'raw capture still has a live file consumer'
            except (FileNotFoundError, PermissionError):
                pass
    except (FileNotFoundError, PermissionError):
        pass
# Compression may still be processing a different immutable file; never retire
# its current input. All source content remains unchanged by that operation.
rows = [json.loads(x) for x in (R / 'owned-file-compression02.jsonl').read_text().splitlines()]
assert not any(x.get('path') == str(p) for x in rows if x['event'] == 'child_started')
os.environ['WSL_INTEROP'] = json.loads((R / 'interop-keeper02.json').read_text())['interop']
ps = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'

def free():
    return {'nativeC': int(subprocess.check_output([ps, '-NoProfile', '-Command',
                                                   '(Get-Volume -DriveLetter C).SizeRemaining'], text=True).strip()),
            'WSL': shutil.disk_usage(R).free}

before = free()
p.unlink()
after = free()
record = {'schema': 'verified_offdevice_closed_raw_capture_preservation_v1',
          'unix': time.time(), 'source': source, 'preserved_mac_copy': receipt,
          'restore_before_raw_capture_reuse': True,
          'raw_capture_receipts_and_selected_training_inputs_preserved_on_PC': True,
          'valid_or_paid_checkpoint_removed': False, 'before': before, 'after': after,
          'observed_free_delta': {k: after[k] - before[k] for k in before}}
target = R / 'raw-capture-offdevice-preservation01.json'
assert not target.exists()
target.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
