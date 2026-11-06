"""Supervised finite development screen; completion wakes the root review."""
from pathlib import Path
import json
import subprocess
import sys
import time

project = Path('/home/philip/binary-eagle-decoding')
run = Path(__file__).resolve().parent
command = [sys.executable, '-B', str(project / 'experiments/qwen35-correctness-matrix-20261101/run_matrix.py'),
           '--project', str(project), '--run', str(run), '--manifest', str(run / 'screen-freeze.json'),
           '--inputs', str(run / 'screen-inputs'), '--phase', 'screen', '--output', 'screen-01']
start = time.perf_counter()
try:
    result = subprocess.run(command, cwd=project)
    (run / 'screen-launch-receipt.json').write_text(json.dumps({'command': command, 'exit_code': result.returncode,
        'process_wall_seconds': time.perf_counter() - start, 'training_seconds': 0}, indent=2) + '\n')
    raise SystemExit(result.returncode)
finally:
    subprocess.run(['tmux', '-L', 'binary-eagle-runtime', 'wait-for', '-S', 'qwen35-matrix-screen-20261101-complete'], check=False)
