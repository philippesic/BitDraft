"""Restart an interrupted finite supplement; every attempt uses remote_job cleanup."""
import json
import pathlib
import subprocess
import sys
import time

P = pathlib.Path('/home/philip/he4-capture-2080ti')
S = P/'supplemental'
ROOT = pathlib.Path(__file__).resolve().parents[2]
for attempt in range(100):
    if (S/'completion.json').exists():
        break
    run_id = f'he4-supplement-{int(time.time())}-{attempt}'
    child = subprocess.run([sys.executable,ROOT/'scripts/remote_job.py',run_id,'--',sys.executable,ROOT/'scripts/he4_capture/run_supplement.py'])
    if (S/'completion.json').exists():
        break
    (S/'restart.json').write_text(json.dumps(dict(attempt=attempt+1, exit_code=child.returncode, run_id=run_id)))
    time.sleep(3)
else:
    raise RuntimeError('100 unexpected supervisor restarts; manual diagnosis needed')
