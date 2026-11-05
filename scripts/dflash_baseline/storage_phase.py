"""CPU-only bounded storage maintenance between native timing blocks."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for name in ('archive_pilot.py','cache_zero.py'):
 subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline'/name)],check=True)
