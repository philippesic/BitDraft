"""Enter the frozen source under the canonical remote_job supervisor."""
import os
import runpy
import sys
from pathlib import Path

source = Path('/home/philip/binary-eagle-decoding/runs/he4broad-20261110-01/source')
os.chdir(source)
sys.path.insert(0,str(source))
runpy.run_module('scripts.he4fresh.workflow',run_name='__main__')
