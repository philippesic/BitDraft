"""Enter the frozen source under the canonical remote_job supervisor."""
import os
import runpy
import sys
from pathlib import Path

import json
import hashlib
run = Path('/home/philip/binary-eagle-decoding/runs/he4broad-20261110-01')
diagnostic='--loader-smoke' in sys.argv
if diagnostic:sys.argv.remove('--loader-smoke')
selection=json.load(open(run/('candidate-source.json' if diagnostic else 'active-source.json')))
assert selection['status']==('CPU_TESTED_LOCAL_SOURCE' if diagnostic else 'QUALIFIED_LOCAL_SOURCE')
source = Path(selection['path'])
assert source.is_relative_to(run) and source.is_dir()
manifest=selection['manifest']
assert hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest()==manifest['sha256']
os.chdir(source)
sys.path.insert(0,str(source))
runpy.run_module('scripts.he4fresh.smoke_local' if diagnostic else 'scripts.he4fresh.workflow',run_name='__main__')
