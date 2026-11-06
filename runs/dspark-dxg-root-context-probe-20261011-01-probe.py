import datetime
import json
import os
import subprocess

code = "import os,json; s={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in open('/proc/self/status') if x.startswith(('Uid:','Gid:','Groups:','CapEff:','CapPrm:','CapBnd:','NoNewPrivs:','NSpid:'))}; e=None; n=None\ntry:\n n=len(os.listdir('/proc/1/fd'))\nexcept Exception as x:\n e=type(x).__name__+': '+str(x)\nprint(json.dumps({'euid':os.geteuid(),'egid':os.getegid(),'pid_namespace':os.readlink('/proc/self/ns/pid'),'status':s,'proc1_fd_count':n,'proc1_fd_error':e},sort_keys=True))"
env = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C.UTF-8'}
if os.environ.get('WSL_INTEROP'):
    env['WSL_INTEROP'] = os.environ['WSL_INTEROP']
cmd = ['/mnt/c/Windows/System32/wsl.exe', '-d', 'Ubuntu', '-u', 'root', '--', '/usr/bin/python3', '-B', '-I', '-c', code]
result = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15, cwd='/', env=env)
print(json.dumps({'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'outer_uid': os.geteuid(), 'outer_wsl_interop': env.get('WSL_INTEROP'), 'outer_wsl_interop_exists': os.path.exists(env.get('WSL_INTEROP', '/nonexistent')), 'root_helper_argv': cmd, 'root_helper_returncode': result.returncode, 'root_helper_stdout': result.stdout, 'root_helper_stderr': result.stderr, 'gpu_queried': False, 'model_loaded': False}, sort_keys=True))
