from pathlib import Path
import subprocess,json,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';start=time.monotonic();stages=[]
for name in ['run-union-all-native.py','qualify-reachability.py']:
 t=time.monotonic();p=subprocess.run([str(P/'.venv/bin/python'),str(R/name)]);assert p.returncode==0,(name,p.returncode);stages.append({'stage':name,'wall_seconds':time.monotonic()-t,'exit_code':p.returncode});(R/'reachability-admission-progress.json').write_text(json.dumps({'stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
(R/'reachability-admission-result.json').write_text(json.dumps({'status':'ALL549_NATIVE_AND28_HTTP_ADMISSION_COMPLETE_NOT_REPEATED_TIMING','stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
