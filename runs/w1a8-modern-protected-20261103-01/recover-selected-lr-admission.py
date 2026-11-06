from pathlib import Path
import subprocess,json,time
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';start=time.monotonic();stages=[]
for name in ['recover-selected-lr1e3-all-native.py', 'run-selected-lr3e3-all-native.py', 'qualify-selected-lr.py']:
 t=time.monotonic();p=subprocess.run([str(P/'.venv/bin/python'),str(R/name)]);assert p.returncode==0,(name,p.returncode);stages.append({'stage':name,'wall_seconds':time.monotonic()-t,'exit_code':p.returncode});(R/'selected-lr-admission-recovery-progress.json').write_text(json.dumps({'stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
(R/'selected-lr-admission-recovery-result.json').write_text(json.dumps({'status':'PAIRED_MATURE_LR_ALL549_NATIVE_AND_HTTP_QUALIFICATION_COMPLETE_NOT_THROUGHPUT','stages':stages,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
