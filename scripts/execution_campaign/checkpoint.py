"""Snapshot authenticated ownership and a bounded storage inventory."""
from pathlib import Path
import json,os,subprocess,time,shutil
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01'
rows=[]
for name in sorted(x.name for x in (P/'runs').glob('execution-*-20261105-*') if (x/'state.json').exists()):
 f=P/'runs'/name/'state.json'
 if not f.exists():continue
 d=json.load(open(f));d['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 for key in ('pid','supervisor_pid'):
  try:
   stat=Path(f'/proc/{d[key]}/stat').read_text();d[key+'_identity']=dict(pid=d[key],stat=stat,startticks=stat[stat.rfind(')')+2:].split()[19])
  except FileNotFoundError:d[key+'_identity']=None
 rows.append(d)
(R/'live-ownership.json').write_text(json.dumps(dict(time_unix_ns=time.time_ns(),worktree=str(Path(__file__).resolve().parents[2]),tmux_socket='binary-eagle-runtime',jobs=rows,stop='Authenticate boot/startticks against these identities; TERM owned supervisor, await terminal receipt and group absence; do not stop unrelated processes.',space={x:shutil.disk_usage(x)._asdict() for x in ['/', '/mnt/c']}),indent=2)+'\n')
