from pathlib import Path
import subprocess,json,time,os,shlex,sys,datetime
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/design-readiness-20261104-01';script=sys.argv[1];run=sys.argv[2]
assert not (P/'runs'/run).exists()
c=json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080'];assert not c.get('pause_requested') and not c.get('stop_all_requested')
gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
cmd=[str(P/'.venv/bin/python'),str(P/'scripts/remote_job.py'),run,'--stop-grace-seconds','180','--',str(P/'.venv/bin/python'),str(R/script)]
subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s',run,'-c',str(P),shlex.join(cmd)],check=True)
for _ in range(50):
 q=P/'runs'/run/'state.json'
 if q.exists() and json.loads(q.read_text()).get('pid'):break
 time.sleep(.1)
s=json.loads(q.read_text());record=dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),tmux_socket='binary-eagle-runtime',tmux_session=run,command=cmd,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identities=[],stop='Check boot/start ticks against each identity, TERM supervisor;180s graceful checkpoint; verify all owned groups and native server release receipts.')
for pid in [s['supervisor_pid'],s['pid']]:
 p=Path('/proc')/str(pid);record['identities'].append(dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
(R/(run+'-ownership.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
