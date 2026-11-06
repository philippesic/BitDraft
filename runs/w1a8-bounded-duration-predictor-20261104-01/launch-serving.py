from pathlib import Path
import json,subprocess,time,os,shlex,datetime
P=Path('/home/philip/binary-eagle-decoding');N=P/'runs/w1a8-bounded-duration-predictor-20261104-01';train=json.loads((P/'runs/local-w1a8-bounded-old5760-20261104-02/state.json').read_text());assert train['status']=='finished' and train['exit_code']==0
assert json.loads((N/'old3840-parity.json').read_text())['all_exact'];assert (N/'result.json').exists()
run='local-w1a8-bounded-serving-20261104-01';remaining=int(datetime.datetime(2026,11,4,1,43,tzinfo=datetime.timezone.utc).timestamp()-time.time());assert remaining>300
cmd=[str(P/'.venv/bin/python'),str(P/'scripts/remote_job.py'),run,'--stop-grace-seconds','10','--','timeout','--signal=TERM','--kill-after=15s',str(remaining)+'s',str(P/'.venv/bin/python'),str(N/'serve.py')]
subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s',run,'-c',str(P),shlex.join(cmd)],check=True)
for _ in range(50):
 q=P/'runs'/run/'state.json'
 if q.exists() and json.loads(q.read_text()).get('pid'):break
 time.sleep(.1)
state=json.loads(q.read_text());receipt=dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),tmux_socket='binary-eagle-runtime',tmux_session=run,identities=[],command=cmd,started_at_utc=state['started_at_utc'],sole_gpu_operator=True,stop_procedure='Authenticate boot/birth. SIGTERM supervisor; Requests catches signal and reaps active native server session; verify release receipts and all owned groups absent. Timeout01:43UTC / totalcap01:47UTC.')
for pid in [state['supervisor_pid'],state['pid']]:
 p=Path('/proc')/str(pid);receipt['identities'].append(dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
(N/'serving-ownership.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
