"""Launch a pinned plan through the existing local supervisor and capture identities."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time
import sys
from runner import atomic,identity,pin,read,require,validate


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--through',type=int,required=True);p.add_argument('--run-id',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 plan=read(a.plan);validate(plan);require(not a.receipt.exists(),'receipt exists');require(not (a.project/'runs'/a.run_id).exists(),'run id exists')
 if plan.get('gpu'):
  control=read(plan['gpu']['control'])['rtx5080'];require(not control.get('pause_requested') and not control.get('stop_all_requested'),'GPU paused')
  busy=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);require(not busy.strip(),'GPU compute process already present')
 command=[str(a.project/'.venv/bin/python'),str(a.project/'scripts/remote_job.py'),a.run_id,'--stop-grace-seconds',str(plan['grace_seconds']+60),'--',sys.executable,str(Path(__file__).with_name('runner.py')),'--plan',str(a.plan.resolve()),'--output',str(a.output.resolve()),'--through',str(a.through)]
 if plan.get('gpu'):command+=['--allow-gpu']
 subprocess.run(['tmux','-L','binary-eagle-runtime','new-session','-d','-s',a.run_id,'-c',str(a.project),shlex.join(command)],check=True)
 state=a.project/'runs'/a.run_id/'state.json';deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  if state.exists():
   try:
    value=read(state)
    if value.get('pid'):break
   except ValueError:pass
  time.sleep(.05)
 else:raise RuntimeError('supervisor did not publish identity; inspect tmux before any retry')
 identities=[]
 for pid in [value['supervisor_pid'],value['pid']]:
  try:identities.append(identity(pid))
  except FileNotFoundError:pass
 atomic(a.receipt,dict(plan=pin(a.plan),command=command,tmux_socket='binary-eagle-runtime',session=a.run_id,supervisor_state=str(state),identities=identities,stop='Authenticate receipt identity, TERM supervisor, wait cleanup grace, verify runner task groups and native release receipts.'))
 print(str(a.receipt))

if __name__=='__main__':main()
