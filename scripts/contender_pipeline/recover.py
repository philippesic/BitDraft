"""Explicit cleanup of an abandoned runner task, using recorded boot/birth identity.

Never infer completion from abandoned output. Preserve the conservative reserved
cost and require a new recovery plan with prior_state after cleanup.
"""
import argparse
import os
from pathlib import Path
import signal
import time
from runner import atomic,identity,identity_alive,group_live,locked,read,require,positive


def recover(state_path, job_id, grace=180):
 require(positive(grace),'invalid cleanup grace')
 state_path=Path(state_path)
 with locked(state_path.parent/'run.lock'):
  state=read(state_path);row=state['jobs'][job_id]
  require(row['status']=='RUNNING','only abandoned running jobs need recovery')
  saved=row['identity'];require(saved['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'boot differs; inspect prior-host state manually')
  try:actual=identity(saved['pid'])
  except FileNotFoundError:actual=None
  require(actual is None or actual==saved,'PID reused; refusing signal')
  if actual is None:
   require(not group_live(saved['pgid']),'leader absent but group remains; manual descendant identity review required')
  else:
   require(saved['pgid']==saved['pid'],'unexpected process group')
   os.killpg(saved['pgid'],signal.SIGTERM)
   deadline=time.monotonic()+grace
   while group_live(saved['pgid']) and time.monotonic()<deadline:time.sleep(.1)
   if group_live(saved['pgid']):
    # Verify leader identity if still present before escalation.
    try:require(identity(saved['pid'])==saved,'PID changed during cleanup')
    except FileNotFoundError:pass
    os.killpg(saved['pgid'],signal.SIGKILL)
    deadline=time.monotonic()+5
    while group_live(saved['pgid']) and time.monotonic()<deadline:time.sleep(.1)
   require(not group_live(saved['pgid']),'group not released')
  known=row.get('descendants',[])
  for sig,limit in [(signal.SIGTERM,grace),(signal.SIGKILL,5)]:
   for ref in known:
    if identity_alive(ref):
     try:os.kill(ref['pid'],sig)
     except ProcessLookupError:pass
   deadline=time.monotonic()+limit
   while any(identity_alive(ref) for ref in known) and time.monotonic()<deadline:time.sleep(.1)
   if not any(identity_alive(ref) for ref in known):break
  require(not any(identity_alive(ref) for ref in known),'recorded descendants remain')
  row.update(status='ABANDONED_RECOVERED',group_released=True,recovery='Conservative full reservation retained; output is not admitted.')
  state['status']='RECOVERED_REQUIRES_EXPLICIT_PLAN'
  atomic(state_path,state)
 return row


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--state',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--grace',type=float,default=180);a=p.parse_args();recover(a.state,a.job,a.grace)
if __name__=='__main__':main()
