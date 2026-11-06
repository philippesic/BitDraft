"""A queue may die before a checkpointing child; grace belongs to the group."""
import json,os,subprocess,sys,time
from .common import *
from scripts.remote_job import stop_group

def main():
 out=R/'qualification/graceful-group-stop';out.mkdir(exist_ok=False)
 child=out/'child.py';child.write_text("import signal,time,sys\nfrom pathlib import Path\ndef stop(*_):\n time.sleep(.4)\n Path(sys.argv[2]).write_text('saved')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nPath(sys.argv[1]).write_text('ready')\nwhile True:time.sleep(.05)\n")
 ready=out/'ready';saved=out/'saved'
 command=[sys.executable,'-c',"import subprocess,sys,time;subprocess.Popen(sys.argv[1:]);time.sleep(60)",sys.executable,str(child),str(ready),str(saved)]
 leader=subprocess.Popen(command,start_new_session=True)
 try:
  deadline=time.monotonic()+5
  while not ready.exists():
   if time.monotonic()>deadline:raise TimeoutError('checkpoint child did not become ready')
   time.sleep(.02)
  began=time.monotonic();stop_group(leader.pid,leader,2);elapsed=time.monotonic()-began
  if not saved.exists() or elapsed<.4:raise ValueError('queue leader exit truncated descendant checkpoint grace')
  save(out/'result.json',dict(status='DESCENDANT_CHECKPOINT_GRACE_PASS',saved_marker=pin(saved),elapsed_seconds=elapsed,leader_exit=leader.returncode,source=pin(ROOT/'scripts/remote_job.py'),fixture=pin(__file__),scope='CPU child simulates checkpoint save after leader dies; no GPU task interrupted. Current already-running supervisors retain old code and require worker-first graceful stop. New supervisors use corrected full group grace.'))
  print('GROUP_GRACE_PASS',elapsed,flush=True)
 finally:
  try:os.killpg(leader.pid,9)
  except ProcessLookupError:pass
  leader.wait()

if __name__=='__main__':main()
