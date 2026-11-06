"""Bounded same-owner dependency handoff; never starts a competing GPU process."""
import argparse,json,os,subprocess,sys,time
from .common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after',type=Path,required=True);a=ap.parse_args();started=time.monotonic()
 save(R/'successor-owner.json',dict(identity=identity(),after=str(a.after.resolve()),status='WAITING_FOR_OWNED_HEAD_STAGE',source=pin(__file__)))
 while True:
  pause();record=json.load(open(a.after))
  if record['status'] in ('finished','failed','interrupted'):
   if record['status']!='finished' or record['exit_code']!=0:raise RuntimeError('head predecessor needs repair; no competing launch')
   break
  time.sleep(10)
 # remote_job publishes terminal status only after stopping its owned child group.
 try:os.killpg(record['pgid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('predecessor group still exists; cannot take GPU lane')
 save(R/'successor-handoff.json',dict(identity=identity(),predecessor=record,wait_seconds=time.monotonic()-started,hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','-q'],text=True)))
 def run(module):subprocess.run([sys.executable,'-m','scripts.component_campaign.'+module],cwd=ROOT,check=True)
 if not (R/'checkpoint-policy.json').exists():run('qualify_tensor_checkpoint')
 run('head_more')
 save(R/'isolated-stage-complete.json',dict(status='PRIMARY_COMPONENT_WINDOWS_COMPLETE_COMBINATIONS_AND_CONFIRMATION_PENDING',owner=identity()))

if __name__=='__main__':main()
