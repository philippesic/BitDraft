"""Serialized post-primary qualification and measured selection; combinations remain required."""
import argparse,json,os,subprocess,sys,time
from .common import *
from .measure import completed

def wait_release(path):
 while True:
  pause();state=json.load(open(path))
  if state['status'] in ('failed','interrupted'):
   raise RuntimeError('predecessor terminated without success; investigate before next GPU work')
  if state['status']=='running' and not Path('/proc',str(state['supervisor_pid'])).exists():
   raise RuntimeError('predecessor supervisor is absent; stale receipt cannot release GPU ownership')
  if state['status']=='finished':
   if state['exit_code']!=0:raise RuntimeError('predecessor failed; investigate before next GPU work')
   try:os.killpg(state['pgid'],0)
   except ProcessLookupError:break
   raise RuntimeError('predecessor group remains after terminal state')
  time.sleep(5)

def run(module,*args):
 print('POST_PRIMARY',module,*map(str,args),flush=True)
 subprocess.run([sys.executable,'-m','scripts.component_campaign.'+module,*map(str,args)],cwd=ROOT,check=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after',type=Path,required=True);a=ap.parse_args();wait_release(a.after)
 save(R/'post-primary-owner.json',dict(owner=identity(),predecessor=pin(a.after),hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),scope='One GPU lane; qualified selection measurements then root makes evidence-led combination/rescue decision.'))
 run('clip_storage_qualify')
 names=list(completed());selection=R/'measurements/selection-primary'
 if not (selection/'result.json').exists():run('measure','selection-primary','--reps','1','--cohort','natural12')
 models=dict(q4=str(Q4),B0=str(B/'model-final.gguf'));cache={}
 for name,entry in completed().items():models[name]=entry['model'];cache[name]=entry['spec'].get('kv',['f16','f16'])
 save(R/'panels/replay-primary-arms.json',models);save(R/'panels/replay-primary-cache.json',cache)
 replay=R/'measurements/replay-primary-selection'
 if not (replay/'result.json').exists():run('replay',R/'panels/replay-primary-arms.json','--cache-map',R/'panels/replay-primary-cache.json','--out',replay,'--runtime',R/'runtime04','--reps','1')
 audit=R/'qualification/physical-tie'
 if not (audit/'result.json').exists():
  save(R/'panels/physical-tie-arms.json',{name:models[name] for name in ('H0','H6','H4','E6','E4','HE4')})
  run('replay',R/'panels/physical-tie-arms.json','--out',audit,'--runtime',R/'runtime04','--reps','1','--audit')
 run('allocation_audit')
 run('formal_summary','--native',selection,'--replay',replay,'--out',R/'selection-primary-summary.json')
 save(R/'publication-index.json',dict(status='IN_PROGRESS',summary=str(R/'selection-primary-summary.json'),selection_panel=selection.name,gpu_panel=replay.name,arm_status={n:'4096 R1 window complete; measured selection, final frontier pending' for n in names}))
 run('publish')
 save(R/'post-primary-result.json',dict(status='SELECTION_MEASUREMENTS_COMPLETE_COMBINATIONS_REQUIRED',owner=identity(),selection=pin(R/'selection-primary-summary.json')))
 print('POST_PRIMARY_SELECTION_COMPLETE_NOT_CAMPAIGN_COMPLETION',flush=True)

if __name__=='__main__':main()
