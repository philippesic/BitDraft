"""Run a recorded bounded R2 recipe with an actual full-state resume pilot."""
import argparse,json,subprocess,sys
from .common import *
from .queue import panel

def run(*args):
 print('RECOVERY',*map(str,args),flush=True)
 subprocess.run([sys.executable,'-m','scripts.component_campaign.train',*map(str,args)],cwd=ROOT,check=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');ap.add_argument('--pilot-first',action='store_true');a=ap.parse_args()
 decision=R/'selection-evidence.json'
 if not decision.exists():raise ValueError('Recovery selection evidence must precede new recipes')
 save(R/'recovery-owner.json',dict(owner=identity(),decision=pin(decision),arms=a.arms,source=pin(__file__)))
 for index,name in enumerate(a.arms):
  pause();specpath=R/'specs'/(name+'.json');spec=json.load(open(specpath));out=R/'arms'/name
  if not spec.get('r2_vocab_scale'):raise ValueError('This runner requires an explicit R2 recipe')
  pilot=R/'arms'/(name+'-pilot');resume=[]
  if a.pilot_first and index==0:
   if not (pilot/'result.json').exists():
    if pilot.exists():raise RuntimeError('Interrupted pilot requires authenticated resume, not overwrite')
    run(specpath,'--updates','32','--out-name',pilot.name,'--pilot')
   result=json.load(open(pilot/'result.json'))
   if result['status']!='COMPLETE' or result['progress']['updates']!=32:raise ValueError('Full-head CUDA/optimizer pilot incomplete')
   diagnostics=result['readouts'][-1]['head_qat_diagnostics']
   if not diagnostics['gradient_finite'] or not diagnostics['adam_moments_valid'] or diagnostics['adam_step']!=32:raise ValueError('Pilot gradient/optimizer qualification failed')
   for step in (0,32):panel(name+'-pilot-'+str(step),pilot/f'model-{step}.gguf',spec.get('kv',('f16','f16')))
   receipt=json.load(open(pilot/'recovery.json'));resume=['--resume',receipt['archive']]
  if not (out/'result.json').exists():
   if out.exists():raise RuntimeError('Interrupted recovery requires authenticated resume, not overwrite')
   run(specpath,'--updates','4096',*resume)
  result=json.load(open(out/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']!=4096:raise ValueError('Full recovery allocation incomplete')
  for step in (0,256,1024,4096):
   model=pilot/'model-0.gguf' if step==0 and pilot.exists() else out/f'model-{step}.gguf'
   panel(name+'-'+str(step),model,spec.get('kv',('f16','f16')))
  save(out/'resume-native-qualification.json',dict(status='FULL_R2_RECOVERY_AND_NATIVE_READOUTS_COMPLETE',result=pin(out/'result.json'),pilot_result=pin(pilot/'result.json') if pilot.exists() else None,source=pin(__file__),decision=pin(decision),policy='Same B0, data order/exposure and R2 optimizer policy as declared matched control; pilot checkpoints add no optimizer updates. Full frozen native codes, trained block scales, upstream gradients and actual native requests qualified.'))
 print('BOUNDED_R2_RECOVERY_COMPLETE',flush=True)

if __name__=='__main__':main()
