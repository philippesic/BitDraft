"""Serialized qualified arms. Each child is inside the one remote_job process group."""
import argparse,json,os,subprocess,sys
from .common import *
from .archive import reference,available

def run(argv):
 print('COMMAND',list(map(str,argv)),flush=True)
 subprocess.run(list(map(str,argv)),cwd=ROOT,check=True)

def panel(name,model,kv=('f16','f16')):
 folder=R/('smoke-'+name)
 if (folder/'result.json').exists():return
 arms=dict(target=None,q4=str(Q4),**{name:str(model)})
 save(R/'panels'/(name+'.json'),arms);save(R/'panels'/(name+'-cache.json'),{name:kv})
 run([sys.executable,'-m','scripts.component_campaign.native','--arms',R/'panels'/(name+'.json'),'--draft-cache-map',R/'panels'/(name+'-cache.json'),'--runtime',R/'runtime03','--out',folder,'--cohort','capture3','--reps','1','--output-cap','64'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');a=ap.parse_args()
 for name in a.arms:
  pause();spec=json.load(open(R/'specs'/(name+'.json')));out=R/'arms'/name
  if not (out/'result.json').exists():
   if out.exists():raise RuntimeError('Interrupted arm requires authenticated resume/new output path: '+name)
   extra=[];source=R/'specs'/(name+'.json')
   if name=='H0':
    source=R/'specs/H0-pilot.json';receipt=json.load(open(R/'arms/H0-pilot/recovery.json'));extra=['--resume',receipt['archive'],'--out-name','H0']
   run([sys.executable,'-m','scripts.component_campaign.train',source,'--updates','4096',*extra])
  result=json.load(open(out/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']!=4096:raise RuntimeError('Primary recovery allocation incomplete: '+name)
  # Periodic native evaluations are deferred until the trainer releases the GPU.
  # All four actual exports are retained; each gets the same consumed-DEV smoke.
  for step in (0,256,1024,4096):
   model=R/'arms/H0-pilot/model-0.gguf' if name=='H0' and step==0 else out/f'model-{step}.gguf'
   panel(name+'-'+str(step),model,spec.get('kv',('f16','f16')))
  save(R/'queue-live.json',dict(last_completed=name,next_pending=a.arms[a.arms.index(name)+1:],owner=identity(),training_result=pin(out/'result.json')))
 print('PRIMARY_QUEUE_COMPLETE',a.arms,flush=True)

if __name__=='__main__':main()
