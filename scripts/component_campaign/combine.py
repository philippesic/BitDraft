"""Compose chosen component contracts, always training cleanly from the frozen B0."""
import argparse,copy,json,subprocess,sys
from .common import *

def compose(name,components):
 result={'name':name};parents={}
 for component in components:
  spec=json.load(open(R/'specs'/(component+'.json')))
  if spec.get('r2_vocab_scale'):raise ValueError('R2 combinations require an explicit matched-recipe decision, not silent recipe mixing')
  parents[component]=pin(R/'arms'/component/'result.json')
  for key,value in spec.items():
   if key=='name':continue
   if key=='scalar':result.setdefault(key,{}).update(value)
   elif key in result and result[key]!=value:raise ValueError('conflicting component contracts '+key)
   else:result[key]=copy.deepcopy(value)
 if result.get('tied') and result.get('head')!=result.get('embedding'):raise ValueError('tied native matrix requires one identical precision/grid')
 if result.get('scalar',{}).get('clip'):result['clip_bits_storage']=True
 return result,parents

def main():
 ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('components',nargs='+');ap.add_argument('--run',action='store_true');a=ap.parse_args()
 spec,parents=compose(a.name,a.components);path=R/'specs'/(a.name+'.json')
 if path.exists() and json.load(open(path))!=spec:raise ValueError('combination name already has a different contract')
 save(path,spec);save(R/'combination-decisions'/(a.name+'.json'),dict(spec=spec,component_evidence=parents,common_parent=pin(B/'checkpoint.pt.bsz'),recovery='4096additional updates from B0; identical data cursor/order and R1 optimizer reset; no sequential adaptation history',selection=pin(R/'selection-evidence.json'),source=pin(__file__)))
 if a.run:subprocess.run([sys.executable,'-m','scripts.component_campaign.queue',a.name],cwd=ROOT,check=True)
 print('COMBINATION',a.name,flush=True)

if __name__=='__main__':main()
