"""Proper-warmup repeated DEV confirmation after the finite search is locked."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-04'
def main():
 lock=json.load(open(r/'search-lock.json'))
 if lock['status']!='INITIAL_SEARCH_LOCKED_FOR_CONFIRMATION':raise ValueError('search decisions must lock before final confirmation')
 registry=json.load(open(r/'tranche.json'));arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),'vanilla-zero':str(P/'runs/dflash-modern-20261104-02/zero.gguf')}
 original=P/'runs/precision-vanilla-eval-20261104-01/panel'
 for label in ('300','900','1800'):arms['vanilla-'+label]=str(original/('trained-'+label+'.gguf'))
 arms['foundation-zero']=str(r/'zero-arms/G8D8.gguf')
 for name in ('vanilla-matched','no-midpoint','no-gain'):arms['foundation-'+name]=str(r/name/'model-final.gguf')
 for name,spec in registry['models'].items():
  if not spec['adapt']:continue
  out=r/'adapted'/name;result=json.load(open(out/'result.json'))
  if result['status']!='COMPLETE' or result['progress']['updates']!=1024:raise ValueError('standard matched arm incomplete: '+name)
  arms[name+'-1024']=str(out/'model-final.gguf')
 for name,model in lock.get('additional_confirmations',{}).items():
  if not Path(model).is_file():raise ValueError('additional locked model missing')
  arms[name]=model
 af=r/'confirmation-arms.json'
 if af.exists() and json.load(open(af))!=arms:raise ValueError('confirmation arms changed')
 af.write_text(json.dumps(arms,indent=2)+'\n')
 env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'))
 subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline/precision_benchmark.py'),'--arms',str(af),'--runtime',str(runtime),'--out',str(r/'confirmation-dev12'),'--cohort','natural12','--reps','5','--warmups','2','--paired-trace'],env=env,cwd=ROOT,check=True)
if __name__=='__main__':main()
