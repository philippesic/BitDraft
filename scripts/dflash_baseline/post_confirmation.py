"""Serial post-DEV numerical/admission checks, locked heldout and native serving."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
def run(name,*args):subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline'/name),*map(str,args)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')),check=True)
def main():
 state=json.load(open(P/'runs/precision-confirmation-20261105-01/state.json'))
 if state.get('status')!='finished':raise ValueError('formal confirmation must finish first')
 if state.get('exit_code')!=0:
  recovery=json.load(open(R/'confirmation-dev12/counter-recovery.json'))
  from scripts.dflash_baseline.precision_benchmark import pin
  result=json.load(open(R/'confirmation-dev12/result.json'))
  if not (state.get('exit_code')==1 and 'RuntimeError: instrumentation changed native counters' in (P/'runs/precision-confirmation-20261105-01/stdout.log').read_text() and recovery['status']=='FINAL_NATIVE_LOG_COUNTERS_RECOVERED' and recovery['expected_cells']==530 and recovery['rows']==7420 and result.get('counter_recovery')==pin(R/'confirmation-dev12/counter-recovery.json')):raise ValueError('failed formal job needs authenticated full log recovery')
 run('qualify_slots.py');run('native_tokenize.py','prepare');run('lock_heldout.py')
 run('prepare_holdout.py',R/'heldout-selection.json',R/'heldout24')
 run('precision_benchmark.py','--arms',R/'heldout-arms.json','--runtime',P/'runs/precision-native-20261104-04','--out',R/'heldout-panel','--prompts',R/'heldout24/prompts.json','--cohort','natural24','--reps','5','--warmups','2','--paired-trace')
if __name__=='__main__':main()
