"""Remeasure all affected zero/anchor references after the gain-stream correction."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-04';env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'))
def run(args):subprocess.run(list(map(str,args)),env=env,check=True)
def main():
 assert json.load(open(P/'runs/precision-gain-op-20261104-02/state.json'))['exit_code']==0
 run([sys.executable,ROOT/'scripts/dflash_baseline/relocate_controls.py'])
 run([sys.executable,ROOT/'scripts/dflash_baseline/prepare_precision.py'])
 run([sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',r/'zero-arms.json','--runtime',runtime,'--out',r/'zero-panel-runtime04','--cohort','capture3','--paired-trace'])
 arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf')}
 for name in ('G8D8','G4D8','G2D8','G1D8'):
  arms.update({name+'-zero':str(r/'zero-arms'/(name+'.gguf')),name+'-256':str(r/'adapted'/name/'model-256.gguf'),name+'-1024':str(r/'adapted'/name/'model-final.gguf')})
 for name in ('vanilla-matched','no-midpoint','no-gain'):arms['foundation-'+name]=str(r/name/'model-final.gguf')
 af=r/'runtime04-reference-arms.json';af.write_text(json.dumps(arms,indent=2)+'\n')
 run([sys.executable,ROOT/'scripts/dflash_baseline/precision_benchmark.py','--arms',af,'--runtime',runtime,'--out',r/'reference-panel-runtime04','--cohort','natural12','--paired-trace'])
 # Zero-cache compatibility is preserved byte-for-byte during regeneration.
 run([sys.executable,ROOT/'scripts/dflash_baseline/cache_zero.py'])
 (r/'active-runtime.json').write_text(json.dumps(dict(path=str(runtime),qualification='GAINED_STREAM_CUDA36_AND_NATIVE_ZERO_ANCHOR_REFERENCE_PANELS_PASS',old_runtime03='Superseded: generic gain op flushes tiny values; historical evidence retained.',target_arithmetic_unchanged=True),indent=2)+'\n')
if __name__=='__main__':main()
