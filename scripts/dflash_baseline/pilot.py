"""Bounded essential-feature pilot, executed in the sole supervised GPU lane."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.export_precision import export
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01'
assert json.load(open(r/'native-qualification/result.json'))['status']=='COMPLETE'
env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'))
subprocess.run([sys.executable,'-m','scripts.dflash_baseline.pipeline',str(r/'qualification-config.json'),'--out',str(r/'foundation-pilot'),'--updates','128','--max-windows','128','--device','cuda','--baseline-parent','--snapshots','0','32','64'],env=env,check=True)
for n in ('0','32','64','final'):
 checkpoint=r/'foundation-pilot'/('checkpoint.pt' if n=='final' else 'checkpoint-'+n+'.pt');export(checkpoint,r/'foundation-pilot'/('model-'+n+'.gguf'))
# First narrow pilot native points, before precision candidates are selected.
arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),**{'foundation-'+n:str(r/'foundation-pilot'/('model-'+n+'.gguf')) for n in ('0','32','64','final')}}
(r/'pilot-arms.json').write_text(json.dumps(arms,indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline/precision_benchmark.py'),'--arms',str(r/'pilot-arms.json'),'--runtime',str(P/'runs/precision-native-20261104-03'),'--out',str(r/'pilot-panel'),'--cohort','capture3'],env=env,check=True)
