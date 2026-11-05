"""Frozen essential foundation and matched feature-removal controls."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.checkpoint_io import archive
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src'))
base=json.load(open(r/'qualification-config.json'));arms={}
for name,features in [('foundation',{}),('vanilla-matched',{'midpoint':False,'gain':False}),('no-midpoint',{'midpoint':False}),('no-gain',{'gain':False})]:
 c=dict(base,features=features);cfg=r/(name+'-config.json');cfg.write_text(json.dumps(c,indent=2)+'\n');out=r/name
 if not (out/'result.json').exists():
  extra=['--resume',str(r/'foundation-pilot/checkpoint.pt')] if name=='foundation' else ['--baseline-parent','--native-points','0','512']
  # Resume the exact pilot config, including absence of a feature override.
  if name=='foundation':cfg=r/'qualification-config.json';extra+=['--native-points','512','1024']
  subprocess.run([sys.executable,'-m','scripts.dflash_baseline.pipeline',str(cfg),'--out',str(out),'--updates','2048','--max-windows','2048','--device','cuda',*extra],env=env,check=True)
 if json.load(open(out/'result.json'))['progress']['updates']!=2048:raise RuntimeError('foundation allocation incomplete')
 if not (out/'model-final.gguf').exists():export(out/'checkpoint.pt',out/'model-final.gguf')
 if not (out/'checkpoint.pt.zst').exists():archive(out/'checkpoint.pt')
 arms[name]=str(out/'model-final.gguf');print('Foundation arm complete',name,flush=True)
# The precision parent is frozen; stage0 controls are evaluated together on broader DEV.
arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),**arms};(r/'foundation-arms.json').write_text(json.dumps(arms,indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline/precision_benchmark.py'),'--arms',str(r/'foundation-arms.json'),'--runtime',str(P/'runs/precision-native-20261104-03'),'--out',str(r/'foundation-panel'),'--cohort','natural12'],env=env,check=True)
