from pathlib import Path
import subprocess,sys,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import build,P,F
R=P/'runs/execution-campaign-20261105-01'
for name,dot in [('cross256','cross'),('mma256','mma')]:
 build(R/('runtime-'+name),copy=True,bound=True,threads=256,dot=dot);print('BUILT',name,flush=True)
for name,args in [('silu',['--silu']),('residual',['--residual']),('generic',['--generic']),('fused',['--silu','--residual','--generic'])]:
 subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign/build_fusion.py'),str(R/('runtime-'+name)),*args],check=True);print('BUILT',name,flush=True)
