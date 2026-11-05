from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[2];R=Path('/home/philip/binary-eagle-decoding/runs/execution-campaign-20261105-01')
for name,args in [('silu03',['--silu']),('residual03',['--residual']),('narrow03',['--silu','--residual','--joint']),('prenorm',['--prenorm'])]:
 subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign/build_fusion.py'),str(R/('runtime-'+name)),*args],check=True);print('BUILT',name,flush=True)
