from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[2];R=Path('/home/philip/binary-eagle-decoding/runs/execution-campaign-20261105-01')
for script,name,args in [('build_lut.py','lut02',[]),('build_fusion.py','silu04',['--silu'])]:
 subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign'/script),str(R/('runtime-'+name)),*args],check=True);print('BUILT',name,flush=True)
subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign/run_replay6.py')],check=True)
