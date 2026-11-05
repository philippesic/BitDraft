from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[2];R=Path('/home/philip/binary-eagle-decoding/runs/execution-campaign-20261105-01')
for script,name,args in [('build_combined.py','combined02',[]),('build_combined.py','combined-last02',['--last-ffn']),('build_lut.py','lut',[]),('build_fusion.py','silu04',['--silu'])]:
 subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign'/script),str(R/('runtime-'+name)),*args],check=True);print('BUILT',name,flush=True)
