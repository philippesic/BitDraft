from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[2];R=Path('/home/philip/binary-eagle-decoding/runs/execution-campaign-20261105-01')
for script,name,args in [('build_head.py','head02',[]),('build_head.py','lastffn02',['--last-ffn']),('build_fusion.py','silu02',['--silu']),('build_fusion.py','residual02',['--residual']),('build_fusion.py','joint',['--joint']),('build_fusion.py','narrow',['--silu','--residual','--joint'])]:
 subprocess.run([sys.executable,str(ROOT/'scripts/execution_campaign'/script),str(R/('runtime-'+name)),*args],check=True);print('BUILT',name,flush=True)
