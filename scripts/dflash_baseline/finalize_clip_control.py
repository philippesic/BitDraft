"""New followup data audit and charts; visual review/publication remain separate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.post_confirmation import run
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
if __name__=='__main__':
 run('publish_clip_control.py','data')
 run('chart_precision.py',R,R/'charts-clip-control')
 print('Followup audited chart package exists; inspect before separate publication.',flush=True)
