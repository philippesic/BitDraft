"""Final independent export audit, raw-data gate and actual chart generation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.post_confirmation import run
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01'
def main():
 if not (R/'final-component-code-audit.json').exists():run('audit_component_weights.py',R/'finalists/G4D8-learned-small/model-1536.gguf',R/'finalists/G4D8-learned-small/model-final.gguf','--out',R/'final-component-code-audit.json')
 run('publish_precision.py','data')
 run('chart_precision.py',R,R/'charts-initial')
 print('Audited initial chart package exists; inspect actual plots/controller before separate publish gate.',flush=True)
if __name__=='__main__':main()
