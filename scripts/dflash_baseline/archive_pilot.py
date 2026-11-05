"""Preserve every paid pilot state losslessly while reducing owned disk usage."""
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.checkpoint_io import archive
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01';before=shutil.disk_usage(P).free;receipts=[]
for name in ('checkpoint-0.pt','checkpoint-32.pt','checkpoint-64.pt','checkpoint.pt'):
 p=r/'foundation-pilot'/name
 if p.exists():receipts.append(archive(p));print('Archived pilot',name,flush=True)
(r/'pilot-archive-storage.json').write_text(json.dumps(dict(before_free_bytes=before,after_free_bytes=shutil.disk_usage(P).free,archives=receipts,reason='Lossless archive only; all parameter/optimizer/RNG/data state recoverable with original raw SHA256. No paid state dropped.'),indent=2)+'\n')
