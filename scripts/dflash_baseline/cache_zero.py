"""Verify reproducible zero GGUF cache before removing unneeded large exports."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.prototype import resolve
from scripts.dflash_baseline.pipeline import sha
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.model import GGUFReader
P=Path('/home/philip/binary-eagle-decoding');r=P/'runs/precision-campaign-20261104-01'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after-all-adaptation',action='store_true');a=ap.parse_args();registry=json.load(open(r/'tranche.json'));zero=r/'zero-arms'
 if not (r/'zero-panel/result.json').exists():raise ValueError('zero screen must finish first')
 # Byte-for-byte reproduction gate includes the initial writer metadata variant.
 proof=r/'zero-cache-reproduction.json'
 if not proof.exists():
  original=zero/'G8D4.gguf';fields=GGUFReader(original).fields;flag='dflash.w1ax.private_head' in fields;config=resolve(json.load(open(r/'locked-configs/G8D4.json')));dest=r/'cache-proof-G8D4.gguf'
  if dest.exists():raise FileExistsError(dest)
  export(registry['parent']['path'],dest,config=config,include_private_head_metadata=flag)
  expected=sha(original);actual=sha(dest)
  if expected!=actual:raise ValueError('zero export is not byte-reproducible')
  proof.write_text(json.dumps(dict(original=str(original),expected=expected,recreated=actual,byte_exact=True,writer_private_head_metadata=flag,parent=registry['parent'],source=str(ROOT/'scripts/dflash_baseline/export_precision.py')),indent=2)+'\n');dest.unlink()
 receipts=[];before=shutil.disk_usage(P).free
 for name,spec in registry['models'].items():
  if name in ('G8D8','G4D8','G2D8','G1D8'):continue
  if spec['adapt'] and not a.after_all_adaptation:continue
  if spec['adapt'] and not (r/'adapted'/name/'panel/result.json').exists():continue
  p=zero/(name+'.gguf')
  if not p.exists():continue
  fields=GGUFReader(p).fields;receipts.append(dict(name=name,path=str(p),sha256=sha(p),bytes=p.stat().st_size,config=str(r/'locked-configs'/(name+'.json')),writer_private_head_metadata='dflash.w1ax.private_head' in fields,parent=registry['parent'],native_evidence=str(r/'zero-panel/result.json'),retention='Native results/config/parent preserved; exact initial model regenerable. No trained checkpoint removed.'));p.unlink()
 record=dict(before_free_bytes=before,after_free_bytes=shutil.disk_usage(P).free,removed=receipts,gate=str(proof));(r/('zero-cache-release-final.json' if a.after_all_adaptation else 'zero-cache-release-first.json')).write_text(json.dumps(record,indent=2)+'\n');print('Reproducible zero cache removed',len(receipts),flush=True)
if __name__=='__main__':main()
