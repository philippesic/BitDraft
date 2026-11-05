"""Archive owned native exports, retain every trained byte and qualify hydration."""
import gc,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline.native_archive import archive,dictionary,pin,paths
from scripts.dflash_baseline.precision_benchmark import save
from scripts.dflash_baseline.audit_component_weights import audit
from scripts.dflash_baseline.prototype import resolve
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.checkpoint_io import load
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';BASIS=R/'zero-arms/G8D8.gguf'
def archive_exports():
 if json.load(open(P/'runs/precision-reserved-20261105-01/state.json')).get('exit_code')!=0:raise ValueError('reserved GPU owner must finish first')
 before={n:shutil.disk_usage(p).free for n,p in [('wsl',P),('windows','/mnt/c')]};records=[];audits=[];basis_pin=pin(BASIS);prepared=dictionary(BASIS)
 import zstandard as z
 params=z.ZstdCompressionParameters.from_level(3,window_log=30,hash_log=27,chain_log=27,strategy=z.STRATEGY_GREEDY);prepared.precompute_compress(compression_params=params)
 # Every member is an owned completed export, never a training checkpoint.
 candidates=sorted(p for folder in ('adapted','interactions','extensions','foundation','foundation-pilot','vanilla-matched','no-midpoint','no-gain','zero-arms') for p in (R/folder).rglob('*.gguf') if p!=BASIS)
 for model in candidates:
  metadata=model.with_suffix('.json')
  if metadata.exists() and json.load(open(metadata)).get('config',{}).get('components'):audits.append(audit(model))
  records.append(archive(model,BASIS,prepared,basis_pin));print('Native bytes retained in lossless archive',model.relative_to(R),records[-1]['original']['bytes'],records[-1]['archive']['bytes'],flush=True)
 # Restore old reproducible zeros byte-exactly, validate their component codes,
 # then archive too. Their original native measurements remain frozen.
 cache={}
 for receipt in R.glob('zero-cache-release-*.json'):
  for item in json.load(open(receipt)).get('removed',[]):cache[item['name']]=item
 registry=json.load(open(R/'tranche.json'));state=load(registry['parent']['path'])
 for name,spec in registry['models'].items():
  model=R/'zero-arms'/(name+'.gguf');compressed,receipt=paths(model)
  if model==BASIS or compressed.exists():continue
  old=cache[name];metadata=model.with_suffix('.json');backup=metadata.with_suffix('.initial.json')
  if metadata.exists() and not backup.exists():shutil.copyfile(metadata,backup)
  result=export(registry['parent']['path'],model,config=resolve(spec['request']),state=state,checkpoint_pin=registry['parent'],include_private_head_metadata=old['writer_private_head_metadata'])
  if result['sha256']!=old['sha256']:raise ValueError('historical native zero did not reproduce exactly: '+name)
  if spec['request'].get('components'):audits.append(audit(model))
  records.append(archive(model,BASIS,prepared,basis_pin));print('Historical zero recovered/audited/archived exactly',name,flush=True)
 del state,prepared;gc.collect()
 after={n:shutil.disk_usage(p).free for n,p in [('wsl',P),('windows','/mnt/c')]}
 save(R/'native-export-archives.json',dict(status='ALL_PAID_NATIVE_BYTES_RETAINED',basis=basis_pin,before_free_bytes=before,after_free_bytes=after,records=records,scope='Lossless storage-only archival; full SHA256 roundtrip per export; optimizer/parameter/RNG checkpoint archives untouched. No Windows reclaim or VHDX compaction claim.'))
 save(R/'component-code-audit.json',dict(status='DEPLOYED_COMPONENT_CODES_EXACT',models=audits))
def main():
 archive_exports()
 arms={'target':None,'q4':str(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'),'G4D8-learned-clip-1024':str(R/'adapted/G4D8-learned-clip/model-final.gguf'),'G1D8-private-head-1024':str(R/'adapted/G1D8-private-head/model-final.gguf')};af=R/'archive-qualified-arms.json';save(af,arms)
 subprocess.run([sys.executable,str(ROOT/'scripts/dflash_baseline/precision_benchmark.py'),'--arms',str(af),'--runtime',str(P/'runs/precision-native-20261104-04'),'--out',str(R/'archive-native-qualification'),'--cohort','capture3','--warmups','2','--paired-trace'],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'src')),check=True)
if __name__=='__main__':main()
