"""Retire only authenticated redundant NEW campaign storage after byte recovery gates."""
import json,shutil,hashlib
from .common import *
from . import archive
from scripts.dflash_baseline import native_archive as old
from .quant import matrix_data

def main():
 before={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')};records=[]
 if json.load(open(R/'qualification/result.json'))['status']!='CUDA_CACHE_STORAGE_GRADIENT_FULL_CHECKPOINT_EXPORT_PASS':raise ValueError('recovery gate required')
 for name in ('E6','E4','HE4'):
  model=R/'zero'/(name+'.gguf');a,r=old.paths(model)
  if not r.exists():continue
  original=old.reference(model);created=old.hydrate(model)
  receipt=archive.archive(model)
  if receipt['original']!=original:raise ValueError('archive conversion changed original')
  # New exact recovery is proven before retiring the redundant old container.
  record=dict(retired=pin(a),original=original,replacement=receipt['archive']);a.unlink();r.unlink();records.append(record)
 pilot=R/'arms/H0-pilot/model-32.gguf'
 if pilot.exists():archive.archive(pilot)
 matrix=R/'matrices/Q4_K.bin'
 if matrix.exists():
  from scripts.dflash_baseline.model import GGUFReader
  basis=R/'bases/Q4_K.gguf';t=next(t for t in GGUFReader(basis).tensors if t.name=='output.weight')
  if pin(matrix)['sha256']!=hashlib.sha256(t.data).hexdigest():raise ValueError('matrix basis changed')
  records.append(dict(retired=pin(matrix),recoverable_slice=dict(basis=pin(basis),offset=int(t.data_offset),bytes=t.data.nbytes)));matrix.unlink()
 receipt=json.load(open(R/'arms/H0-pilot/recovery.json'));directory=Path(receipt['archive']).parent
 if not (directory/'rolling.json').exists():save(directory/'rolling.json',dict(arm='H0-pilot',update=32,receipt=receipt))
 save(R/'consolidation.json',dict(before_free=before,after_free={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')},records=records,protected='All pre-campaign artifacts unchanged; exact original native byte streams preserved by authenticated slice archives.'))
 print('CONSOLIDATION_COMPLETE',flush=True)

if __name__=='__main__':main()
