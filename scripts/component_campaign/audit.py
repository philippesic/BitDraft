"""Stored tensor map, actual sign movement and scalar damage from native readouts."""
import argparse,hashlib,json
import numpy as np
from .common import *
from .archive import hydrate,release
from scripts.dflash_baseline.model import GGUFReader

def audit(name):
 directory=R/'arms'/name;records=json.load(open(directory/'readouts.json'))
 base={t.name:t for t in GGUFReader(R/'bases/shared.gguf').tensors};counts=np.array([i.bit_count() for i in range(256)],dtype=np.uint8);rows=[]
 for record in records:
  model=Path(record['model']['path']);created=hydrate(model)
  try:
   tensors=[]
   for t in GGUFReader(model).tensors:
    entry=dict(name=t.name,type=t.tensor_type.name,shape=list(map(int,t.shape)),bytes=t.data.nbytes,sha256=hashlib.sha256(t.data).hexdigest())
    old=base.get(t.name)
    if old is not None and t.name.endswith('.w1a1_packed'):
     xor=np.bitwise_xor(t.data.view(np.uint8),old.data.view(np.uint8));entry['binary_sign_flips_from_B0']=int(counts[xor].sum());entry['binary_sign_count']=xor.size*8
    if old is not None and (t.name.endswith(('.w1a1_scale','.w1ax_midpoint','.w1ax_gain')) or 'norm' in t.name):
     delta=t.data.astype(np.float32)-old.data.astype(np.float32)
     entry.update(changed_values=int((delta!=0).sum()),max_absolute_change=float(np.abs(delta).max()),nonfinite_values=int((~np.isfinite(t.data)).sum()),zeros=int((t.data==0).sum()))
    tensors.append(entry)
   rows.append(dict(update=record['update'],model=record['model'],tensor_bytes=sum(t['bytes'] for t in tensors),binary_sign_flips_from_B0=sum(t.get('binary_sign_flips_from_B0',0) for t in tensors),tensors=tensors))
  finally:release(model,created)
 save(directory/'native-tensor-audit.json',dict(status='COMPLETE',records=rows,source=pin(__file__),note='Stored model tensor bytes and persistent native bit changes, not compressed archive size or training-master bytes. Scalar comparisons include storage rounding when applicable.'))
 print('TENSOR_AUDIT',name,[x['binary_sign_flips_from_B0'] for x in rows],flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');a=ap.parse_args()
 for name in a.arms:audit(name)
