"""Finite post-H4 compact probe, then embedding/cache/scalar primary recovery."""
import json,subprocess,sys,hashlib
from .common import *
from .quant import make_matrix
from .export import rewrite
from .archive import archive
from .queue import panel,run

def main():
 control=json.load(open(R/'arms/H0/result.json'));h4=json.load(open(R/'arms/H4/result.json'))
 if any(x['status']!='COMPLETE' or x['progress']['updates']!=4096 for x in (control,h4)):raise ValueError('H4 first recovery must finish before finite probe')
 # An exact private-copy control reuses the mathematically identical Q6 training forward.
 run([sys.executable,'-m','scripts.component_campaign.copy_controls'])
 run([sys.executable,'-m','scripts.component_campaign.queue','H6','E6'])
 spec=dict(name='H3',head='Q3_K');save(R/'specs/H3.json',spec)
 save(R/'head-probe-decision.json',dict(status='FINITE_Q3_PROBE_SELECTED_AFTER_H4_RECOVERY',H0=pin(R/'arms/H0/result.json'),H4=pin(R/'arms/H4/result.json'),spec=spec,updates=4096,policy='Q3_K is the next predeclared compact head; no initial acceptance elimination. Q5_K bridge remains conditional on actual H4/H3 recovery and native tradeoff evidence; no Cartesian format sweep.'))
 path=R/'matrices/Q3_K.bin';make_matrix(TARGET,'Q3_K',path)
 model=R/'zero/H3.gguf';basis=R/'bases/Q3_K.gguf'
 if not model.exists() and not basis.exists():
  rewrite(R/'bases/shared.gguf',model,spec);basis.hardlink_to(model)
  from scripts.dflash_baseline.model import GGUFReader
  t=next(t for t in GGUFReader(basis).tensors if t.name=='output.weight')
  if pin(path)['sha256']!=hashlib.sha256(t.data).hexdigest():raise ValueError('Q3 basis mismatch')
  save(R/'Q3-matrix-retention.json',dict(matrix=pin(path),recoverable_basis=pin(basis),offset=int(t.data_offset),bytes=t.data.nbytes));path.unlink()
 panel('H3-zero',model)
 run([sys.executable,'-m','scripts.component_campaign.queue','H3','E4','HE4','K8F','KF8','K88','K48','K84','K44','Sscale','Smidpoint','Sgain','Sclip','Snorm','Sall'])
 print('ISOLATED_COMPONENTS_COMPLETE_COMBINATIONS_REQUIRED',flush=True)

if __name__=='__main__':main()
