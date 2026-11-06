import json,os,subprocess,shutil
from .common import *

def main():
 R.mkdir(exist_ok=False)
 metadata=json.load(open(B/'model-final.json'));receipt=json.load(open(B/'checkpoint.pt.json'))
 checkpoint=B/'checkpoint.pt.bsz'
 if pin(checkpoint)['sha256']!=receipt['archive_sha256']:raise ValueError('B0 archive authentication failed')
 if metadata['checkpoint']['sha256']!=receipt['raw_sha256']:raise ValueError('B0 inference/full-checkpoint ancestry mismatch')
 save(R/'freeze.json',dict(owner=identity(),B0_native=json.load(open(B/'model-final.gguf.native.json')),B0_checkpoint=pin(checkpoint),B0_checkpoint_receipt=pin(B/'checkpoint.pt.json'),B0_config=metadata['config'],B0_progress={k:v for k,v in metadata['progress'].items() if k!='order'},optimized_runtime=pin(OPT/'build.json'),target=pin(TARGET),external_baseline=pin(Q4),baseline_registry=pin(ROOT/'experiments/w1a8-standing-20261102/baseline.json'),hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','-q'],text=True),free_bytes={p:shutil.disk_usage(p).free for p in ('/','/mnt/c')},policy='No target/verifier changes. All new arms branch exact learned-small2048; R1 resets Adam identically and preserves parent data cursor/order. Native exports retained; bounded rolling NEW checkpoints, final frontier checkpoint retained; previous paid states protected. Stop supervisor SIGTERM with90s grace and verify group release.'))
 from .build import build
 build()
 from .quant import make_matrix
 for format in ('Q4_K',):
  print('Quantizing',format,flush=True);make_matrix(TARGET,format,R/'matrices'/f'{format}.bin')
 print('PREPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
