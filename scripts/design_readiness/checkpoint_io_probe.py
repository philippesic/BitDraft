"""Measure serialization of synthetic full-sign state at the current geometry.

CPU/storage probe only: no model is trained and GPU/activation memory is not
qualified. The large synthetic file is always removed; receipts remain.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import resource
import shutil
import time
import torch

P=Path(__file__).resolve().parents[2]
ARTIFACTS=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')


def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def free():return {'WSL':shutil.disk_usage(P).free,'C':shutil.disk_usage('/mnt/c').free}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True);parser.add_argument('--allow-large-scratch',action='store_true');a=parser.parse_args()
    assert a.allow_large_scratch and a.scratch.resolve().is_relative_to(ARTIFACTS.resolve())
    assert not a.scratch.exists() and not a.output.exists()
    shapes=json.loads((P/'experiments/design-readiness-20261104/qat-source-audit.json').read_text())['current_candidate']['projection_shapes']
    count=sum(m*k+m for m,k in shapes.values());payload=3*4*count
    assert available()>payload+3*1024**3,'insufficient actual host memory reserve'
    assert shutil.disk_usage(a.scratch.parent).free>payload+3*1024**3,'insufficient scratch reserve'
    torch.set_num_threads(2);a.output.parent.mkdir(exist_ok=True,parents=True);before=free();started=time.perf_counter()
    state={name:{key:torch.zeros(shape,dtype=torch.float32) for key in ('master','adam_m','adam_v')} for name,shape in shapes.items()}
    state['scales']={name:{key:torch.zeros((shape[0],),dtype=torch.float32) for key in ('master','adam_m','adam_v')} for name,shape in shapes.items()}
    allocation_seconds=time.perf_counter()-started
    try:
        begin=time.perf_counter()
        with a.scratch.open('xb') as stream:
            torch.save({'synthetic_resource_probe':True,'state':state},stream);stream.flush();os.fsync(stream.fileno())
        save_seconds=time.perf_counter()-begin;size=a.scratch.stat().st_size;after_write=free()
        with a.scratch.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        del state;gc.collect();begin=time.perf_counter();restored=torch.load(a.scratch,map_location='cpu',weights_only=False);load_seconds=time.perf_counter()-begin
        assert restored['synthetic_resource_probe'] is True
        actual=0
        for name,shape in shapes.items():
            for tensor in list(restored['state'][name].values())+list(restored['state']['scales'][name].values()):
                assert tensor.dtype==torch.float32 and torch.count_nonzero(tensor)==0
                actual+=tensor.numel()*tensor.element_size()
        assert actual==payload
        del restored;gc.collect()
    finally:
        a.scratch.unlink(missing_ok=True)
    result=dict(status='SYNTHETIC_CURRENT_GEOMETRY_FULL_SIGN_STATE_IO_PASS',destination=str(a.scratch),synthetic_file_removed=not a.scratch.exists(),projection_shapes=shapes,synthetic_payload_bytes=payload,serialized_bytes=size,serialized_sha256=sha,allocation_seconds=allocation_seconds,save_and_fsync_seconds=save_seconds,load_seconds=load_seconds,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,free_before=before,free_after_write=after_write,free_after_cleanup=free(),scope='F32 masters and two Adam moments for current sign and row-scale shapes. No GPU, backward/activation fit, real optimizer resume/export, actual model checkpoint or future architecture qualification.',wall_seconds=time.perf_counter()-started)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
