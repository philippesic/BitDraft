"""Run bounded current-candidate recipes sequentially in one CUDA process.

The current model/data/precision stay fixed. A derived source changes only the
project-path bootstrap and makes thread initialization idempotent. Every arm
rebuilds model/data and restores its paid parent/RNG/Adam independently. This
amortizes process/CUDA startup; it is not a universal architecture trainer.
"""
from pathlib import Path
import argparse
import difflib
import gc
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time

P=Path(__file__).resolve().parents[2]
ARTIFACTS=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts')


def pin(p):
    p=Path(p).resolve()
    with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--run-output',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);p.add_argument('--recipes',nargs='+',required=True);p.add_argument('--final-updates',type=int,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args()
    assert a.allow_cuda and 5760<a.final_updates<=7680
    assert a.out_root.resolve().is_relative_to(ARTIFACTS.resolve())
    assert a.run_output.resolve().is_relative_to((P/'runs').resolve())
    assert not a.run_output.exists() and not a.out_root.exists()
    assert 1<=len(a.recipes)<=4 and all(r in ('control','depth','smooth','combined') for r in a.recipes)
    freeze=json.loads(a.freeze.read_text());source=P/'scripts/design_readiness/fixed_objective.py'
    assert pin(source) in freeze['sources']
    original=source.read_text();needle='    torch.set_num_interop_threads(1)\n';assert original.count(needle)==1
    code=original.replace(needle,'    if torch.get_num_interop_threads() != 1:\n        torch.set_num_interop_threads(1)\n')
    needle='PROJECT = Path(__file__).resolve().parents[2]';assert code.count(needle)==1
    code=code.replace(needle,'PROJECT = Path('+repr(str(P))+')')
    a.run_output.mkdir(parents=True);a.out_root.mkdir(parents=True)
    derived=a.run_output/'fixed_objective_process_reuse.py';derived.write_text(code)
    (a.run_output/'source.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile=str(source),tofile=str(derived))))
    plan=dict(source=pin(source),derived_source=pin(derived),freeze=pin(a.freeze),recipes=a.recipes,final_updates=a.final_updates,source_change='Idempotent identical thread count + explicit unchanged project bootstrap only',out_root=str(a.out_root))
    (a.run_output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('_fixed_objective_process_reuse',derived);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    import torch
    started=time.perf_counter();rows=[]
    def stop(*unused):raise KeyboardInterrupt('Process reuse queue interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for i,recipe in enumerate(a.recipes):
            out=a.out_root/f'arm-{i:02d}-{recipe}'
            if torch.cuda.is_initialized():torch.cuda.reset_peak_memory_stats()
            argv=[str(derived),'--recipe',recipe,'--final-updates',str(a.final_updates),'--out',str(out),'--freeze',str(a.freeze),'--allow-cuda'];saved_argv=sys.argv;t=time.perf_counter()
            try:sys.argv=argv;module.main()
            finally:sys.argv=saved_argv
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            result=json.loads((out/'result.json').read_text());gc.collect();torch.cuda.empty_cache()
            row=dict(index=i,recipe=recipe,output=str(out),wall_seconds=time.perf_counter()-t,trainer_seconds=result['local_trainer_seconds'],post_gc_cuda_allocated_bytes=torch.cuda.memory_allocated(),checkpoint=result['checkpoint'],model=result['model']);rows.append(row)
            (a.run_output/'progress.json').write_text(json.dumps(dict(plan=plan,completed=rows),indent=2)+'\n');print(json.dumps(row),flush=True)
    except BaseException:
        (a.run_output/'interrupted.json').write_text(json.dumps(dict(completed=rows,scope='Preserve completed and recovery outputs; no automatic restart.'),indent=2)+'\n');raise
    report=dict(status='BOUNDED_RECIPE_QUEUE_COMPLETED_NOT_INDEPENDENTLY_QUALIFIED',plan=plan,arms=rows,wall_seconds=time.perf_counter()-started,scope='Each arm is independently reinitialized. CUDA process/library state is deliberately warm after the first arm. Compare exact optimizer/RNG/model fields and native export before reuse; arm process wall is not a model-speed metric.')
    (a.run_output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
