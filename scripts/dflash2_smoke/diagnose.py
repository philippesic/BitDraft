"""One bounded selector failure probe and independent native head/lattice audit."""
import json
import argparse
import os
import subprocess
import sys
from scripts.component_campaign.common import P, ROOT, identity, pin, save
from scripts.dflash2_smoke.train import R
from scripts.dflash2_smoke.build import RUNTIME


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--numeric',action='store_true');parser.add_argument('--fallback',action='store_true');args=parser.parse_args()
    if args.fallback:return fallback()
    if args.numeric:return numeric()
    out=R/'diagnostics/selector01';out.mkdir(parents=True,exist_ok=False)
    save(out/'owner.json',identity())
    draft=R/'models/dflash2-4b-Q4_K_M.gguf'
    target=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'
    save(out/'arms.json',dict(target=None,dflash2=str(draft)))
    command=[sys.executable,'-m','scripts.component_campaign.native','--arms',str(out/'arms.json'),
        '--runtime',str(RUNTIME),'--out',str(out/'probe'),'--reps','1','--warmups','0','--limit','1',
        '--output-cap','32','--trace','--selector-audit',str(R/'libselector-probe.so')]
    save(out/'probe-command.json',dict(argv=command,library=pin(R/'libselector-probe.so'),source=pin(ROOT/'scripts/dflash2_smoke/selector_probe.cpp')))
    with (out/'probe-worker.log').open('w') as log:
        probe=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture'
    anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5
    command=[str(R/'replay'),str(target),str(draft),str(capture),str(anchor),'1',str(out/'audit.json'),'1','2']
    env=dict(os.environ,LD_LIBRARY_PATH=str(RUNTIME/'bin'),
        LD_PRELOAD=str(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),
        GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',EXEC_REQUEST_MASKS='1',
        COMPONENT_K='f16',COMPONENT_V='f16')
    save(out/'audit-command.json',dict(argv=command,source=pin(ROOT/'scripts/dflash2_smoke/replay.cpp'),executable=pin(R/'replay')))
    with (out/'audit.log').open('w') as log:
        audit=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(out/'result.json',dict(probe_returncode=probe.returncode,audit_returncode=audit.returncode,
        source=pin(__file__),scope='Diagnostic only; no latency accepted from these passes.'))
    print('SELECTOR_DIAGNOSTICS_FINISHED',probe.returncode,audit.returncode,flush=True)


def numeric():
    out=R/'diagnostics/selector02';out.mkdir(parents=True,exist_ok=False)
    save(out/'owner.json',identity());records=[]
    target=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'
    capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture'
    anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5
    env=dict(os.environ,LD_LIBRARY_PATH=str(RUNTIME/'bin'),
        LD_PRELOAD=str(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),
        GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',EXEC_REQUEST_MASKS='1',COMPONENT_K='f16',COMPONENT_V='f16')
    for precision in ('Q4_K_M','F16'):
        draft=R/'models'/f'dflash2-4b-{precision}.gguf'
        command=[str(R/'replay-numeric'),str(target),str(draft),str(capture),str(anchor),'1',str(out/f'{precision}.json'),'1','2']
        save(out/f'{precision}-command.json',dict(argv=command,source=pin(ROOT/'scripts/dflash2_smoke/replay.cpp'),executable=pin(R/'replay-numeric'),model=pin(draft)))
        with (out/f'{precision}.log').open('w') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(precision=precision,returncode=result.returncode))
    save(out/'result.json',dict(records=records,scope='Bounded Q4-versus-retained-F16 compatibility diagnosis; no clean latency claims.'))
    print('NUMERIC_DIAGNOSTICS_FINISHED',records,flush=True)


def fallback():
    out=R/'diagnostics/selector03';out.mkdir(parents=True,exist_ok=False)
    save(out/'owner.json',identity())
    draft=R/'models/dflash2-4b-Q4-F32down.gguf'
    target=P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf'
    capture=P/'runs/dflash-modern-capture-20261104-04/data/development-code-00/capture'
    anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5
    env=dict(os.environ,LD_LIBRARY_PATH=str(RUNTIME/'bin'),LD_PRELOAD=str(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),
        GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',EXEC_REQUEST_MASKS='1',COMPONENT_K='f16',COMPONENT_V='f16')
    command=[str(R/'replay-numeric'),str(target),str(draft),str(capture),str(anchor),'1',str(out/'audit.json'),'1','2']
    save(out/'command.json',dict(argv=command,source=pin(ROOT/'scripts/dflash2_smoke/replay.cpp'),model=pin(draft)))
    with (out/'audit.log').open('w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(out/'result.json',dict(returncode=result.returncode,scope='One declared F32-down precision fallback; no latency accepted here.'))
    print('FALLBACK_AUDIT_FINISHED',result.returncode,flush=True)


if __name__=='__main__':main()
