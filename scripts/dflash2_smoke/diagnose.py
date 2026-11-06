"""One bounded selector failure probe and independent native head/lattice audit."""
import json
import os
import subprocess
import sys
from scripts.component_campaign.common import P, ROOT, identity, pin, save
from scripts.dflash2_smoke.train import R
from scripts.dflash2_smoke.build import RUNTIME


def main():
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


if __name__=='__main__':main()
