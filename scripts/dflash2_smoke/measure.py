"""One GPU lane: exact-state recovery, frozen matrix smoke, separate diagnostics."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from scripts.component_campaign.common import P, ROOT, identity, pause, pin, save
from scripts.component_campaign.archive import hydrate, release
from scripts.dflash2_smoke.train import R, OUT, COMPONENT
from scripts.dflash2_smoke.build import RUNTIME


def run(command, dest):
    save(dest.with_suffix('.command.json'),dict(argv=list(map(str,command)),owner=identity()))
    with dest.with_suffix('.log').open('w') as log:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cells',nargs='*')
    ap.add_argument('--skip-recovery',action='store_true')
    ap.add_argument('--dflash2-model',type=Path,default=R/'models/dflash2-4b-Q4_K_M.gguf')
    ap.add_argument('--phase',choices=['all','preflight','native','trace','gpu','audit'],default='all')
    a=ap.parse_args()
    pause()
    if not a.skip_recovery and not (OUT/'final-recovery/result.json').exists():
        run([sys.executable,'-m','scripts.dflash2_smoke.verify_final'],R/'verify-final')
    final=json.load(open(OUT/'result.json'))
    assert final['status']=='COMPLETE'
    registry=json.load(open(ROOT/'experiments/w1a8-standing-20261102/baseline.json'))
    save(R/'measurement-owner.json',identity())
    statuses={}
    for row in registry['rows']:
        cell=row['target']
        if a.cells and cell not in a.cells:continue
        dest=R/'matrix'/cell;dest.mkdir(parents=True,exist_ok=True)
        assert pin(row['target_pin']['path'])==row['target_pin']
        assert pin(row['draft_pin']['path'])==row['draft_pin']
        arms=dict(target=None,q4=row['draft_pin']['path'])
        cache={}
        if cell.startswith('4B'):
            arms.update(before=str(OUT/'model-0.gguf'),mid300=str(OUT/'model-300.gguf'),
                        mid900=str(OUT/'model-900.gguf'),after=final['readouts'][-1]['model']['path'],
                        HE4=str(COMPONENT/'arms/HE4/model-4096.gguf'),
                        dflash2=str(a.dflash2_model))
            cache.update({name:['q4_0','q4_0'] for name in ('before','mid300','mid900','after')})
            cache['HE4']=['f16','f16']
        save(dest/'arms.json',arms);save(dest/'cache.json',cache)
        save(dest/'plan.json',dict(registry=row,runtime=pin(RUNTIME/'build.json'),arms=arms,cache=cache,
            dflash2_variant=dict(model=pin(a.dflash2_model),f32_down_fallback='F32down' in a.dflash2_model.name),
            source=pin(__file__),repetitions=2,warmups=2,cohort='capture3 consumed DEV; no sealed heldout',
            dflash2_status='PENDING_NATIVE' if 'dflash2' in arms else 'UNSUPPORTED_NO_COMPATIBLE_9B_RELEASE',
            candidate_status='PENDING_NATIVE' if 'before' in arms else 'UNTESTED_NO_9B_CANDIDATE_WEIGHTS'))
        common=[sys.executable,'-m','scripts.component_campaign.native','--arms',dest/'arms.json',
            '--draft-cache-map',dest/'cache.json','--runtime',RUNTIME,'--target',row['target_pin']['path'],
            '--depth',str(row['draft_depth']),'--cohort','capture3','--cohort-label','consumed-DEV-smoke3']
        for phase in (['preflight','native','trace','gpu','audit'] if a.phase=='all' else [a.phase]):
            pause()
            if (dest/phase/'result.json').exists():continue
            try:
                if phase in ('gpu','audit'):
                    gpu_replay(dest, row, arms, cache, audit=phase=='audit')
                else:
                    command=common+['--out',dest/phase,'--reps','1' if phase in ('preflight','trace') else '2',
                        '--warmups','0' if phase=='preflight' else '2']
                    if phase=='preflight':command+=['--limit','1','--output-cap','32']
                    if phase=='trace':command+=['--trace','--selector-audit',R/'libselector-audit.so']
                    run(command,dest/(phase+'-worker'))
                statuses[cell+'-'+phase]='COMPLETE'
            except subprocess.CalledProcessError as error:
                statuses[cell+'-'+phase]=dict(status='FAILED',error=str(error),log=str(dest/(phase+'-worker.log')))
                save(dest/(phase+'-failure.json'),statuses[cell+'-'+phase])
                # Preserve incomplete evidence. Resolve failures before acceptance;
                # other target cells remain useful independent work.
                break
            save(R/'matrix-progress.json',statuses)
    save(R/'matrix-progress.json',statuses)
    print('MATRIX_PHASES_FINISHED',statuses,flush=True)


def gpu_replay(dest, row, arms, cache, audit=False):
    out=dest/('audit' if audit else 'gpu');out.mkdir(exist_ok=True)
    records=json.load(open(out/'progress.json')) if (out/'progress.json').exists() else []
    for rep in range(1 if audit else 2):
        order=list(arms) if rep%2==0 else list(reversed(arms))
        for name in order:
            if name=='target' or (audit and name!='dflash2'):continue
            pause();created=hydrate(arms[name])
            try:
                for domain in (('code',) if audit else ('code','reasoning','prose')):
                    capture=P/f'runs/dflash-modern-capture-20261104-04/data/development-{domain}-00/capture'
                    anchor=json.load(open(capture/'capture.json'))['prompt_tokens']+5
                    result=out/f'{rep:02d}-{name}-{domain}.json'
                    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','CUDA_','LD_','W1A8_','LLAMA_'))}
                    env.update(LD_LIBRARY_PATH=str(RUNTIME/'bin'),
                        LD_PRELOAD=str(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),
                        GGML_CUDA_DECODE_CONSISTENT='32',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',
                        EXEC_REQUEST_MASKS='1',COMPONENT_K=cache.get(name,['f16','f16'])[0],
                        COMPONENT_V=cache.get(name,['f16','f16'])[1])
                    command=[str(R/'replay'),row['target_pin']['path'],arms[name],str(capture),str(anchor),
                             '1',str(result),'1' if audit else '0',str(row['draft_depth'])]
                    if result.exists():
                        previous=json.load(open(result.with_suffix('.command.json')))
                        assert previous['argv']==command and previous['runtime']==pin(RUNTIME/'build.json')
                        assert previous['source']==pin(ROOT/'scripts/dflash2_smoke/replay.cpp')
                    else:
                        save(result.with_suffix('.command.json'),dict(argv=command,environment={k:v for k,v in env.items() if k.startswith(('GGML_','CUDA_','LD_','OMP_','EXEC_','COMPONENT_'))},
                            runtime=pin(RUNTIME/'build.json'),source=pin(ROOT/'scripts/dflash2_smoke/replay.cpp')))
                        with result.with_suffix('.log').open('w') as log:
                            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    data=json.load(open(result))
                    assert len(data['rows'])==9 and len([x for x in data['rows'] if not x['warmup']])==3
                    if not audit:
                        assert all(x['graph_count']>=1 and x['graph_event_ms']>0 for x in data['rows'] if not x['warmup'])
                    else:
                        for x in data['rows']:
                            assert any(t.get('kind')=='finite_full_vocabulary_logits' for t in x['audit'])
                            assert any(t.get('kind')=='selector_unary_candidate_identity' and t['pass'] for t in x['audit'])
                    if not any(x['rep']==rep and x['arm']==name and x['domain']==domain for x in records):
                        records.append(dict(rep=rep,arm=name,domain=domain,model=pin(arms[name]),result=data))
                    save(out/'progress.json',records)
            finally:release(arms[name],created)
    save(out/'result.json',dict(status='COMPLETE' if records else 'UNSUPPORTED_NO_MATCHED_DFLASH2',records=records,
        diagnostic_audit=audit,
        runtime=pin(RUNTIME/'build.json'),executable=pin(R/'replay'),
        timer_library=pin(P/'runs/execution-campaign-20261105-01/libgraph-timer.so'),
        source=pin(ROOT/'scripts/dflash2_smoke/replay.cpp'),target=row['target_pin'],cache_formats=cache,
        scope='Two alternating process reps; two same-process warmup passes; three changing native target histories per domain. Target capture excluded. CUDA-event sum of captured graph regions per draft invocation; CPU/inter-region work and copies outside graph regions excluded. DFlash2 has multiple regions; host proposal selection separate. Existing complete, source/command-matched cells may resume without replay.'))


if __name__=='__main__': main()
