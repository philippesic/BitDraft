"""Explicit adapter for the already-qualified row-scale W1A8 reference.

This is a rehearsal adapter, not a claim of support for unknown architectures.
Uses original frozen source/data and never lifts its bounded training horizon.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from runner import atomic, pin, read, require


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['interrupt','resume','benchmark'])
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--training-output',type=Path)
    parser.add_argument('--interruption',type=Path)
    parser.add_argument('--updates',type=int,default=5808)
    parser.add_argument('--models',type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--results',type=Path,nargs='+')
    parser.add_argument('--tier',choices=['smoke','contender'],default='contender')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    project=args.project.resolve()
    trainer=project/'scripts/design_readiness/fixed_objective.py'
    if args.mode in ('interrupt','resume'):
        argv=[sys.executable,str(trainer),'--recipe','control','--final-updates',str(args.updates),'--out',str(args.training_output),'--freeze',str(args.freeze),'--allow-cuda']
        if args.mode=='resume':
            receipt=read(args.interruption)
            require(receipt['status']=='INTERRUPTION_CHECKPOINT_SAVED','interruption missing')
            ref=receipt['checkpoint'];require(pin(ref['path'])==ref,'recovery checkpoint changed')
            argv+=['--resume',ref['path']]
            subprocess.run(argv,check=True)
            result=read(args.training_output/'result.json')
            atomic(args.output/'result.json',dict(status='RESUMED_AND_EXPORTED',training=pin(args.training_output/'result.json'),checkpoint=result['checkpoint'],model=result['model']))
            return
        # Same group as adapter: the enclosing runner also cleans this child.
        with (args.output/'trainer.log').open('xb') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=project)
            sent=False
            deadline=time.monotonic()+900
            try:
                while process.poll() is None:
                    updates=args.training_output/'updates.jsonl'
                    if not sent and updates.exists() and updates.stat().st_size>0:
                        process.send_signal(signal.SIGTERM);sent=True
                    require(time.monotonic()<deadline,'interruption probe exceeded deadline')
                    time.sleep(.02)
                require(sent and process.returncode!=0,'expected real interrupted training')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=120)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
        import torch
        checkpoints=list(args.training_output.glob('checkpoint-*.pt'))
        require(len(checkpoints)==1,'one interrupted checkpoint expected')
        saved=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
        require(saved['status']=='INTERRUPTED' and 5760<saved['cursor']<args.updates,'checkpoint did not capture interrupted updates')
        atomic(args.output/'result.json',dict(status='INTERRUPTION_CHECKPOINT_SAVED',cursor=saved['cursor'],checkpoint=pin(checkpoints[0]),trainer_returncode=process.returncode))
        return
    sys.path[:0]=[str(project/'scripts/design_readiness'),str(project/'scripts/w1a8_generalization')]
    from native_panel import Panel
    from audit_panel import audit
    models=read(args.models)
    template=copy.deepcopy(next(a for a in models['arms'] if a['kind']=='w1a8'))
    models['arms']=[a for a in models['arms'] if a['id'] in ('target_only','dflash_Q4KM','w1a8_initial','paid5760')]
    require(len(models['arms'])==4,'mandatory target/DFlash/initializer/paid controls absent')
    for index,path in enumerate(args.results):
        result=read(path);require(result['status']=='TRAINED_EXPORTED_NOT_NATIVE_ADMITTED','training result not qualified')
        require(pin(result['model']['path'])==result['model'],'export changed')
        arm=copy.deepcopy(template);arm['id']=f'candidate_{index}';arm['model']=result['model'];arm['runtime']['log_verbosity']=4;arm['runtime'].pop('loader_qualification',None)
        models['arms'].append(arm)
    models['source_pins'] += [pin(__file__)]+[pin(p) for p in args.results]
    verbose=args.output/'models-verbose.json';atomic(verbose,models)
    qualification=args.output/'qualification'
    panel=Panel(argparse.Namespace(models=verbose,inputs=args.inputs,output=qualification,repetitions=5,allow_cuda=True))
    panel.stopping=False;panel.records=[]
    signal.signal(signal.SIGTERM,panel.request_stop);signal.signal(signal.SIGINT,panel.request_stop)
    panel.authenticate()
    for arm in models['arms']:
        if not arm['id'].startswith('candidate_'):continue
        before=len(panel.records);panel.cell(0,arm);records=panel.records[before:]
        cell=qualification/('rep-00-'+arm['id']);release=read(cell/'release.json')
        require(release['owned_process_reaped'] and release['owned_process_group_absent'] and release['returncode']==0,'native group not released')
        require(len(records)==14 and all(r['token_text_stop_parity'] for r in records),'native exact output failure')
        proof=dict(schema='w1a8_model_runtime_native_loader_qualification_v1',model=arm['model'],runtime={k:v for k,v in arm['runtime'].items() if k not in ['log_verbosity','loader_qualification']},native_outputs_audited=len(records),raw_log=pin(cell/'server.log'),release=pin(cell/'release.json'),command=pin(cell/'command.json'),maps=[pin(cell/'loaded-images-warm.json'),pin(cell/'loaded-images-final.json')],models=pin(verbose),responses=[r['response'] for r in records])
        path=args.output/(arm['id']+'-qualification.json');atomic(path,proof)
        arm['runtime']['log_verbosity']=3;arm['runtime']['loader_qualification']=pin(path)
    atomic(qualification/'result.json',dict(status='NATIVE_LOADER_PARITY_PASS',records=panel.records,timing_claim=False))
    quiet=args.output/'models-quiet.json';atomic(quiet,models)
    if args.tier=='smoke':
        atomic(args.output/'result.json',dict(status='SMOKE_PASS_NO_RANKING',qualification=pin(qualification/'result.json'),timing_claim=False))
        return
    serving=args.output/'serving'
    Panel(argparse.Namespace(models=quiet,inputs=args.inputs,output=serving,repetitions=5,allow_cuda=True)).run()
    result=audit(serving,'paid5760','paid5760')
    atomic(args.output/'audit.json',result)
    atomic(args.output/'result.json',dict(status='CONTENDER_PANEL_AUDITED',audit=pin(args.output/'audit.json'),qualification=pin(qualification/'result.json'),scope='Consumed DEV. Infrastructure rehearsal; no selection or generalization claim.'))

if __name__=='__main__':main()
