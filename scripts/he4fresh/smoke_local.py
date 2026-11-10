"""Short matched old-remote/new-local training diagnostic from one saved state.

Both arms are scratch continuations; neither overwrites/adopts the paused run.
The user authorized this diagnostic while sustained training remains paused.
"""
import argparse
import collections
import contextlib
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .common import *


def measured_capture(factory,row):
    """Time the actual constructors, restoring all patched functions immediately."""
    costs=collections.Counter();depth=collections.Counter();patches=[]
    def wrap(obj,name,category):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            outer=depth[category]==0;depth[category]+=1;start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                depth[category]-=1
                if outer:costs[category]+=time.perf_counter()-start
        patches.append((obj,name,original));setattr(obj,name,measured)
    try:
        wrap(Path,'read_bytes','file_reads');wrap(hashlib,'sha256','sha256')
        for name in ('load','frombuffer','concatenate'):wrap(np,name,'array_parse_and_copy')
        wrap(torch,'from_numpy','array_parse_and_copy')
        wrap(torch,'isfinite','finite_feature_check');wrap(torch.Tensor,'all','finite_feature_check')
        start=time.perf_counter();capture=factory(row);elapsed=time.perf_counter()-start
    finally:
        for obj,name,original in reversed(patches):setattr(obj,name,original)
    costs['metadata_and_python']=max(0.,elapsed-sum(costs.values()))
    return capture,dict(costs)


def worker(mode,out,steps):
    from . import data as adapter
    from .train import source_model,tensor_hash
    from scripts.component_campaign.tensor_checkpoint import memory_load
    from scripts.component_campaign.gpu_resume import isolated_optimizer_state
    from scripts.component_campaign.qualify_tensor_checkpoint import fingerprint
    from scripts.dflash_baseline.pipeline import load_parameters,restore_rng,rng,update
    from scripts.dflash_baseline.prototype_model import optimizer
    from scripts.dflash_baseline.prototype import digest
    began=time.perf_counter();torch.set_num_threads(2)
    torch.manual_seed(8101);random.seed(8101);np.random.seed(8101)
    torch.backends.cuda.matmul.allow_tf32=False
    recipe=json.load(open(R/'recipe.json'));config=recipe['config']
    adapter.local_admission(config)
    index=json.load(open(config['data']['path']))
    windows=[(i,a) for i,row in enumerate(index['rows']) for a in row['anchors']]
    parent=json.load(open(OUT/'recovery.json'))
    assert pin(parent['archive'])['sha256']==parent['archive_sha256']
    model=source_model(config,recipe['spec']).to('cuda');opt=optimizer(model)
    state=memory_load(parent['archive'])
    assert state['config_hash']==digest(config) and state['component_spec']==recipe['spec']
    load_parameters(model,state['parameters'])
    opt.load_state_dict(isolated_optimizer_state(state['optimizer']))
    assert all(torch.equal(p.detach().cpu(),state['parameters'][n]) for n,p in model.named_parameters())
    restored=opt.state_dict();assert restored['param_groups']==state['optimizer']['param_groups']
    for key,fields in state['optimizer']['state'].items():
        for name,expected in fields.items():
            actual=restored['state'][key][name]
            assert torch.equal(actual.cpu(),expected.cpu()) if torch.is_tensor(expected) else actual==expected
    progress=copy.deepcopy(state['progress']);restore_rng(state['rng'])
    assert fingerprint(rng())==fingerprint(state['rng'])
    initial_updates=progress['updates'];assert {int(v['step']) for v in opt.state.values()}=={initial_updates}
    del state,restored;gc.collect()
    adapter.TEACHER=R/'teacher-remote-comparator' if mode=='legacy' else TEACHER
    factory=adapter.TeacherCapture if mode=='legacy' else adapter.LocalTeacherCapture
    current_costs={}
    def make(row):
        capture,costs=measured_capture(factory,row);current_costs.update(costs);return capture
    pool=adapter.LazyTeachers(index['rows'],make)
    setup_seconds=time.perf_counter()-began;records=[];loop=time.perf_counter()
    for j in range(steps):
        current_costs.clear();start=time.perf_counter()
        sequence,anchor=windows[progress['order'][progress['cursor']%len(windows)]]
        capture=pool[sequence];data_seconds=time.perf_counter()-start
        torch.cuda.synchronize();start_update=time.perf_counter()
        loss=update(model,opt,lambda:capture.loss(model,anchor)[0]);torch.cuda.synchronize()
        update_seconds=time.perf_counter()-start_update
        assert math.isfinite(loss) and math.isfinite(model.last_gradient_norm)
        progress['cursor']+=1;progress['updates']+=1
        assert {int(v['step']) for v in opt.state.values()}=={progress['updates']}
        record=dict(step=j+1,sequence=sequence,anchor=anchor,loss=loss,gradient_norm=model.last_gradient_norm,
            loader_seconds=data_seconds,update_seconds=update_seconds,categories=current_costs.copy())
        records.append(record)
        with (out/(mode+'-steps.jsonl')).open('a') as stream:stream.write(json.dumps(record)+'\n')
    loop_seconds=time.perf_counter()-loop
    # State identity diagnostics are outside the timed training loop.
    final=dict(parameters={n:tensor_hash(p) for n,p in model.named_parameters()},
        groups=opt.state_dict()['param_groups'],optimizer={},rng=fingerprint(rng()))
    for key,fields in opt.state_dict()['state'].items():
        final['optimizer'][key]={n:tensor_hash(v) if torch.is_tensor(v) else v for n,v in fields.items()}
    totals=collections.Counter()
    for row in records:
        for k,v in row['categories'].items():totals[k]+=v
    result=dict(status='PASS',mode=mode,parent=parent,parent_updates=initial_updates,steps=steps,
        final_updates=progress['updates'],final_cursor=progress['cursor'],setup_seconds=setup_seconds,
        training_loop_seconds=loop_seconds,loader_seconds=sum(x['loader_seconds'] for x in records),
        update_seconds=sum(x['update_seconds'] for x in records),categories=dict(totals),records=records,
        final_state_digest=hashlib.sha256(json.dumps(final,sort_keys=True).encode()).hexdigest(),
        full_parameters_adam_rng_restored=True,all_adam_steps=progress['updates'],production_state_modified=False,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    save(out/(mode+'-result.json'),result)
    print(mode,'SMOKE_PASS',loop_seconds,flush=True)


def controller(out,steps):
    assert not out.exists();out.mkdir(parents=True)
    from .data import local_admission
    local_admission(json.load(open(R/'recipe.json'))['config'])
    control=json.load(open(Path.home()/'.config/binary-eagle-decoding/gpu-control.json'))
    assert control['rtx5080']['pause_requested'], 'Sustained training must stay paused'
    before=pin(OUT/'recovery.json');locks=[];remote=R/'teacher-remote-comparator'
    try:
        for path in (P/'runs/contender-gpu.lock',Path('/home/philip/.cache/semantic-offload/gpu-0.lock'),Path('/tmp/vllm-semantic-cache-gpu.lock')):
            f=path.open('a+');locks.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        save(out/'gpu-owner.json',dict(owner=identity(),locks=[str(f.name) for f in locks],
            authorization='Philip requested a very short old/new local-loader smoke while sustained training is paused.'))
        remote.mkdir(exist_ok=True)
        assert not remote.is_mount()
        subprocess.run([str(R/'tools/unpacked/usr/bin/sshfs'),'pstack-2080:/home/philip/he4-capture-2080ti',str(remote),
            '-o','ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3'],check=True)
        for mode in ('legacy','local'):
            with (out/(mode+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-m','scripts.he4fresh.smoke_local','--worker',mode,
                    '--out',str(out),'--steps',str(steps)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        old=json.load(open(out/'legacy-result.json'));new=json.load(open(out/'local-result.json'))
        assert old['parent']==new['parent'] and old['final_cursor']==new['final_cursor']
        assert [(x['sequence'],x['anchor']) for x in old['records']]==[(x['sequence'],x['anchor']) for x in new['records']]
        assert np.allclose([x['loss'] for x in old['records']],[x['loss'] for x in new['records']],rtol=1e-6,atol=1e-6)
        assert pin(OUT/'recovery.json')==before
        save(out/'result.json',dict(status='MATCHED_LOCAL_LOADER_SMOKE_PASS',steps=steps,
            old=pin(out/'legacy-result.json'),new=pin(out/'local-result.json'),
            exact_final_state_equal=old['final_state_digest']==new['final_state_digest'],losses_match=True,
            loop_speedup=old['training_loop_seconds']/new['training_loop_seconds'],
            loader_speedup=old['loader_seconds']/new['loader_seconds'],
            parent_recovery_receipt_unchanged=True,sustained_training_paused=True,
            note='Scratch diagnostic updates only; neither arm adopts state or resets/extends the original training allocation.'))
    finally:
        if remote.is_mount():subprocess.run(['fusermount3','-u',str(remote)],check=True)
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
        save(out/'gpu-release.json',dict(checked_unix=time.time(),owner=identity(),
            clients=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['legacy','local']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=32);args=ap.parse_args()
    assert 1<=args.steps<=64
    worker(args.worker,args.out,args.steps) if args.worker else controller(args.out,args.steps)
