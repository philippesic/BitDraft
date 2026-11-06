"""Continue the retained Csmallbroad state for synchronized update wall time."""
import copy
import gc
import json
import resource
import shutil
import signal
import time
from pathlib import Path

import torch

from scripts.component_campaign.common import P, ROOT, R as COMPONENT, identity, pause, pin, save
from scripts.component_campaign.tensor_checkpoint import load
from scripts.component_campaign.checkpoint import save_state
from scripts.component_campaign.deployment import values
from scripts.component_campaign.export import rewrite
from scripts.component_campaign.gpu_resume import qualify
from scripts.component_campaign.qualify_tensor_checkpoint import fingerprint
from scripts.component_campaign.simulation import install as install_component
from scripts.component_campaign.train_impl import evaluation
from scripts.dflash_baseline.components import install_components
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.model import Capture, Model
from scripts.dflash_baseline.pipeline import load_parameters, parameters, restore_rng, rng, update
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.prototype_model import install, optimizer

R = P / 'runs/dflash2-smoke-20261106-01'
OUT = R / 'candidate'
DEST = Path('/mnt/c/binary-eagle-dflash2-smoke-20261106-01')


def main():
    began = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    DEST.mkdir(exist_ok=True)
    if shutil.disk_usage(DEST).free < 4_800_000_000:
        raise RuntimeError('Reserve final-state headroom before training')
    save(OUT / 'ownership.json', identity())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    retained = json.load(open(COMPONENT / 'frontier-retention.json'))
    receipt = next(v for v in retained['states'].values() if v['original_arm'] == 'Csmallbroad')
    source = Path(receipt['archive'])
    actual = pin(source)
    assert actual['sha256'] == receipt['archive_sha256'] and actual['bytes'] == receipt['archive_bytes']
    state = load(source)
    assert state['resume_capable'] and state['config_hash'] == digest(state['config'])
    assert state['progress']['updates'] == 4096 and len(state['parameters']) == 104
    assert {int(v['step']) for v in state['optimizer']['state'].values()} == {4096}
    spec, config = state['component_spec'], state['config']
    base = OUT / 'zero-base.gguf'
    export(None, base, state=state, checkpoint_pin=actual)
    rewrite(base, OUT / 'model-0.gguf', spec, deployment_exps=state['deployment_exps'])
    assert pin(OUT / 'model-0.gguf')['sha256'] == receipt['native']['sha256']
    base.unlink(); base.with_suffix('.json').unlink()
    qualification = qualify(state, 'Csmallbroad', OUT / 'qualification')
    save(OUT / 'zero-resume-proof.json', dict(status='PASS', native_byte_identity=True,
        parent=actual, qualification=qualification, qualification_charged_to_budget=False))
    model = Model(config['source']['path'], config['evaluation']['target']['path'])
    install(model, config, {})
    install_components(model, config)
    install_component(model, spec)
    model.to('cuda')
    opt = optimizer(model)
    load_parameters(model, state['parameters'])
    opt.load_state_dict(state['optimizer'])
    assert all(torch.equal(p.detach().cpu(), state['parameters'][n]) for n, p in model.named_parameters())
    restored = opt.state_dict()
    assert restored['param_groups'] == state['optimizer']['param_groups']
    for key, saved in state['optimizer']['state'].items():
        for field, expected in saved.items():
            current = restored['state'][key][field]
            assert torch.equal(current.detach().cpu(), expected) if torch.is_tensor(expected) else current == expected
    del restored
    progress = copy.deepcopy(state['progress'])
    before_progress = copy.deepcopy(progress)
    manifest = json.load(open(config['data']['path']))
    rows = [r for r in manifest['records'] if r['split'] == 'TRAIN']
    devrows = [r for r in manifest['records'] if r['split'] == 'consumed_DEV']
    for ref in [config['source'], config['data'], config['evaluation']['target']] + [p for r in rows + devrows for p in [r['metadata'], *r['files']]]:
        assert pin(ref['path'])['sha256'] == ref['sha256']
    train = [Capture(Path(r['metadata']['path']).parent) for r in rows]
    dev = [Capture(Path(r['metadata']['path']).parent) for r in devrows]
    for cap, row in zip(train, rows, strict=True):
        if 'training_last_input_position' in row:
            cap.anchors = [a for a in cap.anchors if a + 15 <= row['training_last_input_position']]
    windows = [(i, a) for i, cap in enumerate(train) for a in cap.anchors]
    assert len(train) == 40 and len(windows) == 3234
    assert sorted(progress['order']) == list(range(len(windows)))
    unique = {(str(cap.path), a+j-1) for cap in train for a in cap.anchors for j in range(1,16)}
    assert len(unique) == 3794
    saved_rng = state['rng']
    lineage = copy.deepcopy(state['lineage'])
    lineage['continuation'] = dict(parent=actual, budget='1800 seconds synchronized forward/backward/optimizer updates',
        optimizer='Restored all Adam state, groups and learning rates; no reset', source=pin(__file__))
    save(OUT / 'resolved.json', dict(config=config, spec=spec, parent=actual, lineage=lineage,
        windows=len(windows), unique_labels=len(unique), sequences=len(train),
        source_pins={str(p.relative_to(ROOT)): pin(p) for d in ('component_campaign','dflash_baseline','dflash2_smoke') for p in (ROOT/'scripts'/d).glob('*.py')},
        parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype),trainable=p.requires_grad) for n,p in model.named_parameters()}))
    del state
    gc.collect()
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    spent = 0.
    readouts = []
    def point(label, reuse=False):
        started = time.perf_counter()
        metrics = evaluation(model, dev)
        evaluation_seconds = time.perf_counter()-started
        stored, _ = values(model, spec)
        if not reuse:
            export_state = dict(version=2, config=config, config_hash=digest(config), parameters=parameters(model),
                                progress=copy.deepcopy(progress), deployment_exps=stored, lineage=lineage)
            base = OUT / f'base-{label}.gguf'
            export(None, base, state=export_state, checkpoint_pin=actual)
            rewrite(base, OUT / f'model-{label}.gguf', spec, deployment_exps=stored)
            base.unlink(); base.with_suffix('.json').unlink()
        readouts.append(dict(label=label, progress={k:v for k,v in progress.items() if k!='order'},
            additional_update_seconds=spent, dev=metrics, model=pin(OUT/f'model-{label}.gguf'),
            evaluation_seconds=evaluation_seconds, readout_seconds=time.perf_counter()-started,
            elapsed_process_seconds=time.perf_counter()-began))
        save(OUT/'readouts.json', readouts)
        print('READOUT', label, metrics['ce'], metrics['agreement'], flush=True)
    point('0', reuse=True)
    restore_rng(saved_rng)
    assert fingerprint(rng()) == fingerprint(saved_rng)
    setup_seconds = time.perf_counter()-began
    thresholds = iter([300,900,1800]); next_point = next(thresholds)
    times = []; exposed = set()
    def recovery_snapshot(label):
        torch.cuda.synchronize()
        stored, _ = values(model, spec)
        snapshot = dict(version=2,config=config,config_hash=digest(config),parameters=parameters(model),
            optimizer=opt.state_dict(),progress=copy.deepcopy(progress),rng=rng(),lineage=lineage,
            component_spec=spec,resume_capable=True,deployment_exps=stored)
        receipt = save_state(DEST / f'Csmallbroad-{label}-{progress["updates"]}.pt', snapshot)
        save(OUT/'recovery.json', receipt)
        return receipt
    while spent < 1800:
        try: pause()
        except RuntimeError: stopped = True
        if stopped: break
        i, anchor = windows[progress['order'][progress['cursor'] % len(windows)]]
        torch.cuda.synchronize(); start = time.perf_counter()
        loss = update(model, opt, lambda: train[i].loss(model, anchor)[0])
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        times.append(elapsed); spent += elapsed
        progress['update_seconds'] += elapsed; progress['updates'] += 1
        progress['cursor'] += 1; progress['supervised_tokens'] += 15
        exposed.update((str(train[i].path),anchor+j-1) for j in range(1,16))
        with (OUT/'updates.jsonl').open('a') as f:
            f.write(json.dumps(dict(update=progress['updates'],loss=loss,gradient_norm=model.last_gradient_norm,
                seconds=elapsed,cumulative_seconds=spent,sequence=str(train[i].path),anchor=anchor))+'\n')
        save(OUT/'live.json',dict(update=progress['updates'],additional_seconds=spent,last_loss=loss,owner=identity()))
        print('UPDATE', progress['updates'], spent, loss, flush=True)
        if spent >= next_point:
            try:
                point(str(next_point))
            except BaseException as error:
                recovery = recovery_snapshot('readout-failed')
                save(OUT/'readout-failure.json', dict(error=repr(error), actual_update_seconds=spent,
                    additional_updates=len(times), recovery=recovery, progress=progress))
                raise
            next_point = next(thresholds, float('inf'))
    if stopped:
        recovery_snapshot('paused')
        point('paused')
    torch.cuda.synchronize()
    stored, _ = values(model,spec)
    final_state = dict(version=2,config=config,config_hash=digest(config),parameters=parameters(model),
        optimizer=opt.state_dict(),progress=copy.deepcopy(progress),rng=rng(),lineage=lineage,
        component_spec=spec,resume_capable=True,deployment_exps=stored)
    save_started = time.perf_counter()
    recovery = save_state(DEST / f'Csmallbroad-plus1800-{progress["updates"]}.pt', final_state)
    save(OUT/'recovery.json', recovery)
    save(OUT/'result.json',dict(status='PAUSED' if stopped else 'COMPLETE',parent=actual,
        starting_progress={k:v for k,v in before_progress.items() if k!='order'},
        final_progress={k:v for k,v in progress.items() if k!='order'},
        additional_updates=len(times),actual_update_seconds=spent,budget_seconds=1800,
        one_update_overshoot_seconds=max(0,spent-1800),last_update_seconds=times[-1] if times else None,
        supervised_positions=len(times)*15,unique_positions_exposed=len(exposed),unique_windows=3234,
        total_unique_label_positions=3794,sequences=40,setup_and_qualification_seconds=setup_seconds,
        final_checkpoint_seconds=time.perf_counter()-save_started,process_wall_seconds=time.perf_counter()-began,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        readouts=readouts,recovery=recovery,source=pin(__file__)))
    print('CONTINUATION_COMPLETE',spent,len(times),flush=True)


if __name__ == '__main__': main()
