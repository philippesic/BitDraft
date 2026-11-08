"""Continue the retained HE4 state for synchronized update wall time."""
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
from scripts.component_campaign.gpu_resume import qualify, isolated_optimizer_state
from scripts.component_campaign.qualify_tensor_checkpoint import fingerprint
from scripts.component_campaign.simulation import install as install_component
from scripts.component_campaign.train_impl import evaluation
from scripts.dflash_baseline.components import install_components
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.model import Capture, Model
from scripts.dflash_baseline.pipeline import load_parameters, parameters, restore_rng, rng, update
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.prototype_model import install, optimizer

R = P / 'runs/he4-continuation-20261107-01'
OUT = R / 'candidate'
DEST = R / 'checkpoints'
BUDGET = 21600
POINTS = [300, 1800, 3600, 7200, 10800, 14400, 18000, BUDGET]


def main():
    began = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    DEST.mkdir(exist_ok=False)
    if shutil.disk_usage(DEST).free < 48_000_000_000:
        raise RuntimeError('Reserve final-state headroom before training')
    save(OUT / 'ownership.json', identity())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    retained = json.load(open(COMPONENT / 'frontier-retention.json'))
    receipt = next(v for v in retained['states'].values() if v['original_arm'] == 'HE4')
    assert receipt['archive'] == '/mnt/c/binary-eagle-component-20261105-01/HE4-confirm-4096.pt.bsz'
    assert receipt['archive_sha256'] == '0a33150b21316d0bfa4392309ab670a3fcb4e21d5264909cd4b0421651a4dad4'
    assert receipt['archive_bytes'] == 4250228027
    assert receipt['raw_sha256'] == '96b396a1c32bb53a08d87d746c9aa08555b1ea6a013f616e7e936abec37d3188'
    assert receipt['native']['sha256'] == 'd24a366aac5c1c3a8757b0533ad03bc87facd386b91350c42831b054ca72a17b'
    assert receipt['same_history_reproduction'] is True
    source = Path(receipt['archive'])
    actual = pin(source)
    assert actual['sha256'] == receipt['archive_sha256'] and actual['bytes'] == receipt['archive_bytes']
    from scripts.component_campaign.checkpoint import validate
    validate(source, receipt['raw_sha256'], receipt['raw_bytes'])
    assert pin(receipt['native']['path']) == receipt['native']
    assert pin(receipt['confirmed_result']['path']) == receipt['confirmed_result']
    state = load(source)
    assert state['resume_capable'] and state['config_hash'] == digest(state['config'])
    assert state['progress']['updates'] == 4096 and len(state['parameters']) == 104
    assert {int(v['step']) for v in state['optimizer']['state'].values()} == {4096}
    spec, config = state['component_spec'], state['config']
    assert spec == dict(name='HE4', head='Q4_K', embedding='Q4_K', tied=True)
    assert config['schedule'] == 'constant'
    assert len(config['coverage']) == 18
    assert all(v == dict(weight='W1', bits=8 if n.endswith('down') else 4) for n,v in config['coverage'].items())
    assert config['components'] == dict(fusion='Q4_0', attention_qk='Q4_0', attention_vo='Q4_0')
    assert all(v.dtype == torch.float32 for v in state['parameters'].values())
    base = OUT / 'zero-base.gguf'
    export(None, base, state=state, checkpoint_pin=actual)
    rewrite(base, OUT / 'model-0.gguf', spec, deployment_exps=state['deployment_exps'])
    assert pin(OUT / 'model-0.gguf')['sha256'] == receipt['native']['sha256']
    base.unlink(); base.with_suffix('.json').unlink()
    qualification = qualify(state, 'HE4', OUT / 'qualification')
    assert {int(v['step']) for v in state['optimizer']['state'].values()} == {4096}
    save(OUT / 'zero-resume-proof.json', dict(status='PASS', native_byte_identity=True,
        parent=actual, qualification=qualification, qualification_charged_to_budget=False))
    model = Model(config['source']['path'], config['evaluation']['target']['path'])
    install(model, config, {})
    install_components(model, config)
    install_component(model, spec)
    model.to('cuda')
    opt = optimizer(model)
    load_parameters(model, state['parameters'])
    opt.load_state_dict(isolated_optimizer_state(state['optimizer']))
    assert {int(v['step']) for v in opt.state.values()} == {4096}
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
    assert len(train) == 24 and len(windows) == 1426
    assert sorted(progress['order']) == list(range(len(windows)))
    unique = {(str(cap.path), a+j-1) for cap in train for a in cap.anchors for j in range(1,16)}
    assert len(unique) == 1762
    saved_rng = state['rng']
    lineage = copy.deepcopy(state['lineage'])
    lineage['continuation'] = dict(parent=actual, budget='21600 seconds synchronized forward/backward/optimizer updates',
        optimizer='Restored all Adam state, groups and learning rates; no reset', source=pin(__file__))
    save(OUT / 'resolved.json', dict(config=config, spec=spec, parent=actual, lineage=lineage,
        windows=len(windows), unique_labels=len(unique), sequences=len(train),
        source_pins={str(p.relative_to(ROOT)): pin(p) for d in ('component_campaign','dflash_baseline','he4_continuation') for p in (ROOT/'scripts'/d).glob('*.py')},
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
    thresholds = iter(POINTS); next_point = next(thresholds)
    times = []; exposed = set()
    def recovery_snapshot(label):
        if shutil.disk_usage(DEST).free < 6_000_000_000:
            raise RuntimeError('Checkpoint space reserve exhausted; previous good checkpoint preserved')
        torch.cuda.synchronize()
        stored, _ = values(model, spec)
        snapshot = dict(version=2,config=config,config_hash=digest(config),parameters=parameters(model),
            optimizer=opt.state_dict(),progress=copy.deepcopy(progress),rng=rng(),lineage=lineage,
            component_spec=spec,resume_capable=True,deployment_exps=stored)
        receipt = save_state(DEST / f'HE4-{label}-{progress["updates"]}.pt', snapshot)
        save(OUT/'recovery.json', receipt)
        return receipt
    recovery = receipt
    checkpoint_seconds = 0.
    while spent < BUDGET:
        try: pause()
        except RuntimeError: stopped = True
        if stopped: break
        i, anchor = windows[progress['order'][progress['cursor'] % len(windows)]]
        torch.cuda.synchronize(); start = time.perf_counter()
        try:
            loss = update(model, opt, lambda: train[i].loss(model, anchor)[0])
        except BaseException as error:
            save(OUT/'training-failure.json', dict(error=repr(error), last_good_recovery=recovery,
                actual_completed_update_seconds=spent, completed_additional_updates=len(times)))
            raise
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        times.append(elapsed); spent += elapsed
        progress['update_seconds'] += elapsed; progress['updates'] += 1
        progress['cursor'] += 1; progress['supervised_tokens'] += 15
        if {int(v['step']) for v in opt.state.values()} != {progress['updates']}:
            recovery_snapshot('invalid-step')
            raise ValueError('Live Adam steps differ from actual counted updates')
        exposed.update((str(train[i].path),anchor+j-1) for j in range(1,16))
        with (OUT/'updates.jsonl').open('a') as f:
            f.write(json.dumps(dict(update=progress['updates'],loss=loss,gradient_norm=model.last_gradient_norm,
                seconds=elapsed,cumulative_seconds=spent,sequence=str(train[i].path),anchor=anchor))+'\n')
        if progress['updates'] % 100 == 0:
            if shutil.disk_usage(DEST).free < 12_000_000_000:
                stopped = True
                save(OUT/'resource-stop.json', dict(reason='Disk reserve', free_bytes=shutil.disk_usage(DEST).free))
        save(OUT/'live.json',dict(update=progress['updates'],additional_seconds=spent,last_loss=loss,owner=identity()))
        print('UPDATE', progress['updates'], spent, loss, flush=True)
        if spent >= next_point:
            try:
                checkpoint_started = time.perf_counter()
                recovery = recovery_snapshot('plus'+str(next_point))
                checkpoint_seconds += time.perf_counter()-checkpoint_started
                point(str(next_point))
                readouts[-1]['recovery'] = recovery
                save(OUT/'readouts.json', readouts)
            except BaseException as error:
                save(OUT/'readout-failure.json', dict(error=repr(error), actual_update_seconds=spent,
                    additional_updates=len(times), recovery=recovery, progress=progress))
                raise
            next_point = next(thresholds, float('inf'))
    save_started = time.perf_counter()
    if stopped:
        recovery = recovery_snapshot('exact-paused')
        checkpoint_seconds += time.perf_counter()-save_started
    if stopped: point('paused')
    assert pin(source) == actual, 'Immutable parent changed during training'
    save(OUT/'result.json',dict(status='PAUSED' if stopped else 'COMPLETE',parent=actual,
        starting_progress={k:v for k,v in before_progress.items() if k!='order'},
        final_progress={k:v for k,v in progress.items() if k!='order'},
        additional_updates=len(times),actual_update_seconds=spent,budget_seconds=21600,
        one_update_overshoot_seconds=max(0,spent-21600),last_update_seconds=times[-1] if times else None,
        supervised_positions=len(times)*15,unique_positions_exposed=len(exposed),unique_windows=len(windows),
        total_unique_label_positions=len(unique),sequences=len(train),setup_and_qualification_seconds=setup_seconds,
        all_checkpoint_seconds=checkpoint_seconds,process_wall_seconds=time.perf_counter()-began,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        readouts=readouts,recovery=recovery,source=pin(__file__)))
    print('CONTINUATION_COMPLETE',spent,len(times),flush=True)


if __name__ == '__main__': main()
