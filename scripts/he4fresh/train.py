"""Thin time-budget segments using the existing QAT, export and checkpoint helpers."""
import argparse
import copy
import gc
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .common import *
from scripts.component_campaign.checkpoint import save_state
from scripts.component_campaign.tensor_checkpoint import memory_load
from scripts.component_campaign.gpu_resume import isolated_optimizer_state
from scripts.component_campaign.simulation import install as install_component
from scripts.component_campaign.train_impl import evaluation
from scripts.component_campaign.export import rewrite
from scripts.dflash_baseline.components import install_components
from scripts.dflash_baseline.export_precision import export
from scripts.dflash_baseline.model import Model, Capture, GGUFReader, dequantize
from scripts.dflash_baseline.pipeline import parameters, load_parameters, rng, restore_rng, update
from scripts.dflash_baseline.prototype import digest
from scripts.dflash_baseline.prototype_model import install, optimizer


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reconcile_log(progress):
    path = OUT/'updates.jsonl'
    if not path.exists():
        assert progress['updates'] == 0
        return
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert len(lines) >= progress['updates'], 'Durable checkpoint update log is incomplete'
    durable = lines[:progress['updates']]
    assert [json.loads(line)['update'] for line in durable] == list(range(1,progress['updates']+1))
    extra = lines[progress['updates']:]
    if not extra: return
    # Preserve all rejected exposure and charge its time to the original budget.
    incident = R/'recovery-incidents'/str(time.time_ns())
    incident.mkdir(parents=True)
    (incident/'uncheckpointed-updates.jsonl').write_bytes(b''.join(extra))
    rows=[]
    for line in extra:
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    lost_wall = max((x.get('training_seconds',0.) for x in rows),default=progress['training_seconds'])-progress['training_seconds']
    lost_update = math.fsum(x['seconds'] for x in rows)
    progress['training_seconds'] += max(0,lost_wall)
    progress['discarded_update_seconds'] = progress.get('discarded_update_seconds',0.)+lost_update
    progress['discarded_updates'] = progress.get('discarded_updates',0)+len(rows)
    save(incident/'accounting.json',dict(discarded_updates=len(rows),update_seconds=lost_update,training_seconds=lost_wall,
        reason='Exact checkpoint rollback; rejected exposures preserved and charged'))
    with path.open('wb') as out:
        out.write(b''.join(durable));out.flush();__import__('os').fsync(out.fileno())


from .data import data


def source_model(config, spec, prove=False):
    # Model already creates BlockBinaryLinear from source. Supply the full map as
    # well, and verify all 18 tensors against Projection after install().
    authenticate(json.load(open(R/'q4-matrix-proof.json'))['basis'])
    model = Model(config['source']['path'], config['evaluation']['target']['path'])
    reader = GGUFReader(config['source']['path'])
    weights = {}
    for tensor in reader.tensors:
        key = tensor.name.removesuffix('.weight')
        if key in config['coverage']:
            assert tensor.tensor_type.name == 'BF16'
            weights[key] = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type).reshape(
                tuple(tensor.shape[::-1])).astype(np.float32))
    assert set(weights) == set(config['coverage']) and len(weights) == 18
    install(model, config, weights)
    proofs = {}
    for key, projection in model.binary_linears().items():
        weight = weights[key]
        assert torch.equal(projection.latent_sign, weight)
        assert torch.allclose(projection.scale, weight.abs().mean(-1), rtol=0, atol=1e-8)
        assert torch.count_nonzero(projection.midpoint) == 0
        assert torch.count_nonzero(projection.log_gain) == 0
        bound = config['quantizer'].get('initial_bounds', {}).get(key, config['quantizer']['clip'])
        assert projection.log_clip.item() == torch.tensor(math.log(bound), dtype=torch.float32).item()
        assert not torch.all((weight == 1) | (weight == -1))
        proofs[key] = dict(source_weight_sha256=tensor_hash(weight),
            latent_sha256=tensor_hash(projection.latent_sign), scale_sha256=tensor_hash(projection.scale),
            midpoint_zero=True, log_gain_zero=True, initial_bound=bound,
            exact_source_latents=True, mean_abs_scale_max_error=float((projection.scale-weight.abs().mean(-1)).abs().max()))
    del weights, reader, weight, projection
    gc.collect()
    install_components(model, config)
    install_component(model, spec)
    if prove:
        # Compare to existing JSON digests only: no old parameter/optimizer tensor
        # is deserialized, even for the negative initialization controls.
        baseline_path = COMPONENT.parent / 'precision-campaign-20261104-01/finalists/G4D8-learned-small/model-final.json'
        audit_path = COMPONENT / 'arms/HE4/native-tensor-audit.json'
        baseline = json.load(open(baseline_path))['tensor_hashes']
        he4 = {t['name']: t['sha256'] for r in json.load(open(audit_path))['records']
               if r['update'] == 4096 for t in r['tensors']}
        comparisons = {}
        for name, reference in [('B0', baseline), ('HE4-4096', he4)]:
            differing = []
            for key, projection in model.binary_linears().items():
                native = key + '.w1a1_scale'
                actual = tensor_hash(projection.scale)
                if actual != reference[native]:
                    differing.append(dict(parameter=key+'.scale', source_init_sha256=actual,
                                          reference_sha256=reference[native]))
            assert differing, 'Source init unexpectedly equals '+name
            comparisons[name] = differing
        save(OUT / 'init-proof.json', dict(status='PASS', initializer='authenticated BF16 source via Model/BlockBinaryLinear/Projection',
            source=config['source'], target=config['evaluation']['target'], projections=proofs,
            old_parameter_tensors_loaded=False, old_optimizer_state_loaded=False, random_signs=False,
            comparison_receipts=[pin(baseline_path), pin(audit_path)], comparisons=comparisons,
            parameter_hashes={n:tensor_hash(p) for n,p in model.named_parameters()},
            q4_matrix=json.load(open(R/'q4-matrix-proof.json'))))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--until', type=int, required=True, choices=POINTS)
    args = ap.parse_args()
    began = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(8101); random.seed(8101); np.random.seed(8101)
    torch.backends.cuda.matmul.allow_tf32 = False
    resolved = json.load(open(R/'recipe.json'))
    config, spec = resolved['config'], resolved['spec']
    train, dev, windows = data(config)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    model = source_model(config, spec, prove=args.until == 0)
    model.to('cuda')
    opt = optimizer(model)  # Exact unchanged defaults, including weight_decay=0.
    rates = {g['name']:g['lr'] for g in opt.param_groups}
    assert rates == dict(sign=1e-3,scale=1e-5,midpoint=1e-5,gain=1e-5,clip=1e-5,norm=1e-5)
    assert all(g['weight_decay'] == 0 for g in opt.param_groups)
    order = list(range(len(windows))); random.Random(8101).shuffle(order)
    progress = dict(updates=0, cursor=0, order=order, update_seconds=0., training_seconds=0., data_io_seconds=0., supervised_tokens=0)
    lineage = dict(source=config['source'], parent=config['source'], data=config['data'],
        recipe='HE4broad fresh quantization on combined 2080Ti data; no trained parameter/optimizer load', optimizer_policy='fresh Adam at zero; exact own-state resumes',
        window_order_seed=8101, objective='existing Capture.loss: 15-token decay-weighted hard-target CE')
    if (OUT/'recovery.json').exists():
        receipt = json.load(open(OUT/'recovery.json'))
        assert pin(receipt['archive'])['sha256'] == receipt['archive_sha256']
        state = memory_load(receipt['archive'])
        assert state['config_hash'] == digest(config) and state['component_spec'] == spec
        load_parameters(model, state['parameters'])
        opt.load_state_dict(isolated_optimizer_state(state['optimizer']))
        progress = copy.deepcopy(state['progress']); lineage = state['lineage']
        restore_rng(state['rng'])
        assert all(torch.equal(p.detach().cpu(), state['parameters'][n]) for n,p in model.named_parameters())
        assert {int(v['step']) for v in opt.state.values()} == ({progress['updates']} if progress['updates'] else set())
        restored = opt.state_dict()
        assert restored['param_groups'] == state['optimizer']['param_groups']
        for key, fields in state['optimizer']['state'].items():
            for field, expected in fields.items():
                actual = restored['state'][key][field]
                assert torch.equal(actual.cpu(),expected.cpu()) if torch.is_tensor(expected) else actual == expected
        save(OUT/f'resume-{args.until}.json', dict(status='PASS', parent=receipt, updates=progress['updates'],
             parameter_identity=True, optimizer_groups=opt.state_dict()['param_groups']))
        del state, restored; gc.collect()
    else:
        assert not opt.state and progress['updates'] == 0 and progress['cursor'] == 0
        save(OUT/'optimizer-zero.json',dict(status='PASS', state_entries=0, step_counts=[], implicit_initial_step=0,
             rates=rates, weight_decay=0, schedule='constant', cursor=0, seed=8101))
    assert progress['order'] == order
    reconcile_log(progress)
    previous = json.load(open(OUT/'readouts.json')) if (OUT/'readouts.json').exists() else []
    last_loss = None
    start_updates = progress['updates']
    with (OUT/'updates.jsonl').open('a', buffering=1) as log:
        while progress['training_seconds'] < args.until and progress['update_seconds'] + progress.get('discarded_update_seconds',0) < BUDGET:
            try: pause()
            except RuntimeError: stopped = True
            if stopped: break
            iteration_started = time.perf_counter()
            index, anchor = windows[order[progress['cursor'] % len(windows)]]
            capture = train[index]  # Remote I/O and hashing are outside GPU update timing.
            data_elapsed = time.perf_counter() - iteration_started
            torch.cuda.synchronize(); started = time.perf_counter()
            try:
                loss = update(model, opt, lambda:capture.loss(model, anchor)[0])
                torch.cuda.synchronize()
            except BaseException as error:
                save(OUT/'update-failure.json',dict(error=repr(error), completed_updates=progress['updates'],
                    completed_update_seconds=progress['update_seconds'], failed_update_counted=False,
                    recovery=json.load(open(OUT/'recovery.json'))))
                raise
            elapsed = time.perf_counter()-started
            assert math.isfinite(loss)
            progress['updates'] += 1; progress['cursor'] += 1
            progress['update_seconds'] += elapsed; progress['supervised_tokens'] += 15
            progress['training_seconds'] += time.perf_counter() - iteration_started
            progress['data_io_seconds'] += data_elapsed
            assert len(opt.state) == sum(len(g['params']) for g in opt.param_groups)
            assert {int(v['step']) for v in opt.state.values()} == {progress['updates']}
            assert {g['name']:g['lr'] for g in opt.param_groups} == rates
            log.write(json.dumps(dict(update=progress['updates'], seconds=elapsed,
                cumulative_seconds=progress['update_seconds'], loss=loss, gradient_norm=model.last_gradient_norm,
                sequence=str(capture.path), anchor=anchor, training_seconds=progress['training_seconds'], data_io_seconds=data_elapsed))+'\n')
            last_loss = loss
            if progress['updates'] == 1:
                save(OUT/'one-update-qualification.json',dict(status='PASS', loss=loss,
                    gradient_norm=model.last_gradient_norm, seconds=elapsed, charged_to_budget=True,
                    all_trainable_parameters_have_adam_state=True, all_adam_steps=1,
                    groups={g['name']:dict(lr=g['lr'], parameters=len(g['params']),
                        nonzero_first_moments=sum(int(torch.count_nonzero(opt.state[p]['exp_avg'])) for p in g['params'])) for g in opt.param_groups}))
            if progress['updates'] % 100 == 0 or progress['updates'] == 1:
                save(OUT/'live.json',dict(status='TRAINING', updates=progress['updates'],
                    additional_update_seconds=progress['update_seconds'], training_seconds=progress['training_seconds'], data_io_seconds=progress['data_io_seconds'], last_loss=loss,
                    last_dev=previous[-1]['dev'] if previous else None, owner=identity()))
                print('UPDATE',progress['updates'],progress['update_seconds'],loss,flush=True)
    if train.cache: train.cache.clear()
    gc.collect()
    label = 'paused-'+str(progress['updates']) if stopped else str(args.until)
    snapshot = dict(version=2, config=config, config_hash=digest(config), parameters=parameters(model),
        optimizer=opt.state_dict(), progress=copy.deepcopy(progress), rng=rng(), lineage=lineage,
        component_spec=spec, resume_capable=True, deployment_exps={})
    checkpoint = DEST/f'HE4broad-{label}-{progress["updates"]}.pt'
    if checkpoint.with_suffix('.pt.bsz').exists():
        recovery = json.load(open(checkpoint.with_suffix('.pt.json')))
        assert pin(recovery['archive'])['sha256'] == recovery['archive_sha256']
    else:
        recovery = save_state(checkpoint,snapshot)
    save(OUT/'recovery.json',recovery)
    eval_start = time.perf_counter()
    metrics = evaluation(model, dev)
    eval_seconds = time.perf_counter()-eval_start
    # Keep RNG exactly at the training boundary across evaluation and I/O.
    base = OUT/f'base-{label}.gguf'
    if not base.exists(): export(None,base,state=snapshot,checkpoint_pin=pin(recovery['archive']))
    native = OUT/f'model-{label}.gguf'
    if not native.exists(): rewrite(base,native,spec,deployment_exps={})
    base.unlink(); base.with_suffix('.json').unlink()
    point = dict(label=label, requested_update_seconds=args.until,
        additional_update_seconds=progress['update_seconds'], training_seconds=progress['training_seconds'], data_io_seconds=progress['data_io_seconds'], updates=progress['updates'],
        dev=metrics, recovery=recovery, model=pin(native), evaluation_seconds=eval_seconds,
        segment_wall_seconds=time.perf_counter()-began, segment_updates=progress['updates']-start_updates)
    previous.append(point); save(OUT/'readouts.json',previous)
    result = dict(status='PAUSED' if stopped else ('COMPLETE' if args.until == BUDGET else 'RUNNING'),
        budget_seconds=BUDGET, training_wall_budget_seconds=BUDGET, actual_update_seconds=progress['update_seconds'], training_seconds=progress['training_seconds'], data_io_seconds=progress['data_io_seconds'], updates=progress['updates'],
        last_loss=last_loss, readouts=previous, recovery=recovery,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(OUT/'result.json',result)
    save(OUT/'live.json',dict(status=result['status'], updates=progress['updates'],
         additional_update_seconds=progress['update_seconds'],training_seconds=progress['training_seconds'],last_loss=last_loss,last_dev=metrics,owner=identity()))
    print('READOUT',label,progress['updates'],metrics['ce'],metrics['agreement'],flush=True)


if __name__ == '__main__':
    main()
