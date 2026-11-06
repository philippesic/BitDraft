"""Qualify an actual next optimizer update without changing a retained frontier."""
import gc
import json
import time
from pathlib import Path

import torch

from .common import R, ROOT, identity, pause, pin, save
from .deployment import values
from .qualify_tensor_checkpoint import fingerprint
from .simulation import install as install_component
from scripts.dflash_baseline.components import install_components
from scripts.dflash_baseline.model import Capture, Model
from scripts.dflash_baseline.pipeline import load_parameters, restore_rng, rng, update
from scripts.dflash_baseline.prototype_model import install, optimizer


def isolated_optimizer_state(saved):
    """Adam retains CPU scalar tensors by reference even when moments move to CUDA."""
    return dict(saved, state={key: {field: value.clone() if torch.is_tensor(value) and value.ndim == 0 else value
                                    for field, value in fields.items()}
                              for key, fields in saved['state'].items()})


def qualify(state, original_arm, destination, training_directory=None):
    pause()
    spec, config, progress = state['component_spec'], state['config'], state['progress']
    training_directory = Path(training_directory or R / 'arms' / original_arm)
    resolved = json.loads((training_directory / 'resolved.json').read_text())
    numerical_sources = [
        'component_campaign/' + name + '.py' for name in ('simulation', 'quant', 'deployment')]
    if spec.get('r2_vocab_scale'):
        numerical_sources.append('component_campaign/head_qat.py')
    numerical_sources += [
        'dflash_baseline/' + name + '.py' for name in
        ('model', 'pipeline', 'prototype', 'prototype_model', 'components')]
    for relative in numerical_sources:
        key = 'scripts/' + relative
        if pin(ROOT / key)['sha256'] != resolved['source_pins'][key]['sha256']:
            raise ValueError('Numerical resume source differs: ' + key)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Model(config['source']['path'], config['evaluation']['target']['path'])
    install(model, config, {})
    install_components(model, config)
    install_component(model, spec)
    if spec.get('r2_vocab_scale'):
        from .head_qat import install as install_head, optimizer as head_optimizer
        install_head(model, spec)
    model.to('cuda')
    opt = head_optimizer(model) if spec.get('r2_vocab_scale') else optimizer(model)
    load_parameters(model, state['parameters'])
    saved_steps = {int(v['step']) for v in state['optimizer']['state'].values()}
    saved_optimizer_fingerprint = fingerprint(state['optimizer'])
    if saved_steps != {progress['updates']}:
        raise ValueError('Saved Adam step differs from recorded updates before qualification')
    opt.load_state_dict(isolated_optimizer_state(state['optimizer']))
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach().cpu(), state['parameters'][name]):
            raise ValueError('GPU master differs: ' + name)
    restored_optimizer = opt.state_dict()
    if restored_optimizer['param_groups'] != state['optimizer']['param_groups']:
        raise ValueError('GPU optimizer groups differ')
    for key, saved in state['optimizer']['state'].items():
        restored = restored_optimizer['state'][key]
        for field, expected in saved.items():
            actual = restored[field]
            if torch.is_tensor(expected):
                if not torch.equal(actual.detach().cpu(), expected):
                    raise ValueError('GPU Adam state differs: ' + str((key, field)))
            elif actual != expected:
                raise ValueError('GPU optimizer metadata differs')
    deployment, _ = values(model, spec)
    if set(deployment) != set(state.get('deployment_exps', {})):
        raise ValueError('GPU deployment scalar coverage differs')
    for key, expected in state.get('deployment_exps', {}).items():
        if not torch.equal(deployment[key], expected):
            raise ValueError('GPU deployment rounding differs: ' + key)
    manifest = json.loads(Path(config['data']['path']).read_text())
    rows = [r for r in manifest['records'] if r['split'] == 'TRAIN']
    for ref in [config['data'], config['source'], config['evaluation']['target']]:
        if pin(ref['path'])['sha256'] != ref['sha256']:
            raise ValueError('Resume input differs')
    captures = []
    for row in rows:
        for ref in [row['metadata'], *row['files']]:
            if pin(ref['path']) != ref:
                raise ValueError('Resume TRAIN input differs')
        capture = Capture(Path(row['metadata']['path']).parent)
        if 'training_last_input_position' in row:
            capture.anchors = [a for a in capture.anchors if a + 15 <= row['training_last_input_position']]
        captures.append(capture)
    windows = [(i, anchor) for i, capture in enumerate(captures) for anchor in capture.anchors]
    if sorted(progress['order']) != list(range(len(windows))):
        raise ValueError('Resume data order differs')
    index, anchor = windows[progress['order'][progress['cursor'] % len(windows)]]
    restore_rng(state['rng'])
    if fingerprint(rng()) != fingerprint(state['rng']):
        raise ValueError('Actual GPU/CPU RNG restoration differs')
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = update(model, opt, lambda: captures[index].loss(model, anchor)[0])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    next_steps = {int(v['step']) for v in opt.state.values()}
    if next_steps != {progress['updates'] + 1}:
        raise ValueError('Actual resumed Adam did not advance exactly one step')
    if {int(v['step']) for v in state['optimizer']['state'].values()} != saved_steps:
        raise ValueError('Scratch qualification mutated the caller recovery state')
    if fingerprint(state['optimizer']) != saved_optimizer_fingerprint:
        raise ValueError('Scratch qualification mutated caller Adam moments/groups')
    changed, nonfinite = [], []
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu()
        if not bool(torch.isfinite(current).all()):
            nonfinite.append(name)
        if not torch.equal(current, state['parameters'][name]):
            changed.append(name)
    if nonfinite or not changed:
        raise ValueError('Resumed update failed finite/change checks')
    result = dict(status='ACTUAL_GPU_MASTER_ADAM_RNG_CURSOR_RESUME_UPDATE_PASS',
        owner=identity(), arm=original_arm, saved_updates=progress['updates'],
        retained_state_trainer=pin(training_directory / 'result.json'),
        qualification_updates=1, next_adam_steps=sorted(next_steps),
        next_cursor=progress['cursor'] + 1, next_capture=str(captures[index].path),
        next_anchor=anchor, loss=loss, gradient_norm=model.last_gradient_norm,
        update_seconds=seconds, changed_master_names=changed,
        gpu_deployment_values_exact=True, caller_optimizer_unchanged=True, source=pin(__file__),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        policy='Actual next TRAIN window with saved Adam/RNG/order/cursor; one qualification update only. Retained4096 state and evaluated native bytes remain unchanged. This scratch update is not another recovery window, independent seed or reported finalist.')
    save(destination / 'gpu-resume.json', result)
    del model, opt, captures, restored_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result
