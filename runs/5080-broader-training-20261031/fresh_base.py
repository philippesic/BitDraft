"""Proposed full48 restart; unused until separate actual CUDA qualification.

All trainable families come from released BF16, with fresh Adam/RNG/cursor.
Ordinary frozen operands and export metadata use the historical native donor.
This never loads a paid checkpoint or resets an existing trajectory's guards.
"""
from dataclasses import asdict
import gc
import hashlib
from pathlib import Path
import random

from training_system.io import authenticate, atomic_json, pin, read


def validate(spec):
    if spec['schema'] != 'training_lane_v1':
        raise ValueError('fresh serving lane required')
    binding = read(authenticate(spec['fresh_initializer']))
    if binding['schema'] != 'released_bf16_full48_serving_restart_v1':
        raise ValueError('explicit full48 released-base restart required')
    if binding['seed'] != 42 or binding['ordinary_donor'] != spec['parent']['model']:
        raise ValueError('fixed fresh seed and historical ordinary donor required')
    if binding['released_model']['sha256'] != '3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13':
        raise ValueError('the declared released BF16 source is required')
    if spec.get('model_cache') or spec.get('historical_parent_binding') or any(
        spec.get(k) for k in ['parameter_scope', 'gradient_intervention',
                             'calibration_intervention', 'paired_plugin']
    ) or spec['objective'] != 'control':
        raise ValueError('separate uncombined fresh EAL experiment required')
    for key in ['released_model', 'ordinary_donor', 'architecture_contract', 'native_model']:
        authenticate(binding[key])
    config = read(binding['architecture_contract']['path'])['representation']['config']
    if config != binding['config']:
        raise ValueError('fresh architecture differs from the declared comparison')
    if (config['family'], config['activation_bits'], config['profile'],
        config['conditioning'], config['block_size']) != (
        'dspark', 8, 'ffn13_down14_stock', 'captured_prefix', 7
    ):
        raise ValueError('original native A8 binary13 architecture required')
    return binding


def build_cpu(binding, setup, *, bootstrap=False):
    """Read-only construction; bootstrap may publish the initial GGUF once."""
    import numpy as np
    import torch
    import gguf
    from shortlist_pilots.objectives.runtime import configure_imports
    configure_imports()
    from source_gate import verify_training_source
    from shortlist_pilots.objectives.runtime import CAMPAIGN
    from w1a1_eagle.block_qat import BlockDrafter, BlockQATConfig
    from w1a1_eagle.block_training import load_block_gguf
    from joint_tools import (joint_parameters, binary_buffers,
                             install_joint_parameters, export_joint)
    from norm_tools import norm_slots
    from native_frozen_actor import install_native_frozen
    from device_sign_baseline import DeviceSignBaseline
    from training_system.initializers import released_bf16_binary
    from training_system.model_cache import validate_frozen

    setup = Path(setup)
    setup.mkdir(parents=True, exist_ok=True)
    if (binding['schema'] != 'released_bf16_full48_serving_restart_v1'
        or binding['released_model']['sha256'] != '3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13'):
        raise ValueError('declared released-source bootstrap required')
    for key in ['released_model', 'ordinary_donor', 'architecture_contract']:
        authenticate(binding[key])
    if read(binding['architecture_contract']['path'])['representation']['config'] != binding['config']:
        raise ValueError('bootstrap architecture differs')
    assert binding['seed'] == 42
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    config = BlockQATConfig(**binding['config'])
    source = binding['released_model']
    tensors = load_block_gguf(source['path'], source['sha256'], config)
    model = BlockDrafter(tensors, config)
    linears = model.binary_linears()
    if len(linears) != 13:
        raise ValueError('all thirteen released binary projections required')
    for name, module in linears.items():
        original = tensors[name + '.weight']
        if not torch.equal(module.latent_sign, original.float()):
            raise ValueError('constructor changed released latent values: ' + name)
        if not torch.equal(module.initial_scale, original.float().abs().mean(-1)):
            raise ValueError('released mean-absolute initial scales differ: ' + name)
    del tensors
    gc.collect()
    donor = binding['ordinary_donor']
    donor_reader = gguf.GGUFReader(donor['path'])
    frozen = install_native_frozen(model, donor_reader)
    released_reader = gguf.GGUFReader(source['path'])
    source_tensors = {t.name: t for t in released_reader.tensors}
    install_joint_parameters(model, source_tensors)
    released_bf16_binary(linears)
    parameters = joint_parameters(model)
    buffers = binary_buffers(model)
    if len(parameters) != 48 or len(buffers) != 13:
        raise ValueError('fresh48/13-buffer ownership differs')
    norms = {n: getattr(owner, attr).detach().cpu().clone()
             for n, (owner, attr) in norm_slots(model).items()}
    for name, value in norms.items():
        expected = torch.from_numpy(source_tensors[name].data.copy())
        if not torch.equal(value, expected):
            raise ValueError('a fresh norm retained another initializer: ' + name)
    baseline = DeviceSignBaseline(model)
    sign_hashes = {n: hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
                   for n, p in parameters.items() if n.endswith('.latent_sign')}
    ordinary = validate_frozen(model, frozen['tensors'])
    trial = setup / 'fresh-initial.gguf'
    exported = export_joint(model, donor['path'], donor['sha256'], trial)
    if not bootstrap:
        authenticate(binding['native_model'])
        if exported['output']['sha256'] != binding['native_model']['sha256']:
            raise ValueError('fresh zero-update export differs from bootstrap')
        trial.unlink()
    audit = {'schema': binding['schema'], 'released_model': source,
             'ordinary_donor': donor, 'fresh_parameters': sorted(parameters),
             'fresh_norms': sorted(norms), 'fresh_initial_scale_buffers': sorted(buffers),
             'optimizer_imported': False, 'paid_checkpoint_loaded': False,
             'cursor_imported': False, 'rng_seed': 42, 'guard_anchor': 'fresh fork zero',
             'export': exported, 'zero_export_retained': bootstrap,
             'source_validation': verify_training_source(CAMPAIGN),
             'ordinary_arithmetic': frozen['scope']}
    atomic_json(setup / 'fresh-initialization.json', audit)
    metadata = {'initial_norms': norms, 'initial_sign_hashes': sign_hashes,
                'TRAIN_seconds': 0., 'parent_step': 0, 'frozen_state': ordinary,
                'fresh_initializer': binding}
    del donor_reader, released_reader, source_tensors
    gc.collect()
    return model, baseline, metadata


def load(setup, spec):
    import torch
    from training_system.devices import configure_for_spec
    from train_nine_model_qat import configure_cuda
    from joint_tools import joint_optimizer, assert_joint_policy, measure_norms
    from w1a1_eagle.block_training import _block_rng_state
    binding = validate(spec)
    model, baseline, metadata = build_cpu(binding, setup)
    hardware = configure_for_spec(spec, configure_cuda)
    model.to('cuda:0')
    baseline.bind_device(model)
    optimizer = joint_optimizer(model)
    assert_joint_policy(model, optimizer)
    if optimizer.state or baseline.measure(model)['fraction'] != 0:
        raise ValueError('fresh optimizer/sign reference is not zero')
    reference = {n: t.to(model.token_embd.device) for n, t in metadata['initial_norms'].items()}
    drift = measure_norms(model, reference)
    if drift['max_relative_l2'] != 0 or drift['max_absolute_change'] != 0:
        raise ValueError('fresh norm guard is not zero')
    random.seed(42)
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    metadata['rng'] = _block_rng_state(model.token_embd.device)
    atomic_json(Path(setup) / 'parent-loader.json',
                {'hardware': hardware, 'fresh_initializer': spec['fresh_initializer'],
                 'paid_checkpoint_loaded': False, 'Adam_imported': False,
                 'fresh48_original_guard_policy': True})
    model.train()
    return model, optimizer, baseline, metadata
