"""CPU-only zero-update teacher-forced and recurrent reference; not native actor qualification."""
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import json
import os
import resource
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before Torch import'
import numpy as np
import torch
from torch import nn
from safetensors import safe_open

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-recurrent-actor-20261103-01'
sys.path[:0] = [str(P / 'src'), str(P / 'scripts'), str(P / 'scripts/w1a8_generalization'),
               str(P / 'third_party/llama.cpp/gguf-py')]
from gguf import GGUFReader, GGMLQuantizationType as Type
from audit_eagle_w1a1_gguf import SOURCE_NAMES, gguf_qk_row_order
from modern_admission import build_actor_steps, require_actor_steps
from w1a1_eagle.native_step import ExternalNativeF32Embedding, NativeStepAdapter, NATIVE_NORM_PATHS, bind_frozen_norms
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, shared_round_hard_signs

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
started = time.perf_counter()
phases = {}


def read(path):
    return json.loads(Path(path).read_text())


baseline = read(P / 'experiments/w1a8-standing-20261102/baseline.json')
blocked_target_paths = {str(Path(row['target_pin']['path']).resolve()) for row in baseline['rows']}
authenticated = {}


def pin(path):
    path = Path(path).resolve()
    assert str(path) not in blocked_target_paths, 'target weights must never be opened or hashed'
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def authenticate(ref):
    path = str(Path(ref['path']).resolve())
    if path in blocked_target_paths:
        expected = next(row['target_pin'] for row in baseline['rows'] if str(Path(row['target_pin']['path']).resolve()) == path)
        assert ref['sha256'] == expected['sha256']
        return  # Ancestry declaration only; parent already authenticated these admissions.
    if path not in authenticated:
        actual = pin(path)
        assert actual['sha256'] == ref['sha256'], path
        if 'bytes' in ref:
            assert actual['bytes'] == ref['bytes'], path
        authenticated[path] = actual
    else:
        assert authenticated[path]['sha256'] == ref['sha256'], path


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            yield value
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def unpermute_rows(value, heads):
    order = np.concatenate((np.arange(40).reshape(2, 20).T.reshape(-1), np.arange(40, 160)))
    return np.ascontiguousarray(value.reshape(heads, 160, -1)[:, np.argsort(order), :].reshape(value.shape))


def norm():
    module = nn.Module()
    module.weight = nn.Parameter(torch.ones(2560, dtype=torch.float32), requires_grad=False)
    module.variance_epsilon = 1e-6
    return module


bundle = P / 'runs/w1a8-modern-packed-init-20261103-01'
freeze_path, export_path, audit_path = [bundle / name for name in ('freeze.json', 'export.json', 'independent-audit.json')]
freeze, exported, audited = read(freeze_path), read(export_path), read(audit_path)
for name in ('source', 'repair_receipt', 'exporter', 'auditor'):
    authenticate(freeze[name])
authenticate(exported['output'])
assert audited['source_sha256'] == freeze['source']['sha256']
assert audited['gguf_sha256'] == exported['output']['sha256']
assert audited['packed_tensors'] == 9 and audited['audited_rows'] == 76880
assert all(row['packed_word_mismatches'] == 0 and row['max_abs_scale_error'] == 0 for row in audited['records'])
assert exported['serialization_audit_passed'] and exported['activation_bits'] == 8 and exported['scale_layout'] == 'row'
config_path = P / 'runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json'
config = read(config_path)
repair = next(row for row in read(freeze['repair_receipt']['path'])['records'] if row['size'] == '4B')
assert pin(config_path)['sha256'] == repair['config_sha256']
assert {name: config[name] for name in ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'num_hidden_layers', 'draft_vocab_size', 'vocab_size')} == {
    'hidden_size': 2560, 'intermediate_size': 7680, 'num_attention_heads': 16, 'num_key_value_heads': 4,
    'head_dim': 160, 'num_hidden_layers': 1, 'draft_vocab_size': 50000, 'vocab_size': 248320}
assert config['rope_parameters'] == {'partial_rotary_factor': .25, 'rope_theta': 10000000, 'rope_type': 'default'}
reader = GGUFReader(exported['output']['path'])
tensors = {tensor.name: tensor for tensor in reader.tensors}
assert len(tensors) == len(reader.tensors) == 23
assert not any(name.endswith('.weight') and name.removesuffix('.weight') in SOURCE_NAMES for name in tensors)
assert 'token_embd.weight' not in tensors
assert reader.fields['eagle3.rope.dimension_count'].contents() == 40
assert reader.fields['eagle3.attention.head_count'].contents() == 16
assert reader.fields['eagle3.attention.head_count_kv'].contents() == 4
phases['authenticate_source_and_native_initializer_seconds'] = time.perf_counter() - started

drafter = nn.Module()
drafter.config = SimpleNamespace(**config)
drafter.early_stop_method = None
drafter.tree_mask = None
drafter.midlayer = nn.Module()
drafter.midlayer.self_attn = nn.Module()
drafter.midlayer.mlp = nn.Module()
for path in NATIVE_NORM_PATHS.values():
    parent_path, name = path.rsplit('.', 1) if '.' in path else ('', path)
    setattr(drafter.get_submodule(parent_path) if parent_path else drafter, name, norm())
linears = {}
norm_arrays = {name: torch.from_numpy(np.array(tensors[name].data, copy=True)) for name in NATIVE_NORM_PATHS}
native_map = np.array(tensors['d2t'].data, dtype=np.int64, copy=True)
assert tensors['d2t'].tensor_type == Type.I64 and native_map.shape == (50000,)
source_records = {}
stage = time.perf_counter()
with safe_open(freeze['source']['path'], framework='pt', device='cpu') as source:
    expected_keys = set(SOURCE_NAMES.values()) | {path + '.weight' for path in NATIVE_NORM_PATHS.values()} | {'d2t', 't2d'}
    assert set(source.keys()) == expected_keys
    offset_map = source.get_tensor('d2t')
    assert offset_map.dtype == torch.int64 and tuple(offset_map.shape) == (50000,)
    absolute_map = offset_map.numpy() + np.arange(50000, dtype=np.int64)
    assert np.array_equal(absolute_map, native_map)
    assert len(np.unique(absolute_map)) == 50000 and absolute_map.min() >= 0 and absolute_map.max() < 248320
    source_membership = source.get_tensor('t2d')
    expected_membership = np.zeros(248320, dtype=bool)
    expected_membership[absolute_map] = True
    assert source_membership.dtype == torch.bool and np.array_equal(source_membership.numpy(), expected_membership)
    reverse = np.full(248320, -1, dtype=np.int64)
    reverse[absolute_map] = np.arange(50000)
    drafter.register_buffer('d2t', offset_map.clone())
    drafter.register_buffer('t2d', source_membership.clone())
    for name, path in NATIVE_NORM_PATHS.items():
        assert torch.equal(source.get_tensor(path + '.weight').float(), norm_arrays[name])
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        weight = source.get_tensor(SOURCE_NAMES[base])
        assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == tuple(exported['projections'][base]['shape'])
        assert bool(torch.isfinite(weight).all())
        latent = torch.where(weight < 0, torch.tensor(-.5), torch.tensor(.5))
        scales = np.array(tensors[base + '.w1a1_scale'].data, dtype=np.float32, copy=True)
        heads = 16 if base == 'blk.0.attn_q' else 4 if base == 'blk.0.attn_k' else None
        if heads is not None:
            scales = unpermute_rows(scales[:, None], heads)[:, 0].copy()
        assert hashlib.sha256(scales.tobytes()).hexdigest() == exported['projections'][base]['source_scale_sha256']
        ordered = gguf_qk_row_order(latent.numpy(), heads, 40) if heads is not None else latent.numpy()
        packed = np.ascontiguousarray(np.packbits(ordered >= 0, axis=1, bitorder='little')).view('<i4')
        assert np.array_equal(packed, tensors[base + '.w1a1_packed'].data)
        module = RowBinaryLinear(latent, torch.from_numpy(scales), W1AxContract(8))
        parent_path, name = path.rsplit('.', 1) if '.' in path else ('', path)
        setattr(drafter.get_submodule(parent_path) if parent_path else drafter, name, module)
        linears[path] = module
        source_records[base] = {'shape': list(weight.shape), 'initial_signs_match_native_packed': True,
                                'row_scales_source': 'exact native initializer scales, inverse partial Q/K permutation',
                                'source_scale_sha256': exported['projections'][base]['source_scale_sha256']}
        del weight, latent, scales, ordered, packed
        gc.collect()
bind_frozen_norms(drafter, norm_arrays)
phases['construct_nine_linears_and_maps_seconds'] = time.perf_counter() - stage
assert all(parameter.device.type == 'cpu' for parameter in drafter.parameters())
assert not hasattr(drafter, 'embed_tokens')

admission_root = P / 'runs/w1a8-modern-admission-20261103-01'
recipe_pins = [pin(__file__), pin(config_path), pin(freeze_path), pin(export_path), pin(audit_path)]
recipe_pins += [pin(P / 'src/w1a1_eagle' / name) for name in ('native_step.py', 'recurrent_qat.py', 'recurrent_binary.py')]
recipe_pins += [pin(P / 'scripts/w1a8_generalization/modern_admission.py')]
cases = {}


def save_f32(name, rows, width):
    path = R / name
    values = np.stack(rows).astype('<f4', copy=False) if rows else np.empty((0, width), dtype='<f4')
    assert values.ndim == 2 and values.shape[1] == width and np.isfinite(values).all()
    with path.open('xb') as stream:
        stream.write(values.tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    return {'file': pin(path), 'shape': list(values.shape)}


def prediction(logits, label):
    values = logits.detach().float().cpu().numpy().copy()
    assert values.shape == (50000,) and np.isfinite(values).all()
    draft_argmax = int(values.argmax())
    proposed = int(absolute_map[draft_argmax])
    draft_label = int(reverse[label])
    best_wrong = None
    margin = None
    if draft_label >= 0:
        other = values.copy()
        other[draft_label] = -np.inf
        best_wrong = int(other.argmax())
        margin = float(values[draft_label] - other[best_wrong])
    return values, {'draft_argmax': draft_argmax, 'proposed_target_token_id': proposed,
                    'label_token_id': int(label), 'label_supported': draft_label >= 0,
                    'draft_label': draft_label, 'best_wrong_draft_id': best_wrong,
                    'label_margin': margin, 'correct': proposed == label}


for prompt_id in (2, 4):
    admission_path = admission_root / ('prompt-%02d-admitted.json' % prompt_id)
    receipt = read(admission_path)
    recipe_pins.append(pin(admission_path))
    for ref in references(receipt['evidence']):
        authenticate(ref)
    packet = receipt['evidence']
    meta = read(packet['capture']['path'])
    shape = tuple(meta['feature_shape'])
    arrays = {name: np.memmap(ref['path'], dtype='<f4', mode='r', shape=shape)
              for name, ref in packet['arrays'].items() if name.startswith('layer-')}
    features = torch.from_numpy(np.concatenate([arrays['layer-%d.f32' % tap] for tap in (2, 16, 29)], axis=1))
    rows_by_token = {}
    for index, token in enumerate(meta['input_tokens']):
        row = torch.from_numpy(np.array(arrays['layer-0.f32'][index], copy=True))
        if token in rows_by_token:
            assert torch.equal(row, rows_by_token[token])
        else:
            rows_by_token[token] = row
    lookup = ExternalNativeF32Embedding(lookup=rows_by_token.__getitem__, vocab_size=248320, hidden_size=2560,
                target_identity='Qwen3.5-4B-Q4_K_M native get_rows', target_sha256=packet['target']['sha256'],
                capture_sha256=packet['capture']['sha256'])
    adapter = NativeStepAdapter(drafter, embedding_lookup=lookup, embedding_precision='external_native_f32')
    assert adapter.head_dim == 160 and adapter.rotary_dim == 40 and adapter.rope_theta == 1e7
    steps = build_actor_steps(receipt)
    assert len(steps) == 8
    labels = [row['serving_label'] for row in receipt['rows']]
    assert len(labels) == 9
    eligible_ids = [step['actor_step_id'] for step in steps if step['supervision_eligible']]
    assert len(eligible_ids) == (7 if prompt_id == 2 else 6)
    require_actor_steps(receipt, eligible_ids)
    require_actor_steps(receipt, eligible_ids, purpose='protection_numeric')
    stage = time.perf_counter()
    teacher_records, teacher_logits, teacher_states, teacher_encoded = [], [], [], []
    recurrent_records, recurrent_logits, recurrent_states = [], [], []
    with torch.inference_mode(), shared_round_hard_signs(linears):
        count = meta['prompt_tokens'] - 1
        cache = adapter.build_context_cache(meta['input_tokens'][1:count + 1], features[:count], chunk_size=32)
        assert cache.key.shape == cache.value.shape == (4, count, 160)
        teacher_context_seconds = time.perf_counter() - stage
        teacher_started = time.perf_counter()
        for step in steps:
            position = step['decoder_position']
            assert position == step['feature_input_position'] and position + 1 == step['input_position']
            assert meta['input_tokens'][step['input_position']] == step['input_token_id']
            encoded = adapter.encode_feature(features[position])
            result = adapter.decode_step(step['input_token_id'], encoded, position, cache)
            cache = result.cache
            values, details = prediction(result.logits, step['label_token_id'])
            teacher_logits.append(values)
            teacher_states.append(result.pre_norm.detach().numpy().copy())
            teacher_encoded.append(encoded.detach().numpy().copy())
            teacher_records.append({**step, **details, 'logits_row_index': len(teacher_logits) - 1,
                                    'state_row_index': len(teacher_states) - 1,
                                    'encoded_row_index': len(teacher_encoded) - 1,
                                    'actor_native_qualification_proven': False})
        teacher_seconds = time.perf_counter() - teacher_started
        del cache, result
        recurrent_started = time.perf_counter()
        for anchor_index, initial_step in enumerate(steps):
            anchor = initial_step['decoder_position']
            context_started = time.perf_counter()
            cache = adapter.build_context_cache(meta['input_tokens'][1:anchor + 1], features[:anchor], chunk_size=32)
            assert cache.key.shape == cache.value.shape == (4, anchor, 160)
            context_seconds = time.perf_counter() - context_started
            state = adapter.encode_feature(features[anchor])
            token = initial_step['input_token_id']
            chain = []
            correct_prefix = 0
            protectable_prefix = 0
            numeric_reachable = True
            stop_reason = 'max_depth'
            for depth in range(2):
                teacher_index = anchor_index + depth
                label_index = teacher_index + 1
                if label_index >= len(labels):
                    stop_reason = 'label_boundary'
                    break
                expected_input = labels[teacher_index]
                assert token == expected_input, 'alternative prefix must never use gold labels'
                if token not in rows_by_token:
                    stop_reason = 'embedding_unavailable'
                    break
                numeric_eligible = True
                try:
                    require_actor_steps(receipt, [teacher_index])
                    require_actor_steps(receipt, [teacher_index], purpose='protection_numeric')
                except ValueError:
                    numeric_eligible = False
                numeric_reachable = numeric_reachable and numeric_eligible
                position = anchor + depth
                assert cache.key.shape[1] == position
                result = adapter.decode_step(token, state, position, cache)
                values, details = prediction(result.logits, labels[label_index])
                recurrent_logits.append(values)
                recurrent_states.append(result.pre_norm.detach().numpy().copy())
                record = {'anchor_index': anchor_index, 'anchor_position': anchor, 'depth': depth,
                          'decoder_position': position, 'input_position': position + 1,
                          'label_position': position + 2, 'input_token_id': int(token),
                          'source_teacher_row_id': teacher_index, 'label_teacher_row_id': label_index,
                          'numeric_eligible': numeric_eligible, 'numeric_reachable': numeric_reachable,
                          'state_source': 'encoded_target_feature' if depth == 0 else 'previous_actor_pre_norm',
                          'state_row_index': len(recurrent_states) - 1,
                          'logits_row_index': len(recurrent_logits) - 1,
                          'parent_success_protection_proven': False,
                          'actor_native_qualification_proven': False, **details}
                chain.append(record)
                if not details['correct']:
                    stop_reason = 'first_divergence'
                    break
                correct_prefix += 1
                if numeric_reachable:
                    protectable_prefix += 1
                token = details['proposed_target_token_id']
                state = result.pre_norm
                cache = result.cache
            recurrent_records.append({'anchor_index': anchor_index, 'anchor_position': anchor,
                                      'max_depth': 2, 'steps': chain, 'stop_reason': stop_reason,
                                      'correct_prefix': correct_prefix,
                                      'numeric_protectable_prefix': protectable_prefix,
                                      'context_pairs': anchor, 'context_seconds': context_seconds,
                                      'parent_success_protection_proven': False})
            del cache, state, result
        recurrent_seconds = time.perf_counter() - recurrent_started
    teacher_files = {'logits': save_f32('prompt-%02d-teacher-logits.f32' % prompt_id, teacher_logits, 50000),
                     'pre_norm': save_f32('prompt-%02d-teacher-prenorm.f32' % prompt_id, teacher_states, 2560),
                     'encoded': save_f32('prompt-%02d-encoded.f32' % prompt_id, teacher_encoded, 2560)}
    recurrent_files = {'logits': save_f32('prompt-%02d-recurrent-logits.f32' % prompt_id, recurrent_logits, 50000),
                      'pre_norm': save_f32('prompt-%02d-recurrent-prenorm.f32' % prompt_id, recurrent_states, 2560)}
    scored = [row for row in teacher_records if row['supervision_eligible']]
    flat = [step for chain in recurrent_records for step in chain['steps']]
    case = {'admission': pin(admission_path), 'capture': packet['capture'],
            'teacher_forced_steps': teacher_records, 'teacher_files': teacher_files,
            'recurrent': recurrent_records, 'recurrent_files': recurrent_files,
            'numeric_eligible_teacher_denominator': len(scored),
            'eligible_teacher_correct': sum(row['correct'] for row in scored),
            'eligible_teacher_unsupported_labels': sum(not row['label_supported'] for row in scored),
            'recurrent_diagnostic_steps': len(flat),
            'recurrent_numeric_eligible_steps': sum(row['numeric_eligible'] for row in flat),
            'recurrent_unsupported_labels_in_diagnostic_denominator': sum(not row['label_supported'] for row in flat),
            'correct_prefix_sum': sum(chain['correct_prefix'] for chain in recurrent_records),
            'numeric_protectable_prefix_sum': sum(chain['numeric_protectable_prefix'] for chain in recurrent_records),
            'teacher_context_seconds': teacher_context_seconds, 'teacher_seconds': teacher_seconds,
            'recurrent_seconds': recurrent_seconds, 'parent_success_protection_proven': False,
            'actor_native_qualification_proven': False}
    cases[str(prompt_id)] = case
    write(R / ('prompt-%02d-result.json' % prompt_id), case)
    print(json.dumps({k: v for k, v in case.items() if k in ('numeric_eligible_teacher_denominator',
          'eligible_teacher_correct', 'recurrent_diagnostic_steps', 'correct_prefix_sum',
          'numeric_protectable_prefix_sum', 'recurrent_seconds')}), flush=True)
    del adapter, rows_by_token, lookup, arrays, features, teacher_logits, teacher_states, teacher_encoded
    del recurrent_logits, recurrent_states
    gc.collect()

peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert peak_bytes < 6 * 1024**3, 'CPU readout exceeded6GiB RSS target'
report = {'status': 'COMPLETE', 'scope': 'CPU-only zero-update teacher-forced and first-divergence-limited recurrent actor reference on reused development histories; not native actor parity, native acceptance, training improvement, untouched quality or throughput qualification',
          'arithmetic_caveat': 'Existing RowBinaryLinear A8 dots dequantized activation values then row-scales; new native packed operation scales after integer dot. Arithmetic unchanged and unqualified here.',
          'source_bundle_freeze': pin(freeze_path), 'source_and_native_pins': recipe_pins,
          'authenticated_non_target_dependencies': list(authenticated.values()),
          'target_model_opened_or_hashed': False, 'target_ancestry': 'Previously parent-admitted immutable packet pins; target weight paths blocked',
          'nine_linear_initialization': source_records, 'native_d2t_absolute_matches_source_offsets': True,
          'source_t2d_membership_matches_native_map': True, 'native_four_norms_exact': True,
          'cases': cases, 'phases': phases, 'peak_rss_bytes': peak_bytes,
          'cpu_threads': torch.get_num_threads(), 'wall_seconds': time.perf_counter() - started,
          'training_updates': 0, 'optimizer_created': False, 'full_checkpoint_created': False,
          'gpu_seconds': 0, 'private_target_embedding_created': False, 'failed_attempts': []}
write(R / 'report.json', report)
print(json.dumps({'status': 'COMPLETE', 'wall_seconds': report['wall_seconds'],
                  'peak_rss_bytes': peak_bytes, 'training_updates': 0, 'gpu_seconds': 0}), flush=True)
