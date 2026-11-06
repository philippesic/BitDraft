"""Saved-output audit plus selected recurrent-state negative control; CPU only."""
from pathlib import Path
import gc
import hashlib
import json
import os
import resource
import sys
import time

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-recurrent-actor-20261103-01'
sys.path[:0] = [str(P / 'third_party/llama.cpp/gguf-py')]
from gguf import GGUFReader

started = time.perf_counter()
report = json.loads((R / 'report.json').read_text())
baseline = json.loads((P / 'experiments/w1a8-standing-20261102/baseline.json').read_text())
blocked = {str(Path(x['target_pin']['path']).resolve()) for x in baseline['rows']}


def pin(path):
    path = Path(path).resolve()
    assert str(path) not in blocked
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def checked(ref):
    value = pin(ref['path'])
    assert value['sha256'] == ref['sha256']
    if 'bytes' in ref:
        assert value['bytes'] == ref['bytes']
    return Path(ref['path'])


for ref in report['source_and_native_pins'] + report['authenticated_non_target_dependencies']:
    checked(ref)
bundle = json.loads((P / 'runs/w1a8-modern-packed-init-20261103-01/export.json').read_text())
reader = GGUFReader(checked(bundle['output']))
absolute_map = np.array(next(x for x in reader.tensors if x.name == 'd2t').data, dtype=np.int64)
assert absolute_map.shape == (50000,) and len(np.unique(absolute_map)) == 50000
inverse = np.full(248320, -1, dtype=np.int64)
inverse[absolute_map] = np.arange(50000)
summary = {}


def matrix(spec):
    path = checked(spec['file'])
    values = np.fromfile(path, dtype='<f4').reshape(spec['shape'])
    assert np.isfinite(values).all()
    return values


def prediction_check(row, values, label):
    index = int(values.argmax())
    assert row['draft_argmax'] == index
    assert row['proposed_target_token_id'] == int(absolute_map[index])
    supported = int(inverse[label]) >= 0
    assert row['label_supported'] == supported
    assert row['draft_label'] == int(inverse[label])
    assert row['label_token_id'] == label
    assert row['correct'] == (row['proposed_target_token_id'] == label)
    if supported:
        other = values.copy()
        other[row['draft_label']] = -np.inf
        best = int(other.argmax())
        assert row['best_wrong_draft_id'] == best
        assert row['label_margin'] == float(values[row['draft_label']] - other[best])


for pid, case in report['cases'].items():
    receipt = json.loads(checked(case['admission']).read_text())
    teacher_rows = receipt['rows']
    teacher = matrix(case['teacher_files']['logits'])
    teacher_state = matrix(case['teacher_files']['pre_norm'])
    encoded = matrix(case['teacher_files']['encoded'])
    recurrent = matrix(case['recurrent_files']['logits'])
    recurrent_state = matrix(case['recurrent_files']['pre_norm'])
    assert teacher.shape == (8, 50000) and teacher_state.shape == encoded.shape == (8, 2560)
    old = np.fromfile(P / 'runs/w1a8-modern-actor-readout-20261103-01' / ('prompt-%02d-logits.f32' % int(pid)), dtype='<f4').reshape(8, 50000)
    assert np.array_equal(teacher, old), 'unchanged teacher-forced control changed'
    for index, row in enumerate(case['teacher_forced_steps']):
        assert row['source_teacher_row_id'] == index and row['label_teacher_row_id'] == index + 1
        assert row['feature_input_position'] == row['decoder_position'] == teacher_rows[index]['input_position']
        assert row['input_token_id'] == teacher_rows[index]['serving_label']
        prediction_check(row, teacher[index], teacher_rows[index + 1]['serving_label'])
    used = []
    seed_drift = []
    for chain in case['recurrent']:
        i = chain['anchor_index']
        anchor = teacher_rows[i]['input_position']
        assert chain['anchor_position'] == chain['context_pairs'] == anchor
        correct_prefix = protectable = 0
        numeric_reachable = True
        for depth, row in enumerate(chain['steps']):
            j = i + depth
            assert row['depth'] == depth and row['anchor_position'] == anchor
            assert row['decoder_position'] == anchor + depth
            assert row['input_position'] == anchor + depth + 1 and row['label_position'] == anchor + depth + 2
            assert row['input_token_id'] == teacher_rows[j]['serving_label']
            assert row['source_teacher_row_id'] == j and row['label_teacher_row_id'] == j + 1
            assert row['state_source'] == ('encoded_target_feature' if depth == 0 else 'previous_actor_pre_norm')
            numeric = all(teacher_rows[k]['supervision_eligible'] and teacher_rows[k]['protection_numeric_eligible']
                          and teacher_rows[k]['native_argmax'] == teacher_rows[k]['serving_label']
                          and k not in receipt['excluded_row_ids'] for k in (j, j + 1))
            assert row['numeric_eligible'] == numeric
            numeric_reachable = numeric_reachable and numeric
            assert row['numeric_reachable'] == numeric_reachable
            index = row['logits_row_index']
            assert index == row['state_row_index'] and index not in used
            used.append(index)
            prediction_check(row, recurrent[index], teacher_rows[j + 1]['serving_label'])
            assert row['parent_success_protection_proven'] is False
            if depth == 0:
                delta = recurrent[index].astype(np.float64) - teacher[i]
                seed_drift.append({'anchor_index': i, 'argmax_matches_teacher_forced': row['draft_argmax'] == case['teacher_forced_steps'][i]['draft_argmax'],
                                   'logits_max_abs': float(np.abs(delta).max()),
                                   'logits_relative_rms': float(np.linalg.norm(delta) / np.linalg.norm(teacher[i]))})
            if row['correct']:
                correct_prefix += 1
                if numeric_reachable:
                    protectable += 1
            else:
                assert depth == len(chain['steps']) - 1 and chain['stop_reason'] == 'first_divergence'
            if depth:
                assert chain['steps'][depth - 1]['correct'], 'gold label used after divergence'
                assert row['input_token_id'] == chain['steps'][depth - 1]['proposed_target_token_id']
        assert chain['correct_prefix'] == correct_prefix and chain['numeric_protectable_prefix'] == protectable
        assert len(chain['steps']) <= min(2, 8 - i)
    assert sorted(used) == list(range(len(recurrent))) and recurrent_state.shape == (len(recurrent), 2560)
    assert case['correct_prefix_sum'] == sum(x['correct_prefix'] for x in case['recurrent'])
    assert case['numeric_protectable_prefix_sum'] == sum(x['numeric_protectable_prefix'] for x in case['recurrent'])
    summary[pid] = {'teacher_control_byteexact': True, 'recurrent_steps': len(recurrent),
                    'correct_prefix_sum': case['correct_prefix_sum'], 'numeric_protectable_prefix_sum': case['numeric_protectable_prefix_sum'],
                    'saved_logits_maps_shift_and_first_divergence_pass': True, 'seed_teacher_drift': seed_drift}

saved_output_audit_seconds = time.perf_counter() - started

# Reconstruct one actual actor/cache using the already-approved constructor, then
# explicitly feed the saved previous pre-norm state. A target-feature replacement
# is a known-wrong recurrence control and must produce different logits.
source = (R / 'readout.py').read_text()
namespace = {'__file__': str(R / 'readout.py'), '__name__': 'actor_recomputation'}
exec(compile(source[:source.index('cases = {}')], str(R / 'readout.py'), 'exec'), namespace)
torch = namespace['torch']
receipt = json.loads((P / 'runs/w1a8-modern-admission-20261103-01/prompt-02-admitted.json').read_text())
meta = json.loads(Path(receipt['evidence']['capture']['path']).read_text())
shape = tuple(meta['feature_shape'])
arrays = {name: np.memmap(ref['path'], dtype='<f4', mode='r', shape=shape)
          for name, ref in receipt['evidence']['arrays'].items() if name.startswith('layer-')}
features = torch.from_numpy(np.concatenate([arrays['layer-%d.f32' % tap] for tap in (2, 16, 29)], axis=1))
embedding = {}
for i, token in enumerate(meta['input_tokens']):
    row = torch.from_numpy(np.array(arrays['layer-0.f32'][i], copy=True))
    assert token not in embedding or torch.equal(embedding[token], row)
    embedding[token] = row
lookup = namespace['ExternalNativeF32Embedding'](lookup=embedding.__getitem__, vocab_size=248320, hidden_size=2560,
    target_identity='Qwen3.5-4B-Q4_K_M native get_rows', target_sha256=receipt['evidence']['target']['sha256'],
    capture_sha256=receipt['evidence']['capture']['sha256'])
adapter = namespace['NativeStepAdapter'](namespace['drafter'], embedding_lookup=lookup, embedding_precision='external_native_f32')
case = report['cases']['2']
chain = case['recurrent'][3]
assert len(chain['steps']) == 2 and chain['steps'][0]['correct'] and not chain['steps'][1]['correct']
anchor = chain['anchor_position']
saved = matrix(case['recurrent_files']['logits'])
saved_state = matrix(case['recurrent_files']['pre_norm'])
stage = time.perf_counter()
with torch.inference_mode(), namespace['shared_round_hard_signs'](namespace['linears']):
    cache = adapter.build_context_cache(meta['input_tokens'][1:anchor + 1], features[:anchor], chunk_size=32)
    initial = adapter.encode_feature(features[anchor])
    first = adapter.decode_step(chain['steps'][0]['input_token_id'], initial, anchor, cache)
    assert np.array_equal(first.logits.numpy(), saved[chain['steps'][0]['logits_row_index']])
    assert np.array_equal(first.pre_norm.numpy(), saved_state[chain['steps'][0]['state_row_index']])
    token = chain['steps'][1]['input_token_id']
    recurrent = adapter.decode_step(token, first.pre_norm, anchor + 1, first.cache)
    assert np.array_equal(recurrent.logits.numpy(), saved[chain['steps'][1]['logits_row_index']])
    replacement = adapter.decode_step(token, adapter.encode_feature(features[anchor + 1]), anchor + 1, first.cache)
    wrong_feature_max_difference = float(torch.max(torch.abs(replacement.logits - recurrent.logits)))
    assert wrong_feature_max_difference > 1e-3, 'known-wrong target-feature recurrence was not distinguishable'
    try:
        adapter.decode_step(token, first.pre_norm, anchor + 2, first.cache)
    except ValueError:
        shifted_position_rejected = True
    else:
        raise AssertionError('wrong decoder/cache position admitted')

recomputation_seconds = time.perf_counter() - stage
peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert peak_bytes < 6 * 1024**3
result = {'status': 'SAVED_OUTPUT_AND_RECURRENT_STATE_AUDIT_PASS', 'producer_report': pin(R / 'report.json'),
          'audit_recipe': pin(__file__), 'cases': summary, 'selected_second_step_recomputed_byteexact': True,
          'known_wrong_target_feature_replacement_max_abs_logits_difference': wrong_feature_max_difference,
          'wrong_decoder_cache_position_rejected': shifted_position_rejected,
          'saved_output_audit_seconds': saved_output_audit_seconds, 'selected_recomputation_seconds': recomputation_seconds,
          'wall_seconds': time.perf_counter() - started, 'peak_rss_bytes': peak_bytes, 'training_updates': 0,
          'gpu_seconds': 0, 'target_model_opened_or_hashed': False, 'native_actor_qualified': False}
with (R / 'independent-audit.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({key: result[key] for key in ('status', 'wall_seconds', 'peak_rss_bytes',
      'known_wrong_target_feature_replacement_max_abs_logits_difference')}), flush=True)
