"""CPU-only saved native/CPU recurrent replay comparison; never imports a model."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import numpy as np

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/w1a8-modern-native-replay-20261103-01'
C = P / 'runs/w1a8-modern-recurrent-actor-20261103-01'
B = P / 'runs/w1a8-modern-replay-build-20261103-01'
H, D, V = 2560, 50000, 248320


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True, type=Path, help='exact helper --cases JSON')
    parser.add_argument('--native', type=Path, default=R / 'native01')
    parser.add_argument('--out', type=Path, default=R / 'comparison.json')
    args = parser.parse_args()
    started = time.perf_counter()
    require(args.out.resolve().is_relative_to(R.resolve()), 'output must stay inside assigned directory')
    require(not args.out.exists(), 'comparison output already exists; choose a new --out')
    baseline = json.loads((P / 'experiments/w1a8-standing-20261102/baseline.json').read_text())
    blocked = {str(Path(x['target_pin']['path']).resolve()) for x in baseline['rows']}
    pins = {}

    def pin(path):
        path = Path(path).resolve()
        require(str(path) not in blocked, 'target model access forbidden')
        # Model/weight contents are outside this comparison task. Use prior audit ancestry.
        require(path.suffix not in ('.gguf', '.safetensors', '.npz'), 'model/weight access forbidden')
        if str(path) not in pins:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            pins[str(path)] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
        return pins[str(path)]

    def checked(ref):
        actual = pin(ref['path'])
        require(actual['sha256'] == ref['sha256'], 'pin changed: ' + str(ref['path']))
        if 'bytes' in ref:
            require(actual['bytes'] == ref['bytes'], 'size changed: ' + str(ref['path']))
        return Path(ref['path'])

    def read(path):
        pin(path)
        return json.loads(Path(path).read_text())

    def array(path, shape, ref=None):
        if ref is not None:
            checked(ref)
        else:
            pin(path)
        require(Path(path).stat().st_size == int(np.prod(shape)) * 4, 'wrong F32 size: ' + str(path))
        values = np.fromfile(path, dtype='<f4').reshape(shape)
        require(np.isfinite(values).all(), 'nonfinite array: ' + str(path))
        return values

    def cpu_array(spec):
        return array(spec['file']['path'], tuple(spec['shape']), spec['file'])

    def difference(a, b):
        require(a.shape == b.shape and a.size > 0, 'difference shape/empty mismatch')
        av, bv = a.astype(np.float64), b.astype(np.float64)
        delta = av - bv
        denom = float(np.linalg.norm(bv))
        return {'max_abs': float(np.abs(delta).max()), 'mean_abs': float(np.abs(delta).mean()),
                'rms': float(np.sqrt(np.mean(delta * delta))),
                'relative_l2_cpu_denominator': float(np.linalg.norm(delta) / max(denom, 1e-30)),
                'byteexact': a.tobytes() == b.tobytes()}

    report = read(C / 'report.json')
    audit = read(C / 'independent-audit.json')
    require(audit['status'] == 'SAVED_OUTPUT_AND_RECURRENT_STATE_AUDIT_PASS', 'CPU reference not audited')
    checked(audit['producer_report'])
    checked(audit['audit_recipe'])
    for ref in report['source_and_native_pins']:
        checked(ref)
    build = read(B / 'build.json')
    for key in ('source', 'builder', 'executable', 'candidate_build', 'candidate_source_pins'):
        checked(build[key])
    for ref in build['candidate_binaries'] + build['headers']:
        checked(ref)
    export = read(P / 'runs/w1a8-modern-packed-init-20261103-01/export.json')
    prepared = read(R / 'prepared.json')
    for key in ('cases', 'draft_map', 'baseline', 'build'):
        checked(prepared[key])
    require(Path(prepared['cases']['path']).resolve() == args.packet.resolve(), 'packet differs from preparation')
    require(prepared['draft_model']['sha256'] == export['output']['sha256'], 'draft ancestry differs from frozen export')
    packet = read(args.packet)
    require(packet['schema'] == 'w1a8_modern_native_replay_v1', 'packet schema differs')
    map_path = Path(packet['draft_map'])
    pin(map_path)
    absolute = np.loadtxt(map_path, dtype=np.int64, ndmin=1)
    require(absolute.shape == (D,) and len(np.unique(absolute)) == D and
            absolute.min() >= 0 and absolute.max() < V, 'absolute map invalid')
    map_raw_hash = hashlib.sha256(absolute.astype('<i8').tobytes()).hexdigest()
    require(map_raw_hash == export['preserved_tensors']['d2t']['raw_sha256'], 'map differs from frozen native map')
    inverse = np.full(V, -1, dtype=np.int64)
    inverse[absolute] = np.arange(D)
    native = read(args.native / 'replay.json')
    require(native['status'] == 'NATIVE_REPLAY_COMPLETE' and native['correctness_mode'] == 32 and
            native['training_updates'] == 0 and native['kv'] == 'F16' and native['embedding_trace'] is True,
            'native run incomplete/wrong runtime')
    execution = read(R / 'result01.json')
    checked(execution['result'])
    require(Path(execution['result']['path']).resolve() == (args.native / 'replay.json').resolve() and
            execution['status'] == 'NATIVE_REPLAY_EXECUTED', 'native execution receipt differs')
    for ref in execution['mapped_images']:
        checked(ref)
    library_maps = (args.native / 'loaded-library-maps.txt')
    pin(library_maps)
    maps_text = library_maps.read_text()
    for ref in build['candidate_binaries']:
        if Path(ref['path']).name.startswith(('libllama.so.', 'libggml-cpu.so.', 'libggml-cuda.so.', 'libggml-base.so.', 'libggml.so.')):
            require(str(Path(ref['path']).resolve()) in maps_text, 'expected native library absent from live maps')
    packet_cases = {x['id']: x for x in packet['cases']}
    require(set(packet_cases) == set(native['cases']) == {'prompt-02', 'prompt-04'}, 'case set mismatch')
    result = {}
    for cid in sorted(packet_cases):
        item, ncase = packet_cases[cid], native['cases'][cid]
        case = report['cases'][str(int(cid.split('-')[1]))]
        receipt = read(checked(case['admission']))
        evidence = receipt['evidence']
        require(evidence['target']['sha256'] == prepared['target']['sha256'] and
                evidence['target']['path'] == prepared['target']['path'], 'admitted/native target ancestry differs')
        capture = read(checked(evidence['capture']))
        tokens = capture['input_tokens']
        prompt = capture['prompt_tokens']
        teacher_rows = receipt['rows']
        labels = [x['serving_label'] for x in teacher_rows]
        anchors = [prompt - 1 + i for i in range(8)]
        require(item['tokens'] == tokens and item['prompt_tokens'] == prompt and item['labels'] == labels and
                item['anchors'] == anchors and item['depth'] == 2, 'packet/admission geometry mismatch: ' + cid)
        require(ncase['tokens'] == len(tokens) and ncase['prompt_tokens'] == prompt, 'native token geometry differs')
        require(item['features'] == [evidence['arrays']['layer-%d.f32' % t]['path'] for t in (2, 16, 29)],
                'native feature tap order differs')
        for t in (0, 2, 16, 29):
            checked(evidence['arrays']['layer-%d.f32' % t])
        emb_ref = evidence['arrays']['layer-0.f32']
        layer0 = array(emb_ref['path'], (len(tokens), H), emb_ref)
        embedding = {}
        for i, token in enumerate(tokens):
            require(token not in embedding or np.array_equal(embedding[token], layer0[i]), 'captured duplicate token embedding differs')
            embedding[token] = layer0[i]
        encoded = array(args.native / (cid + '-encoded.f32'), (len(tokens), H))
        cencoded = cpu_array(case['teacher_files']['encoded'])
        encoded_rows = [{'anchor': a, **difference(encoded[a], cencoded[i])} for i, a in enumerate(anchors)]

        def numeric(label_index):
            ids = (label_index - 1, label_index)
            return all(teacher_rows[k]['supervision_eligible'] and teacher_rows[k]['protection_numeric_eligible'] and
                       teacher_rows[k]['native_argmax'] == teacher_rows[k]['serving_label'] and
                       k not in receipt['excluded_row_ids'] for k in ids)

        modes = {}
        for mode in ('teacher', 'recurrent'):
            nm = ncase[mode]
            records = nm['records']
            require(nm['rows'] == len(records) and [x['row'] for x in records] == list(range(len(records))), 'native rows missing/reordered')
            logits = array(args.native / (cid + '-' + mode + '-logits.f32'), (len(records), D))
            prenorm = array(args.native / (cid + '-' + mode + '-prenorm.f32'), (len(records), H))
            observed_embeddings = array(args.native / (cid + '-' + mode + '-embeddings.f32'), (len(records), H))
            files = case['teacher_files' if mode == 'teacher' else 'recurrent_files']
            clogits, cstates = cpu_array(files['logits']), cpu_array(files['pre_norm'])
            crows = case['teacher_forced_steps'] if mode == 'teacher' else [s for w in case['recurrent'] for s in w['steps']]
            cm = {(x['decoder_position'] if mode == 'teacher' else x['anchor_position'], x.get('depth', 0), x['decoder_position']): x for x in crows}
            require(len(cm) == len(crows), 'duplicate CPU join key')
            nkeys, comparisons, unpaired = set(), [], []
            windows = {w['anchor']: w for w in nm['windows']}
            require(len(windows) == len(nm['windows']) == 8 and set(windows) == set(anchors), 'native window missing/duplicate')
            window_result = []
            for anchor in anchors:
                group = [x for x in records if x['anchor'] == anchor]
                require(group and [x['depth'] for x in group] == list(range(len(group))), 'missing/reordered native step')
                require(len(group) <= (1 if mode == 'teacher' else min(2, len(labels) - (anchor - prompt + 2))), 'native steps exceed admitted limit')
                prefix = protectable = 0
                reachable = True
                for depth, x in enumerate(group):
                    index = anchor - prompt + 2 + depth
                    require(x['decoder_position'] == x['cache_position_max'] == anchor + depth and
                            x['label_index'] == index and x['label'] == labels[index], 'native row position/label misaligned')
                    require(x['input_token'] == labels[index - 1] and
                            (not depth or group[depth - 1]['correct'] and x['input_token'] == group[depth - 1]['proposed']),
                            'native gold label after alternative prefix')
                    values = logits[x['row']]
                    subset_proposed = int(absolute[int(values.argmax())])
                    require(x['proposed'] == subset_proposed, 'full native proposal differs from mapped saved logits')
                    require(x['correct'] == (x['proposed'] == x['label']), 'native recorded correctness wrong')
                    require(x['input_token'] in embedding, 'native embedding token absent from admitted capture')
                    embdiff = difference(observed_embeddings[x['row']], embedding[x['input_token']])
                    eligible = numeric(index)
                    reachable = reachable and eligible
                    if x['correct']:
                        prefix += 1
                        protectable += int(reachable)
                    else:
                        require(depth == len(group) - 1, 'native continued after first divergence')
                    key = (anchor, depth, anchor + depth)
                    require(key not in nkeys, 'duplicate native join key')
                    nkeys.add(key)
                    inverse_label = int(inverse[x['label']])
                    margin = None
                    if inverse_label >= 0:
                        other = values.copy(); other[inverse_label] = -np.inf
                        margin = float(values[inverse_label] - other.max())
                    detail = {'anchor': anchor, 'anchor_index': anchor - prompt + 1, 'depth': depth,
                              'decoder_position': anchor + depth, 'numeric_eligible': eligible,
                              'numeric_reachable': reachable, 'label': x['label'], 'label_supported': inverse_label >= 0,
                              'native_proposed': x['proposed'], 'native_correct': x['correct'], 'native_label_margin': margin,
                              'native_embedding_vs_admitted_layer0': embdiff}
                    if key in cm:
                        c = cm[key]
                        require(c['input_token_id'] == x['input_token'] and c['label_token_id'] == x['label'], 'CPU/native tokens misaligned')
                        require(bool(c.get('numeric_eligible', c.get('protection_numeric_eligible'))) == eligible, 'numeric eligibility differs')
                        li, si = c['logits_row_index'], c['state_row_index']
                        require(int(absolute[int(clogits[li].argmax())]) == c['proposed_target_token_id'], 'CPU saved logits proposal mismatch')
                        detail.update({'cpu_proposed': c['proposed_target_token_id'], 'cpu_correct': c['correct'],
                                       'proposal_matches': c['proposed_target_token_id'] == x['proposed'],
                                       'correctness_matches': c['correct'] == x['correct'], 'cpu_label_margin': c['label_margin'],
                                       'logits_native_minus_cpu': difference(values, clogits[li]),
                                       'prenorm_native_minus_cpu': difference(prenorm[x['row']], cstates[si])})
                        comparisons.append(detail)
                    else:
                        require(mode == 'recurrent' and depth > 0 and (anchor, depth - 1, anchor + depth - 1) in cm,
                                'unexplained native row without CPU match')
                        prior = cm[(anchor, depth - 1, anchor + depth - 1)]
                        require(not prior['correct'] and group[depth - 1]['correct'], 'unpaired native step lacks earlier correctness divergence')
                        unpaired.append({'side': 'native_only_after_correctness_divergence', **detail})
                win = windows[anchor]
                require(win['correct_prefix'] == prefix, 'native correct prefix metadata wrong')
                expected_stop = ('first_gold_divergence' if not group[-1]['correct'] else
                                 ('teacher_forced_restart' if mode == 'teacher' else
                                  ('no_admitted_next_teacher_row' if group[-1]['label_index'] + 1 >= len(labels) and len(group) < 2 else 'depth_limit')))
                require(win['stop_reason'] == expected_stop, 'native stop reason disagrees with saved rows')
                cpu_prefix = (int(case['teacher_forced_steps'][anchor - prompt + 1]['correct']) if mode == 'teacher' else
                              case['recurrent'][anchor - prompt + 1]['correct_prefix'])
                window_result.append({'anchor': anchor, 'native_correct_prefix': prefix, 'cpu_correct_prefix': cpu_prefix,
                                      'prefix_matches': prefix == cpu_prefix, 'native_numeric_eligible_prefix': protectable,
                                      'stop_reason': win['stop_reason'], 'native_diagnostic_steps': len(group)})
            require(len(nkeys) == len(records), 'native rows outside expected anchors')
            for key, c in cm.items():
                if key not in nkeys:
                    anchor, depth, position = key
                    require(mode == 'recurrent' and depth > 0, 'unexplained missing native row')
                    prior = next((x for x in records if x['anchor'] == anchor and x['depth'] == depth - 1), None)
                    require(prior is not None and not prior['correct'] and cm[(anchor, depth - 1, position - 1)]['correct'],
                            'CPU-only row lacks earlier native correctness divergence')
                    unpaired.append({'side': 'cpu_only_after_correctness_divergence', 'anchor': anchor, 'depth': depth,
                                     'decoder_position': position, 'cpu_correct': c['correct'],
                                     'cpu_proposed': c['proposed_target_token_id'], 'numeric_eligible': c['numeric_eligible']})
            modes[mode] = {'joined_rows': comparisons, 'unpaired_rows': unpaired, 'windows': window_result,
                           'native_diagnostic_rows': len(records), 'cpu_diagnostic_rows': len(crows),
                           'joined_proposal_changes': sum(not x['proposal_matches'] for x in comparisons),
                           'joined_correctness_changes': sum(not x['correctness_matches'] for x in comparisons),
                           'native_correct_prefix_sum': sum(x['native_correct_prefix'] for x in window_result),
                           'cpu_correct_prefix_sum': sum(x['cpu_correct_prefix'] for x in window_result),
                           'native_numeric_eligible_prefix_sum': sum(x['native_numeric_eligible_prefix'] for x in window_result)}
        result[cid] = {'encoded_fc_anchor_native_minus_cpu': encoded_rows, 'modes': modes}
    output = {'status': 'COMPARISON_COMPLETE_WITH_DECISION_IMPACT_QUANTIFIED',
              'scope': 'Reused development histories; numeric native replay comparison only; no training, untouched generalization, serving verifier, quality or throughput qualification',
              'arithmetic': 'CPU dequantized-activation matmul and native integer-dot scale order unchanged; discrepancies reported by decision impact',
              'join': 'actual anchor, depth, decoder position; only explicit earlier correctness divergence permits unpaired rows',
              'absolute_map_raw_i64_sha256': map_raw_hash, 'cases': result,
              'all_input_pins': list(pins.values()), 'comparison_recipe': pin(__file__),
              'target_model_opened_or_hashed': False, 'model_or_optimizer_loaded': False,
              'native_execution_ancestry': 'Pinned helper build, packet, output and loaded-library maps; parent retains actual invocation/target ancestry evidence',
              'native_actor_qualified': False, 'parent_success_protection_proven': False,
              'wall_seconds': time.perf_counter() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'output': str(args.out), 'wall_seconds': output['wall_seconds'],
                      'cases': {cid: {m: {k: v for k, v in data['modes'][m].items() if k not in ('joined_rows', 'unpaired_rows', 'windows')}
                                      for m in ('teacher', 'recurrent')} for cid, data in result.items()}}, indent=2))


if __name__ == '__main__':
    main()
