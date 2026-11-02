"""Audit frozen raw results and export the full baseline matrix and intervals."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from run_matrix import pin, read, sha, signature, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-pins', action='store_true')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(exist_ok=True)
    began = time.perf_counter()
    freeze = read(run / 'final-freeze.json')
    inputs = read(run / 'final-inputs/receipt.json')
    assert inputs['manifest_sha256'] == sha(run / 'final-freeze.json')
    assert freeze['frozen_at_utc'] < inputs['created_at_utc']
    for name, digest in inputs['files'].items():
        assert sha(run / 'final-inputs' / name) == digest
    selection = read(run / 'development-merged.json')['selection']
    assert {(row['target_id'], row['selected']) for row in selection} == {(row['target_id'], row['id']) for row in freeze['arms']}
    receipt = read(run / 'final-01/receipt.json')
    assert receipt['completed'] and not receipt['error'] and len(receipt['records']) == 290
    assert all(row['status'] == 'completed' for row in receipt['records'])
    if args.verify_pins:
        for ref in freeze['pins']:
            assert Path(ref['path']).stat().st_size == ref['bytes'] and sha(ref['path']) == ref['sha256'], ref['path']
    raw_path = run / 'final-01/requests.jsonl'
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(rows) == 8352
    assert sum(row['phase'] == 'performance' for row in rows) == 6960
    assert sum(row['phase'] == 'quality' for row in rows) == 1392
    refs = {(row['target_id'], row['rep'], row['phase'], row['prompt_id']): row
            for row in rows if row['family'] == 'target_only'}
    by_arm = defaultdict(list)
    first = {}
    speculative_matches = 0
    repeated_matches = 0
    stop_types = defaultdict(int)
    for row in rows:
        result = row['result']
        reference = refs[row['target_id'], row['rep'], row['phase'], row['prompt_id']]['result']
        assert signature(result) == signature(reference)
        assert result['tokens_predicted'] == len(result['tokens']) == row['output_tokens']
        assert result['timings']['cache_n'] == 0 and not result['truncated']
        assert result['generation_settings']['temperature'] == 0 and result['generation_settings']['seed'] == 1234
        if row['family'] != 'target_only':
            speculative_matches += 1
            assert row['speculative']['accepted'] == result['timings']['draft_n_accepted']
            assert row['speculative']['proposed'] == result['timings']['draft_n']
        key = row['target_id'], row['arm_id'], row['phase'], row['prompt_id']
        if key in first:
            assert signature(first[key]) == signature(result)
            repeated_matches += 1
        else:
            first[key] = result
        by_arm[row['target_id'], row['arm_id']].append(row)
        stop_types[row['phase'] + '/' + result['stop_type']] += 1
    assert speculative_matches == 7776 and repeated_matches == 5568
    original_analysis = read(run / 'final-analysis.json')
    assert all(arm['admitted'] for arm in original_analysis['arms'])
    original_by_arm = {(arm['target_id'], arm['id']): arm for arm in original_analysis['arms']}
    matrix = []
    clusters = {}
    for arm in freeze['arms']:
        key = arm['target_id'], arm['id']
        data = by_arm[key]
        performance = [row for row in data if row['phase'] == 'performance']
        quality = [row for row in data if row['phase'] == 'quality']
        assert len(performance) == 120 and len(quality) == 24
        tokens = sum(row['output_tokens'] for row in performance)
        wall = sum(row['full_request_seconds'] for row in performance)
        counters = {k: sum(row['speculative'][k] for row in performance) for k in ('accepted', 'proposed', 'rounds')}
        recorded = original_by_arm[key]
        assert tokens == recorded['tokens'] and counters == recorded['speculative']
        assert math.isclose(wall, recorded['full_request_seconds'], rel_tol=1e-12)
        assert math.isclose(tokens / wall, recorded['tokens_per_second'], rel_tol=1e-12)
        if arm['family'] != 'target_only':
            assert counters['proposed'] > 0
        reps = [sum(r['full_request_seconds'] for r in performance if r['rep'] == rep) for rep in range(5)]
        rep_tps = [sum(r['output_tokens'] for r in performance if r['rep'] == rep) / reps[rep] for rep in range(5)]
        prompt_groups = defaultdict(lambda: [0, 0.])
        for row in performance:
            prompt_groups[row['prompt_id']][0] += row['output_tokens']
            prompt_groups[row['prompt_id']][1] += row['full_request_seconds']
        clusters[key] = prompt_groups
        entry = {'target': arm['target_id'], 'family': arm['family'], 'draft_precision': arm['dtype'], 'draft_depth': arm['depth'],
                 'arm_id': arm['id'], 'tokens_per_second': tokens / wall, 'repetition_tokens_per_second': rep_tps,
                 'repetition_min_tps': min(rep_tps), 'repetition_max_tps': max(rep_tps),
                 'repetition_wall_cv_percent': statistics.stdev(reps) / statistics.mean(reps) * 100,
                 'accepted': counters['accepted'], 'proposed': counters['proposed'], 'rounds': counters['rounds'],
                 'acceptance_fraction': counters['accepted'] / counters['proposed'] if counters['proposed'] else None,
                 'quality_passed': sum(row['judgement']['passed'] for row in quality), 'quality_cases': 24,
                 'admitted': True}
        matrix.append(entry)
        for rep in range(5):
            cell = run / 'final-01' / arm['target_id'] / f"rep-{rep}-{arm['id']}"
            command = read(cell / 'command.json')
            assert command['environment']['GGML_CUDA_DECODE_CONSISTENT'] == '32'
            argv = command['command']
            assert argv[argv.index('-m') + 1] == freeze['targets'][arm['target_id']]['path']
            assert argv[argv.index('--spec-type') + 1] == arm['spec_type']
            if arm['draft']:
                assert argv[argv.index('-md') + 1] == arm['draft']['path']
            if arm['family'] != 'target_only':
                assert int(argv[argv.index('--spec-draft-n-max') + 1]) == arm['depth']
            assert not any('synth' in token for token in argv)
            maps = (cell / 'loaded-library-maps.txt').read_text()
            cuda = sorted({line.split()[-1] for line in maps.splitlines() if 'libggml-cuda.so' in line})
            assert cuda == [freeze['cuda']['path']]
    lookup = {(row['target'], row['arm_id']): row for row in matrix}
    targets = {}
    rng = np.random.default_rng(20261101)
    draws = rng.integers(0, 24, size=(20000, 24))

    def interval(target, candidate, control):
        a, b = clusters[target, candidate], clusters[target, control]
        keys = sorted(a)
        assert keys == sorted(b) and all(a[k][0] == b[k][0] for k in keys)
        aa = np.array([a[k][1] for k in keys])
        bb = np.array([b[k][1] for k in keys])
        values = bb[draws].sum(axis=1) / aa[draws].sum(axis=1)
        return {'point': float(bb.sum() / aa.sum()), 'low': float(np.quantile(values, .025)),
                'high': float(np.quantile(values, .975)), 'prompt_clusters': 24, 'resamples': 20000,
                'scope': 'Conditional paired prompt-cluster interval; repetitions stay together, not a production or multiple-comparison winner guarantee'}

    for target in freeze['targets']:
        options = sorted((row for row in matrix if row['target'] == target), key=lambda row: -row['tokens_per_second'])
        reference = next(row for row in options if row['family'] == 'target_only')
        primary = next(row for row in options if row['family'] == 'eagle3' and row['draft_precision'] == 'Q4_0')
        best = options[0]
        runner_up = options[1]
        mtp = next(row for row in options if row['family'] == 'mtp')
        family_best = {family: max((row for row in options if row['family'] == family), key=lambda row: row['tokens_per_second'])
                       for family in dict.fromkeys(row['family'] for row in options)}
        comparisons = {'best_vs_primary': interval(target, best['arm_id'], primary['arm_id']),
                       'best_vs_target_only': interval(target, best['arm_id'], reference['arm_id']),
                       'best_vs_runner_up': interval(target, best['arm_id'], runner_up['arm_id']),
                       'best_vs_mtp': interval(target, best['arm_id'], mtp['arm_id']),
                       'primary_vs_target_only': interval(target, primary['arm_id'], reference['arm_id'])}
        for row in options:
            row['speedup_vs_target_only'] = row['tokens_per_second'] / reference['tokens_per_second']
            row['speedup_vs_primary_q4_0'] = row['tokens_per_second'] / primary['tokens_per_second']
        competitive = [row['arm_id'] for row in options if interval(target, best['arm_id'], row['arm_id'])['low'] <= 1]
        targets[target] = {'target_only': reference, 'primary_q4_0_eagle': primary, 'fastest_observed': best,
                           'runner_up': runner_up, 'family_best': family_best, 'comparisons': comparisons,
                           'not_separated_from_best_by_prompt_interval': competitive,
                           'quality_passed': reference['quality_passed'], 'quality_cases': 24}
    phases = {}
    for directory in ['pilot-01', 'screen-01', *sorted(p.name for p in run.glob('refinement-*') if p.is_dir()), 'final-01']:
        path = run / directory / 'receipt.json'
        if path.exists():
            phase_receipt = read(path)
            if 'costs' in phase_receipt:
                phases[directory] = phase_receipt['costs']
    campaign = read(run / 'confirmation-campaign-receipt.json')
    audit = {'passed': True, 'raw_requests': len(rows), 'performance_requests': 6960, 'quality_requests': 1392,
             'exact_speculative_comparisons': speculative_matches, 'exact_repeat_comparisons': repeated_matches,
             'cells': 290, 'groups': 58, 'input_hashes_rechecked': True,
             'all_frozen_pin_hashes_rechecked': args.verify_pins, 'frozen_pin_count': len(freeze['pins']),
             'frozen_pin_bytes': sum(p['bytes'] for p in freeze['pins']), 'all_loaded_cuda_maps_match': True,
             'stop_types': dict(stop_types), 'verification_wall_seconds': time.perf_counter() - began,
             'report_source': pin(__file__),
             'final_freeze': pin(run / 'final-freeze.json'), 'input_receipt': pin(run / 'final-inputs/receipt.json'),
             'raw_requests_file': pin(raw_path)}
    write(run / 'root-final-audit.json', audit)
    summary = {'scope': 'Qwen3.5 quant4group correctness-mode matched full-HTTP-request baseline matrix on RTX5080; not W1A8 or stock equivalence',
               'targets': targets, 'audit': audit, 'costs': {'phases': phases,
                   'refinement_and_confirmation_inclusive_wall_seconds': campaign['total_wall_seconds'], 'training_seconds': 0,
                   'overlap_note': 'Inclusive campaign and constituent phase walls overlap; do not add them.'},
               'sources': {'final_freeze_sha256': sha(run / 'final-freeze.json'), 'raw_requests_sha256': sha(raw_path)},
               'statistics': 'TPS is pooled generated tokens / summed full HTTP request wall. Paired prompt-cluster bootstrap uses20000draws; intervals are conditional on this panel and chosen comparisons, not a globally optimal model claim.'}
    write(out / 'summary.json', summary)
    fields = [key for key in matrix[0] if key != 'repetition_tokens_per_second']
    with (out / 'matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in matrix:
            writer.writerow({k: row[k] for k in fields})
    for target, data in targets.items():
        print(target, 'best', data['fastest_observed']['arm_id'], round(data['fastest_observed']['tokens_per_second'], 3),
              'best/runner-upCI', data['comparisons']['best_vs_runner_up'], flush=True)


if __name__ == '__main__':
    main()
