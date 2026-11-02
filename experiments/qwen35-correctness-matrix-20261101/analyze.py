"""Recompute baseline correctness, quality, timing and development selections."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from run_matrix import load_module, pin, read, sha, signature, write


def analyze(manifest_path, inputs_path, output_path):
    manifest = read(manifest_path)
    receipt = read(output_path / 'receipt.json')
    assert receipt['completed'], receipt['error']
    assert sha(manifest_path) == read(inputs_path / 'receipt.json')['manifest_sha256']
    judge = load_module('analysis_quality_judge', manifest['judge']['path'])
    assert sha(manifest['judge']['path']) == manifest['judge']['sha256']
    cases = {row['id']: row for row in read(inputs_path / 'quality-cases.json')}
    prompt_count = len(read(inputs_path / 'requests.json'))
    rows = [json.loads(line) for line in (output_path / 'requests.jsonl').read_text().splitlines()]
    by = {(r['target_id'], r['arm_id'], r['rep'], r['phase'], r['prompt_id']): r for r in rows}
    assert len(by) == len(rows)
    completed = {(r['target_id'], r['arm_id'], r['rep']): r for r in receipt['records']}
    target_reference = {target: next(a for a in manifest['arms'] if a['target_id'] == target and a['family'] == 'target_only')['id']
                        for target in manifest['targets']}
    arms, selected, boundaries = [], [], []
    groups = defaultdict(list)
    for arm in manifest['arms']:
        target, arm_id = arm['target_id'], arm['id']
        matching = [r for r in rows if r['target_id'] == target and r['arm_id'] == arm_id]
        performance = [r for r in matching if r['phase'] == 'performance']
        quality = [r for r in matching if r['phase'] == 'quality']
        failures = [r for r in receipt['records'] if r['target_id'] == target and r['arm_id'] == arm_id and r['status'] != 'completed']
        mismatches, repeat_mismatches, quality_mismatches = [], [], []
        reference_missing = False
        for row in matching:
            key = (target, target_reference[target], row['rep'], row['phase'], row['prompt_id'])
            reference = by.get(key)
            if reference is None:
                reference_missing = True
                continue
            if signature(row['result']) != signature(reference['result']):
                item = {'rep': row['rep'], 'prompt_id': row['prompt_id']}
                a, b = row['result']['tokens'], reference['result']['tokens']
                item['first_token_difference'] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None)
                (quality_mismatches if row['phase'] == 'quality' else mismatches).append(item)
            first = by.get((target, arm_id, 0, row['phase'], row['prompt_id']))
            if first and signature(first['result']) != signature(row['result']):
                repeat_mismatches.append({'rep': row['rep'], 'prompt_id': row['prompt_id']})
            if row['phase'] == 'quality':
                assert judge.judge(cases[row['prompt_id']], row['result']['content']) == row['judgement']
        complete = (not failures and len(performance) == prompt_count * manifest['repetitions'] and len(quality) == len(cases)
                    and all((target, arm_id, rep) in completed for rep in range(manifest['repetitions'])))
        counters = {k: sum(r['speculative'][k] for r in performance) for k in ('accepted', 'proposed', 'rounds')}
        active = arm['family'] == 'target_only' or counters['proposed'] > 0
        admitted = complete and not reference_missing and not mismatches and not quality_mismatches and not repeat_mismatches and active
        seconds = sum(r['full_request_seconds'] for r in performance)
        tokens = sum(r['output_tokens'] for r in performance)
        repetition_rates = []
        repetition_seconds = []
        for rep in range(manifest['repetitions']):
            rr = [r for r in performance if r['rep'] == rep]
            wall = sum(r['full_request_seconds'] for r in rr)
            repetition_seconds.append(wall)
            repetition_rates.append(sum(r['output_tokens'] for r in rr) / wall if wall else None)
        arm_summary = {**arm, 'complete': complete, 'admitted': admitted, 'active_speculation': active,
                       'performance_requests': len(performance), 'quality_requests': len(quality),
                       'quality_passed': sum(r['judgement']['passed'] for r in quality),
                       'quality_failures': {r['prompt_id']: r['judgement']['reason'] for r in quality if not r['judgement']['passed']},
                       'performance_mismatches': mismatches, 'quality_mismatches': quality_mismatches,
                       'repeat_mismatches': repeat_mismatches, 'failures': failures,
                       'tokens': tokens, 'full_request_seconds': seconds, 'tokens_per_second': tokens / seconds if seconds else None,
                       'repetition_tokens_per_second': repetition_rates, 'repetition_seconds': repetition_seconds,
                       'speculative': counters, 'acceptance_fraction': counters['accepted'] / counters['proposed'] if counters['proposed'] else None}
        if complete and len(repetition_seconds) >= 2:
            arm_summary['repetition_wall_cv_percent'] = statistics.stdev(repetition_seconds) / statistics.mean(repetition_seconds) * 100
        arms.append(arm_summary)
        groups[target, arm['group']].append(arm_summary)
    for (target, group), candidates in groups.items():
        admitted = [a for a in candidates if a['admitted']]
        if not admitted:
            selected.append({'target_id': target, 'group': group, 'selected': None, 'reason': 'No complete correct active development configuration'})
            continue
        best = min(admitted, key=lambda a: a['full_request_seconds'])
        selected.append({'target_id': target, 'group': group, 'selected': best['id'], 'development_tokens_per_second': best['tokens_per_second']})
        depths = sorted(a['depth'] for a in candidates)
        if len(depths) > 1 and best['depth'] == depths[-1]:
            boundaries.append({'target_id': target, 'group': group, 'depth': best['depth'],
                               'note': 'Review whether an additional supported depth is informative before final freeze; block-trained maxima must not be exceeded.'})
    practical = {}
    for target in manifest['targets']:
        valid = [a for a in arms if a['target_id'] == target and a['admitted']]
        winners = sorted(valid, key=lambda a: -a['tokens_per_second'])
        practical[target] = {'fastest_valid_arm': winners[0]['id'] if winners else None,
                             'fastest_valid_tps': winners[0]['tokens_per_second'] if winners else None,
                             'primary_q4_0_eagle': [a['id'] for a in valid if a['family'] == 'eagle3' and a['dtype'] == 'Q4_0'],
                             'valid_arm_count': len(valid)}
    result = {'scope': manifest['scope'], 'manifest': pin(manifest_path), 'arms': arms,
              'selection': selected, 'boundary_review': boundaries, 'practical_baselines': practical,
              'costs': receipt['costs'], 'training_seconds': 0}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.inputs, args.output)
    write(args.summary, result)
    print(json.dumps({'arms': len(result['arms']), 'admitted': sum(a['admitted'] for a in result['arms']),
                      'practical_baselines': result['practical_baselines'], 'boundary_review': result['boundary_review']}), flush=True)


if __name__ == '__main__':
    main()
