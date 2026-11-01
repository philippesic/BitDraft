"""Recompute the frozen ordinary-vLLM qualification; no engine imports or GPU use."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def load(path):
    return json.loads(path.read_text())


def signature(row):
    return tuple(row[k] if k != 'token_ids' else tuple(row[k])
                 for k in ('token_ids', 'text', 'finish_reason', 'stop_reason'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/clean-engine-qualification-20261101'))
    args = parser.parse_args()
    root = args.root
    freeze = load(root / 'final-freeze.json')
    inputs = load(root / 'final-inputs/receipt.json')
    assert hashlib.sha256((root / 'final-freeze.json').read_bytes()).hexdigest() == inputs['freeze_sha256']
    for entry in freeze['sources']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    for name, sha in inputs['files'].items():
        assert hashlib.sha256((root / 'final-inputs' / name).read_bytes()).hexdigest() == sha
    assert freeze['frozen_at_utc'] < inputs['created_at_utc']
    controls = load(root / 'final-inputs/quality-controls.json')
    assert len(controls) == 24
    assert all(r['correct_control_passed'] and r['known_wrong_control_rejected'] for r in controls)
    receipt = load(root / 'final-01/receipt.json')
    assert receipt['completed'] and not receipt['failure'] and len(receipt['records']) == 20
    judge_path = Path(freeze['sources'][-1]['path'])
    spec = importlib.util.spec_from_file_location('frozen_judge', judge_path)
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    cases = {r['id']: r for r in load(root / 'final-inputs/quality-cases.json')}
    cells = {}
    costs = {'startup_seconds': 0, 'excluded_warmup_seconds': 0,
             'performance_generate_seconds': 0, 'quality_generate_seconds': 0,
             'phase_wall_seconds': receipt['total_wall_seconds'], 'training_seconds': 0}
    for record in receipt['records']:
        assert record['exit_code'] == 0
        mode, arm, rep = (record[k] for k in ('mode', 'arm', 'rep'))
        cell = root / 'final-01' / f'{rep}-{mode}-{arm}'
        expected = dict(freeze['engine_kwargs'])
        if arm == 'eagle3':
            expected['speculative_config'] = freeze['speculative_config']
        assert load(cell / 'kwargs.json') == expected
        summary = load(cell / 'summary.json')
        costs['startup_seconds'] += summary['startup_seconds']
        costs['excluded_warmup_seconds'] += summary['excluded_warmup_seconds']
        rows = [json.loads(line) for line in (cell / 'requests.jsonl').read_text().splitlines()]
        performance = {r['id']: r for r in rows if r['phase'] == 'performance'}
        quality = {r['id']: r for r in rows if r['phase'] == 'quality'}
        assert len(performance) == 24 and len(quality) == (24 if rep == 0 else 0)
        assert all(len(r['token_ids']) == 128 for r in performance.values())
        assert summary['requests'] == len(rows) and summary['completed']
        for row in rows:
            costs[row['phase'] + '_generate_seconds'] += row['full_generate_wall_seconds']
            if row['phase'] == 'quality':
                assert judge.judge(cases[row['id']], row['text']) == row['judgement']
        counters = {}
        for name in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
            metric = 'vllm:spec_decode_' + name
            total = 0
            for row in performance.values():
                before = {m['name']: m for m in row['metrics_before']}
                after = {m['name']: m for m in row['metrics_after']}
                delta = after.get(metric, {}).get('value', 0) - before.get(metric, {}).get('value', 0)
                assert delta >= 0
                total += delta
            counters[name] = total
        if arm == 'eagle3':
            assert counters['num_draft_tokens'] == 3 * counters['num_drafts']
            assert 0 < counters['num_accepted_tokens'] <= counters['num_draft_tokens']
        cells[mode, arm, rep] = {'performance': performance, 'quality': quality, 'counters': counters}
    modes = {}
    for mode in ('invariant', 'stock'):
        arms = {}
        paired = []
        for rep in range(5):
            target = cells[mode, 'target_only', rep]['performance']
            draft = cells[mode, 'eagle3', rep]['performance']
            paired.append([key for key in target if signature(target[key]) != signature(draft[key])])
        for arm in ('target_only', 'eagle3'):
            reps = [cells[mode, arm, rep] for rep in range(5)]
            times = [sum(r['full_generate_wall_seconds'] for r in c['performance'].values()) for c in reps]
            tokens = [sum(len(r['token_ids']) for r in c['performance'].values()) for c in reps]
            for c in reps[1:]:
                assert all(signature(row) == signature(reps[0]['performance'][key]) for key, row in c['performance'].items())
            quality = reps[0]['quality']
            arms[arm] = {'rep_seconds': times, 'rep_output_tokens': tokens,
                         'mean_seconds': statistics.mean(times),
                         'cv_percent': statistics.stdev(times) / statistics.mean(times) * 100,
                         'aggregate_tokens_per_second': sum(tokens) / sum(times),
                         'quality_passed': sum(r['judgement']['passed'] for r in quality.values()),
                         'quality_failures': {k: r['judgement']['reason'] for k, r in quality.items() if not r['judgement']['passed']},
                         'acceptance': {name: sum(c['counters'][name] for c in reps) for name in reps[0]['counters']}}
        target_quality = cells[mode, 'target_only', 0]['quality']
        draft_quality = cells[mode, 'eagle3', 0]['quality']
        quality_output_matches = sum(signature(r) == signature(draft_quality[k]) for k, r in target_quality.items())
        modes[mode] = {'arms': arms, 'paired_mismatch_ids_per_rep': paired,
                       'quality_exact_output_matches': quality_output_matches,
                       'request_speedup': arms['target_only']['mean_seconds'] / arms['eagle3']['mean_seconds']}
    assert not any(modes['invariant']['paired_mismatch_ids_per_rep'])
    assert modes['invariant']['quality_exact_output_matches'] == 24
    result = {'scope': 'ordinary upstream BF16 EAGLE3 foundation, not Binary or Q4_0 qualification',
              'modes': modes, 'costs': costs,
              'source_and_input_hashes_rechecked': True,
              'recorded_quality_controls': {'correct_passed': 24, 'known_wrong_rejected': 24},
              'jit_caveat': 'Inference JIT warnings cannot all be assigned to warmups versus measured calls. These are observed full-generate wall times, not isolated steady GPU timings.',
              'cross_mode_target_mismatch_ids': [k for k, r in cells['stock', 'target_only', 0]['performance'].items() if signature(r) != signature(cells['invariant', 'target_only', 0]['performance'][k])],
              'cross_mode_eagle_mismatch_ids': [k for k, r in cells['stock', 'eagle3', 0]['performance'].items() if signature(r) != signature(cells['invariant', 'eagle3', 0]['performance'][k])]}
    result['observed_mode_wall_increase_percent'] = {
        arm: (modes['invariant']['arms'][arm]['mean_seconds'] / modes['stock']['arms'][arm]['mean_seconds'] - 1) * 100
        for arm in ('target_only', 'eagle3')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
