"""Finish development refinement, then freeze and execute fresh confirmation.

All target/runtime/model files stay fixed. Development timing is used only to
choose settings; final inputs do not exist until that choice is frozen.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from analyze import analyze
from run_matrix import pin, read, sha, signature, write


ARM_KEYS = ('target_id', 'id', 'group', 'family', 'dtype', 'draft', 'spec_type', 'depth')


def arm_config(row, depth=None):
    result = {key: row[key] for key in ARM_KEYS}
    if depth is not None:
        result['depth'] = depth
        result['id'] = result['group'] + f'-d{depth}'
    return result


def merge_development(run, summaries):
    # A repeated setting uses its latest matched phase. Every timing remains an
    # actual observation; only the selection score is normalized to that phase's
    # target-only timing to reduce between-phase clock/load effects.
    configurations = {}
    originals = None
    provenance = []
    for summary_path in summaries:
        data = read(summary_path)
        assert all(row['admitted'] for row in data['arms']), 'Stop on development admission failures'
        phase_path = Path(data['raw_output'])
        rows = [json.loads(line) for line in (phase_path / 'requests.jsonl').read_text().splitlines()]
        refs = {(row['target_id'], row['phase'], row['prompt_id']): row['result']
                for row in rows if row['family'] == 'target_only'}
        if originals is None:
            originals = refs
        for row in rows:
            key = row['target_id'], row['phase'], row['prompt_id']
            assert signature(row['result']) == signature(originals[key]), 'Development reference/output changed across phases'
        controls = {row['target_id']: row['tokens_per_second'] for row in data['arms'] if row['family'] == 'target_only'}
        for row in data['arms']:
            record = copy.deepcopy(row)
            record['development_phase'] = str(phase_path)
            record['selection_speedup_vs_phase_target'] = row['tokens_per_second'] / controls[row['target_id']]
            configurations[row['target_id'], row['id']] = record
        provenance.append(pin(summary_path))
    groups = {}
    for row in configurations.values():
        groups.setdefault((row['target_id'], row['group']), []).append(row)
    assert len(groups) == 58
    selection = []
    for (target, group), rows in groups.items():
        best = max(rows, key=lambda row: row['selection_speedup_vs_phase_target'])
        selection.append({'target_id': target, 'group': group, 'selected': best['id'],
                          'development_tokens_per_second': best['tokens_per_second'],
                          'selection_speedup_vs_phase_target': best['selection_speedup_vs_phase_target']})
    result = {'scope': 'Development selection only, phase-normalized; not a final performance claim',
              'arms': list(configurations.values()), 'selection': selection, 'phase_provenance': provenance,
              'selection_rule': 'Latest observation per repeated setting; maximize speedup versus its matched phase target-only. All outputs must equal original development references. No final outputs inspected.',
              'training_seconds': 0}
    write(run / 'development-merged.json', result)
    return result


def winners(summary):
    return [next(row for row in summary['arms'] if row['target_id'] == choice['target_id'] and row['id'] == choice['selected'])
            for choice in summary['selection']]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project, run = args.project.resolve(), args.run.resolve()
    code = project / 'experiments/qwen35-correctness-matrix-20261101'
    base = read(run / 'screen-freeze.json')
    assert not (run / 'final-freeze.json').exists() and not (run / 'final-inputs').exists()
    assert all(row['admitted'] for row in read(run / 'passive-screen-analysis.json')['arms'])
    initial = read(run / 'passive-screen-analysis.json')
    initial['raw_output'] = str(run / 'screen-01')
    write(run / 'development-screen-source.json', initial)
    summaries = [run / 'development-screen-source.json']
    summary = merge_development(run, summaries)
    records = []
    started = time.perf_counter()

    def execute(arguments):
        begin = time.perf_counter()
        result = subprocess.run([sys.executable, '-B', *map(str, arguments)], cwd=project)
        records.append({'command': list(map(str, arguments)), 'exit_code': result.returncode,
                        'wall_seconds': time.perf_counter() - begin})
        write(run / 'confirmation-campaign-progress.json', records)
        assert result.returncode == 0, 'Preserve failed phase for root repair; do not proceed to held-out inputs'

    def run_refinement(name, requested):
        targets = [arm_config(row) for row in winners(summary) if row['family'] == 'target_only']
        requested = list({(row['target_id'], row['id']): arm_config(row) for row in requested}.values())
        targets.extend(row for row in requested if row['family'] != 'target_only')
        manifest = copy.deepcopy(base)
        manifest.update(scope=f'Development {name}; not held-out or a final ranking', arms=targets,
                        frozen_at_utc=datetime.now(timezone.utc).isoformat(), repetitions=1)
        manifest['sources'].append(pin(__file__))
        manifest['pins'].append(pin(__file__))
        manifest_path = run / f'{name}-freeze.json'
        write(manifest_path, manifest)
        inputs = run / f'{name}-inputs'
        inputs.mkdir(exist_ok=False)
        for filename in ('requests.json', 'quality-cases.json', 'quality-controls.json'):
            (inputs / filename).write_bytes((run / 'screen-inputs' / filename).read_bytes())
        write(inputs / 'receipt.json', {'manifest_sha256': sha(manifest_path),
                                      'files': {file.name: sha(file) for file in inputs.glob('*.json')}})
        output = run / name
        execute([code / 'run_matrix.py', '--project', project, '--run', run, '--manifest', manifest_path,
                 '--inputs', inputs, '--phase', 'screen', '--output', name])
        result = analyze(manifest_path, inputs, output)
        result['raw_output'] = str(output)
        summary_path = run / f'{name}-analysis.json'
        write(summary_path, result)
        assert all(row['admitted'] for row in result['arms']), 'Do not freeze around failed configurations'
        summaries.append(summary_path)
        return merge_development(run, summaries)

    # Inspect both immediate neighbors of the dominant depth-three result for
    # every format, and the missing small-depth boundary for block drafters.
    requested = []
    current = winners(summary)
    for best in current:
        if best['family'] == 'target_only':
            continue
        requested += [arm_config(best, depth) for depth in (2, 4)]
        if best['family'] in ('dflash', 'dspark'):
            requested.append(arm_config(best, 1))
        elif best['depth'] == 7:
            requested.append(arm_config(best, 15))
    # Repeat the primary and best architecture controls within this phase.
    for target in base['targets']:
        relevant = [row for row in current if row['target_id'] == target]
        requested += [arm_config(row) for row in relevant if row['family'] == 'mtp' or row['group'] == 'eagle3-Q4_0']
        for family in ('dflash', 'dspark'):
            candidates = [row for row in relevant if row['family'] == family]
            if candidates:
                requested.append(arm_config(max(candidates, key=lambda row: row['selection_speedup_vs_phase_target'])))
    summary = run_refinement('refinement-01', requested)

    # Continue only informative local searches: the required Q4_0 EAGLE control
    # and configurations that already beat target-only. Below-target secondary
    # paths retain their best tested settings rather than consume an open sweep.
    # 31 is the actual max verified draft length for this width-32 runtime;
    # block-model limits are the trained maxima, not arbitrary job budgets.
    iteration = 2
    while True:
        tested = {(row['target_id'], row['group'], row['depth']) for row in summary['arms']}
        requested = []
        for best in winners(summary):
            if best['family'] == 'target_only':
                continue
            if best['group'] != 'eagle3-Q4_0' and best['selection_speedup_vs_phase_target'] <= 1:
                continue
            maximum = {'dflash': 15, 'dspark': 16}.get(best['family'], 31)
            missing = [depth for depth in (best['depth'] - 1, best['depth'] + 1)
                       if 1 <= depth <= maximum and (best['target_id'], best['group'], depth) not in tested]
            if missing:
                requested.append(arm_config(best))
                requested += [arm_config(best, depth) for depth in missing]
        if not requested:
            break
        summary = run_refinement(f'refinement-{iteration:02d}', requested)
        iteration += 1
    write(run / 'development-completion.json', {'completed': True, 'phases': [pin(path) for path in summaries],
        'selection': summary['selection'], 'training_seconds': 0,
        'limits': 'Best tested development settings; neighbor-complete for primary and target-beating paths. No global optimality claim. All groups retain admission.'})
    execute([code / 'select_final.py', '--run', run, '--development-summary', run / 'development-merged.json'])
    final_path = run / 'final-freeze.json'
    final = read(final_path)
    final['sources'].append(pin(__file__))
    final['pins'].append(pin(__file__))
    final['development_phase_provenance'] = [pin(path) for path in summaries]
    final['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(final_path, final)
    execute([code / 'generate_inputs.py', '--project', project, '--run', run, '--manifest', final_path])
    execute([code / 'run_matrix.py', '--project', project, '--run', run, '--manifest', final_path,
             '--inputs', run / 'final-inputs', '--phase', 'final', '--output', 'final-01'])
    final_summary = analyze(final_path, run / 'final-inputs', run / 'final-01')
    write(run / 'final-analysis.json', final_summary)
    write(run / 'confirmation-campaign-receipt.json', {'completed': True, 'records': records,
        'total_wall_seconds': time.perf_counter() - started, 'training_seconds': 0,
        'final_admitted': sum(row['admitted'] for row in final_summary['arms']), 'final_groups': len(final_summary['arms']),
        'scope': 'Execution complete; root must audit held-out results, variability and release before declaring matrix success.'})


if __name__ == '__main__':
    def interrupted(*_):
        raise KeyboardInterrupt('supervisor interruption')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except BaseException as exc:
        if '--run' in sys.argv:
            run = Path(sys.argv[sys.argv.index('--run') + 1])
            write(run / 'confirmation-campaign-error.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                  'at_utc': datetime.now(timezone.utc).isoformat(), 'training_seconds': 0})
        raise
    finally:
        subprocess.run(['tmux', '-L', 'binary-eagle-runtime', 'wait-for', '-S', 'qwen35-matrix-confirmation-20261101-complete'], check=False)
