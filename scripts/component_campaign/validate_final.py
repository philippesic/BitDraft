"""Validate the completed formal packet, keeping distinct cohorts and timings separate."""
import collections
import json

from .common import R, pin, save


def main():
    decision = json.loads((R / 'final-recipe-decision.json').read_text())
    expected = {'target', 'q4', 'B0', *decision['controls'], *decision['candidates']}
    panels = {}
    common_runtime = None
    for name, prompts_per_cell in (
            ('final-dev', 12), ('final-host', 3), ('final-context-128', 3),
            ('final-context-1536', 3), ('final-context-3840', 3), ('final-heldout24', 24)):
        folder = R / 'measurements' / name
        result = json.loads((folder / 'result.json').read_text())
        freeze = json.loads((folder / 'freeze.json').read_text())
        if result['status'] != 'COMPLETE' or set(freeze['arms']) != expected:
            raise ValueError('Incomplete formal arm packet: ' + name)
        if freeze['warmup_policy']['per_server_cell'] != 2 or freeze['warmup_policy']['measured_repetitions'] != 5:
            raise ValueError('Formal repetitions/warmups differ: ' + name)
        runtime = {key: value['sha256'] for key, value in freeze['runtime'].items()}
        if common_runtime is None:
            common_runtime = runtime
        if runtime != common_runtime:
            raise ValueError('Formal runtime changed: ' + name)
        for arm, ref in decision['models'].items():
            if freeze['arms'][arm]['sha256'] != ref['sha256']:
                raise ValueError('Formal weights differ from locked recipe: ' + name + '/' + arm)
        rows = result['rows']
        measured = [row for row in rows if not row['warmup']]
        warmups = [row for row in rows if row['warmup']]
        expected_cells = {(arm, rep) for arm in expected for rep in range(5)}
        for subset, count in ((measured, prompts_per_cell), (warmups, 2)):
            cells = collections.Counter((row['arm'], row['rep']) for row in subset)
            if set(cells) != expected_cells or set(cells.values()) != {count}:
                raise ValueError('Missing or duplicate formal cells: ' + name)
        for row in rows:
            if not row['correct'] or bool(row['instrumented']) != (name == 'final-host'):
                raise ValueError('Wrong output/timing scope: ' + name)
            if row['arm'] == 'target':
                continue
            native = row['native']
            prefix = native['prefix_survival_counts']
            if not (0 <= native['accepted'] <= native['proposed'] and native['rounds'] > 0
                    and len(prefix) == 2 and 0 <= prefix[1] <= prefix[0] <= native['rounds']
                    and sum(prefix) == native['accepted']):
                raise ValueError('Invalid actual acceptance/prefix accounting: ' + name)
        for arm, rep in expected_cells:
            cell = folder / (f'{rep:02d}-' + arm + ('-traced' if name == 'final-host' else ''))
            if not json.loads((cell / 'release.json').read_text())['reaped']:
                raise ValueError('Native cell was not reaped: ' + str(cell))
        panels[name] = dict(result=pin(folder / 'result.json'), freeze=pin(folder / 'freeze.json'),
                            measured_requests=len(measured), warmups=len(warmups),
                            distinct_prompts=prompts_per_cell, repetitions=5,
                            instrumented=name == 'final-host')
    replay_file = R / 'measurements/final-replay/result.json'
    replay = json.loads(replay_file.read_text())
    if replay['status'] != 'COMPLETE' or replay['audit'] or replay['profiled']:
        raise ValueError('Final replay is not clean timing')
    replay_expected = {(arm, rep, domain) for arm in expected - {'target'}
                       for rep in range(5) for domain in ('code', 'prose', 'reasoning')}
    actual = [(record['arm'], record['rep'], record['domain']) for record in replay['records']]
    if set(actual) != replay_expected or len(actual) != len(replay_expected):
        raise ValueError('Incomplete clean replay cells')
    for record in replay['records']:
        measured = [row for row in record['result']['rows'] if not row['warmup']]
        if len(measured) != 3 or any(row['graph_count'] != 1 or row['graph_event_ms'] <= 0 for row in measured):
            raise ValueError('Clean replay region/context missing')
    trace_file = R / 'qualification/final-trace/result.json'
    if not json.loads(trace_file.read_text())['paired_trace_counters_equal']:
        raise ValueError('Trace instrumentation changed native proposal counts')
    recovery_file = R / 'qualification/frontier-recovery/result.json'
    recovery = json.loads(recovery_file.read_text())
    if {row['original_arm'] for row in recovery['records']} != set(decision['retain']):
        raise ValueError('Retained frontier coverage differs from decision')
    for record in recovery['records']:
        if record['gpu_resume']['status'] != 'ACTUAL_GPU_MASTER_ADAM_RNG_CURSOR_RESUME_UPDATE_PASS':
            raise ValueError('Actual retained GPU resume did not pass')
        if record['native']['sha256'] != decision['models'][record['original_arm']]['sha256']:
            raise ValueError('Evaluated native differs from retained full state')
    selection = pin(R / 'final-selection.json')
    heldout_freeze = json.loads((R / 'heldout24/selection-freeze.json').read_text())
    consumption = json.loads((R / 'heldout24/consumption.json').read_text())
    if heldout_freeze['selection'] != selection or consumption['selection'] != selection or len(consumption['records']) != 24:
        raise ValueError('Fresh heldout lock/consumption ancestry differs')
    save(R / 'qualification/final-protocol.json', dict(
        status='FULL_FORMAL_PACKET_AND_RETAINED_STATE_PROTOCOL_PASS', panels=panels,
        clean_replay=pin(replay_file), paired_trace=pin(trace_file),
        retained_state_qualification=pin(recovery_file), selection=selection,
        heldout_consumption=pin(R / 'heldout24/consumption.json'), source=pin(__file__),
        policy='Distinct native cohorts/contexts and instrumented timings remain separate. All measured cells, excluded warmups, greedy output checks and prefix totals verified. Prompt/process repetitions are not independent training seeds.'))
    print('FULL_FORMAL_PACKET_PROTOCOL_PASS', flush=True)


if __name__ == '__main__':
    main()
