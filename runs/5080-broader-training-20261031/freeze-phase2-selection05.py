"""Freeze valid fresh native choices before phase2 serving requests."""
from pathlib import Path
import datetime
import sys

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
sys.path.insert(0, str(R / 'source03/scripts'))
from training_system.io import read, pin, authenticate, atomic_json, digest
from training_system.train import continuation_contract

assert not (R / 'phase2-selection05.json').exists()
assert not (W / 'request-final02').exists() and not (W / 'request-final-config02.json').exists()
for name in ['fresh-qualify01', 'fresh-finish03']:
    state = read(W / 'supervisors' / name / 'state.json')
    assert state['status'] == 'finished' and state['exit_code'] == 0
old = read(W / 'supervisors/fresh-training01/state.json')
assert old['status'] == 'interrupted' and old['exit_code'] == -15
recovery = read(R / 'fresh-training-completion-recovery03.json')
assert recovery['optimizer_updates'] == recovery['TRAIN_seconds'] == 0 and recovery['job_unchanged']
assert recovery['completed_curve'] == pin(W / 'fresh-training01/state.json')
curve = read(W / 'fresh-training01/state.json')
assert curve['status'] == 'COMPLETE' and curve['native']['14400'] == recovery['native_result']
job = read(Path(curve['job']) / 'job.json')
assert job['owner'] is None and job['status'] != 'RUNNING'
manifest = job['spec']['manifest']
previous = read(R / 'final-selection.json')
assert previous['manifest'] == manifest
eligible = []
for target, ref in curve['native'].items():
    result = read(authenticate(ref))
    events = [e for e in curve['events'] if e['action'] == 'native_complete_independently_verified'
              and e['target'] == int(target)]
    assert len(events) == 1 and events[0]['metrics'] == result['metrics']
    assert events[0]['proof']['result'] == ref and events[0]['proof']['raw_files'] == 288
    producers = []
    for attempt in job['attempts']:
        f = Path(attempt['output']) / 'finish.json'
        if f.exists():
            finish = read(f)
            if finish.get('model') == result['config']['model'] and finish['status'] == 'EXPOSURE_COMPLETE':
                producers.append((pin(f), finish))
    assert len(producers) == 1 and producers[0][1]['local_updates'] == int(target)
    eligible.append({'model': result['config']['model'], 'native_result': ref,
                     'local_updates': int(target), 'producer_finish': producers[0][0],
                     'native_metrics': result['metrics']['development'], 'origin': 'long fresh milestone'})
pre = R / 'fresh-valid-prestop-receipt01.json'
if pre.exists():
    state = read(W / 'supervisors/fresh-prestop01/state.json')
    assert state['status'] == 'finished' and state['exit_code'] == 0
    receipt = read(pre)
    assert receipt['optimizer_updates'] == receipt['TRAIN_seconds'] == 0
    assert receipt['complete_state_unchanged'] and not receipt['guard_stopped_checkpoint_loaded']
    result = read(authenticate(receipt['native_result']))
    assert result['config']['model'] == receipt['model']
    eligible.append({'model': receipt['model'], 'native_result': receipt['native_result'],
                     'local_updates': receipt['local_updates'], 'producer_finish': receipt['zero_export_finish'],
                     'valid_pre_stop_receipt': pin(pre), 'native_metrics': result['metrics']['development'],
                     'origin': 'one predeclared valid fresh pre-guard endpoint; zero additional updates'})
if not eligible:
    assert job['status'] == 'GUARD_STOPPED'
    assert (R / 'fresh-prestop-qualified-reference01.json').exists()
    qualification = read(W / 'fresh-qualification01/result.json')
    record = qualification['records']['control']
    source = read(authenticate(record['qualification_proofs']['source']))
    assert record['adapter_state_key'] == digest(continuation_contract(job['spec'])) == digest(continuation_contract(source))
    assert record['state_exact'] and record['native_exact'] and record['resumed']['local_updates'] == 8
    ref = record['qualification_proofs']['native_result']
    result = read(authenticate(ref))
    assert result['config']['model'] == record['resumed']['model']
    eligible.append({'model': record['resumed']['model'], 'native_result': ref, 'local_updates': 8,
                     'native_metrics': result['metrics']['development'],
                     'origin': 'predeclared qualified8 reference: long run guarded before an eligible positive milestone; not a long milestone',
                     'reference_plan': pin(R / 'fresh-prestop-qualified-reference01.json')})
for item in eligible:
    result = read(item['native_result']['path'])
    authenticate(item['model'])
    assert result['config']['manifest'] == manifest and result['windows'] == 288
    assert item['native_metrics']['windows'] == 192 and item['native_metrics']['groups'] == 48
best = max(eligible, key=lambda x: (x['native_metrics']['prefix5_sum'],
                                   x['native_metrics']['first_error_margin_sum'], -x['local_updates']))
design = read(R / 'phase2-serving-design02.json')
assert design['phase2_requests_not_started'] and design['fresh14k_native_not_yet_observed']
early = design['diagnostic_arms']['fresh_early96']
assert early['local_updates'] == 96 and best['local_updates'] != 96
assert best['model']['sha256'] != early['model']['sha256']
assert any(item['model'] == early['model'] and item['native_result'] == early['native_result'] for item in eligible)
authenticate(early['model'])
authenticate(early['native_result'])
prior = previous['candidates']['eal_best']
authenticate(prior['model'])
authenticate(prior['native_result'])
selection = {'selection_basis': 'frozen native development; no request-based checkpoint choice',
             'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'manifest': manifest, 'candidates': {'previous_eal_best': prior, 'fresh_best': best, 'fresh_early96': early},
             'eligible_endpoints': {'fresh': eligible}, 'plan': pin(R / 'fresh-restart-plan04.json'),
             'closed_fresh_curve': pin(W / 'fresh-training01/state.json'),
             'previous_selection': pin(R / 'final-selection.json'),
             'prospective_serving_design': pin(R / 'phase2-serving-design02.json'),
             'native_only_completion': pin(R / 'fresh-training-completion-recovery03.json'),
             'limitations': 'Repeated development selection, all historical groups parent-seen; no pristine unseen claim. Fresh origin/update count disclosed.'}
atomic_json(R / 'phase2-selection05.json', selection)
print('PHASE2_SELECTION_FROZEN', best['local_updates'], best['native_metrics']['prefix5_sum'], flush=True)
