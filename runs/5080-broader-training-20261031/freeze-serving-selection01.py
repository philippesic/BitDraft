"""Freeze phase1 choices before reading any phase1 request result."""
from pathlib import Path
import datetime
import sys

P = Path('/home/philip/binary-eagle-decoding')
R = P / 'runs/5080-broader-training-20261031'
W = Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031')
sys.path.insert(0, str(R / 'source02/scripts'))
from training_system.io import read, pin, authenticate, atomic_json

assert not (R / 'final-selection.json').exists()
assert not (W / 'request-final01/summary.json').exists()
assert not (W / 'request-final01/requests.jsonl').exists()
state = read(W / 'supervisors/main07/state.json')
assert state['status'] == 'finished' and state['exit_code'] == 0
controller = read(W / 'main-controller02/controller.json')
assert controller['owner'] is None and controller['status'] in ['PAUSED', 'COMPLETE', 'BLOCKED']
manifest = read(W / 'base-spec04.json')['manifest']
assert manifest['sha256'] == '0d8cd5e914fd0c9c92ace025c61b9a904a98f42b881252eb961e30f5754e4475'
authenticate(manifest)
selection = {'selection_basis': 'frozen native development; no request-based checkpoint choice',
             'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'manifest': manifest, 'candidates': {}, 'eligible_endpoints': {},
             'analysis_plan': pin(R / 'native-analysis-plan01.json'),
             'contingency_plan': pin(R / 'native-analysis-addendum02.json'),
             'phase_order': pin(R / 'serving-gate-plan03.json'),
             'limitations': 'Parent-seen repeated development selection; no untouched release or matched32400 diversity causal claim.'}
for objective, name in [('control', 'eal_best'), ('dpal', 'dpal_best')]:
    candidate = controller['candidates'][objective]
    job = read(Path(candidate['job']) / 'job.json')
    valid = []
    for ref in candidate['native'].values():
        native = read(authenticate(ref))
        model = native['config']['model']
        authenticate(model)
        producers = []
        for attempt in job['attempts']:
            path = Path(attempt['output']) / 'finish.json'
            if path.exists():
                finish = read(path)
                if finish.get('model') == model and finish['status'] in ['PAUSED', 'EXPOSURE_COMPLETE']:
                    producers.append((path, finish))
        assert len(producers) == 1
        path, finish = producers[0]
        valid.append({'model': model, 'native_result': ref, 'producer_finish': pin(path),
                      'local_updates': finish['local_updates'], 'native_metrics': native['metrics']['development']})
    receipt = R / ('valid-prestop-' + objective + '-receipt01.json')
    if receipt.exists():
        saved = read(receipt)
        assert saved['objective'] == objective and saved['optimizer_updates'] == saved['TRAIN_seconds'] == 0
        assert saved['complete_state_unchanged'] and not saved['guard_stopped_checkpoint_loaded']
        native = read(authenticate(saved['native_result']))
        finish = read(authenticate(saved['zero_export_finish']))
        assert finish['status'] == 'PAUSED' and finish['model'] == saved['model'] == native['config']['model']
        authenticate(saved['model'])
        valid.append({'model': saved['model'], 'native_result': saved['native_result'],
                      'producer_finish': saved['zero_export_finish'], 'valid_pre_stop_receipt': pin(receipt),
                      'local_updates': saved['local_updates'], 'native_metrics': native['metrics']['development']})
    assert valid
    for item in valid:
        result = read(item['native_result']['path'])
        assert result['config']['manifest'] == manifest and result['windows'] == 288
        assert item['native_metrics']['windows'] == 192 and item['native_metrics']['groups'] == 48
    best = max(valid, key=lambda item: (item['native_metrics']['prefix5_sum'],
                                       item['native_metrics']['first_error_margin_sum'],
                                       -item['local_updates']))
    selection['eligible_endpoints'][objective] = valid
    selection['candidates'][name] = best
narrow = pin(W / 'narrow32400-native01/result.json')
result = read(authenticate(narrow))
assert result['config']['manifest'] == manifest and result['windows'] == 288
authenticate(result['config']['model'])
selection['candidates']['narrow_32400'] = {'model': result['config']['model'], 'native_result': narrow,
                                         'role': 'historical narrower-data practical reference; exposure differs from selected broader endpoints'}
atomic_json(R / 'final-selection.json', selection)
print({name: {'updates': item.get('local_updates', 32400),
              'development_prefix': item.get('native_metrics', {}).get('prefix5_sum')}
       for name, item in selection['candidates'].items()}, flush=True)
