"""Independently reconcile the completed trajectory and native request counters."""
import json
import math
from collections import Counter
from pathlib import Path
from scripts.component_campaign.common import P, pin, save
from scripts.he4_continuation.train import R, OUT
from scripts.he4_continuation.publish import REPORT


def main():
    workflow = json.load(open(R / 'workflow-result.json'))
    assert workflow['status'] == 'COMPLETE'
    results = json.load(open(REPORT / 'results.json'))
    assert pin(workflow['published']['path']) == workflow['published']
    training = json.load(open(OUT / 'result.json'))
    updates = [json.loads(line) for line in (OUT / 'updates.jsonl').read_text().splitlines()]
    assert len(updates) == training['additional_updates']
    assert [x['update'] for x in updates] == list(range(4097, training['final_progress']['updates'] + 1))
    assert all(math.isfinite(x[key]) for x in updates for key in ('loss','gradient_norm','seconds','cumulative_seconds'))
    assert all(x['seconds'] > 0 for x in updates)
    seconds = math.fsum(x['seconds'] for x in updates)
    assert abs(seconds - training['actual_update_seconds']) < 1e-6
    assert seconds >= 21600 and seconds - 21600 < updates[-1]['seconds']
    assert training['final_progress']['cursor'] - training['starting_progress']['cursor'] == len(updates)
    assert training['final_progress']['supervised_tokens'] - training['starting_progress']['supervised_tokens'] == 15 * len(updates)
    probe = json.load(open(OUT / 'zero-resume-proof.json'))['qualification']
    first = updates[0]
    crosscheck = dict(update=first['update'], same_loss=first['loss'] == probe['loss'],
                      same_gradient_norm=first['gradient_norm'] == probe['gradient_norm'],
                      same_capture=first['sequence'] == probe['next_capture'], same_anchor=first['anchor'] == probe['next_anchor'])
    assert all(v for k,v in crosscheck.items() if k != 'update')
    parent = results['training']['parent']
    assert pin(parent['path']) == parent
    recoveries = []
    for path in sorted((R / 'checkpoints').glob('*.json')):
        receipt = json.load(open(path))
        actual = pin(receipt['archive'])
        assert actual['sha256'] == receipt['archive_sha256'] and actual['bytes'] == receipt['archive_bytes']
        assert receipt['lossless_verified']
        recoveries.append(receipt)
    assert len(recoveries) == 8
    for readout in training['readouts']:
        assert pin(readout['model']['path']) == readout['model']
    panels = {}
    for label, published in results['panels'].items():
        path = R / label / 'result.json'
        assert pin(path) == published['source']
        raw = json.load(open(path))
        measured = [x for x in raw['rows'] if not x['warmup']]
        arms = published['arms']
        repetitions = 5 if label == 'native-final' else 1
        assert len(measured) == repetitions * 3 * len(arms)
        assert all(x['correct'] and not x['instrumented'] and not x['profiled'] for x in raw['rows'])
        assert len({(x['arm'],x['rep'],x['prompt']) for x in measured}) == len(measured)
        checks = {}
        for arm, summary in arms.items():
            rows = [x for x in measured if x['arm'] == arm]
            assert len(rows) == repetitions * 3
            assert Counter(x['rep'] for x in rows) == Counter({i:3 for i in range(repetitions)})
            tokens = sum(x['tokens'] for x in rows)
            wall = math.fsum(x['wall_seconds'] for x in rows)
            assert abs(tokens / wall - summary['request_tok_s']) < 1e-9
            accepted = sum(x['native'].get('accepted',0) for x in rows)
            proposed = sum(x['native'].get('proposed',0) for x in rows)
            rounds = sum(x['native'].get('rounds',0) for x in rows)
            assert (accepted,proposed,rounds) == (summary['accepted'],summary['proposed'],summary['rounds'])
            if arm != 'target':
                assert all(x['timings']['draft_n'] == x['native']['proposed'] and
                           x['timings']['draft_n_accepted'] == x['native']['accepted'] for x in rows)
                assert proposed > 0 and abs(accepted/proposed - summary['acceptance']) < 1e-12
            decode_tokens = sum(x['timings']['predicted_n'] for x in rows)
            decode_ms = math.fsum(x['timings']['predicted_ms'] for x in rows)
            prefix = [sum(x['native'].get('prefix_survival_counts',[0,0])[i] for x in rows) for i in (0,1)]
            assert arm == 'target' or all(0 <= prefix[i] <= rounds for i in (0,1))
            samples = []
            for rep in range(repetitions):
                cell = R / label / f'{rep:02d}-{arm}'
                assert json.load(open(cell / 'release.json'))['reaped']
                samples += json.load(open(cell / 'memory.json'))
            checks[arm] = dict(responses=len(rows), accepted=accepted, proposed=proposed,
                acceptance=accepted/proposed if proposed else None, verification_rounds=rounds,
                prefix_survival_counts=prefix, prefix_method='Sum of existing per-request rounded survival counts; unchanged protocol.',
                emitted_tokens_per_round=tokens/rounds if rounds else None,
                server_decode_tok_s=1000*decode_tokens/decode_ms,
                complete_request_tok_s=tokens/wall,
                sampled_combined_device_peak_bytes=max((x['MiB']*2**20 for x in samples),default=None),
                sampled_host_peak_rss_bytes=max((x.get('VmHWM',0) for x in samples),default=None))
        panels[label] = checks
    audit = dict(status='INDEPENDENT_UPDATE_NATIVE_COUNTER_AND_ARTIFACT_AUDIT_PASS', source=pin(__file__),
        original_results=pin(REPORT / 'results.json'), workflow=pin(R / 'workflow-result.json'),
        additional_updates=len(updates), actual_update_seconds=seconds, main_resume_crosscheck=crosscheck,
        unchanged_parent=parent, verified_new_recovery_archives=recoveries, panels=panels,
        limitations='Native device memory is sampled total usage including desktop and target; Torch allocator peak is separately in the training result. Server decode time excludes prompt processing; complete request wall includes it. Repeated greedy responses are correlated.')
    save(REPORT / 'audit.json', audit)
    print('HE4_INDEPENDENT_AUDIT_PASS', len(updates), seconds, flush=True)


if __name__ == '__main__':
    main()
