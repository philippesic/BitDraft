"""Publish compact HE4 continuation evidence using the campaign metric definitions."""
import csv
import json
from scripts.component_campaign.common import P, pin, save
from scripts.component_campaign.metrics import summarize, paired_ratio
from scripts.he4_continuation.train import R, OUT

REPORT = P / 'experiments/he4-continuation-20261107'


def main():
    training = json.load(open(OUT / 'result.json'))
    recovery = json.load(open(OUT / 'final-recovery/result.json'))
    assert training['status'] == 'COMPLETE'
    assert recovery['status'] == 'FULL_FINAL_STATE_NATIVE_BYTE_IDENTITY_AND_GPU_RESUME_PASS'
    updates = [json.loads(line) for line in (OUT / 'updates.jsonl').read_text().splitlines()]
    assert len(updates) == training['additional_updates']
    assert updates[0]['update'] == 4097
    assert [x['update'] for x in updates] == list(range(4097, training['final_progress']['updates'] + 1))
    assert abs(sum(x['seconds'] for x in updates) - training['actual_update_seconds']) < 1e-6
    assert training['actual_update_seconds'] >= 21600
    assert training['one_update_overshoot_seconds'] < training['last_update_seconds']
    panels = {}
    for label in ['native-final', 'native-plus1800', 'native-plus7200', 'native-plus14400']:
        path = R / label / 'result.json'
        native = json.load(open(path))
        assert native['status'] == 'COMPLETE' and all(x['correct'] for x in native['rows'])
        freeze = json.load(open(path.parent / 'freeze.json'))
        assert freeze['warmup_policy']['measured_repetitions'] == (5 if label == 'native-final' else 1)
        panels[label] = dict(source=pin(path), freeze=pin(path.parent / 'freeze.json'),
                             arms=summarize(path), cohort=freeze['cohort'])
    path = R / 'native-final/result.json'
    parent_ratio = paired_ratio(path, 'after', 'parent')
    q4_ratio = paired_ratio(path, 'after', 'q4')
    arms = panels['native-final']['arms']
    gain = arms['after']['acceptance'] - arms['parent']['acceptance']
    lower, upper = parent_ratio['paired_95_interval']
    finding = ('Useful native headroom appears in this consumed-DEV panel.' if gain > 0 and lower > 1 else
               'Acceptance improves, but useful request-throughput headroom remains unresolved.' if gain > 0 else
               'This continuation does not demonstrate useful native headroom on this consumed-DEV panel.')
    limitations = ('Three previously consumed DEV prompts, one training trajectory, and five alternating process repetitions; '
                   'timing repetitions are not independent prompts or training seeds. No new heldout, context matrix, or other target cell was opened. '
                   'Intermediate 64-token panels are one-repetition smoke diagnostics and cannot be pooled with the final 256-token comparison. '
                   'Inherited QAT/native arithmetic parity limitations remain; native complete requests determine throughput. '
                   'External DFlash Q4_K_M was rerun in the matched final panel and remains the deployment incumbent. '
                   'No mixed-weight experiment or deployment promotion was performed.')
    result = dict(status='COMPLETE', finding=finding, training=training, final_recovery=recovery,
                  panels=panels, after_vs_parent=parent_ratio, after_vs_q4=q4_ratio,
                  acceptance_change= gain, limitations=limitations,
                  parent_resume_proof=json.load(open(OUT / 'zero-resume-proof.json')),
                  trajectory_log=pin(OUT / 'updates.jsonl'), source=pin(__file__))
    save(REPORT / 'results.json', result)
    with (REPORT / 'native.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['panel','arm','acceptance','accepted','proposed','rounds','emitted_per_round','request_tok_s','responses','correct'])
        writer.writeheader()
        for label, panel in panels.items():
            for arm, values in panel['arms'].items():
                writer.writerow(dict(panel=label, arm=arm, **{key:values[key] for key in writer.fieldnames if key not in ('panel','arm')}))
    with (REPORT / 'trajectory.csv').open('w') as stream:
        writer = csv.writer(stream);writer.writerow(['additional_update_seconds','updates','dev_ce','dev_agreement','native_path','native_sha256'])
        for row in training['readouts']:
            writer.writerow([row['additional_update_seconds'],row['progress']['updates'],row['dev']['ce'],row['dev']['agreement'],row['model']['path'],row['model']['sha256']])
    document = f'''# HE4 overnight continuation, November 7–8, 2026

{finding}

The unchanged HE4 trajectory used **{training['actual_update_seconds']:.3f} synchronized update seconds**, advancing from **4096 to {training['final_progress']['updates']} updates** ({training['additional_updates']} additional updates). Acceptance changed by **{gain*100:+.2f} percentage points**; complete-request throughput relative to HE4-4096 was **{parent_ratio['ratio']:.4f}×**, with paired timing interval **[{lower:.4f}, {upper:.4f}]**. Relative to external DFlash Q4_K_M it was **{q4_ratio['ratio']:.4f}×**. These are preliminary consumed-DEV measurements.

| Arm | Accepted / proposed | Acceptance | Request tokens/s |
| --- | ---: | ---: | ---: |
'''
    for name, values in arms.items():
        acceptance = '—' if values['acceptance'] is None else f"{values['acceptance']*100:.2f}%"
        document += f"| {name} | {values['accepted']} / {values['proposed']} | {acceptance} | {values['request_tok_s']:.2f} |\n"
    document += f'''
The final panel uses the existing component native runner, runtime04, Qwen3.5-4B Q4_K_M target, F16 target and draft K/V, context 4096, greedy depth 2, 256-token output reserve, two excluded warmups per server cell, and five alternating measured repetitions over code/reasoning/prose DEV prompts. All native requests match target-only outputs. Acceptance is total accepted/proposed; request speed is total output tokens / complete request wall seconds, including prompt processing. Prefix and decode counters remain in the raw native receipts.

The frozen recipe remains W1 across all 18 FFN matrices, A4 gate/up and A8 down inputs, Q4_0 attention/fusion, one tied private Q4_K vocabulary matrix, F16 draft K/V and F32 compensation/norm storage. The parent's 24 TRAIN sequences, 1,426 windows, 1,762 unique label positions, saved order/cursor/RNG, Adam moments/groups and constant learning rates were preserved. Repeated exposure is not new data.

## Recovery and resources

Exact parent archive/raw/native hashes were verified before training. Native byte recovery and an actual scratch GPU update from 4096 to 4097 passed before the update budget. Qualification did not mutate parent Adam counters. The final full checkpoint was decoded, re-exported to identical native bytes, and passed an actual next GPU update. Qualification time is separate from the 21,600 update-second budget.

Peak Torch allocated/reserved memory: {training['peak_allocated_bytes']/2**30:.3f} / {training['peak_reserved_bytes']/2**30:.3f} GiB; peak host RSS: {training['peak_host_rss_bytes']/2**30:.3f} GiB. Setup/qualification: {training['setup_and_qualification_seconds']:.3f}s; checkpoints: {training['all_checkpoint_seconds']:.3f}s; trainer process wall: {training['process_wall_seconds']:.3f}s. One-update budget overshoot: {training['one_update_overshoot_seconds']:.6f}s. Readout/export costs are recorded separately at each point. All intermediate checkpoints and native exports are retained in the new run directory.

Final checkpoint: `{training['recovery']['archive']}`. Final native export: `{training['readouts'][-1]['model']['path']}`. Full hashes and ancestry are in [machine-readable results](results.json); run accounting, source closure, owner identities, commands, per-update logs and raw native receipts are under `{R}`. The original parent and frontier registry were preserved.

## Interpretation limits

{limitations}

[Native table](native.csv) · [DEV trajectory](trajectory.csv) · [Protocol](../../docs/CANDIDATE_BENCHMARKING.md) · [Run plan](manifest.json).
'''
    (REPORT / 'report.md').write_text(document)
    # Standalone exportable chart; the agent can display its HTML on handoff.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['parent', 'after', 'q4']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    axes[0].bar(labels, [arms[x]['acceptance']*100 for x in labels], color=['#53708d','#1f8669','#c57f28'])
    axes[0].set_ylabel('Accepted / proposed (%)')
    axes[1].bar(labels, [arms[x]['request_tok_s'] for x in labels], color=['#53708d','#1f8669','#c57f28'])
    axes[1].set_ylabel('Complete request tokens/s')
    fig.suptitle('HE4 unchanged continuation · consumed DEV3 · 5 timing repetitions')
    fig.savefig(REPORT / 'comparison.png', dpi=160)
    fig.savefig(REPORT / 'comparison.svg')
    plt.close(fig)
    svg = (REPORT / 'comparison.svg').read_text()
    (REPORT / 'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>HE4 continuation</title>' + svg + '<p>' + limitations + '</p>')
    print('HE4_REPORT_COMPLETE', finding, flush=True)


if __name__ == '__main__':
    main()
