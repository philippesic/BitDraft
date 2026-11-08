"""Reconcile completed updates and publish compact live/final receipts."""
import json
import math
from .common import *
from scripts.component_campaign.metrics import summarize, paired_ratio


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    training = json.load(open(OUT/'result.json')) if (OUT/'result.json').exists() else {}
    updates = [json.loads(line) for line in (OUT/'updates.jsonl').read_text().splitlines()] if (OUT/'updates.jsonl').exists() else []
    assert [x['update'] for x in updates] == list(range(1,len(updates)+1))
    assert all(math.isfinite(x[k]) for x in updates for k in ('loss','gradient_norm','seconds','cumulative_seconds'))
    assert all(x['seconds'] > 0 for x in updates)
    seconds = math.fsum(x['seconds'] for x in updates)
    assert not updates or abs(seconds-updates[-1]['cumulative_seconds']) < 1e-6
    durable_updates = training.get('updates',0)
    durable_seconds = math.fsum(x['seconds'] for x in updates[:durable_updates])
    assert abs(durable_seconds-training.get('actual_update_seconds',0)) < 1e-6
    audit = dict(status='PASS', completed_updates=len(updates),actual_update_seconds=seconds,
        checkpointed_updates=durable_updates, checkpointed_update_seconds=durable_seconds,
        uncheckpointed_completed_updates=len(updates)-durable_updates,
        supervised_positions=len(updates)*15,unique_windows_exposed=len({(x['sequence'],x['anchor']) for x in updates}),
        unique_labels_exposed=len({(x['sequence'],x['anchor']+j-1) for x in updates for j in range(1,16)}),
        update_log=pin(OUT/'updates.jsonl') if updates else None)
    save(REPORT/'audit.json',audit)
    panels = {}
    for point in POINTS:
        for kind in ('smoke','formal'):
            label = kind+'-'+str(point)
            path = R/label/'result.json'
            if not path.exists(): continue
            raw = json.load(open(path))
            assert raw['status'] == 'COMPLETE' and all(x['correct'] for x in raw['rows'])
            freeze = json.load(open(path.parent/'freeze.json'))
            assert freeze['warmup_policy']['measured_repetitions'] == (5 if kind == 'formal' else 1)
            panels[label] = dict(source=pin(path),freeze=pin(path.parent/'freeze.json'),arms=summarize(path))
            if kind == 'formal':
                panels[label]['vs_q4'] = paired_ratio(path,'HE4broad','q4')
                panels[label]['vs_target'] = paired_ratio(path,'HE4broad','target')
    workflow = json.load(open(R/'workflow-result.json')) if (R/'workflow-result.json').exists() else {}
    status = workflow.get('status',training.get('status','PREPARING'))
    final_verified = (R/'final-recovery.json').exists()
    complete = (training.get('status') == 'COMPLETE' and seconds >= BUDGET and final_verified
        and all(label in panels for label in ('formal-0','formal-28800','smoke-0','smoke-14400','smoke-28800')))
    if complete:
        assert len(updates) == durable_updates
        assert seconds-BUDGET < updates[-1]['seconds']
        assert [x['requested_update_seconds'] for x in training['readouts']] == POINTS
        status = 'COMPLETE'
    elif status == 'COMPLETE': status = 'FINAL_VALIDATION_PENDING'
    results = dict(status=status, recipe=json.load(open(R/'recipe.json')),training=training,audit=audit,panels=panels,
        init=json.load(open(OUT/'init-proof.json')) if (OUT/'init-proof.json').exists() else None,
        data=json.load(open(OUT/'data-proof.json')) if (OUT/'data-proof.json').exists() else None,
        optimizer=json.load(open(OUT/'optimizer-zero.json')) if (OUT/'optimizer-zero.json').exists() else None,
        qualification=json.load(open(OUT/'one-update-qualification.json')) if (OUT/'one-update-qualification.json').exists() else None,
        final_recovery=json.load(open(R/'final-recovery.json')) if final_verified else None,
        limits='Consumed DEV3 only; one source-initialized seed and training trajectory. Timing repetitions are not new prompts or training seeds. No heldout24/fresh24 opened. QAT CE/agreement are diagnostics; native throughput uses complete request wall time. External DFlash Q4_K_M remains the deployment incumbent; no promotion claimed.')
    save(REPORT/'results.json',results)
    doc = f'''# HE4broad source-initialized QAT, November 8, 2026

Status: **{status}**. Completed **{len(updates):,} updates** using **{seconds:.3f} / 28,800 actual update seconds**. The last recoverable checkpoint covers {durable_updates:,} updates. Setup, evaluation, serialization and native panels are outside this budget.

This new recipe starts from authenticated source BF16 DFlash tensors and target Qwen3.5-4B Q4_K_M tensors. No B0, HE4-4096 or overnight continuation parameters/Adam state are loaded. Exact source latent tensors, mean-absolute row scales, zero midpoints/log gains and HE4 initial clipping bounds are asserted before training. Existing saved JSON tensor digests establish distinct trainable scales from both B0 and HE4-4096. The private tied Q4_K vocabulary is installed from its authenticated target-derived matrix.

The architecture matches HE4: all 18 FFN projections W1; gate/up A4, down A8; Q4_0 attention/fusion; one private tied Q4_K head/embedding; F16 draft KV; F32 scales, midpoints, gains, clipping and norms. TRAIN asserts 40 sequences, 3,234 windows and 3,794 unique label positions, with the exact Csmallbroad/H0broad manifest. consumed_DEV is evaluation-only. Fresh Adam uses the unchanged optimizer defaults: signs 0.001; scale/midpoint/gain/clip/norm 0.00001; constant schedule, weight decay zero. Every group takes a real Adam step each window. Window shuffle seed 8101, initial cursor zero. Loss is the existing 15-position decay-weighted hard-target CE.

| Requested update seconds | Actual update seconds | Updates | DEV CE | DEV top-1 agreement |
| ---: | ---: | ---: | ---: | ---: |
'''
    for row in training.get('readouts',[]):
        doc += f"| {row['requested_update_seconds']} | {row['additional_update_seconds']:.3f} | {row['updates']} | {row['dev']['ce']:.6f} | {row['dev']['agreement']:.4%} |\n"
    doc += '\nNative panels use runtime04, consumed DEV3, depth 2, greedy sampling, context 4096 and F16 target/draft KV. Formal panels use two excluded warmups per server cell, five alternating repetitions, and 256-token output reserve. Smoke panels use one repetition, no warmups and 64 tokens. Training exits before each native panel.\n\n'
    for label, panel in panels.items():
        doc += f"### {label}\n\n| Arm | Acceptance | Complete-request tokens/s | Correct |\n| --- | ---: | ---: | --- |\n"
        for arm, values in panel['arms'].items():
            acceptance = '—' if values['acceptance'] is None else f"{values['acceptance']:.4%}"
            doc += f"| {arm} | {acceptance} | {values['request_tok_s']:.3f} | {values['correct']} |\n"
        doc += '\n'
    doc += f"Full checkpoints (including zero and every hourly state) are retained under `{DEST}`; exports, immutable source closure, commands, process identity and update log are under `{R}`. Storage is physically reserved before CUDA work; unused reservation files are released at shutdown. SIGTERM is honored between complete updates. An update exception preserves the last good checkpoint and is excluded from completed-update accounting.\n\n"
    if complete:
        doc += f"Final checkpoint: `{training['recovery']['archive']}`. Final export: `{training['readouts'][-1]['model']['path']}`. Full hashes and native-byte recovery proof are in [results](results.json).\n\n"
    doc += results['limits']+'\n\n[Results and hashes](results.json) · [Completed-update audit](audit.json) · [Manifest](manifest.json).\n'
    (REPORT/'report.md').write_text(doc)
    marker = '## HE4broad source-initialized QAT'
    status_path = P/'docs/STATUS.md'
    current = status_path.read_text()
    if marker in current:
        start = current.index(marker)
        end = current.find('\n## ',start+len(marker))
        current = current[:start]+(current[end+1:] if end >= 0 else '')
    block = f"{marker}\n\nNovember 8 user-authorized new recipe: **{status}**, {len(updates):,} completed updates, {seconds:.3f} / 28,800 actual update seconds. Source BF16 initialization; broader authenticated 40/3234/3794 TRAIN set; fresh Adam at unchanged default rates. Hourly recoverable states and native exports retained. [Report](../experiments/he4broad-20261108/report.md) · [Live progress](../runs/he4broad-20261108-01/candidate/live.json) · [Ownership and stop procedure](../experiments/he4broad-20261108/operations.md). External DFlash Q4_K_M remains the deployment incumbent; no promotion.\n\n"
    current = current.replace('# Current project status\n\n','# Current project status\n\n'+block,1)
    status_path.write_text(current)
    print('PUBLISHED',status,len(updates),seconds,flush=True)


if __name__ == '__main__': main()
