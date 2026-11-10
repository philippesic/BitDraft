"""Compact live/final results; keep raw data and full state outside Git."""
import json
import math
from .common import *
from .clock import UPDATE_BUDGET, UPDATE_POINTS, POLICY, panel_label
from scripts.component_campaign.metrics import summarize, paired_ratio


def main():
    training = json.load(open(OUT/'result.json')) if (OUT/'result.json').exists() else {}
    panels = {}
    historical = [('smoke-' if point==60 else 'formal-')+str(point) for point in POINTS]
    labels = historical + [panel_label(point) for point in UPDATE_POINTS]
    for label in labels:
        path = R/label/'result.json'
        if not path.exists(): continue
        raw = json.load(open(path))
        assert raw['status']=='COMPLETE' and all(x['correct'] for x in raw['rows'])
        panels[label] = dict(source=pin(path),freeze=pin(path.parent/'freeze.json'),arms=summarize(path))
        if label!='smoke-60':
            panels[label]['vs_q4'] = paired_ratio(path,'HE4broad','q4')
            panels[label]['vs_target'] = paired_ratio(path,'HE4broad','target')
    workflow = json.load(open(R/'workflow-result.json')) if (R/'workflow-result.json').exists() else {}
    status = workflow.get('status',training.get('status','PREPARING'))
    expected = {panel_label(point) for point in UPDATE_POINTS}
    expected.update(('smoke-' if row['label']=='60' else 'formal-')+row['label']
                    for row in training.get('readouts',[]) if row['label'].isdecimal())
    if status=='COMPLETE' and (not expected.issubset(panels) or not (R/'final-recovery.json').exists()): status='FINAL_VALIDATION_PENDING'
    results = dict(status=status,recipe=json.load(open(R/'recipe.json')),training_policy=POLICY,training=training,panels=panels,
        data=json.load(open(OUT/'data-proof.json')) if (OUT/'data-proof.json').exists() else None,
        final_recovery=json.load(open(R/'final-recovery.json')) if (R/'final-recovery.json').exists() else None,
        limitations='Consumed DEV3, one training trajectory. Timing repetitions are not fresh prompts/seeds. No heldout opened. External Q4 remains deployment incumbent; no promotion.')
    save(REPORT/'results.json',results)
    doc = '# Fresh HE4broad on the combined 2080 Ti captures\n\n'
    doc += f"Status: **{status}**. {training.get('updates',0):,} recoverable updates; {training.get('actual_update_seconds',0):.3f} synchronized update seconds, {training.get('training_seconds',0):.3f} training wall seconds (including {training.get('data_io_seconds',0):.3f} data I/O seconds). Philip's resumed allocation is **{UPDATE_BUDGET:,} total synchronized update seconds (8 hours)**, including the existing 2873.748 seconds. Rejected update time also counts. There is no training-wall cap; data I/O, setup, checkpoints, export and native eval do not consume the update allocation and remain separately reported.\n\n"
    doc += 'Fresh base-drafter HE4 quantization, fresh Adam, combined 15,647 sequences and 1,568,435 windows. Global seeded window shuffle; exact own-state resume. All hourly native panels use depth 2, greedy, F16 KV, context 4096, two warmups and five repetitions per cell with 256-token output reserve. The extra 60-second diagnostic uses one repetition and 64 tokens. Network model transfer is excluded from native request timings.\n\n'
    doc += 'Historical numeric points are training-wall seconds; new `update-*` points are cumulative synchronized update seconds. The original recipe/config is retained verbatim for exact state compatibility; the explicit continuation policy supersedes its old clock/budget.\n\n'
    doc += '| Readout (clock in label) | Actual update seconds | Updates | DEV CE | DEV agreement |\n| --- | ---: | ---: | ---: | ---: |\n'
    for row in training.get('readouts',[]):
        doc += f"| {row['label']} | {row['additional_update_seconds']:.3f} | {row['updates']} | {row['dev']['ce']:.6f} | {row['dev']['agreement']:.4%} |\n"
    doc += '\n| Panel | Arm | Draft acceptance | Request tokens/s | Correct |\n| --- | --- | ---: | ---: | --- |\n'
    for label,panel in panels.items():
        for arm,v in panel['arms'].items():
            acceptance = '—' if v['acceptance'] is None else f"{v['acceptance']:.4%}"
            doc += f"| {label} | {arm} | {acceptance} | {v['request_tok_s']:.3f} | {v['correct']} |\n"
    doc += '\n'+results['limitations']+'\n\n[Results and pins](results.json) · [Operations](operations.md) · [Local-data migration](localization.md).\n'
    (REPORT/'report.md').write_text(doc)
    print('PUBLISHED',status,training.get('updates',0),flush=True)


if __name__=='__main__': main()
