"""Compact live/final results; keep raw data and full state outside Git."""
import json
import math
from .common import *
from scripts.component_campaign.metrics import summarize, paired_ratio


def main():
    training = json.load(open(OUT/'result.json')) if (OUT/'result.json').exists() else {}
    panels = {}
    for point in POINTS:
        label = ('smoke-' if point==60 else 'formal-')+str(point)
        path = R/label/'result.json'
        if not path.exists(): continue
        raw = json.load(open(path))
        assert raw['status']=='COMPLETE' and all(x['correct'] for x in raw['rows'])
        panels[label] = dict(source=pin(path),freeze=pin(path.parent/'freeze.json'),arms=summarize(path))
        if point!=60:
            panels[label]['vs_q4'] = paired_ratio(path,'HE4broad','q4')
            panels[label]['vs_target'] = paired_ratio(path,'HE4broad','target')
    workflow = json.load(open(R/'workflow-result.json')) if (R/'workflow-result.json').exists() else {}
    status = workflow.get('status',training.get('status','PREPARING'))
    if status=='COMPLETE' and (len(panels)!=len(POINTS) or not (R/'final-recovery.json').exists()): status='FINAL_VALIDATION_PENDING'
    results = dict(status=status,recipe=json.load(open(R/'recipe.json')),training=training,panels=panels,
        data=json.load(open(OUT/'data-proof.json')) if (OUT/'data-proof.json').exists() else None,
        final_recovery=json.load(open(R/'final-recovery.json')) if (R/'final-recovery.json').exists() else None,
        limitations='Consumed DEV3, one training trajectory. Timing repetitions are not fresh prompts/seeds. No heldout opened. External Q4 remains deployment incumbent; no promotion.')
    save(REPORT/'results.json',results)
    doc = '# Fresh HE4broad on the combined 2080 Ti captures\n\n'
    doc += f"Status: **{status}**. {training.get('updates',0):,} recoverable updates; {training.get('actual_update_seconds',0):.3f} synchronized update seconds, {training.get('training_seconds',0):.3f} training wall seconds (including {training.get('data_io_seconds',0):.3f} data I/O seconds). Cap: 43,200 training wall seconds and 43,200 actual update seconds. Setup, checkpoints, export and native eval are separate.\n\n"
    doc += 'Fresh base-drafter HE4 quantization, fresh Adam, combined 15,647 sequences and 1,568,435 windows. Global seeded window shuffle; exact own-state resume. All hourly native panels use depth 2, greedy, F16 KV, context 4096, two warmups and five repetitions per cell with 256-token output reserve. The extra 60-second diagnostic uses one repetition and 64 tokens. Network model transfer is excluded from native request timings.\n\n'
    doc += '| Training wall point | Actual update seconds | Updates | DEV CE | DEV agreement |\n| ---: | ---: | ---: | ---: | ---: |\n'
    for row in training.get('readouts',[]):
        doc += f"| {row['label']} | {row['additional_update_seconds']:.3f} | {row['updates']} | {row['dev']['ce']:.6f} | {row['dev']['agreement']:.4%} |\n"
    doc += '\n| Panel | Arm | Draft acceptance | Request tokens/s | Correct |\n| --- | --- | ---: | ---: | --- |\n'
    for label,panel in panels.items():
        for arm,v in panel['arms'].items():
            acceptance = '—' if v['acceptance'] is None else f"{v['acceptance']:.4%}"
            doc += f"| {label} | {arm} | {acceptance} | {v['request_tok_s']:.3f} | {v['correct']} |\n"
    doc += '\n'+results['limitations']+'\n\n[Results and pins](results.json) · [Operations](operations.md).\n'
    (REPORT/'report.md').write_text(doc)
    print('PUBLISHED',status,training.get('updates',0),flush=True)


if __name__=='__main__': main()
