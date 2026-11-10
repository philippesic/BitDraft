"""CPU-only audit of data exposure, resume controls and existing native panels."""
import collections
import copy
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from .common import P, R, OUT, REPORT, COMPONENT, pin, save
from scripts.dflash_baseline.native_counters import reconcile
from scripts.component_campaign.metrics import summarize


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    recipe=json.load(open(R/'recipe.json'));index=json.load(open(recipe['config']['data']['path']))
    assert pin(recipe['config']['data']['path'])==recipe['config']['data']
    rows=index['rows'];windows=[(i,a) for i,row in enumerate(rows) for a in row['anchors']]
    order=list(range(len(windows)));random.Random(8101).shuffle(order)
    sequences={str(Path(row['receipt']['metadata']['path']).parent):i for i,row in enumerate(rows)}
    ids=collections.Counter(row['id'] for row in rows)
    groups=collections.Counter(row['receipt']['group_id'] for row in rows)
    prompt_hashes=collections.Counter(digest(row['metadata']['input_tokens'][:row['metadata']['prompt_tokens']]) for row in rows)
    causal_links=0
    for row in rows:
        m=row['metadata'];assert row['anchors']==m['logit_positions'][:-14]
        for p,label in zip(m['logit_positions'],m['greedy_ids'],strict=True):
            if p+1<len(m['input_tokens']):
                assert m['input_tokens'][p+1]==label
                causal_links+=1
    prior=json.load(open(COMPONENT/'broader-train-capture/manifest.json'))
    dev=[json.load(open(x['metadata']['path'])) for x in prior['records'] if x['split']=='consumed_DEV']
    dev_hashes={digest(x['input_tokens'][:x['prompt_tokens']]) for x in dev}
    assert not (dev_hashes & set(prompt_hashes))
    updates=[]
    with (OUT/'updates.jsonl').open() as stream:
        for line in stream:
            try: row=json.loads(line)
            except json.JSONDecodeError:
                assert not stream.read();break
            updates.append(row)
    for i,row in enumerate(updates):
        assert row['update']==i+1
        assert (sequences[row['sequence']],row['anchor'])==windows[order[i%len(order)]]
        assert all(math.isfinite(row[k]) for k in ('loss','gradient_norm','seconds','training_seconds','data_io_seconds'))
    points=json.load(open(OUT/'readouts.json'))
    exposures=[];used_sequences=set();used_windows=set();used_labels=set();cursor=0;last_update=0;last_wall=0.
    for point in points:
        while cursor<point['updates']:
            row=updates[cursor];used_sequences.add(row['sequence']);used_windows.add((row['sequence'],row['anchor']))
            used_labels.update((row['sequence'],row['anchor']+j) for j in range(15));cursor+=1
        segment=updates[last_update:point['updates']]
        steady=segment[1:] if len(segment)>1 else segment
        exposures.append(dict(label=point['label'],updates=point['updates'],training_wall_seconds=point['training_seconds'],
            actual_update_seconds=point['additional_update_seconds'],distinct_sequences=len(used_sequences),
            distinct_windows=len(used_windows),distinct_label_positions=len(used_labels),
            pool_window_fraction=len(used_windows)/len(windows),label_fraction=len(used_labels)/index['unique_labels_in_windows'],
            update_seconds_median_excluding_segment_first=statistics.median(x['seconds'] for x in steady) if steady else None,
            update_seconds_mean=statistics.mean(x['seconds'] for x in segment) if segment else None,
            data_io_seconds_mean=statistics.mean(x['data_io_seconds'] for x in segment) if segment else None,
            training_loss_median=statistics.median(x['loss'] for x in segment) if segment else None,
            gradient_norm_median=statistics.median(x['gradient_norm'] for x in segment) if segment else None,
            dev_ce=point['dev']['ce'],dev_agreement=point['dev']['agreement'],
            dev_positions=point['dev']['positions']))
        last_update=point['updates'];last_wall=point['training_seconds']
    resumes=[]
    expected_rates=dict(sign=.001,scale=.00001,midpoint=.00001,gain=.00001,clip=.00001,norm=.00001)
    for path in sorted(OUT.glob('resume-*.json')):
        row=json.load(open(path));assert row['status']=='PASS' and row['parameter_identity']
        assert {g['name']:g['lr'] for g in row['optimizer_groups']}==expected_rates
        resumes.append(dict(receipt=pin(path),restored_updates=row['updates'],rates_unchanged=True))
    assert recipe['config']['schedule']=='constant'
    panels=[];baseline=None;prompt_source=None;signatures={};cells_checked=0;responses_checked=0
    immutable_keys=('source','counter_parser','runtime','target','baseline_registry','build','cohort_source',
                    'precision','draft_cache_types','synthetic_context_positions','warmup_policy','sampling')
    for point in points:
        if point['label']=='60':continue
        panel=R/('formal-'+point['label']);path=panel/'result.json'
        if not path.exists():continue
        raw=json.load(open(path));freeze=json.load(open(panel/'freeze.json'));prompts=json.load(open(panel/'prompts.json'))
        assert raw['status']=='COMPLETE' and len(raw['rows'])==75
        immutable={k:freeze[k] for k in immutable_keys};immutable['q4']=freeze['arms']['q4']
        if baseline is None:baseline=immutable;prompt_source=prompts
        assert immutable==baseline and prompts==prompt_source
        assert freeze['arms']['HE4broad']['sha256']==point['model']['sha256']
        measured=[x for x in raw['rows'] if not x.get('warmup')]
        assert len(measured)==45 and not any(x['instrumented'] or x['profiled'] for x in measured)
        assert {(x['arm'],x['rep'],x['prompt']) for x in measured}=={
            (arm,rep,p['id']) for arm in ('target','q4','HE4broad') for rep in range(5) for p in prompts}
        max_timing_error=0.;cache_tokens=[]
        for rep in range(5):
            for arm in ('target','q4','HE4broad'):
                cell=panel/f'{rep:02d}-{arm}';cellrows=[x for x in raw['rows'] if x['rep']==rep and x['arm']==arm]
                command=json.load(open(cell/'command.json'));argv=command['argv']
                assert argv[argv.index('-c')+1]=='4096'
                if arm!='target':
                    assert argv[argv.index('--spec-draft-n-max')+1]=='2'
                    assert not reconcile(cell,copy.deepcopy(cellrows))
                for row in cellrows:
                    response=json.load(open(cell/(row['prompt']+'-response.json')))
                    request=json.load(open(cell/(row['prompt']+'-request.json')))
                    assert request['n_predict']==256 and request['temperature']==0 and not request['cache_prompt']
                    signature={k:response[k] for k in ('tokens','content','stop_type','stopping_word','truncated')}
                    signatures.setdefault(row['prompt'],signature)
                    assert signatures[row['prompt']]==signature and row['correct']
                    assert row['tokens']==len(response['tokens'])
                    max_timing_error=max(max_timing_error,abs(row['wall_seconds']-(row['bounds_ns'][1]-row['bounds_ns'][0])/1e9))
                    if not row['warmup']:cache_tokens.append(response['timings']['cache_n'])
                    responses_checked+=1
                cells_checked+=1
        assert max_timing_error<.005 and not any(cache_tokens)
        metrics=summarize(path);per_prompt={}
        for prompt in prompts:
            selected=[x for x in measured if x['arm']=='HE4broad' and x['prompt']==prompt['id']]
            native=lambda key:sum(x['native'][key] for x in selected)
            per_prompt[prompt['id']]=dict(acceptance=native('accepted')/native('proposed'),
                request_tok_s=sum(x['tokens'] for x in selected)/sum(x['wall_seconds'] for x in selected),
                prefix_survival=[sum(x['native']['prefix_survival_counts'][j] for x in selected) for j in range(2)],
                rounds=native('rounds'))
        panels.append(dict(label=point['label'],metrics=metrics,per_prompt=per_prompt,
            prompts=len(prompts),measured_requests=len(measured),all_responses_equal_to_fixed_target=True,
            native_counters_reconciled=True,max_request_timing_boundary_error_seconds=max_timing_error,
            controls_and_requests_unchanged=True))
    distinct_exposure_sequences={x['sequence'] for x in updates}
    distinct_exposure_windows={(x['sequence'],x['anchor']) for x in updates}
    result=dict(status='CPU_AUDIT_PASS_WITH_LIMITATIONS',checked_unix=time.time(),
        decision='Check obvious data/resume/evaluation errors before attributing the early flattening to model limits.',
        budget='Read-only CPU/metadata/log analysis; no GPU run, no active trainer/source/config changes.',
        source=pin(__file__),recipe=pin(R/'recipe.json'),index=recipe['config']['data'],
        dataset=dict(sequences=len(rows),windows=len(windows),unique_label_positions=index['unique_labels_in_windows'],
            unique_ids=len(ids),unique_group_ids=len(groups),unique_prompt_token_sequences=len(prompt_hashes),
            largest_exact_prompt_duplicate_group=max(prompt_hashes.values()),causal_next_token_links_checked=causal_links,
            domain_sequences=dict(collections.Counter(x['receipt']['domain'] for x in rows)),
            domain_windows=dict(collections.Counter(domain for row in rows for domain in [row['receipt']['domain']]*len(row['anchors']))),
            dev_prompt_overlap=0),
        exposure=dict(completed_updates=len(updates),distinct_sequences=len(distinct_exposure_sequences),
            distinct_windows=len(distinct_exposure_windows),repeated_windows=len(updates)-len(distinct_exposure_windows),
            exact_seeded_global_order_checked=True,window_epochs=len(updates)/len(windows)),
        learning_curve=exposures,resumes=resumes,formal_panels=panels,cells_checked=cells_checked,responses_checked=responses_checked,
        objective=dict(training_future_positions=15,native_proposal_depth=2,dev_teacher_forced_positions_per_readout=135),
        limits=['Three fixed consumed DEV prompts; timing repetitions do not add prompt diversity.',
            'Stratified bootstrap has one prompt per domain here, so intervals do not establish generalization across new prompts.',
            'Different shuffled training examples at each step: training-loss medians are not a paired overfitting test.',
            'Verified target/label/causal index identities are not a fresh full QAT-versus-native numerical parity replay.',
            '15-token teacher-forced objective and depth-two autoregressive acceptance are different measurements.'])
    save(REPORT/'setup-audit.json',result)
    print(json.dumps(dict(status=result['status'],dataset=result['dataset'],exposure=result['exposure'],
        curve=result['learning_curve'],cells_checked=cells_checked,responses_checked=responses_checked),indent=2))


if __name__=='__main__': main()
