"""Fail-closed corrected-D5 diagnostic cost accounting. Never runs a model.

CPU wall trace spans support descriptive same-pass fractions and a round-only
frozen-trajectory counterfactual. They do not, by themselves, prove removable
critical-path GPU work or a clean full-request speedup bound.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_w1ax_break_even import summarize_counterfactual
from analyze_w1ax_diagnostics import TRACE_SPAN_STAGES

SOURCES=('tools/server/server-context.cpp','common/speculative.cpp','common/sampling.cpp',
         'src/llama-context.cpp','ggml/src/ggml.c')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pin(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def read(path):return json.loads(Path(path).read_text())


def checked(ref):
    need(sha(ref['path'])==ref['sha256'],'source/input hash changed: '+ref['path'])
    return Path(ref['path'])


def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def trace_groups(rows):
    groups=[];seen=set();last_end=None
    for row in rows:
        key=(row['task_id'],row.get('parent_task_id'),row['slot_id'])
        if not groups or groups[-1][0]!=key:
            need(key not in seen,'noncontiguous request/task trace group')
            seen.add(key);groups.append((key,[]))
        start,end=row.get('round_start_us'),row.get('round_end_us')
        need(isinstance(start,(int,float)) and isinstance(end,(int,float)) and start<=end,'valid monotonic round bounds required')
        if last_end is not None:need(start>=last_end,'round windows overlap; single-request exclusivity unproven')
        last_end=end;groups[-1][1].append(row)
    return [rows for _,rows in groups]


def inspect_source(config, source_root=None, manifest_path=None):
    build=read(checked(config['build']))
    manifest=read(checked(build['source_manifest'])) if manifest_path is None else read(manifest_path)
    if manifest_path is not None:
        need(sha(manifest_path)==build['source_manifest']['sha256'],'copied source manifest differs from native build')
    root=Path(source_root or build['source_root']);text={};pins={}
    for name in SOURCES:
        path=root/name;expected=manifest[name]
        if isinstance(expected,dict):expected=expected['sha256']
        need(sha(path)==expected,'source differs from native build: '+name)
        text[name]=path.read_text();pins[name]=pin(path)
    server=text[SOURCES[0]];common=text[SOURCES[1]];context=text['src/llama-context.cpp']
    draft_marker='// generate the actual drafts (if any)'
    begin=server.index(draft_marker);outer=server[begin:begin+1800]
    getter=context[context.index('float * llama_get_logits_ith('):][:450]
    need('common_speculative_draft(spec.get())' in outer and 'yield_to_queue' in outer,
         'unreviewed current draft trace boundary')
    need('ctx->synchronize();' in getter,'draft getter completion source changed')
    need('if (sync_component_timings) llama_synchronize(ctx_dft);' in common,
         'explicit DSpark component sync source changed')
    need('clock_gettime(CLOCK_MONOTONIC, &ts)' in text['ggml/src/ggml.c'],
         'native Linux monotonic clock source changed')
    return {'native_build':config['build'],'source_manifest':build['source_manifest'],'source_pins':pins,
            'clock':'native Linux ggml_time_us uses CLOCK_MONOTONIC microseconds; Python request_wall_s records only a perf_counter elapsed duration, without request boundary timestamps',
            'outer_draft_scope':'server yield_to_queue(common_speculative_draft); queue/sampling/wait host time included',
            'post_draft_completion':'ordinary sampler logits getter synchronizes ctx_dft; optional component mode also fences seed decode',
            'process_injection_completion':'only explicitly fenced under DSPARK_SYNC_COMPONENT_TIMINGS when process tracing enabled',
            'pre_draft_explicit_fence_recorded':False,
            'request_start_end_in_trace_clock_recorded':False,
            'all_draft_work_inside_outer_draft_span':False,
            'gaps':['no recorded explicit pre-draft fence; carry-in injection work may be charged to a later wait',
                    'request artifacts record elapsed HTTP wall, not start/end in the native trace clock',
                    'draft-context injection/cache catch-up also lives in process spans outside outer proposal call',
                    'yield_to_queue may include scheduling work; interval exclusivity alone is not causal removability',
                    'separate clean timings cannot be reduced by diagnostic spans']}


def analyze_request(rows, record, expected_tokens, *, explicit_sync_requested):
    result=record['result'];tokens=result['generated_token_ids']
    need(tokens==expected_tokens and result['completion_tokens']==len(tokens),'actual generated IDs/count differ from fixed target')
    need(math.isfinite(result['request_wall_s']) and result['request_wall_s']>0,'positive same-pass request wall required')
    need(all(row.get('n_draft_configured')==5 for row in rows),'corrected D5 configured trace required; D7 not reusable')
    need(all(type(row.get('n_proposed')) is int and 0<=row['n_proposed']<=5 for row in rows),'actual proposals exceed D5')
    need(all(row.get('draft_p_min')==0 for row in rows),'greedy D5 confidence policy differs')
    need(all(row.get('clock')=='ggml_time_us_cpu_wall' for row in rows),'native CPU wall clock required')
    explicit=all(row.get('draft_component_explicit_sync') is True for row in rows)
    need(explicit==explicit_sync_requested,'recorded explicit sync disagrees with exact cell environment')
    accounting=summarize_counterfactual(rows,{})
    emitted=[t for row in rows if row['status']!='checkpoint_replay' for t in row['emitted_token_ids']]
    need(emitted in (tokens,tokens[1:]),'native round emissions do not map to this exact request')
    complete=[row for row in rows if row['status']=='complete' and not row['replay']]
    counts={'accepted':sum(row['n_accepted'] for row in complete),
            'proposed':sum(row['n_proposed'] for row in complete),'rounds':len(complete)}
    need(counts==result['speculative'],'trace counters differ from actual native request metrics')
    # Cross-round nonoverlap is checked independently of each round's union.
    for previous,current in zip(rows,rows[1:]):
        need(previous['round_end_us']<=current['round_start_us'],'overlapping round windows')
    wall_us=result['request_wall_s']*1e6;round_us=accounting['timing_us']['C_all_rounds']
    need(round_us<=wall_us,'trace round sum exceeds enclosing HTTP elapsed wall; no same-pass containment')
    stage_us={stage:sum(row[stage+'_us'] for row in rows) for stage in TRACE_SPAN_STAGES}
    return {'prompt_id':record['prompt_id'],'domain':record['domain'],'actual_tokens':len(tokens),
            'actual_request_wall_s':result['request_wall_s'],'actual_tokens_per_s':len(tokens)/result['request_wall_s'],
            'actual_native_counts':counts,'target_ids_exact':True,'native_trace_ids_and_counts_exact':True,
            'omitted_initial_target_seed':int(emitted==tokens[1:]),'explicit_component_sync':explicit,
            'same_pass_round_cpu_wall_us':round_us,'same_pass_named_stage_cpu_wall_us':stage_us,
            'named_stage_fractions_of_round_cpu_wall':{k:v/round_us for k,v in stage_us.items()},
            'exclusive_outer_draft_host_us':accounting['timing_us']['D_removable_exclusive_draft'],
            'exclusive_outer_draft_fraction_of_DIAGNOSTIC_HTTP_elapsed':accounting['timing_us']['D_removable_exclusive_draft']/wall_us,
            'draft_overlap_other_named_spans_us':accounting['timing_us']['draft_overlap_other_union'],
            'draft_coverage':'outer proposal call only; process injection and cache catch-up retained',
            'round_only_frozen_trajectory_counterfactual':{
                'observed_round_cpu_wall_us':round_us,
                'exclusive_outer_call_removed_round_cpu_wall_us':accounting['timing_us']['C_minus_D'],
                'scope':'host-interval arithmetic only; proposals/emissions/acceptance/other costs fixed; not a causal runtime bound'},
            'full_request_runtime_upper_bound':None,
            'full_request_subtraction_permitted':False}


def analyze(run, *, source_root=None, manifest_path=None):
    summary=read(run/'summary.json');config=read(run/'config.json');requests=jsonl(run/'requests.jsonl')
    need(summary.get('status')=='COMPLETE' and summary.get('diagnostic') is True,'complete separate DIAGNOSTIC evaluation required; clean timing cannot be mixed')
    need(config['schema']=='shortlist_fixed_request_evaluator_v1' and config['target_width_mask']==3 and config['draft_limit']==5,
         'current corrected D5/mask3 evaluation configuration required')
    proof=inspect_source(config,source_root,manifest_path)
    selected={};files=[]
    for record in requests:
        key=(record['panel'],record['repetition'],record['arm'])
        selected.setdefault(key,[]).append(record)
    results=[]
    for (panel,rep,arm),records in selected.items():
        stage=run/panel/f'rep-{rep:02d}'/arm
        trace=stage/'rounds.jsonl';groups=trace_groups(jsonl(trace));files.append(pin(trace))
        command=read(stage/'command.json');environment=command['environment']
        need(environment.get('DSPARK_REQUIRE_AUTHOR_LAYOUT')=='1' and environment.get('DSPARK_TARGET_WIDTH_MASK')=='3','current author-layout/mask3 runtime required')
        args=command['argv'];need(args[args.index('--spec-draft-n-max')+1]=='5','actual native command is not D5')
        need(args[args.index('--parallel')+1]=='1','shared/multisequence round spans cannot be treated exclusive')
        protocol=read(checked(config['panels'][panel]['protocol']))
        need(protocol['concurrency']==1,'single-request source protocol required')
        reference=read(checked(config['panels'][panel]['target_reference']))
        target={(r['repetition'],r['prompt_id']):r['generated_token_ids'] for r in reference['records']}
        warmups=config['warmups_per_cell'];need(warmups==2,'original excluded warmup boundary required')
        need(len(groups)==warmups+len(records),'trace/request mapping lacks complete warmup+measured groups')
        for group,record in zip(groups[warmups:],records):
            result=analyze_request(group,record,target[(rep,record['prompt_id'])],
                                   explicit_sync_requested='DSPARK_SYNC_COMPONENT_TIMINGS' in environment)
            result.update(panel=panel,arm=arm,repetition=rep);results.append(result)
    need(results,'no measured diagnostic requests')
    def totals(rows):
        wall=sum(x['actual_request_wall_s'] for x in rows)
        rounds=sum(x['same_pass_round_cpu_wall_us'] for x in rows)
        exclusive=sum(x['exclusive_outer_draft_host_us'] for x in rows)
        return {'requests':len(rows),'actual_tokens':sum(x['actual_tokens'] for x in rows),
                'DIAGNOSTIC_HTTP_elapsed_s':wall,'round_cpu_wall_us':rounds,
                'native_counts':{key:sum(x['actual_native_counts'][key] for x in rows) for key in ('accepted','proposed','rounds')},
                'exclusive_outer_draft_host_us':exclusive,
                'exclusive_outer_draft_host_fraction_of_DIAGNOSTIC_HTTP_elapsed':exclusive/(wall*1e6),
                'named_stage_cpu_wall_us':{key:sum(x['same_pass_named_stage_cpu_wall_us'][key] for x in rows) for key in TRACE_SPAN_STAGES},
                'all_explicit_component_sync':all(x['explicit_component_sync'] for x in rows),
                'full_request_subtraction_permitted':False}
    cells={}
    for panel,arm in sorted({(x['panel'],x['arm']) for x in results}):
        rows=[x for x in results if x['panel']==panel and x['arm']==arm]
        cells[panel+'/'+arm]={**totals(rows),
            'by_domain':{domain:totals([x for x in rows if x['domain']==domain]) for domain in sorted({x['domain'] for x in rows})}}
    return {'schema':'corrected_d5_diagnostic_cost_accounting_v1','status':'DIAGNOSTIC_ACCOUNTING_PASS_FULL_REQUEST_BOUND_REJECTED',
            'source_proof':proof,'diagnostic_cells':cells,'requests':results,'inputs':[pin(run/'config.json'),pin(run/'summary.json'),pin(run/'requests.jsonl'),*files],
            'full_request_runtime_upper_bound':None,'full_request_subtraction_permitted':False,
            'interpretation':'Fractions describe the same instrumented request. They are not maximum removable draft cost, clean serving bounds, or predicted speedups. Acceptance and emitted tokens remain fixed in the round-only counterfactual.',
            'count_and_target_scope':'actual returned IDs == frozen F16 target reference; actual native counters == reconstructed native trace counts',
            'new_model_or_acceptance_change_predicted':False,'old_D7_numbers_used':False}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True,help='complete evaluate.py --diagnostic output')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,help='optional byte-identical local copy of current native04 source')
    p.add_argument('--source-manifest',type=Path,help='optional copied source manifest; original build pin still checked')
    args=p.parse_args()
    try:
        result=analyze(args.run,source_root=args.source_root,manifest_path=args.source_manifest)
    except (ValueError,KeyError,IndexError,FileNotFoundError) as error:
        result={'schema':'corrected_d5_diagnostic_cost_accounting_v1','status':'REJECTED_UNPROVEN_OR_INCOMPLETE',
                'reason':str(error),'full_request_runtime_upper_bound':None,'full_request_subtraction_permitted':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'full_request_subtraction_permitted':False}))
    return 0 if result['status'].startswith('DIAGNOSTIC_ACCOUNTING_PASS') else 2


if __name__=='__main__':raise SystemExit(main())
