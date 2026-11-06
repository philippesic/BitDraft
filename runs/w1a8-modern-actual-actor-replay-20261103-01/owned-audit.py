"""Independent saved-array/chronology/protection audit; no model or GPU access."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import time

os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"
import numpy as np

P=Path('/home/philip/binary-eagle-decoding')
R=P/'runs/w1a8-modern-actual-actor-replay-20261103-01'
T=P/'runs/w1a8-modern-serving-trace-20261103-03/prompt-02'
started=time.perf_counter()
pins={}


def pin(path):
    path=Path(path).resolve()
    assert path.suffix not in ('.gguf','.safetensors','.npz')
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    value={'path':str(path),'sha256':digest,'bytes':path.stat().st_size};pins[str(path)]=value
    return value


def read(path):
    pin(path);return json.loads(Path(path).read_text())


def check(ref):
    value=pin(ref['path']);assert value['sha256']==ref['sha256']
    if 'bytes' in ref:assert value['bytes']==ref['bytes']


def data(spec):
    p=T/'trace'/spec['file'];pin(p)
    assert p.parent.resolve()==(T/'trace').resolve()
    a=np.fromfile(p,dtype='<f4').reshape(spec['shape']);assert np.isfinite(a).all()
    return a


def metrics(a,b):
    aa,bb=a.astype(np.float64),b.astype(np.float64);delta=aa-bb
    return {'max_abs':float(np.abs(delta).max()),'mean_abs':float(np.abs(delta).mean()),
            'rms':float(np.sqrt(np.mean(delta*delta))),
            'relative_l2_native_denominator':float(np.linalg.norm(delta)/max(np.linalg.norm(bb),1e-30)),
            'byteexact':a.tobytes()==b.tobytes()}


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).astype('<f4').tobytes()).hexdigest()


report=read(R/'owned-report.json')
for ref in report['input_and_output_pins']:check(ref)
assert report['status']=='ACTOR_OWNED_STATE_CPU_REPLAY_COMPLETE'
freeze=read(T.parent/'freeze.json')
for ref in freeze['pins']:
    if Path(ref['path']).suffix!='.gguf':check(ref)
assert freeze['draft']['sha256']==report['initializer']['sha256']
assert freeze['target']['sha256']=='3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630'
parser_path=P/'scripts/w1a8_generalization/modern_serving_trace.py';pin(parser_path)
spec=importlib.util.spec_from_file_location('serving_parser',parser_path)
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
response=read(T/'response.json');pin(T/'server.log')
protection=json.loads(json.dumps(parser.audit_trace((T/'server.log').read_text(),response,expected_depth=2)))
assert protection['draft_proposed']==110 and protection['emitted_draft_successes']==20
if (R/'protection-independent.json').exists():
    assert read(R/'protection-independent.json')==protection
else:
    with (R/'protection-independent.json').open('x') as stream:json.dump(protection,stream,indent=2);stream.write('\n')
root_protection=read(T/'protection.json')
for key in ('protections','rounds','protection_mask'):
    assert protection[key]==root_protection[key]
log=T/'trace/calls.jsonl';pin(log)
events=[json.loads(x) for x in log.read_text().splitlines()]
assert [x['event'] for x in events]==list(range(1,len(events)+1))
befores={x['call']:x for x in events if x['phase']=='before'}
afters={x['call']:x for x in events if x['phase']=='after'}
assert len(befores)==len(afters) and set(befores)==set(afters)
assert all(x['result']==0 for x in afters.values())
mapping_path=Path(report['trace_manifest']['draft_map']);pin(mapping_path)
mapping=np.loadtxt(mapping_path,dtype=np.int64,ndmin=1)
inverse={int(x):i for i,x in enumerate(mapping)}
cl=np.fromfile(R/'owned-cpu-selected-logits.f32',dtype='<f4').reshape(110,50000)
cs=np.fromfile(R/'owned-cpu-selected-prenorm.f32',dtype='<f4').reshape(110,2560)
cf=np.fromfile(R/'owned-cpu-encoder-fc.f32',dtype='<f4').reshape(-1,2560)
assert np.isfinite(cl).all() and np.isfinite(cs).all() and np.isfinite(cf).all()
rows=report['decoder_results'];by_call={}
for row in rows:by_call.setdefault(row['call'],[]).append(row)
enc_report={x['call']:x for x in report['encoder_results']}
source_hash={};cache=[];seen_selected=[];rollbacks=[];fc_offset=0
for event in events:
    call=event['call']
    if event['architecture']!='eagle3':continue
    if event['phase']=='before':
        batch=event['batch']
        if call in report['warmup_calls_skipped_without_supplied_embedding']:
            assert 'embeddings' not in batch;continue
        supplied=data(batch['embeddings'])
        if event['process_type_name']=='encode':
            er=enc_report[call];native=data(afters[call]['encoded_fc'])
            assert metrics(cf[fc_offset:fc_offset+len(native)],native)==er['whole_fc_cpu_minus_native']
            for i,rr in enumerate(er['rows']):
                assert metrics(cf[fc_offset+i],native[i])==rr['fc_cpu_minus_native']
                source_hash.setdefault(digest(native[i]),[]).append({'source':'encoder_fc','call':call,'row':i,
                    'target_call':er['upstream_target_call'],'target_position':rr['actual_target_position']})
            fc_offset+=len(native)
            continue
        assert len(by_call[call])==len(batch['rows'])
        first=batch['rows'][0]['positions'][0]
        assert first<=len(cache)
        if first<len(cache):
            rollbacks.append({'call':call,'cache_before':len(cache),'first_position':first,'discarded_rows':len(cache)-first})
            cache=cache[:first]
        selected={x['batch_row']:x for x in afters[call].get('selected',[])}
        nstates=data(afters[call]['prenorm']) if selected else None
        for i,(native,record) in enumerate(zip(batch['rows'],by_call[call],strict=True)):
            assert native['positions'][0]==len(cache)==record['decoder_position']
            assert native['seq_ids']==[0] and native['token']==record['input_token'] and native['saved_embedding_row']==i
            assert record['input_fc_or_prenorm_sources']==source_hash[digest(supplied[i])]
            assert all(x['call']<call for x in record['input_fc_or_prenorm_sources'])
            cache.append((call,i));assert len(cache)==record['cache_positions_after']
            assert native['output']==record['output_selected']==(i in selected)
            if i not in selected:continue
            n=selected[i];row=record['cpu_saved_row'];assert row==len(seen_selected)
            assert metrics(cs[row],nstates[n['saved_row']])==record['prenorm_cpu_minus_native']
            ids=np.asarray(n['candidate_ids'],dtype=np.int64);nl=data(n['candidate_logits']).reshape(-1)
            assert n['logits_mode']=='backend_explicit_candidate_pairs' and len(ids)==len(nl)==n['candidate_count']
            assert metrics(cl[row,[inverse[int(t)] for t in ids]],nl)==record['cpu_minus_native_candidate_logits']
            assert int(mapping[int(cl[row].argmax())])==record['cpu_proposed']
            assert int(ids[int(nl.argmax())])==record['native_proposed']
            seen_selected.append(record)
    elif event['phase']=='after' and event.get('prenorm'):
        states=data(event['prenorm'])
        for row in event['selected']:
            source_hash.setdefault(digest(states[row['saved_row']]),[]).append({'source':'native_decoder_prenorm','call':call,'row':row['saved_row']})
assert fc_offset==len(cf) and len(seen_selected)==110 and len(rows)==274
assert rollbacks==report['cache_rollbacks'] and len(rollbacks)==107
# Join native selected call rows to independently parsed actual round/depth/response.
joined=[];cursor=0;native_successes=[];mismatch=[];rounds=[];eligible_gold_rows=0
native_mask={(x['round'],x['proposal_depth']) for x in protection['protections']}
for round_record in protection['rounds']:
    accepted=round_record['verified_accepted'];cpu_prefix=0;prior_gold=True
    for depth,proposal in enumerate(round_record['proposals']):
        row=seen_selected[cursor];cpu=cl[cursor];cursor+=1
        index=round_record['first_proposal_response_index']+depth
        assert row['decoder_position']==round_record['anchor_position']+depth and row['native_proposed']==proposal
        assert row['input_token']==(round_record['boundary_token'] if depth==0 else round_record['proposals'][depth-1])
        n=afters[row['call']]['selected'][row['batch_row']]
        ids=n['candidate_ids'];nl=data(n['candidate_logits']).reshape(-1)
        order=np.argsort(nl)[::-1];native_margin=float(nl[order[0]]-nl[order[1]])
        native_class=inverse[proposal];wrong=cpu.copy();wrong[native_class]=-np.inf
        cpu_margin=float(cpu[native_class]-wrong.max())
        success=(round_record['round'],depth) in native_mask
        valid_gold=prior_gold and index<len(response['tokens'])
        eligible_gold_rows+=int(valid_gold)
        entry={'call':row['call'],'round':round_record['round'],'depth':depth,'response_index':index,
               'decoder_position':row['decoder_position'],'input_token':row['input_token'],
               'native_proposed':proposal,'cpu_proposed':row['cpu_proposed'],
               'response_token':response['tokens'][index] if index<len(response['tokens']) else None,
               'native_emitted_success':success,'gold_label_reachable_from_native_prefix':valid_gold,
               'native_top1_margin':native_margin,'cpu_native_label_minus_best_wrong':cpu_margin,
               'cpu_correct_on_reachable_gold':valid_gold and row['cpu_proposed']==response['tokens'][index]}
        if prior_gold and row['cpu_proposed']==entry['response_token']:cpu_prefix+=1
        prior_gold=prior_gold and proposal==entry['response_token']
        if not row['proposal_matches']:
            entry['native_winner_minus_cpu_winner']=float(nl[ids.index(proposal)]-nl[ids.index(row['cpu_proposed'])])
            mismatch.append(entry)
        if success:
            assert proposal==entry['response_token']
            native_successes.append(entry)
        joined.append(entry)
    assert cpu_prefix==accepted
    rounds.append({'round':round_record['round'],'native_accepted_prefix':accepted,'cpu_same_recorded_state_prefix':cpu_prefix})
assert cursor==110 and len(mismatch)==sum(not x['proposal_matches'] for x in seen_selected) and len(native_successes)==20
assert sum(x['cpu_proposed']!=x['native_proposed'] for x in native_successes)==0
old=read(P/'runs/w1a8-modern-recurrent-actor-20261103-01/report.json')['cases']['2']
conditional={(s['decoder_position'],s['proposed_target_token_id']) for w in old['recurrent']
             for s in w['steps'] if s['correct'] and s['numeric_reachable']}
actual={(s['decoder_position'],s['native_proposed']) for s in native_successes}
intersection=sorted(conditional&actual)
assert conditional=={(57,2702),(60,93008)} and intersection==[(57,2702)]
# Native margins can differ from the CPU surrogate. If frozen CPU parent margins
# are used, the unchanged parent has zero initial penalty by construction.
protected_margin_pairs=[{'call':x['call'],'round':x['round'],'depth':x['depth'],
                         'native':x['native_top1_margin'],'cpu':x['cpu_native_label_minus_best_wrong'],
                         'native_target_would_activate_initial_penalty':min(2.0,x['native_top1_margin'])>x['cpu_native_label_minus_best_wrong']}
                        for x in native_successes]
output={'status':'INDEPENDENT_ACTOR_OWNED_STATE_AND_PROTECTION_AUDIT_PASS',
        'producer_report':pin(R/'owned-report.json'),'audit_recipe':pin(__file__),'inputs':list(pins.values()),
        'selected_call_rows':110,'actual_decoder_rows':274,'encoder_rows':fc_offset,
        'input_source_causality_raw_bytes_pass':True,'cache_rollback_invariants_pass':True,
        'all_saved_fc_prenorm_candidate_logit_metrics_recomputed':True,
        'all_native_candidates_join_actual_round_depth_and_response':True,
        'native_emitted_successes':20,'native_successes_cpu_matched':20,'native_successes_lost_by_cpu':[],
        'native_round_prefixes_preserved':55,'rounds':rounds,'mismatches':mismatch,
        'native_protections':native_successes,'protected_margin_pairs':protected_margin_pairs,
        'reachable_gold_label_diagnostic_rows':eligible_gold_rows,
        'mismatches_on_reachable_gold':sum(x['gold_label_reachable_from_native_prefix'] for x in mismatch),
        'conditional_protected_tokens':sorted(conditional),'conditional_actual_protection_intersection':intersection,
        'actual_protections_outside_conditional_set':len(actual-conditional),
        'practical_gate':{'decision_disagreements':len(mismatch),'denominator':110,'fraction':len(mismatch)/110,
                          'threshold':0.05,'aggregate_pass':len(mismatch)/110<=0.05,
                          'mean_absolute_native_accepted_prefix_error':0.0,
                          'native_protection_preservation_pass':True,
                          'qualified_scope':'This reused actual P2 history only; continue native admission per history'},
        'loss_and_gradient_implications':{
            'gold_labels':'External emitted/target token IDs unchanged by CPU argmax; only attach future gold labels while native prefix remains correct',
            'mismatches':'All native-unaccepted. Gold labels remain external and unchanged; any depth1 row after a wrong native or CPU first token remains diagnostic-only',
            'protection_mask':'All20 native successes retain correct CPU predictions; keep all20 native protections and every110 native proposal in evaluation',
            'margin_gradients':'Numeric surrogate margins differ; native-margin targets can produce a nonzero unchanged-parent retention penalty. Freeze parent reference margins from the same CPU actor/states, with immutable native success mask; native evaluation remains authoritative',
            'native_gradient_equivalence_proven':False,
            'masked_native_successes_because_cpu_failed':0},
        'recommendation':'Actor-owned CPU FC and recurrent pre-norm pass this one-history forward/protection gate. Proceed to actual-serving-state training, preserve every native success, and validate native continuations after updates. No further parity chase is warranted.',
        'cpu_and_native_cache_trajectories':'Actual native raw FC input is recomputed through CPU FC; every native FC or pre-norm source is replaced by its matching CPU-owned result. Native tokens/cache overwrite path remain fixed; protected two-step unrolls are valid only until first gold divergence. No free-running claim after divergence; autograd was not exercised in this inference pass',
        'failed_attempts':[],'model_loaded':False,'gpu_seconds':0,
        'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
with (R/'owned-independent-audit.json').open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
print(json.dumps({k:output[k] for k in ('status','native_emitted_successes','native_successes_cpu_matched',
      'reachable_gold_label_diagnostic_rows','mismatches_on_reachable_gold','actual_protections_outside_conditional_set',
      'practical_gate','wall_seconds','peak_rss_bytes')},indent=2))
