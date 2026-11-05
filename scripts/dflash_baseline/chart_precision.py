"""Recompute campaign tables and standalone charts solely from native receipts."""
import argparse,csv,hashlib,html,json,re,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/philip/binary-eagle-decoding')
def read(p):return json.loads(Path(p).read_text())
@lru_cache(None)
def shared_tensor(path):
 sys.path.insert(0,str(P/'third_party/llama.cpp/gguf-py'))
 from gguf import GGUFReader
 tensors={t.name:t for t in GGUFReader(path).tensors};t=tensors.get('output.weight',tensors['token_embd.weight'])
 return dict(name=t.name,format=t.tensor_type.name,bytes=t.data.nbytes,shape=t.shape.tolist())
def reference(path):
 path=Path(path);return dict(path=str(path.resolve()),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),bytes=path.stat().st_size)
def summarize(panel,root):
 result=read(panel/'result.json');rows=result['rows'];freeze=read(panel/'freeze.json') if (panel/'freeze.json').exists() else {};cohort=freeze.get('cohort','capture3');points=[]
 provenance=dict(result=reference(panel/'result.json'),prompts=reference(panel/'prompts.json'),freeze=reference(panel/'freeze.json') if (panel/'freeze.json').exists() else None,**{k:freeze.get(k) for k in ('source','counter_parser','cohort_source','prompt_source','baseline_registry','build','trace_library','warmup_policy')})
 prompt_domains={p['id']:p.get('domain',next((d for d in ('code','reasoning','prose') if d in p['id'].split('-')),'unspecified')) for p in read(panel/'prompts.json')}
 for arm in dict.fromkeys(x['arm'] for x in rows):
  plain=[x for x in rows if x['arm']==arm and not x['warmup'] and not x.get('instrumented')];traced=[x for x in rows if x['arm']==arm and not x['warmup'] and x.get('instrumented')]
  if not plain:continue
  counts=[x.get('native',{}) for x in plain];tokens=sum(x['tokens'] for x in plain);wall=sum(x['wall_seconds'] for x in plain);decode=sum(x['timings']['predicted_ms'] for x in plain)/1000
  accepted=sum(x.get('accepted',y['timings'].get('draft_n_accepted',0)) for x,y in zip(counts,plain));proposed=sum(x.get('proposed',y['timings'].get('draft_n',0)) for x,y in zip(counts,plain));rounds=sum(x.get('rounds',0) for x in counts)
  meta=freeze.get('arms',{}).get(arm);model=meta.get('path') if meta else None;sidecar=Path(model).with_suffix('.json') if model else None;info=read(sidecar) if sidecar and sidecar.exists() else {};config=info.get('config',{});coverage=config.get('coverage',{});bits=[coverage.get(f'blk.{i}.ffn_gate',{}).get('bits') for i in range(6)];down=[coverage.get(f'blk.{i}.ffn_down',{}).get('bits') for i in range(6)]
  family='G'+str(bits[0])+'/D'+str(down[0]) if bits and bits[0] else 'external' if arm in ('target','q4') else 'vanilla W1A8'
  memories=[];allocator=[];device_samples=[];model_buffers=[]
  for cell in panel.glob('*-'+arm):
   mp=cell/'memory.json'
   if mp.exists():
    samples=read(mp);memories.extend(x['MiB'] for x in samples);rep=int(cell.name.split('-',1)[0]);intervals=[x['bounds_ns'] for x in plain if x['rep']==rep and 'bounds_ns' in x]
    device_samples.extend(x for x in samples if any(begin<=x['monotonic']*1e9<=end for begin,end in intervals))
   log=cell/'server.log'
   if log.exists():
    log_text=log.read_text(errors='replace');loads=re.findall(r'load_tensors:\s+CUDA0 model buffer size\s*=\s*([\d.]+) MiB',log_text)
    if loads:model_buffers.append(sum(map(float,loads)))
    match=re.findall(r'CUDA0.*?\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)',log_text)
    if match:allocator.append(dict(zip(('self_MiB','model_MiB','context_MiB','compute_MiB'),map(int,match[-1]))))
  trace_rounds=sum(x['native'].get('rounds',0) for x in traced);full=sum(x.get('draft_full_ms',0) for x in traced)/trace_rounds if trace_rounds else None
  prefix=[]
  for n in counts:
   for i,v in enumerate(n.get('prefix_survival_counts',[])):
    if i==len(prefix):prefix.append(0)
    prefix[i]+=v
  resource_note=read(panel/'resource-note.json') if (panel/'resource-note.json').exists() else None
  runtime_pin=freeze.get('runtime',{}).get('libllama.so.0.8.0',{})
  runtime_series=Path(runtime_pin['path']).parent.parent.name if runtime_pin else 'historical-unpinned-runtime'
  point=dict(runtime_series=runtime_series,execution='Native INT8-code DP4A / packed W1 FFN' if coverage else 'Native external model',resource_note=resource_note,id=str(panel.relative_to(root))+'::'+arm,panel=str(panel),arm=arm,cohort=cohort,scope='confirmation' if len({x['rep'] for x in plain})>=5 else 'diagnostic smoke',family=family,config=config,model=model,model_bytes=meta.get('bytes') if meta else (Path(model).stat().st_size if model else 0),correct=all(x['correct'] for x in plain+traced),requests=len(plain),prompts=len({x['prompt'] for x in plain}),repetitions=len({x['rep'] for x in plain}),accepted=accepted,proposed=proposed,rounds=rounds,acceptance=accepted/proposed if proposed else None,prefix_counts=prefix,prefix_survival=[x/rounds for x in prefix] if rounds else [],emitted_per_round=tokens/rounds if rounds else None,request_tps=tokens/wall,decode_tps=tokens/decode,full_draft_ms=full,narrow_generate_ms=sum(x.get('draft_ms',0) for x in counts)/rounds if rounds else None,total_peak_device_MiB=max(memories) if memories else None,main_context_allocator_shutdown_MiB=allocator[-1] if allocator else None,training_updates=info.get('progress',{}).get('updates',info.get('updates')),training_seconds=info.get('progress',{}).get('update_seconds'),lineage=info.get('lineage'),runtime=freeze.get('runtime'),instrumented_requests=len(traced),verified_paired_counters=result.get('paired_trace_counters_equal',False),target_verify_and_sample_ms_per_round=sum(x.get('target_verify_and_sample_ms',0) for x in traced)/trace_rounds if trace_rounds else None)
  target_ref=freeze.get('target',{});target_bytes=target_ref.get('bytes')
  point['model_reference']=meta;point['panel_provenance']=provenance;point['native_export_metadata']=reference(sidecar) if sidecar and sidecar.exists() else None;point['training_checkpoint_reference']=info.get('checkpoint')
  point['target_model_bytes']=target_bytes;point['combined_deployment_bytes']=target_bytes+point['model_bytes'] if target_bytes is not None else None
  point['shared_target_head']=shared_tensor(target_ref['path']) if target_ref else None
  point['private_head_format']=config.get('components',{}).get('head')
  point['size_accounting']='Actual target and drafter GGUF bytes include metadata; target tied embedding/head charged once in combined bytes. Any drafter private head is additional inside its GGUF. Allocator/peak measurements separately include execution buffers.'
  point['main_context_allocator_scope']='Main target context only; separate drafter allocations appear outside this context. This breakdown is not the process total or peak.'
  point['total_cuda_model_buffers_MiB']=max(model_buffers) if model_buffers else None
  point['observed_request_device_state']={}
  for key in ('sm_clock_MHz','temperature_C','power_W','utilization_percent'):
   values=[x[key] for x in device_samples if x.get(key) is not None]
   point['observed_request_device_state'][key]=dict(min=min(values),max=max(values),median=float(np.median(values))) if values else None
  progress=info.get('progress',{});point['supervised_positions']=progress.get('supervised_tokens');point['data_cursor']=progress.get('cursor');point['immediate_parent']=config.get('parent')
  if point['training_seconds'] is None:point['training_seconds']=info.get('update_seconds')
  if arm=='vanilla-zero':point['training_updates']=0
  if point['family']=='vanilla W1A8':point['execution']='Native packed W1 / INT8-code DP4A (vanilla reference)'
  point['training_scope']='No campaign training' if arm in ('target','q4') else 'Vanilla reference updates from source' if not coverage else 'Precision updates after frozen common foundation' if '/foundation/checkpoint' in config.get('parent',{}).get('path','') else 'Foundation/control updates after frozen vanilla1800'
  point['point_status']='NATIVE_EXTERNAL_CONTROL' if arm in ('target','q4') else 'MEASURED_ZERO_PHASE_UPDATES' if point['training_updates']==0 else 'MEASURED_TRAINED_PHASE'
  policy=freeze.get('warmup_policy',{})
  if point['repetitions']>=5 and not (policy.get('same_process') and policy.get('per_server_cell',0)>=2):point['scope']='repeated diagnostic; warmup unqualified'
  point['quantizer_mode']=config.get('quantizer',{}).get('mode','native baseline')
  components=config.get('components',{})
  point['protected_weight_formats']='; '.join(k+'='+components.get(k,'BF16') for k in ('fusion','attention_qk','attention_vo'))+'; head='+components.get('head','shared target Q6_K')
  if arm=='q4':point['protected_weight_formats']='Frozen mixed Q4_K_M GGUF; target-shared Q6_K head/embedding'
  if arm=='target':point['protected_weight_formats']='Frozen Q4_K_M target with tied Q6_K head/embedding'
  if coverage:
   components=config.get('components',{});point['formats']=dict(ffn_weights='Packed1 sign + F32 row scale/midpoint; F32 positive channel gains',gate_up_activations='A'+str(bits[0]),down_activations='A'+str(down[0]),activation_execution='INT8-code DP4A for all logical widths',fusion=components.get('fusion','BF16'),attention_qk=components.get('attention_qk','BF16'),attention_vo=components.get('attention_vo','BF16'),norms='F32 original trainable norms',head=components.get('head','shared target Q6_K'),embedding='shared target Q6_K',target_and_draft_KV='F16')
  training_result=Path(model).parent/'result.json' if model else None
  point['training_peak_allocated_bytes']=read(training_result).get('peak_bytes') if training_result and training_result.exists() and (point['training_updates'] or 0)>0 else None
  baseline=[x for x in rows if x['arm']=='q4' and not x['warmup'] and not x.get('instrumented')]
  point['domain_breakdown']={}
  for domain in sorted(set(prompt_domains.values())):
   cases=[x for x in plain if prompt_domains[x['prompt']]==domain];control=[x for x in baseline if prompt_domains[x['prompt']]==domain]
   if not cases:continue
   acc=sum(x.get('native',{}).get('accepted',x['timings'].get('draft_n_accepted',0)) for x in cases);prop=sum(x.get('native',{}).get('proposed',x['timings'].get('draft_n',0)) for x in cases);tokens_domain=sum(x['tokens'] for x in cases);rate=tokens_domain/sum(x['wall_seconds'] for x in cases)
   point['domain_breakdown'][domain]=dict(requests=len(cases),unique_prompts=len({x['prompt'] for x in cases}),accepted=acc,proposed=prop,acceptance=acc/prop if prop else None,request_tps=rate,decode_tps=tokens_domain/(sum(x['timings']['predicted_ms'] for x in cases)/1000),request_vs_q4=rate/(sum(x['tokens'] for x in control)/sum(x['wall_seconds'] for x in control)) if control else None,scope='Descriptive subset; four DEV/eight heldout cases per domain, not independent training replicates.')
  if baseline:
   point['request_vs_q4']=point['request_tps']/(sum(x['tokens'] for x in baseline)/sum(x['wall_seconds'] for x in baseline));point['decode_vs_q4']=point['decode_tps']/(sum(x['tokens'] for x in baseline)/(sum(x['timings']['predicted_ms'] for x in baseline)/1000))
  if baseline and point['repetitions']>=5:
   lookup={(x['rep'],x['prompt']):x for x in baseline};pairs=[(x,lookup[(x['rep'],x['prompt'])]) for x in plain];rng=np.random.default_rng(811);ratios=[]
   acceptance_samples=[];latency_samples=[];overhead_samples=[];decode_ratios=[];trace_by={(x['rep'],x['prompt']):x for x in traced}
   reps=sorted({x['rep'] for x in plain});prompts=sorted({x['prompt'] for x in plain});by={(x['rep'],x['prompt']):(x,y) for x,y in pairs}
   for _ in range(2000):
    sample_reps=rng.choice(reps,len(reps));sample_prompts=rng.choice(prompts,len(prompts));chosen=[by[(r,p)] for r in sample_reps for p in sample_prompts];ca=[x for x,y in chosen];ba=[y for x,y in chosen];ratios.append((sum(x['tokens'] for x in ca)/sum(x['wall_seconds'] for x in ca))/(sum(x['tokens'] for x in ba)/sum(x['wall_seconds'] for x in ba)))
    # Cases are resampled once across all sampled repetitions (two-way clusters).
    decode_ratios.append((sum(x['tokens'] for x in ca)/sum(x['timings']['predicted_ms'] for x in ca))/(sum(x['tokens'] for x in ba)/sum(x['timings']['predicted_ms'] for x in ba)))
    accepted_sum=sum(x.get('native',{}).get('accepted',x['timings'].get('draft_n_accepted',0)) for x in ca);proposed_sum=sum(x.get('native',{}).get('proposed',x['timings'].get('draft_n',0)) for x in ca)
    if proposed_sum:acceptance_samples.append(accepted_sum/proposed_sum)
    if traced:
     ta=[trace_by[(x['rep'],x['prompt'])] for x in ca];trace_round_sum=sum(x['native'].get('rounds',0) for x in ta)
     if trace_round_sum:latency_samples.append(sum(x.get('draft_full_ms',0) for x in ta)/trace_round_sum)
     overhead_samples.append(100*(sum(x['wall_seconds'] for x in ta)/sum(x['wall_seconds'] for x in ca)-1))
   point['request_vs_q4_bootstrap95']=np.quantile(ratios,[.025,.975]).tolist()
   point['decode_vs_q4_bootstrap95']=np.quantile(decode_ratios,[.025,.975]).tolist()
   if acceptance_samples:point['acceptance_bootstrap95']=np.quantile(acceptance_samples,[.025,.975]).tolist()
   if latency_samples:point['full_draft_ms_bootstrap95']=np.quantile(latency_samples,[.025,.975]).tolist()
   if overhead_samples:point['trace_overhead_bootstrap95_percent']=np.quantile(overhead_samples,[.025,.975]).tolist()
  if traced:
   names=sorted({n for x in traced for n in x.get('host_spans_ms',{}).get('generation',{})})
   point['generation_host_ms_per_round']={n:sum(x.get('host_spans_ms',{}).get('generation',{}).get(n,0) for x in traced)/trace_rounds for n in names} if trace_rounds else {}
   point['trace_request_overhead_percent']=100*((sum(x['wall_seconds'] for x in traced)/sum(x['wall_seconds'] for x in plain))-1)
   trace_tokens=sum(x['tokens'] for x in traced);draft_seconds=sum(x.get('draft_full_ms',0) for x in traced)/1000;trace_wall=sum(x['wall_seconds'] for x in traced);trace_decode=sum(x['timings']['predicted_ms'] for x in traced)/1000
   point['host_draft_span_subtraction_request_tps']=trace_tokens/(trace_wall-draft_seconds) if trace_wall>draft_seconds else None
   point['host_draft_span_subtraction_decode_tps']=trace_tokens/(trace_decode-draft_seconds) if trace_decode>draft_seconds else None
   point['host_draft_span_subtraction_scope']='Diagnostic arithmetic subtracting measured draft HOST spans only, assuming unchanged verification/proposals. Asynchronous draft GPU work may reside in other spans; neither a zero-cost drafter prediction nor an upper bound on attainable throughput. Not measured deployment throughput.'
  target_cells=list(panel.glob('*-target'));target_peak=[];target_allocator=[];target_model_buffers=[]
  for cell in target_cells:
   if (cell/'memory.json').exists():target_peak.extend(x['MiB'] for x in read(cell/'memory.json'))
   if (cell/'server.log').exists():
    target_log=(cell/'server.log').read_text(errors='replace');loads=re.findall(r'load_tensors:\s+CUDA0 model buffer size\s*=\s*([\d.]+) MiB',target_log)
    if loads:target_model_buffers.append(sum(map(float,loads)))
    matches=re.findall(r'CUDA0.*?\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)',target_log)
    if matches:target_allocator.append(tuple(map(int,matches[-1])))
  point['incremental_peak_device_MiB_vs_target']=max(memories)-max(target_peak) if memories and target_peak else None
  point['incremental_cuda_model_buffers_MiB_vs_target']=max(model_buffers)-max(target_model_buffers) if model_buffers and target_model_buffers else None
  points.append(point)
 return points

def build(root,out):
 out.mkdir(parents=True,exist_ok=True);points=[]
 for p in sorted(root.rglob('result.json')):
  if p.parent==out:continue
  data=read(p)
  if data.get('status')=='COMPLETE' and isinstance(data.get('rows'),list) and data['rows'] and 'arm' in data['rows'][0]:points+=summarize(p.parent,root)
 # Point-estimate three-objective frontier; retain uncertainty and training histories.
 for point in points:
  peers=[p for p in points if all(p[k]==point[k] for k in ('panel','cohort','runtime_series','scope')) and p['correct'] and p['acceptance'] is not None and p['full_draft_ms'] is not None]
  point['point_estimate_pareto']=None
  if point in peers:
   point['point_estimate_pareto']=not any(p['acceptance']>=point['acceptance'] and p['full_draft_ms']<=point['full_draft_ms'] and p['model_bytes']<=point['model_bytes'] and (p['acceptance']>point['acceptance'] or p['full_draft_ms']<point['full_draft_ms'] or p['model_bytes']<point['model_bytes']) for p in peers)
 followup_marker=root/'clip-control-followup/completion.json'
 if followup_marker.exists() and Path(read(followup_marker)['charts_directory']).resolve()==out.resolve():raise FileExistsError('Published followup charts are frozen')
 published=read(root/'campaign-completion.json') if (root/'campaign-completion.json').exists() else None
 if published and Path(published['charts_directory']).resolve()==out.resolve():raise FileExistsError('Published initial charts are frozen; use a new output directory')
 (out/'generator.py').write_bytes(Path(__file__).read_bytes())
 completion=read(root/'native-data-completion.json') if (root/'native-data-completion.json').exists() else None
 if completion and completion.get('status')!='INITIAL_NATIVE_DATA_AUDITED':completion=None
 payload=dict(status='FULL_INITIAL_COMPARISON_COMPLETE' if completion else 'MEASURED_CHECKPOINT_NOT_CAMPAIGN_COMPLETION',completion=completion,blocked=read(root/'tranche.json').get('blocked',[]) if (root/'tranche.json').exists() else [],points=points,limitations=['RTX5080 / modern Qwen3.5-4B Q4_K_M only.','Smokes and repeated confirmations are separated by panel/cohort.','A1/A2 execute INT8-code DP4A fallback, not hypothetical binary kernels.','Full draft cost uses disjoint instrumented host spans including catchup, generation and accept; asynchronous work may be charged at a later synchronization.','Sampled device-memory peak includes display/unrelated baseline and is not exact allocator peak.','Confidence intervals resample reused prompts/repetitions; no independent-training variance estimate.','Raw rates/host latency can shift between panels; use the remeasured same-panel Q4 ratios and matched-pair effects for comparisons.','Full-model HF/native untrained numeric parity gate failed historically and is not declared repaired. Native features/labels and direct acceptance/throughput are used; byte-exact integer operation checks do not prove full-model parity.'])
 followup_data=root/'clip-control-followup/data-completion.json'
 followup=read(followup_data) if followup_data.exists() else None
 if followup and followup.get('status')=='FOLLOWUP_NATIVE_DATA_AUDITED':
  payload['followup']=dict(receipt=reference(followup_data),data=followup);payload['status']='FULL_INITIAL_AND_CLIP_CONTROL_COMPARISON_COMPLETE'
  payload['untested_variants']=['Full-model calibrated fixed/learned-bound A1/A2 recovery remains untested; no general A1/A2 inferiority claim.','Fixed-versus-learned bounds are isolated only for compact G4/D8 initialized at1 with1024 matched updates. Other families, initializers and longer histories remain untested.']
 payload['analysis']=dict(generator=reference(out/'generator.py'),python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__,bootstrap=dict(seed=811,resamples=2000,method='Two-way cluster bootstrap: sample repetitions and whole prompt components; use the same sampled prompt set across sampled repetitions. Paired model/Q4 ratios; no independent-training variance.'),rates='Totals of actual returned tokens divided by summed plain full-request wall seconds or native predicted_ms; warmups excluded. Traced costs reported separately.')
 payload['frontier_policy']='Point estimates maximize acceptance and minimize full draft latency and actual bytes within the same panel/cohort/runtime/scope. Confidence-overlapping near ties remain in the table; this is not a deployment promotion or matched-history precision attribution.'
 if 'untested_variants' not in payload:payload['untested_variants']=['Full-model fixed/learned-bound A1/A2 recovery, including calibrated bound initializers, remains untested in this initial tranche. CUDA operation/gradient support alone is not a full-model quality result.','Fixed-vs-learned clipping recovery was not isolated; the initial learned recipe changes both range policy and training. No general A1/A2 inferiority or benefit specifically from learning bounds is established.']
 payload['failures_and_recoveries']=[dict(item='runtime03 generic CUDA gain op',reason='Tiny signed inputs flushed by inherited fast-math',resolution='Fused gained-stream runtime04 passes36 actual CUDA cases; controls remeasured, old series retained separately.'),dict(item='first mutation source snapshot',reason='Top-level exporter dependency omitted; import failed before any training',resolution='Complete dependency snapshot import-qualified; replacement serial job retained source/ownership receipts.')]
 payload['failures_and_recoveries'] += [dict(item='First post-DEV wrapper',reason='Snapshot root missing from import path; stopped before native work or fresh source access.',resolution='Wrapper imports qualified with empty inherited PYTHONPATH; new immutable snapshot used.'),dict(item='Second post-DEV header receipt',reason='Checker compiled, but receipt referenced a nonexistent overlay public header; stopped before GPU work or fresh source access.',resolution='Pinned the actual compiler public header; replacement passed all36 CUDA16-slot cases and12 native tokenizer comparisons before heldout lock/selection.')]
 payload['failures_and_recoveries'].append(dict(item='Full-model untrained numeric parity gate',reason='Known amplified float drift after six binary layers:38/45 F32 token agreement and42/45 BF16-input probe; provisional per-context gate failed despite isolated FFN relative error about1.24e-7.',resolution='Parity remains unresolved and is not claimed. Native feature/label ancestry is preserved, gained integer operations independently qualified, and actual native acceptance/throughput measured rather than inferred from target-output correctness.'))
 recovery=root/'confirmation-dev12/counter-recovery.json'
 if recovery.exists():payload['failures_and_recoveries'].append(dict(item='DEV live log sampling race',reason='A response returned before its cumulative statistics line was visible to the live sampler.',resolution='Every final reaped-server boundary reconciled one-for-one to HTTP response accepted/proposed counts; original progress and raw data preserved.',receipt=reference(recovery),corrected_rows=len(read(recovery)['corrections'])))
 fields=['id','cohort','scope','runtime_series','family','execution','quantizer_mode','protected_weight_formats','point_status','training_scope','training_updates','training_seconds','accepted','proposed','rounds','acceptance','acceptance_bootstrap95','prefix_survival','emitted_per_round','full_draft_ms','full_draft_ms_bootstrap95','request_tps','decode_tps','request_vs_q4','request_vs_q4_bootstrap95','decode_vs_q4','decode_vs_q4_bootstrap95','host_draft_span_subtraction_request_tps','model_bytes','combined_deployment_bytes','total_peak_device_MiB','incremental_peak_device_MiB_vs_target','total_cuda_model_buffers_MiB','incremental_cuda_model_buffers_MiB_vs_target','point_estimate_pareto','correct']
 with (out/'comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(points)
 with (out/'domains.csv').open('w') as f:
  domain_fields=['id','cohort','scope','runtime_series','arm','domain','requests','unique_prompts','accepted','proposed','acceptance','request_tps','decode_tps','request_vs_q4'];writer=csv.DictWriter(f,domain_fields,extrasaction='ignore');writer.writeheader()
  for point in points:
   for domain,values in point['domain_breakdown'].items():writer.writerow({**point,**values,'domain':domain,'scope':point['scope']})
 def fmt(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
 table='Measured native points; counts are accepted/proposed, model size is the exact drafter GGUF including metadata/private tensors. Target tied head/embedding is charged once in combined deployment bytes in JSON/CSV. Training updates refer to the named phase; ancestry and complete metadata are in JSON.\n\n'
 compact=[('Arm','arm'),('Phase updates','training_updates'),('Accepted/proposed','counts'),('Acceptance %','acceptance_percent'),('Full draft ms/round','full_draft_ms'),('Request tok/s','request_tps'),('Decode tok/s','decode_tps'),('Request / Q4 [95%]','request_ratio_ci'),('Decode / Q4','decode_vs_q4'),('Drafter MiB','MiB')]
 for group in sorted({(p['cohort'],p['runtime_series'],p['scope'],str(Path(p['panel']).relative_to(root))) for p in points}):
  table+='## '+' · '.join(group)+'\n\n| '+' | '.join(label for label,key in compact)+' |\n|'+'|'.join('---' for _ in compact)+'|\n'
  for point in [p for p in points if (p['cohort'],p['runtime_series'],p['scope'],str(Path(p['panel']).relative_to(root)))==group]:
   row=dict(point,counts=f"{point['accepted']}/{point['proposed']}",acceptance_percent=100*point['acceptance'] if point['acceptance'] is not None else None,MiB=point['model_bytes']/2**20)
   ci=point.get('request_vs_q4_bootstrap95');row['request_ratio_ci']=fmt(point.get('request_vs_q4'))+(' ['+', '.join(f'{x:.3f}' for x in ci)+']' if ci else '')
   table+='| '+' | '.join(fmt(row.get(key)) for label,key in compact)+' |\n'
  table+='\n'
 table+='## Precision and execution formats\n\nCandidate v2 A1/A2/A4/A8 FFN inputs execute native INT8-code DP4A with packed1-bit sign weights and F32 row affine/gain metadata. Legacy vanilla stores packed signs/F32 scales without affine gains or midpoints. A1 has two levels; A2 has three levels. Protected formats below are weight formats; their native backend arithmetic is unchanged. All target/drafter KV caches are F16.\n\n| Family | Bound policy | Midpoint/gain enabled | Protected weights |\n|---|---|---|---|\n'
 format_rows=set()
 for point in points:
  if not point['config']:continue
  features=point['config'].get('features',{});row=(point['family'],point['quantizer_mode'],str(features.get('midpoint'))+'/'+str(features.get('gain')),point['protected_weight_formats']);format_rows.add(row)
 for row in sorted(format_rows):table+='| '+' | '.join(row)+' |\n'
 table+='\n## Blocked and untested paths\n\n'+'\n'.join('- '+(json.dumps(item) if not isinstance(item,str) else item) for item in payload['blocked']+payload['untested_variants'])+'\n\n## Recorded failures and recovery\n\n'+'\n'.join('- '+item['item']+': '+item['reason']+' '+item['resolution'] for item in payload['failures_and_recoveries'])+'\n'
 (out/'full-table.md').write_text(table)
 colors={'G8':'#3269a8','G4':'#38976b','G2':'#d98c22','G1':'#bd4d6c','external':'#676767'}
 def color(p):return next((v for k,v in colors.items() if p['family'].startswith(k)),'#7564ab')
 display=[p for p in points if p['scope']=='confirmation' and p['runtime_series'].endswith('-04')] or [p for p in points if p['runtime_series'].endswith('-04')] or points
 groups=sorted({p['cohort'] for p in display}) or ['none'];fig,axes=plt.subplots(1,len(groups),figsize=(7*len(groups),5),squeeze=False)
 for ax,group in zip(axes[0],groups):
  for p in display:
   if p['cohort']!=group or p['full_draft_ms'] is None or p['acceptance'] is None:continue
   size=25+max(0,(p['model_bytes'] or 0)/1e7);ax.scatter(p['full_draft_ms'],100*p['acceptance'],s=size,color=color(p),alpha=.7,marker='*' if p.get('point_estimate_pareto') else 'o')
   if p.get('point_estimate_pareto') or p['arm']=='q4' or p['arm'] in ('G8D8-1024','G4D8-1024','G2D8-1024','G1D8-1024'):ax.annotate(p['arm'],(p['full_draft_ms'],100*p['acceptance']),fontsize=7)
  ax.set(xlabel='Complete native draft host ms/round (traced)',ylabel='Accepted / proposed (%)',title=group,ylim=(0,100));ax.grid(alpha=.2)
 from matplotlib.lines import Line2D
 handles=[Line2D([],[],marker='o',linestyle='',color=colors[k],label='Gate/up A'+k[1:] if k.startswith('G') else 'External target/Q4') for k in colors]
 handles.append(Line2D([],[],marker='o',linestyle='',color='#7564ab',label='Vanilla packed W1/A8'))
 handles += [axes[0,0].scatter([],[],s=25+mib*2**20/1e7,color='#888888',alpha=.65,label=str(mib)+' MiB raw GGUF') for mib in (200,500,800)]
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.1),ncol=3,fontsize=7)
 fig.suptitle('Logical A1/A2/A4/A8: native INT8-code DP4A · packed W1 FFN · stars: point-estimate frontier',fontsize=10)
 fig.tight_layout();fig.savefig(out/'acceptance-latency.png',dpi=160,bbox_inches='tight');fig.savefig(out/'acceptance-latency.svg',bbox_inches='tight');plt.close(fig)
 selected=[p for p in display if p.get('request_vs_q4') is not None];heights=[max(1,sum(p['cohort']==g for p in selected)) for g in groups];height=max(5,2+.34*sum(heights))
 fig,axes=plt.subplots(len(groups),1,figsize=(11,height),squeeze=False,gridspec_kw={'height_ratios':heights})
 for ax,group in zip(axes[:,0],groups):
  bars=[p for p in selected if p['cohort']==group];y=np.arange(len(bars))
  for offset,key,label,hatch in [(-.18,'request_vs_q4','Full request',None),(.18,'decode_vs_q4','Decode','///')]:
   values=[p[key] for p in bars];ax.barh(y+offset,values,height=.32,color=[color(p) for p in bars],alpha=.8 if hatch is None else .4,hatch=hatch,label=label)
   for i,p in enumerate(bars):
    ci=p.get(key+'_bootstrap95')
    if ci:ax.plot(ci,[y[i]+offset]*2,color='#20252a',lw=1)
  ax.axvline(1,color='black',lw=1);ax.set_yticks(y,[p['arm']+(' ['+Path(p['panel']).name+']' if sum(q['arm']==p['arm'] for q in bars)>1 else '') for p in bars],fontsize=8);ax.invert_yaxis();ax.set(xlabel='Throughput / matched DFlash Q4_K_M',title=group+' · native execution');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
 fig.tight_layout();fig.savefig(out/'throughput.png',dpi=160,bbox_inches='tight');fig.savefig(out/'throughput.svg',bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(len(groups),2,figsize=(13,4.5*len(groups)),squeeze=False)
 for row,group in zip(axes,groups):
  for p in display:
   if p['cohort']!=group:continue
   if p['acceptance'] is not None:row[0].scatter((p['model_bytes'] or 0)/2**20,100*p['acceptance'],color=color(p),alpha=.6)
   if p['total_peak_device_MiB']:
    row[1].scatter(p['total_peak_device_MiB'],p['request_tps'],color=color(p),alpha=.6)
    if p['arm'] in ('target','q4') and p['scope']=='confirmation':row[1].annotate(p['arm'],(p['total_peak_device_MiB'],p['request_tps']),fontsize=8)
  row[0].set(xlabel='Actual drafter GGUF MiB (private tensors included)',ylabel='Acceptance (%)',title=group,ylim=(0,100));row[1].set(xlabel='Sampled total device peak MiB',ylabel='Full request tokens/s',title=group)
 fig.tight_layout();fig.savefig(out/'size-memory.png',dpi=160,bbox_inches='tight');fig.savefig(out/'size-memory.svg',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,6));curves={}
 for p in points:
  match=re.fullmatch(r'(G[1248]D[1248](?:-[a-z-]+)?)-(zero|256|1024|1536|2048)',p['arm'])
  if p['cohort']=='natural12' and p['training_updates'] in (0,256,1024,1536,2048) and p['acceptance'] is not None and match:
   key=(p['runtime_series'],match[1]);prior=curves.setdefault(key,{}).get(p['training_updates'])
   if prior is None or p['repetitions']>prior['repetitions']:curves[key][p['training_updates']]=p
 for (series,name),values in curves.items():
  curve=sorted(values.values(),key=lambda p:p['training_updates']);style='--' if series.endswith('-03') else ':' if 'fixed' in name else '-'
  ax.plot([p['training_updates'] for p in curve],[100*p['acceptance'] for p in curve],marker='D' if 'fixed' in name else 'o',linestyle=style,label=name+' / '+series.rsplit('-',1)[-1],color=color(curve[0]),alpha=.75)
 ax.set(xlabel='Matched additional updates from common frozen foundation',ylabel='Native acceptance (%)',title='Recovery on consumed natural12; runtime series remain distinct');ax.grid(alpha=.2)
 if curves:ax.legend(fontsize=6,ncol=3)
 fig.tight_layout();fig.savefig(out/'recovery.png',dpi=160,bbox_inches='tight');fig.savefig(out/'recovery.svg',bbox_inches='tight');plt.close(fig)
 grid_points=[p for p in points if Path(p['panel']).name=='zero-panel-runtime04'];widths=(8,4,2,1)
 fig,axes=plt.subplots(1,3,figsize=(14,4.8))
 for ax,key,title,multiplier in zip(axes,('acceptance','request_vs_q4','full_draft_ms'),('Acceptance %','Request throughput / Q4','Complete draft host ms/round'),(100,1,1)):
  matrix=np.full((4,4),np.nan)
  for i,g in enumerate(widths):
   for j,d in enumerate(widths):
    found=next((p for p in grid_points if p['arm']==f'G{g}D{d}-zero'),None)
    if found and found.get(key) is not None:matrix[i,j]=found[key]*multiplier
  maximum=1 if key=='request_vs_q4' else float(np.nanmax(matrix))*1.05 if np.isfinite(matrix).any() else 1
  image=ax.imshow(np.ma.masked_invalid(matrix),cmap='viridis',vmin=0,vmax=maximum);ax.set_xticks(range(4),widths);ax.set_yticks(range(4),widths);ax.set(xlabel='Down logical activation bits',ylabel='Gate/up logical activation bits',title=title)
  for i in range(4):
   for j in range(4):
    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white',fontsize=10,bbox=dict(facecolor='black',alpha=.25,edgecolor='none'))
  fig.colorbar(image,ax=ax,shrink=.75)
 fig.suptitle('Zero precision-stage updates after frozen foundation · 4×4 diagnostic / 3 DEV · INT8-code DP4A',fontsize=11);fig.tight_layout();fig.savefig(out/'precision-grid.png',dpi=160,bbox_inches='tight');fig.savefig(out/'precision-grid.svg',bbox_inches='tight');plt.close(fig)
 # Standalone interactive SVG and sortable evidence table: no network/CDN/server.
 data=json.dumps([{k:p.get(k) for k in fields+['arm','panel','immediate_parent']} for p in points]).replace('</','<\\/');columns=json.dumps(fields)
 body='''<!doctype html><html><head><meta charset="utf-8"><title>DFlash precision comparison</title><style>body{font:14px system-ui;margin:24px;color:#172232;background:#f6f8fb}h1{font-size:24px}select,input{padding:7px;margin:5px}svg{width:100%;max-width:1000px;background:white;border:1px solid #ddd}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:7px;border-bottom:1px solid #d6dce6;text-align:right}th{cursor:pointer;position:sticky;top:0;background:#e5ebf3}.scroll{overflow:auto;max-height:650px}#tip{position:fixed;display:none;background:#172232;color:white;padding:10px;max-width:400px;pointer-events:none}img{max-width:100%}</style></head><body><h1>DFlash precision campaign — measured evidence</h1><p id="status">A checkpoint chart is incomplete until the campaign record marks full comparison complete.</p><p>RTX 5080 · Qwen3.5 4B Q4_K_M target · depth 2 · greedy target-sample-and-match. A1/A2 use measured INT8-code DP4A execution. Traced host spans include draft generation, catch-up and acceptance; timings are diagnostic.</p><label>Panel <select id="panel"></select></label><label>X <select id="x"><option value="full_draft_ms">Complete draft ms/round</option value="model_bytes">Actual model bytes</option><option value="total_peak_device_MiB">Sampled total peak MiB</option><option value="training_updates">Segment updates</option></select></label><label>Y <select id="y"><option value="acceptance">Acceptance</option><option value="request_vs_q4">Request throughput / Q4</option><option value="decode_vs_q4">Decode throughput / Q4</option><option value="request_tps">Request tokens/s</option></select></label><svg id="plot" viewBox="0 0 1000 540"></svg><div id="tip"></div><p>Missing measurements are omitted from the selected plot, never plotted as zero. Bubble radius indicates actual GGUF bytes. Hover for counts, lineage and execution details. Click a column to sort.</p><input id="filter" placeholder="Filter rows"><div class="scroll"><table id="table"></table></div>'''
 body=body.replace('A checkpoint chart is incomplete until the campaign record marks full comparison complete.', 'Full initial comparison complete; final DEV and heldout cohorts remain separate.' if completion else 'Measured checkpoint only; full initial confirmation and heldout are still pending.')
 if followup:body=body.replace('Full initial comparison complete; final DEV and heldout cohorts remain separate.','Full initial comparison and bounded clipping control complete; distinct heldout cohorts remain separate.')
 body+='<p>'+html.escape(payload['frontier_policy'])+'</p><p>'+html.escape(json.dumps(payload['blocked']))+'</p>'
 body+='<script>const points='+data+',columns='+columns+''';let sort="id";const panel=document.querySelector('#panel'),sx=document.querySelector('#x'),sy=document.querySelector('#y'),tip=document.querySelector('#tip');panel.innerHTML='<option value="">All panels (mixed diagnostics/cohorts)</option>'+[...new Set(points.map(p=>p.panel))].map(p=>'<option>'+p+'</option>').join('');const defaultPanel=points.find(p=>p.panel.endsWith('/confirmation-dev12'));if(defaultPanel)panel.value=defaultPanel.panel;function draw(){let rows=points.filter(p=>(!panel.value||p.panel===panel.value)&&JSON.stringify(p).toLowerCase().includes(document.querySelector('#filter').value.toLowerCase()));const valid=rows.filter(p=>p[sx.value]!=null&&p[sy.value]!=null);let maxX=Math.max(1,...valid.map(p=>p[sx.value]))*1.08,maxY=Math.max(.01,...valid.map(p=>p[sy.value]))*1.1;const svg=document.querySelector('#plot');svg.innerHTML='<path d="M70 20V470H970" stroke="#789" fill="none"/><text x="500" y="528" text-anchor="middle">'+sx.selectedOptions[0].text+'</text><text x="20" y="240" transform="rotate(-90 20 240)" text-anchor="middle">'+sy.selectedOptions[0].text+'</text>';for(let i=0;i<=5;i++){let x=70+i*180,y=470-i*85;svg.innerHTML+='<text x="'+x+'" y="491" text-anchor="middle">'+(maxX*i/5).toPrecision(3)+'</text><text x="62" y="'+y+'" text-anchor="end">'+(maxY*i/5).toPrecision(3)+'</text>';}valid.forEach(p=>{let c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',70+900*p[sx.value]/maxX);c.setAttribute('cy',470-425*p[sy.value]/maxY);c.setAttribute('r',Math.max(5,Math.sqrt((p.model_bytes||1)/1e7)));c.setAttribute('fill',p.family.startsWith('G1')?'#bd4d6c':p.family.startsWith('G2')?'#d98c22':p.family.startsWith('G4')?'#38976b':p.family.startsWith('G8')?'#3269a8':p.family==='external'?'#676767':'#7564ab');c.setAttribute('opacity','.65');c.onmousemove=e=>{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=p.id+' | '+p.cohort+' | '+p.runtime_series+' | '+p.scope+' | '+p.family+' | '+p.execution+' | '+p.accepted+'/'+p.proposed+' accepted | '+p.request_tps.toFixed(2)+' request tok/s | '+p.protected_weight_formats+' | parent '+JSON.stringify(p.immediate_parent)};c.onmouseleave=()=>tip.style.display='none';svg.append(c)});rows.sort((a,b)=>typeof a[sort]==='number'?(a[sort]||0)-(b[sort]||0):String(a[sort]).localeCompare(String(b[sort])));const esc=x=>String(x??'—').replaceAll('&','&amp;').replaceAll('<','&lt;');document.querySelector('#table').innerHTML='<thead><tr>'+columns.map(k=>'<th data-key="'+k+'">'+k+'</th>').join('')+'</tr></thead><tbody>'+rows.map(p=>'<tr>'+columns.map(k=>'<td>'+esc(typeof p[k]==='number'?Number(p[k].toPrecision(6)):p[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';document.querySelectorAll('th').forEach(t=>t.onclick=()=>{sort=t.dataset.key;draw()})}for(const s of [panel,sx,sy,document.querySelector('#filter')])s.oninput=draw;draw();</script>'''
 body+='<p>Full ancestry and additional metrics are preserved in comparison.json and CSV. Five exportable static figures are included alongside this standalone interactive file.</p>'
 (out/'comparison.html').write_text(body+'</body></html>')
 (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n');return payload
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();result=build(a.root,a.out);print('Native points:',len(result['points']))
