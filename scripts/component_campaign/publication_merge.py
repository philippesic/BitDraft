"""Attach explicitly indexed formal evidence to matching final native readouts."""
import copy,json
from .common import *

def merge(rows):
 index_path=R/'publication-index.json'
 if not index_path.exists():return rows
 index=json.load(open(index_path));summary=json.load(open(index['summary']))
 def panel_for(kind,key,arm):
  names=[index.get(key,''),*index.get(key+'_fallbacks',[])]
  return next((summary[kind][n] for n in names if arm in summary[kind].get(n,{}).get('arms',{})),{})
 from .archive import reference
 inventory=json.load(open(R/'reference-tensor-inventory.json'))
 base=next(r for r in rows if r['arm']=='H0' and r['updates']==0)
 for name,key,model in [('B0','B0',B/'model-final.gguf'),('q4','external-q4',Q4)]:
  row=copy.deepcopy(base);row.update(arm=name,updates=0,component_spec={'reference':name},model=reference(model),model_path=str(model),model_sha256=reference(model)['sha256'],deployed_gguf_bytes=reference(model)['bytes'],tensor_bytes=inventory[key]['tensor_bytes'],native={},status='Fixed immutable reference',training_peak_allocator_bytes=None,training_update_seconds=0.,supervised_positions=0,full_recovery_state_status='Original paid B0 retained' if name=='B0' else 'Frozen external reference',reference_tensor_map=str(R/'reference-tensor-inventory.json'))
  row.update(training_peak_reserved_bytes=None,training_peak_host_rss_bytes=None,readout_elapsed_trainer_wall_seconds=None,final_trainer_process_wall_seconds=None)
  for field in ('accepted','proposed','rounds','output_tokens','request_seconds','request_tok_s','acceptance','accepted_per_round','emitted_per_round','target_identical','responses','native_acceptance_gap_matched_control'):row[field]=None
  if name=='q4':
   for field in ('ce','agreement','mean_rank','mean_margin','ce_gap_matched_control'):row[field]=None
   row.update(ffn_weights='Q4_K_M; full tensor map linked',gate_up_activation='Native Q4 kernel contract',down_activation='Native Q4 kernel contract',fusion_weights='See frozen tensor map',attention_qk_weights='See frozen tensor map',attention_vo_weights='See frozen tensor map',trainable_policy='Frozen external DFlash Q4_K_M reference',source_model_sha256=reference(model)['sha256'],data_manifest_sha256=None,B0_checkpoint_sha256=None,training_resolved_evidence=None,training_sequences=None,row_scale_storage='Native quant-block format; see tensor map',midpoint_storage='No W1 midpoint group',positive_gain_storage='No W1 gain group',clip_storage='No W1 clipping group')
  else:row.update(trainable_policy='Frozen learned-small2048 B0 reference; no extra updates',training_resolved_evidence=str(B/'model-final.json'))
  rows.append(row)
 for row in rows:
  arm=row['arm'];final=max(x['updates'] for x in rows if x['arm']==arm)
  row['status']=index.get('arm_status',{}).get(arm,row['status'])
  if row['updates']!=final:continue
  for key,prefix in [('dev_panel','dev'),('request_panel','final'),('selection_panel','selection')]:
   panel=panel_for('native',key,arm);entry=panel.get('arms',{}).get(arm)
   if not entry:continue
   freeze=json.load(open(panel['freeze']['path']));measured=freeze['arms'][arm]
   if measured['sha256']!=row['model']['sha256']:
    variants=json.load(open(R/'deployment-variants.json')) if (R/'deployment-variants.json').exists() else {}
    variant=variants.get(arm,{})
    if variant.get('model',{}).get('sha256')!=measured['sha256'] or variant.get('parent',{}).get('sha256')!=row['model']['sha256']:raise ValueError('publication model/readout ancestry differs '+arm)
    row['measured_deployment_model']=measured
    row['deployed_gguf_bytes']=measured['bytes']
    if variant['spec'].get('clip_bits_storage'):row['clip_storage']='IEEE F16 bits in UINT16 GGUF metadata; F32 runtime parameter'
   row.update({prefix+'_request_tok_s':entry['request_tok_s'],prefix+'_accepted':entry['accepted'],prefix+'_proposed':entry['proposed'],prefix+'_acceptance':entry['acceptance'],prefix+'_emitted_per_round':entry['emitted_per_round'],prefix+'_request_cohort':panel['cohort'],prefix+'_request_responses':entry['responses'],prefix+'_request_repetitions':entry['timing_repetitions'],prefix+'_request_evidence':panel['source']['path'],prefix+'_vs_q4':entry.get('vs_q4'),prefix+'_prefix_counts':entry['prefix_counts']})
   row.update({prefix+'_request_vs_matched_control':entry.get('vs_matched_control'),prefix+'_acceptance_gap_matched_control':entry.get('acceptance_gap_matched_control'),prefix+'_matched_control':entry.get('matched_control')})
   if prefix=='final':
    if entry['instrumented'] or entry['timing_repetitions']<5 or freeze['warmup_policy']['per_server_cell']<2:raise ValueError('formal final protocol not met')
    row.update(final_native_host_library_sha256=freeze['runtime']['libllama.so.0.8.0']['sha256'],final_native_CUDA_library_sha256=freeze['runtime']['libggml-cuda.so.0.28.0']['sha256'])
   if prefix=='final' or row.get('allocation_evidence') is None or row.get('allocation_scope')=='short_native_readout':
    allocation=entry['allocations'][0]['allocation'];model=allocation['model_buffers'];cache=allocation['cache_buffers']
    row.update(actual_gpu_model_bytes=sum(v for k,v in model.items() if k.startswith('CUDA') and k!='CUDA_Host'),actual_cpu_model_bytes=sum(v for k,v in model.items() if k.startswith('CPU')),draft_cache_bytes=sum(x['total_reported_bytes'] for x in cache[1:]),draft_rotation_bytes=sum(x['reported_bytes'] for x in allocation['rotation_buffers']),sampled_combined_device_peak_bytes=entry['sampled_device_peak_bytes'],allocation_evidence=entry['allocations'][0]['cell'],allocation_rounding='Native logs0.01MiB; persistent tensor bytes exact. Peak includes target/desktop/workspace.')
    row.update(allocation_scope=prefix,runtime_compute_buffer_history=allocation['compute_buffer_history'])
    target=panel['arms']['target']['allocations'][0]['allocation']['model_buffers']
    row['private_gpu_model_bytes']=row['actual_gpu_model_bytes']-sum(v for k,v in target.items() if k.startswith('CUDA') and k!='CUDA_Host')
    row['private_cpu_mapped_model_bytes']=row['actual_cpu_model_bytes']-sum(v for k,v in target.items() if k.startswith('CPU'))
  if row.get('draft_cache_bytes') is not None:row['model_cache_payload_bytes']=row['tensor_bytes']+row['draft_cache_bytes']+row.get('draft_rotation_bytes',0)
  replay=panel_for('replay','gpu_panel',arm);entry=replay.get('arms',{}).get(arm)
  if entry:
   prefix='' if entry['timing_repetitions']>=5 else 'selection_'
   row[prefix+'clean_gpu_forward_ms']=entry['clean_gpu_forward_ms'];row[prefix+'gpu_forward_evidence']=replay['source']['path'];row[prefix+'fixed_replay_forward_host_ms']=entry['fixed_replay_forward_host_ms'];row[prefix+'gpu_forward_repetitions']=entry['timing_repetitions'];row[prefix+'gpu_vs_q4']=entry.get('GPU_vs_q4');row[prefix+'gpu_vs_H0']=entry.get('GPU_vs_H0')
   row[prefix+'gpu_vs_matched_control']=entry.get('GPU_vs_matched_control')
  trace=panel_for('native','trace_panel',arm);entry=trace.get('arms',{}).get(arm)
  if entry:
   if not entry['instrumented']:raise ValueError('trace instrumentation absent')
   prefix='' if entry['timing_repetitions']>=5 else 'selection_'
   row[prefix+'full_draft_host_ms']=entry['full_draft_host_ms_per_round'];row[prefix+'full_draft_host_cohort']=trace['cohort'];row[prefix+'full_draft_host_evidence']=trace['source']['path'];row[prefix+'full_draft_host_repetitions']=entry['timing_repetitions']
   row[prefix+'draft_prefill_host_ms_per_request']=entry.get('draft_prefill_host_ms_per_request');row[prefix+'full_draft_all_phases_host_ms_per_round']=entry.get('full_draft_all_phases_host_ms_per_round');row[prefix+'full_draft_host_scope']=entry['host_scope']
   row[prefix+'full_draft_host_vs_q4']=entry.get('host_vs_q4');row[prefix+'full_draft_host_vs_H0']=entry.get('host_vs_H0')
   row[prefix+'full_draft_host_vs_matched_control']=entry.get('host_vs_matched_control')
 controls={'H4R2':'H6R2','H6R2':'H6R2','HE4R2':'HE6R2','HE6R2':'HE6R2','H0broad':'H0broad','Csmallbroad':'H0broad',**index.get('matched_controls',{})}
 lookup={(r['arm'],r['updates']):r for r in rows}
 for row in rows:
  if row['arm'] in ('B0','q4'):
   row.update(matched_recovery_control=None,ce_gap_matched_control=None,native_acceptance_gap_matched_control=None);continue
  control=controls.get(row['arm'],'H0');c=lookup.get((control,row['updates']))
  row['matched_recovery_control']=control
  if c and row['ce'] is not None:row['ce_gap_matched_control']=row['ce']-c['ce']
  elif row['arm'] in controls:row['ce_gap_matched_control']=None
  if c and row.get('acceptance') is not None and c.get('acceptance') is not None:row['native_acceptance_gap_matched_control']=row['acceptance']-c['acceptance']
  elif row['arm'] in controls:row['native_acceptance_gap_matched_control']=None
 target_bytes=inventory['target']['tensor_bytes']
 shared_vocab=sum(t['bytes'] for t in inventory['target']['tensors'] if t['name']=='token_embd.weight')
 for row in rows:
  spec=row.get('component_spec',{})
  row.update(immutable_target_tensor_bytes=target_bytes,combined_model_tensor_bytes=target_bytes+row['tensor_bytes'],draft_reused_target_vocabulary_bytes=shared_vocab if not spec.get('head') or not spec.get('embedding') else 0)
  if row.get('model_cache_payload_bytes') is not None:
   row['combined_model_and_draft_cache_payload_bytes']=target_bytes+row['model_cache_payload_bytes']
  row['combined_payload_scope']='Immutable target tensors plus private draft tensors and, when measured, draft cache/rotation. Target cache, compute workspace, duplicate device copies and mapped residency remain separate; this is not sampled device usage.'
 return rows
