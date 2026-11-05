"""Attach explicitly indexed formal evidence to matching final native readouts."""
import json
from .common import *

def merge(rows):
 index_path=R/'publication-index.json'
 if not index_path.exists():return rows
 index=json.load(open(index_path));summary=json.load(open(index['summary']))
 for row in rows:
  arm=row['arm'];final=max(x['updates'] for x in rows if x['arm']==arm)
  row['status']=index.get('arm_status',{}).get(arm,row['status'])
  if row['updates']!=final:continue
  for key,prefix in [('dev_panel','dev'),('request_panel','final'),('selection_panel','selection')]:
   panel=summary['native'].get(index.get(key,''),{});entry=panel.get('arms',{}).get(arm)
   if not entry:continue
   freeze=json.load(open(panel['freeze']['path']));measured=freeze['arms'][arm]
   if measured['sha256']!=row['model']['sha256']:
    variants=json.load(open(R/'deployment-variants.json')) if (R/'deployment-variants.json').exists() else {}
    variant=variants.get(arm,{})
    if variant.get('model',{}).get('sha256')!=measured['sha256'] or variant.get('parent',{}).get('sha256')!=row['model']['sha256']:raise ValueError('publication model/readout ancestry differs '+arm)
    row['measured_deployment_model']=measured
    if variant['spec'].get('clip_bits_storage'):row['clip_storage']='IEEE F16 bits in UINT16 GGUF metadata; F32 runtime parameter'
   row.update({prefix+'_request_tok_s':entry['request_tok_s'],prefix+'_accepted':entry['accepted'],prefix+'_proposed':entry['proposed'],prefix+'_acceptance':entry['acceptance'],prefix+'_emitted_per_round':entry['emitted_per_round'],prefix+'_request_cohort':panel['cohort'],prefix+'_request_responses':entry['responses'],prefix+'_request_repetitions':entry['timing_repetitions'],prefix+'_request_evidence':panel['source']['path'],prefix+'_vs_q4':entry.get('vs_q4'),prefix+'_prefix_counts':entry['prefix_counts']})
   if prefix=='final':
    if entry['instrumented'] or entry['timing_repetitions']<5 or freeze['warmup_policy']['per_server_cell']<2:raise ValueError('formal final protocol not met')
    allocation=entry['allocations'][0]['allocation'];model=allocation['model_buffers'];cache=allocation['cache_buffers']
    row.update(actual_gpu_model_bytes=sum(v for k,v in model.items() if k.startswith('CUDA') and k!='CUDA_Host'),actual_cpu_model_bytes=sum(v for k,v in model.items() if k.startswith('CPU')),draft_cache_bytes=sum(x['total_reported_bytes'] for x in cache[1:]),draft_rotation_bytes=sum(x['reported_bytes'] for x in allocation['rotation_buffers']),sampled_combined_device_peak_bytes=entry['sampled_device_peak_bytes'],allocation_evidence=entry['allocations'][0]['cell'],allocation_rounding='Native logs0.01MiB; persistent tensor bytes exact. Peak includes target/desktop/workspace.')
    target=panel['arms']['target']['allocations'][0]['allocation']['model_buffers']
    row['private_gpu_model_bytes']=row['actual_gpu_model_bytes']-sum(v for k,v in target.items() if k.startswith('CUDA') and k!='CUDA_Host')
    row['private_cpu_mapped_model_bytes']=row['actual_cpu_model_bytes']-sum(v for k,v in target.items() if k.startswith('CPU'))
  replay=summary['replay'].get(index.get('gpu_panel',''),{});entry=replay.get('arms',{}).get(arm)
  if entry:
   row['clean_gpu_forward_ms']=entry['clean_gpu_forward_ms'];row['gpu_forward_evidence']=replay['source']['path'];row['fixed_replay_forward_host_ms']=entry['fixed_replay_forward_host_ms']
  trace=summary['native'].get(index.get('trace_panel',''),{});entry=trace.get('arms',{}).get(arm)
  if entry:
   if not entry['instrumented'] or entry['timing_repetitions']<5:raise ValueError('formal trace repetitions absent')
   row['full_draft_host_ms']=entry['full_draft_host_ms_per_round'];row['full_draft_host_cohort']=trace['cohort'];row['full_draft_host_evidence']=trace['source']['path']
 controls=index.get('matched_controls',{})
 lookup={(r['arm'],r['updates']):r for r in rows}
 for row in rows:
  control=controls.get(row['arm'],'H0');c=lookup.get((control,row['updates']))
  row['matched_recovery_control']=control
  if c:row['ce_gap_matched_control']=row['ce']-c['ce']
 return rows
