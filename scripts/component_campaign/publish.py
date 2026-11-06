"""Publish measured readouts, explicit missing fields and standalone interactive charts."""
import csv,json,math,re,statistics
from pathlib import Path
import numpy as np
from .common import *

def native_metrics(folder,arm):
 path=folder/'result.json'
 if not path.exists():return {}
 rows=[x for x in json.load(open(path))['rows'] if x['arm']==arm and not x.get('warmup') and not x.get('instrumented')]
 if not rows:return {}
 n=lambda key:sum(x['native'].get(key,0) for x in rows)
 wall=sum(x['wall_seconds'] for x in rows);tokens=sum(x['tokens'] for x in rows);rounds=n('rounds')
 result=dict(accepted=n('accepted'),proposed=n('proposed'),rounds=rounds,output_tokens=tokens,request_seconds=wall,request_tok_s=tokens/wall,acceptance=n('accepted')/n('proposed') if n('proposed') else None,accepted_per_round=n('accepted')/rounds if rounds else None,emitted_per_round=tokens/rounds if rounds else None,target_identical=all(x['correct'] for x in rows),responses=len(rows),prefix_counts=[sum(x['native'].get('prefix_survival_counts',[0,0])[i] for x in rows) for i in (0,1)])
 return result

def collect():
 rows=[]
 from .metrics import allocations
 retained=json.load(open(R/'frontier-retention.json')).get('states',{}) if (R/'frontier-retention.json').exists() else {}
 for directory in sorted((R/'arms').glob('*')):
  if directory.name.endswith(('-pilot','-confirm')):continue
  path=directory/'readouts.json'
  if not path.exists():continue
  result=json.load(open(directory/'result.json')) if (directory/'result.json').exists() else {}
  spec=json.load(open(directory/'spec.json'));records=json.load(open(path))
  if directory.name=='H0':records=json.load(open(R/'arms/H0-pilot/readouts.json'))[:1]+records
  logs=[directory/'updates.jsonl']
  if directory.name in ('H6','E6'):logs=[R/'arms/H0-pilot/updates.jsonl',R/'arms/H0/updates.jsonl']
  elif directory.name=='H0':logs=[R/'arms/H0-pilot/updates.jsonl',*logs]
  resolved_path=directory/'resolved.json'
  if directory.name in ('H6','E6'):resolved_path=R/'arms/H0/resolved.json'
  resolved={}
  if resolved_path.exists():
   resolved=json.load(open(resolved_path));lineage=resolved.get('lineage',{})
   logs=[*[Path(p)/'updates.jsonl' for p in lineage.get('resume_source_arms',[])],*logs]
   # Resume pilots are physical work, not extra precision arms. Carry their
   # actual standard readouts into the continued arm without inventing exports.
   previous=[]
   for parent in lineage.get('resume_source_arms',[]):
    source=Path(parent)/'readouts.json'
    if source.exists():previous += [p for p in json.load(open(source)) if p['update'] in (0,256,1024,4096)]
   records=list({p['update']:p for p in previous+records}.values());records.sort(key=lambda p:p['update'])
  logs=list(dict.fromkeys(logs))
  events=sorted([json.loads(line) for log in logs if log.exists() for line in log.read_text().splitlines()],key=lambda x:x['update']);seen=set();cursor=0
  for point in records:
   while cursor<len(events) and events[cursor]['update']<=point['update']:
    event=events[cursor];seen.update((event['sequence'],event['anchor']+j-1) for j in range(1,16));cursor+=1
   step=point['update'];sidecar=Path(point['model']['path']).with_suffix('.component.json')
   component=json.load(open(sidecar));nm=native_metrics(R/f'smoke-{directory.name}-{step}',directory.name+'-'+str(step))
   row=dict(arm=directory.name,updates=step,component_spec=spec,head=spec.get('head','target-shared Q6_K'),embedding=spec.get('embedding','target-shared Q6_K'),tied=bool(spec.get('tied')),k=spec.get('kv',['f16','f16'])[0],v=spec.get('kv',['f16','f16'])[1],scalar_f16_groups=','.join(spec.get('scalar',{})),ce=point['dev']['ce'],agreement=point['dev']['agreement'],mean_rank=point['dev']['mean_rank'],mean_margin=point['dev']['mean_margin'],tensor_bytes=component['tensor_bytes'],training_peak_allocator_bytes=point['peak_training_bytes'],training_update_seconds=point['progress']['update_seconds'],supervised_positions=point['progress']['supervised_tokens'],unique_label_positions=result.get('unique_label_positions'),native=nm,status='recovering' if result.get('status')!='COMPLETE' else 'window complete; frontier decision pending',source_readout=str(path),model=point['model'],clean_gpu_forward_ms=None,full_draft_host_ms=None,final_request_tok_s=None,final_accepted=None,final_proposed=None,actual_gpu_model_bytes=None,actual_cpu_model_bytes=None,draft_cache_bytes=None)
   row.update(physical_updates=0 if directory.name in ('H6','E6') else step,training_peak_provenance='Measured H0 identical frozen-forward reuse' if directory.name in ('H6','E6') else 'Measured current trainer',unique_supervised_positions_exposed=len(seen),repeated_supervised_positions=max(0,point['progress']['supervised_tokens']-len(seen)),admitted_train_label_pool=result.get('unique_label_positions',resolved.get('unique_label_positions',1762)),unique_coverage_source=[str(p) for p in logs])
   wall_parents=resolved.get('lineage',{}).get('resume_source_arms',[])
   if directory.name=='H0' and not wall_parents:wall_parents=[str(R/'arms/H0-pilot')]
   parent_wall=sum(json.load(open(Path(p)/'result.json')).get('process_wall_seconds',0) for p in wall_parents if (Path(p)/'result.json').exists())
   row['readout_elapsed_trainer_wall_seconds']=point.get('elapsed_process_wall_seconds')
   if row['readout_elapsed_trainer_wall_seconds'] is not None:row['readout_elapsed_trainer_wall_seconds']+=parent_wall
   row['final_trainer_process_wall_seconds']=(result.get('process_wall_seconds') or 0)+parent_wall if step==result.get('progress',{}).get('updates') and result.get('process_wall_seconds') is not None else None
   row['trainer_wall_scope']='Measured trainer process including setup, DEV, exports, archive/checkpoint I/O and update compute; resumed pilot processes added where recorded. Native evaluation/capture/supervisor gaps separate. Historical intermediate elapsed wall unrecorded, left null.'
   vocabulary_bytes=sum(v['bytes'] for k,v in component['tensors'].items() if k in ('output.weight','token_embd.weight'))
   row.update(deployed_gguf_bytes=point['model']['bytes'],private_vocabulary_tensor_bytes=vocabulary_bytes,on_disk_duplicate_private_vocabulary_bytes=component['tensors'].get('token_embd.weight',{}).get('bytes',0) if spec.get('head') and spec.get('embedding') and not spec.get('tied') else 0,private_vocabulary_note='Target table remains loaded; private vocabulary bytes add to combined deployment. Tied variants contain one output tensor and no duplicate token_embd tensor.')
   config=resolved.get('config',{})
   row.update(source_model_sha256=config.get('source',{}).get('sha256'),data_manifest_sha256=config.get('data',{}).get('sha256'),B0_checkpoint_sha256=config.get('parent',{}).get('sha256'),target_sha256=config.get('evaluation',{}).get('target',{}).get('sha256'),training_resolved_evidence=str(resolved_path),training_sequences=resolved.get('unique_sequences'))
   native_freeze=R/f'smoke-{directory.name}-{step}'/'freeze.json'
   if native_freeze.exists() and (native_freeze.parent/'result.json').exists():
    native_refs=json.load(open(native_freeze))['runtime'];row.update(readout_native_host_library_sha256=native_refs['libllama.so.0.8.0']['sha256'],readout_native_CUDA_library_sha256=native_refs['libggml-cuda.so.0.28.0']['sha256'])
    cell=native_freeze.parent/('00-'+directory.name+'-'+str(step));a=allocations(cell);target=allocations(native_freeze.parent/'00-target')['model_buffers'];gpu=lambda d:sum(v for k,v in d.items() if k.startswith('CUDA') and k!='CUDA_Host');cpu=lambda d:sum(v for k,v in d.items() if k.startswith('CPU'));memory=json.load(open(cell/'memory.json'))
    row.update(actual_gpu_model_bytes=gpu(a['model_buffers']),actual_cpu_model_bytes=cpu(a['model_buffers']),private_gpu_model_bytes=gpu(a['model_buffers'])-gpu(target),private_cpu_mapped_model_bytes=cpu(a['model_buffers'])-cpu(target),draft_cache_bytes=sum(x['total_reported_bytes'] for x in a['cache_buffers'][1:]),draft_rotation_bytes=sum(x['reported_bytes'] for x in a['rotation_buffers']),runtime_compute_buffer_history=a['compute_buffer_history'],sampled_combined_device_peak_bytes=max([x['MiB']*2**20 for x in memory],default=None),allocation_evidence=str(cell),allocation_scope='short_native_readout',allocation_rounding='Native logs0.01MiB; exact persistent tensor bytes separate. Sampled peak includes target/desktop/workspace.')
    row['model_cache_payload_bytes']=row['tensor_bytes']+row['draft_cache_bytes']+row['draft_rotation_bytes']
   row.update({k:v for k,v in nm.items() if k not in ('prefix_counts',)})
   row.update(full_vocabulary_rows=248320,vocabulary_width=2560,additional_trainable_vocabulary_scalars=2483200 if spec.get('r2_vocab_scale') else 0,private_tying=bool(spec.get('tied')),vocabulary_tying='One identical private quantized matrix/device allocation' if spec.get('tied') else 'Frozen source-equivalent tables; target-sharing/private copies per head and embedding fields',ffn_weights='W1 all18',gate_up_activation='A4 learned clipping',down_activation='A8 learned clipping',fusion_weights='Q4_0',attention_qk_weights='Q4_0',attention_vo_weights='Q4_0',row_scale_storage='F16' if spec.get('scalar',{}).get('scale') else 'F32',midpoint_storage='F16' if spec.get('scalar',{}).get('midpoint') else 'F32',positive_gain_storage='F16' if spec.get('scalar',{}).get('gain') else 'F32',norm_gain_storage='F16' if spec.get('scalar',{}).get('norm') else 'F32',clip_storage='IEEE F16 bits in UINT16 GGUF metadata; F32 native parameter' if spec.get('scalar',{}).get('clip') and spec.get('clip_bits_storage') else 'F16-rounded value in F32 GGUF metadata' if spec.get('scalar',{}).get('clip') else 'F32',trainable_policy='R2 native vocabulary block scales plus R1 groups' if spec.get('r2_vocab_scale') else 'R1 FFN sign/scale/midpoint/shared gate-up gain/separate down gain/learned clip/norm',model_path=point['model']['path'],model_sha256=point['model']['sha256'])
   recovery=json.load(open(directory/'recovery.json')) if (directory/'recovery.json').exists() else result.get('recovery',{})
   archived=Path(recovery.get('archive','/nonexistent'));match=re.search(r'-(\d+)\.pt\.(?:bsz|cpt|zst)$',archived.name);saved_step=int(match[1]) if match else None
   row['full_recovery_state_status']=('retained full state for this readout' if saved_step==step else 'latest full arm state retained at '+str(saved_step)+'; this readout has native export only') if archived.exists() else 'retired under NEW rolling policy; B0/native states retained'
   for saved in retained.values():
    if saved['original_arm']==directory.name and saved['native']['sha256']==point['model']['sha256'] and Path(saved['archive']).exists():row.update(full_recovery_state_status='retained authenticated frontier full state; recovery verification separate',frontier_full_state=saved['archive'],frontier_retention_evidence=str(R/'frontier-retention.json'))
   rows.append(row)
 controls={'H4R2':'H6R2','H6R2':'H6R2','HE4R2':'HE6R2','HE6R2':'HE6R2','H0broad':'H0broad','Csmallbroad':'H0broad'};lookup={(r['arm'],r['updates']):r for r in rows}
 for r in rows:
  name=controls.get(r['arm'],'H0');c=lookup.get((name,r['updates']));r['matched_recovery_control']=name;r['ce_gap_matched_control']=r['ce']-c['ce'] if c else None
  r['native_acceptance_gap_matched_control']=r.get('acceptance')-c.get('acceptance') if c and r.get('acceptance') is not None and c.get('acceptance') is not None else None
 from .publication_merge import merge
 return merge(rows)

def charts(rows,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'figure.dpi':140,'font.size':9})
 names=list(dict.fromkeys(r['arm'] for r in rows));colors={n:plt.cm.tab20(i%20) for i,n in enumerate(names)}
 def lines(field,ylabel,title,file):
  fig,ax=plt.subplots(figsize=(11,5.7))
  for name in names:
   points=sorted([r for r in rows if r['arm']==name and r.get(field) is not None],key=lambda x:x['updates'])
   if points:ax.plot([r['updates'] for r in points],[r[field] for r in points],marker='o',ms=3,label=name,color=colors[name])
  ax.set(xlabel='Additional matched updates from frozen learned-small2048 B0',ylabel=ylabel,title=title);ax.grid(alpha=.2)
  if names:ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)
  fig.tight_layout();fig.savefig(out/(file+'.png'));fig.savefig(out/(file+'.svg'));plt.close(fig)
 lines('ce_gap_matched_control','DEV cross entropy gap to matched continuation','Component recovery on consumed DEV; one training history per recipe','recovery-gap')
 lines('acceptance','Native accepted / proposed','Native recovery readouts; short DEV smokes until formal confirmation','native-acceptance')
 lines('ce','Dense DEV cross entropy','Dense full-vocabulary loss; repeated TRAIN exposure, not fresh labels','recovery-loss')
 groups={'Head':['H0','H6','H4','H3',*[n for n in names if n.startswith('H') and 'R2' in n]],'Embedding and tying':['H0','E6','E4','HE4',*[n for n in names if n.startswith(('E','HE')) and 'R2' in n]],'Draft K/V':['H0',*[n for n in names if n.startswith('K')]],'Scalar storage':['H0',*[n for n in names if n.startswith('S')]]}
 for field,label,filename in [('ce_gap_matched_control','DEV cross entropy gap','component-recovery'),('acceptance','Native accepted / proposed','component-native-acceptance')]:
  fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
  for ax,(title,members) in zip(axes.flat,groups.items()):
   for name in members:
    points=sorted([r for r in rows if r['arm']==name and r.get(field) is not None],key=lambda x:x['updates'])
    if points:ax.plot([r['updates'] for r in points],[r[field] for r in points],marker='o',ms=3,label=name,color=colors[name])
   ax.set(title=title,xlabel='Additional updates from B0',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
  fig.suptitle('Matched recovery on consumed DEV; native readouts are short diagnostics');fig.tight_layout();fig.savefig(out/(filename+'.png'));fig.savefig(out/(filename+'.svg'));plt.close(fig)
 fig,axes=plt.subplots(1,4,figsize=(18,4.4))
 final=[max([r for r in rows if r['arm']==n],key=lambda x:x['updates']) for n in names]
 acceptance_field='final_acceptance' if any(r.get('final_acceptance') is not None for r in final) else 'selection_acceptance' if any(r.get('selection_acceptance') is not None for r in final) else 'acceptance'
 acceptance_scope=next((r.get('final_request_cohort') for r in final if r.get('final_acceptance') is not None),next((r.get('selection_request_cohort') for r in final if r.get('selection_acceptance') is not None),'short consumed DEV smoke'))
 memory_includes_cache=any(r.get('draft_cache_bytes') is not None for r in final)
 for r in final:
  n=r['arm'];mem=(r['tensor_bytes']+r['draft_cache_bytes']+r.get('draft_rotation_bytes',0))/2**20 if memory_includes_cache and r.get('draft_cache_bytes') is not None else None if memory_includes_cache else r['tensor_bytes']/2**20
  speed=r.get('final_vs_q4')
  peak=r.get('sampled_combined_device_peak_bytes');peak=peak/2**20 if peak is not None else None
  for ax,x,y in [(axes[0],mem,r.get(acceptance_field)),(axes[1],r.get('clean_gpu_forward_ms'),r.get(acceptance_field)),(axes[2],mem,speed.get('ratio') if speed else None),(axes[3],peak,speed.get('ratio') if speed else None)]:
   if x is not None and y is not None:ax.scatter(x,y,color=colors[n]);ax.annotate(n,(x,y),fontsize=7)
  if mem is not None and speed:
   lo,hi=speed['paired_95_interval'];value=speed['ratio'];axes[2].errorbar(mem,value,yerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='none',color=colors[n],capsize=2)
   if peak is not None:axes[3].errorbar(peak,value,yerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='none',color=colors[n],capsize=2)
 axes[0].set(xlabel='Private model +4096cache/rotation payload MiB' if memory_includes_cache else 'Private model tensor MiB; cache not yet measured',ylabel='Native accepted / proposed',title='Storage / acceptance')
 axes[1].set(xlabel='Clean GPU forward ms',ylabel='Native accepted / proposed',title='Latency / acceptance')
 axes[2].set(xlabel='Private model +4096cache/rotation payload MiB' if memory_includes_cache else 'Private model tensor MiB',ylabel='Formal request speed / external Q4',title='Full-request frontier; paired95% interval');axes[2].axhline(1,color='gray',ls='--',lw=.8)
 axes[3].set(xlabel='Sampled combined device peak MiB',ylabel='Formal request speed / external Q4',title='Target + desktop + workspace included');axes[3].axhline(1,color='gray',ls='--',lw=.8)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Measured component tradeoffs; acceptance cohort: '+acceptance_scope+'; unmeasured fields blank');fig.tight_layout();fig.savefig(out/'tradeoffs.png');fig.savefig(out/'tradeoffs.svg');plt.close(fig)


def main():
 out=R/'charts';out.mkdir(exist_ok=True);rows=collect();index=json.load(open(R/'publication-index.json')) if (R/'publication-index.json').exists() else {}
 save(out/'comparison.json',dict(status=index.get('status','IN_PROGRESS'),rows=rows,freeze=pin(R/'freeze.json'),budget=pin(R/'budget.json'),measurement_note='Null means unmeasured. Short native readouts are not formal final repeated comparisons.'))
 keys=sorted({k for r in rows for k in r if k not in ('native','component_spec','model')})
 with (out/'full-table.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:json.dumps(r.get(k),sort_keys=True) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in keys} for r in rows)
 charts(rows,out)
 from .interactive import render
 (out/'comparison.html').write_text(render(rows))
 print('PUBLISHED',len(rows),'readouts',out,flush=True)

if __name__=='__main__':main()
