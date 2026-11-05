"""Recompute campaign tables and standalone charts solely from native receipts."""
import argparse,base64,csv,html,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/philip/binary-eagle-decoding')
def read(p):return json.loads(Path(p).read_text())
def summarize(panel,root):
 result=read(panel/'result.json');rows=result['rows'];freeze=read(panel/'freeze.json') if (panel/'freeze.json').exists() else {};cohort=freeze.get('cohort','capture3');points=[]
 for arm in dict.fromkeys(x['arm'] for x in rows):
  plain=[x for x in rows if x['arm']==arm and not x['warmup'] and not x.get('instrumented')];traced=[x for x in rows if x['arm']==arm and not x['warmup'] and x.get('instrumented')]
  if not plain:continue
  counts=[x.get('native',{}) for x in plain];tokens=sum(x['tokens'] for x in plain);wall=sum(x['wall_seconds'] for x in plain);decode=sum(x['timings']['predicted_ms'] for x in plain)/1000
  accepted=sum(x.get('accepted',y['timings'].get('draft_n_accepted',0)) for x,y in zip(counts,plain));proposed=sum(x.get('proposed',y['timings'].get('draft_n',0)) for x,y in zip(counts,plain));rounds=sum(x.get('rounds',0) for x in counts)
  meta=freeze.get('arms',{}).get(arm);model=meta.get('path') if meta else None;sidecar=Path(model).with_suffix('.json') if model else None;info=read(sidecar) if sidecar and sidecar.exists() else {};config=info.get('config',{});coverage=config.get('coverage',{});bits=[coverage.get(f'blk.{i}.ffn_gate',{}).get('bits') for i in range(6)];down=[coverage.get(f'blk.{i}.ffn_down',{}).get('bits') for i in range(6)]
  family='G'+str(bits[0])+'/D'+str(down[0]) if bits and bits[0] else 'external' if arm in ('target','q4') else 'vanilla W1A8'
  memories=[];allocator=[]
  for cell in panel.glob('*-'+arm):
   mp=cell/'memory.json'
   if mp.exists():memories.extend(x['MiB'] for x in read(mp))
   log=cell/'server.log'
   if log.exists():
    match=re.findall(r'CUDA0.*?\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)',log.read_text(errors='replace'))
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
  point=dict(runtime_series=runtime_series,execution='Native INT8-code DP4A / packed W1 FFN' if coverage else 'Native external model',resource_note=resource_note,id=str(panel.relative_to(root))+'::'+arm,panel=str(panel),arm=arm,cohort=cohort,scope='confirmation' if len({x['rep'] for x in plain})>=5 else 'diagnostic smoke',family=family,config=config,model=model,model_bytes=meta.get('bytes') if meta else (Path(model).stat().st_size if model else 0),correct=all(x['correct'] for x in plain+traced),requests=len(plain),prompts=len({x['prompt'] for x in plain}),repetitions=len({x['rep'] for x in plain}),accepted=accepted,proposed=proposed,rounds=rounds,acceptance=accepted/proposed if proposed else None,prefix_counts=prefix,prefix_survival=[x/rounds for x in prefix] if rounds else [],emitted_per_round=tokens/rounds if rounds else None,request_tps=tokens/wall,decode_tps=tokens/decode,full_draft_ms=full,narrow_generate_ms=sum(x.get('draft_ms',0) for x in counts)/rounds if rounds else None,total_peak_device_MiB=max(memories) if memories else None,allocator_shutdown_MiB=allocator[-1] if allocator else None,training_updates=info.get('progress',{}).get('updates',info.get('updates')),training_seconds=info.get('progress',{}).get('update_seconds'),lineage=info.get('lineage'),runtime=freeze.get('runtime'),instrumented_requests=len(traced),verified_paired_counters=result.get('paired_trace_counters_equal',False),target_verify_and_sample_ms_per_round=sum(x.get('target_verify_and_sample_ms',0) for x in traced)/trace_rounds if trace_rounds else None)
  baseline=[x for x in rows if x['arm']=='q4' and not x['warmup'] and not x.get('instrumented')]
  if baseline:
   point['request_vs_q4']=point['request_tps']/(sum(x['tokens'] for x in baseline)/sum(x['wall_seconds'] for x in baseline));point['decode_vs_q4']=point['decode_tps']/(sum(x['tokens'] for x in baseline)/(sum(x['timings']['predicted_ms'] for x in baseline)/1000))
  if baseline and point['repetitions']>=5:
   lookup={(x['rep'],x['prompt']):x for x in baseline};pairs=[(x,lookup[(x['rep'],x['prompt'])]) for x in plain];rng=np.random.default_rng(811);ratios=[]
   acceptance_samples=[];latency_samples=[];overhead_samples=[];trace_by={(x['rep'],x['prompt']):x for x in traced}
   reps=sorted({x['rep'] for x in plain});prompts=sorted({x['prompt'] for x in plain});by={(x['rep'],x['prompt']):(x,y) for x,y in pairs}
   for _ in range(2000):
    sample_reps=rng.choice(reps,len(reps));sample_prompts=rng.choice(prompts,len(prompts));chosen=[by[(r,p)] for r in sample_reps for p in sample_prompts];ca=[x for x,y in chosen];ba=[y for x,y in chosen];ratios.append((sum(x['tokens'] for x in ca)/sum(x['wall_seconds'] for x in ca))/(sum(x['tokens'] for x in ba)/sum(x['wall_seconds'] for x in ba)))
      # Cases are resampled once across all sampled repetitions (two-way clusters).
    accepted_sum=sum(x.get('native',{}).get('accepted',x['timings'].get('draft_n_accepted',0)) for x in ca);proposed_sum=sum(x.get('native',{}).get('proposed',x['timings'].get('draft_n',0)) for x in ca)
    if proposed_sum:acceptance_samples.append(accepted_sum/proposed_sum)
    if traced:
     ta=[trace_by[(x['rep'],x['prompt'])] for x in ca];trace_round_sum=sum(x['native'].get('rounds',0) for x in ta)
     if trace_round_sum:latency_samples.append(sum(x.get('draft_full_ms',0) for x in ta)/trace_round_sum)
     overhead_samples.append(100*(sum(x['wall_seconds'] for x in ta)/sum(x['wall_seconds'] for x in ca)-1))
   point['request_vs_q4_bootstrap95']=np.quantile(ratios,[.025,.975]).tolist()
   if acceptance_samples:point['acceptance_bootstrap95']=np.quantile(acceptance_samples,[.025,.975]).tolist()
   if latency_samples:point['full_draft_ms_bootstrap95']=np.quantile(latency_samples,[.025,.975]).tolist()
   if overhead_samples:point['trace_overhead_bootstrap95_percent']=np.quantile(overhead_samples,[.025,.975]).tolist()
  if traced:
   point['trace_request_overhead_percent']=100*((sum(x['wall_seconds'] for x in traced)/sum(x['wall_seconds'] for x in plain))-1)
   trace_tokens=sum(x['tokens'] for x in traced);draft_seconds=sum(x.get('draft_full_ms',0) for x in traced)/1000;trace_wall=sum(x['wall_seconds'] for x in traced);trace_decode=sum(x['timings']['predicted_ms'] for x in traced)/1000
   point['optimistic_zero_draft_request_tps']=trace_tokens/(trace_wall-draft_seconds) if trace_wall>draft_seconds else None
   point['optimistic_zero_draft_decode_tps']=trace_tokens/(trace_decode-draft_seconds) if trace_decode>draft_seconds else None
   point['zero_draft_bound_scope']='Accounting upper bound assuming every traced draft span fully removable, unchanged verification and proposals; async host attribution limits apply. Not measured deployment throughput.'
  target_cells=list(panel.glob('*-target'));target_peak=[];target_allocator=[]
  for cell in target_cells:
   if (cell/'memory.json').exists():target_peak.extend(x['MiB'] for x in read(cell/'memory.json'))
   if (cell/'server.log').exists():
    matches=re.findall(r'CUDA0.*?\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)',(cell/'server.log').read_text(errors='replace'))
    if matches:target_allocator.append(tuple(map(int,matches[-1])))
  point['incremental_peak_device_MiB_vs_target']=max(memories)-max(target_peak) if memories and target_peak else None
  point['incremental_model_allocator_MiB_vs_target']=allocator[-1]['model_MiB']-target_allocator[-1][1] if allocator and target_allocator else None
  points.append(point)
 return points

def build(root,out):
 out.mkdir(parents=True,exist_ok=True);points=[]
 for p in sorted(root.rglob('result.json')):
  if p.parent==out:continue
  try:
   data=read(p)
   if isinstance(data.get('rows'),list) and data['rows'] and 'arm' in data['rows'][0]:points+=summarize(p.parent,root)
  except (KeyError,TypeError):continue
 completion=read(root/'campaign-completion.json') if (root/'campaign-completion.json').exists() else None
 payload=dict(status='FULL_INITIAL_COMPARISON_COMPLETE' if completion else 'MEASURED_CHECKPOINT_NOT_CAMPAIGN_COMPLETION',completion=completion,blocked=read(root/'tranche.json').get('blocked',[]) if (root/'tranche.json').exists() else [],points=points,limitations=['RTX5080 / modern Qwen3.5-4B Q4_K_M only.','Smokes and repeated confirmations are separated by panel/cohort.','A1/A2 execute INT8-code DP4A fallback, not hypothetical binary kernels.','Full draft cost uses disjoint instrumented host spans including catchup, generation and accept; asynchronous work may be charged at a later synchronization.','Sampled device-memory peak includes display/unrelated baseline and is not exact allocator peak.','Confidence intervals resample reused prompts/repetitions; no independent-training variance estimate.'])
 (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n');fields=['id','cohort','scope','runtime_series','family','execution','training_updates','accepted','proposed','rounds','acceptance','emitted_per_round','full_draft_ms','request_tps','decode_tps','request_vs_q4','optimistic_zero_draft_request_tps','model_bytes','total_peak_device_MiB','incremental_peak_device_MiB_vs_target','incremental_model_allocator_MiB_vs_target','correct']
 with (out/'comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(points)
 def fmt(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
 table='| '+' | '.join(fields)+' |\n|'+'|'.join(['---']*len(fields))+'|\n'+''.join('| '+' | '.join(fmt(p.get(k)) for k in fields)+' |\n' for p in points)
 (out/'full-table.md').write_text('Measured native points; scope and cohort remain separate.\n\n'+table)
 colors={'G8':'#3269a8','G4':'#38976b','G2':'#d98c22','G1':'#bd4d6c','external':'#676767'}
 def color(p):return next((v for k,v in colors.items() if p['family'].startswith(k)),'#7564ab')
 display=[p for p in points if p['scope']=='confirmation' and p['runtime_series'].endswith('-04')] or [p for p in points if p['runtime_series'].endswith('-04')] or points
 groups=sorted({p['cohort'] for p in display}) or ['none'];fig,axes=plt.subplots(1,len(groups),figsize=(7*len(groups),5),squeeze=False)
 for ax,group in zip(axes[0],groups):
  for p in display:
   if p['cohort']!=group or p['full_draft_ms'] is None or p['acceptance'] is None:continue
   size=25+max(0,(p['model_bytes'] or 0)/1e7);ax.scatter(p['full_draft_ms'],100*p['acceptance'],s=size,color=color(p),alpha=.7);ax.annotate(p['arm'],(p['full_draft_ms'],100*p['acceptance']),fontsize=7)
  ax.set(xlabel='Complete native draft ms/round (traced)',ylabel='Accepted / proposed (%)',title=group);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'acceptance-latency.png',dpi=160);fig.savefig(out/'acceptance-latency.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(max(10,len(points)*.18),5));selected=[p for p in display if p.get('request_vs_q4') is not None];ax.bar(range(len(selected)),[p['request_vs_q4'] for p in selected],color=[color(p) for p in selected]);ax.axhline(1,color='black',lw=1);ax.set_xticks(range(len(selected)),[p['panel'].split('/')[-1]+' / '+p['arm'] for p in selected],rotation=90,fontsize=6);ax.set(ylabel='Full request throughput / matched Q4',title='Native throughput; each bar uses its own matched panel control');fig.tight_layout();fig.savefig(out/'throughput.png',dpi=160);fig.savefig(out/'throughput.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for p in display:
  if p['acceptance'] is None:continue
  axes[0].scatter((p['model_bytes'] or 0)/2**20,100*p['acceptance'],color=color(p),alpha=.6)
  if p['total_peak_device_MiB']:axes[1].scatter(p['total_peak_device_MiB'],p['request_tps'],color=color(p),alpha=.6)
 axes[0].set(xlabel='Actual drafter GGUF MiB (private tensors included)',ylabel='Acceptance (%)');axes[1].set(xlabel='Sampled total device peak MiB',ylabel='Full request tokens/s');fig.tight_layout();fig.savefig(out/'size-memory.png',dpi=160);fig.savefig(out/'size-memory.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,6));curves={}
 for p in points:
  match=re.fullmatch(r'(G[1248]D[1248](?:-[a-z-]+)?)-(zero|256|1024)',p['arm'])
  if p['cohort']=='natural12' and p['training_updates'] in (0,256,1024) and p['acceptance'] is not None and match:
   key=(p['runtime_series'],match[1]);prior=curves.setdefault(key,{}).get(p['training_updates'])
   if prior is None or p['repetitions']>prior['repetitions']:curves[key][p['training_updates']]=p
 for (series,name),values in curves.items():
  curve=sorted(values.values(),key=lambda p:p['training_updates']);style='--' if series.endswith('-03') else '-'
  ax.plot([p['training_updates'] for p in curve],[100*p['acceptance'] for p in curve],marker='o',linestyle=style,label=name+' / '+series.rsplit('-',1)[-1],color=color(curve[0]),alpha=.75)
 ax.set(xlabel='Matched additional updates from common frozen foundation',ylabel='Native acceptance (%)',title='Recovery on consumed natural12; runtime series remain distinct');ax.grid(alpha=.2)
 if curves:ax.legend(fontsize=6,ncol=3)
 fig.tight_layout();fig.savefig(out/'recovery.png',dpi=160);fig.savefig(out/'recovery.svg');plt.close(fig)
 # Standalone interactive SVG and sortable evidence table: no network/CDN/server.
 data=json.dumps(points).replace('</','<\\/');columns=json.dumps(fields)
 body='''<!doctype html><html><head><meta charset="utf-8"><title>DFlash precision comparison</title><style>body{font:14px system-ui;margin:24px;color:#172232;background:#f6f8fb}h1{font-size:24px}select,input{padding:7px;margin:5px}svg{width:100%;max-width:1000px;background:white;border:1px solid #ddd}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:7px;border-bottom:1px solid #d6dce6;text-align:right}th{cursor:pointer;position:sticky;top:0;background:#e5ebf3}.scroll{overflow:auto;max-height:650px}#tip{position:fixed;display:none;background:#172232;color:white;padding:10px;max-width:400px;pointer-events:none}img{max-width:100%}</style></head><body><h1>DFlash precision campaign — measured evidence</h1><p id="status">A checkpoint chart is incomplete until the campaign record marks full comparison complete.</p><p>RTX 5080 · Qwen3.5 4B Q4_K_M target · depth 2 · greedy target-sample-and-match. A1/A2 use measured INT8-code DP4A execution. Traced host spans include draft generation, catch-up and acceptance; timings are diagnostic.</p><label>Panel <select id="panel"></select></label><label>X <select id="x"><option value="full_draft_ms">Complete draft ms/round</option value="model_bytes">Actual model bytes</option><option value="total_peak_device_MiB">Sampled total peak MiB</option><option value="training_updates">Segment updates</option></select></label><label>Y <select id="y"><option value="acceptance">Acceptance</option><option value="request_vs_q4">Request throughput / Q4</option><option value="decode_vs_q4">Decode throughput / Q4</option><option value="request_tps">Request tokens/s</option></select></label><svg id="plot" viewBox="0 0 1000 540"></svg><div id="tip"></div><p>Missing measurements are omitted from the selected plot, never plotted as zero. Bubble radius indicates actual GGUF bytes. Hover for counts, lineage and execution details. Click a column to sort.</p><input id="filter" placeholder="Filter rows"><div class="scroll"><table id="table"></table></div>'''
 body+='<script>const points='+data+',columns='+columns+''';let sort="id";const panel=document.querySelector('#panel'),sx=document.querySelector('#x'),sy=document.querySelector('#y'),tip=document.querySelector('#tip');panel.innerHTML='<option value="">All panels (different cohorts remain distinct)</option>'+[...new Set(points.map(p=>p.panel))].map(p=>'<option>'+p+'</option>').join('');function draw(){let rows=points.filter(p=>(!panel.value||p.panel===panel.value)&&JSON.stringify(p).toLowerCase().includes(document.querySelector('#filter').value.toLowerCase()));const valid=rows.filter(p=>p[sx.value]!=null&&p[sy.value]!=null);let maxX=Math.max(1,...valid.map(p=>p[sx.value]))*1.08,maxY=Math.max(.01,...valid.map(p=>p[sy.value]))*1.1;const svg=document.querySelector('#plot');svg.innerHTML='<path d="M70 20V470H970" stroke="#789" fill="none"/><text x="500" y="528" text-anchor="middle">'+sx.selectedOptions[0].text+'</text><text x="20" y="240" transform="rotate(-90 20 240)" text-anchor="middle">'+sy.selectedOptions[0].text+'</text>';for(let i=0;i<=5;i++){let x=70+i*180,y=470-i*85;svg.innerHTML+='<text x="'+x+'" y="491" text-anchor="middle">'+(maxX*i/5).toPrecision(3)+'</text><text x="62" y="'+y+'" text-anchor="end">'+(maxY*i/5).toPrecision(3)+'</text>';}valid.forEach(p=>{let c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',70+900*p[sx.value]/maxX);c.setAttribute('cy',470-425*p[sy.value]/maxY);c.setAttribute('r',Math.max(5,Math.sqrt((p.model_bytes||1)/1e7)));c.setAttribute('fill',p.family.startsWith('G1')?'#bd4d6c':p.family.startsWith('G2')?'#d98c22':p.family.startsWith('G4')?'#38976b':'#3269a8');c.setAttribute('opacity','.65');c.onmousemove=e=>{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=p.id+' | '+p.scope+' | '+p.family+' | '+p.accepted+'/'+p.proposed+' accepted | '+p.request_tps.toFixed(2)+' request tok/s | '+JSON.stringify(p.config.components||{})};c.onmouseleave=()=>tip.style.display='none';svg.append(c)});rows.sort((a,b)=>typeof a[sort]==='number'?(a[sort]||0)-(b[sort]||0):String(a[sort]).localeCompare(String(b[sort])));const esc=x=>String(x??'—').replaceAll('&','&amp;').replaceAll('<','&lt;');document.querySelector('#table').innerHTML='<thead><tr>'+columns.map(k=>'<th data-key="'+k+'">'+k+'</th>').join('')+'</tr></thead><tbody>'+rows.map(p=>'<tr>'+columns.map(k=>'<td>'+esc(typeof p[k]==='number'?Number(p[k].toPrecision(6)):p[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';document.querySelectorAll('th').forEach(t=>t.onclick=()=>{sort=t.dataset.key;draw()})}for(const s of [panel,sx,sy,document.querySelector('#filter')])s.oninput=draw;draw();</script>'''
 for name in ('acceptance-latency','throughput','size-memory','recovery'):body+='<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'">'
 (out/'comparison.html').write_text(body+'</body></html>');return payload
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();result=build(a.root,a.out);print('Native points:',len(result['points']))
