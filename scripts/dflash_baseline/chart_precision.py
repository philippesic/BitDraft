"""Recompute campaign tables and standalone charts solely from native receipts."""
import argparse,base64,csv,html,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/philip/binary-eagle-decoding')
def read(p):return json.loads(Path(p).read_text())
def summarize(panel):
 result=read(panel/'result.json');rows=result['rows'];freeze=read(panel/'freeze.json') if (panel/'freeze.json').exists() else {};cohort=freeze.get('cohort','capture3');points=[]
 for arm in dict.fromkeys(x['arm'] for x in rows):
  plain=[x for x in rows if x['arm']==arm and not x['warmup'] and not x.get('instrumented')];traced=[x for x in rows if x['arm']==arm and not x['warmup'] and x.get('instrumented')]
  if not plain:continue
  counts=[x.get('native',{}) for x in plain];tokens=sum(x['tokens'] for x in plain);wall=sum(x['wall_seconds'] for x in plain);decode=sum(x['timings']['predicted_ms'] for x in plain)/1000
  accepted=sum(x.get('accepted',y['timings'].get('draft_n_accepted',0)) for x,y in zip(counts,plain));proposed=sum(x.get('proposed',y['timings'].get('draft_n',0)) for x,y in zip(counts,plain));rounds=sum(x.get('rounds',0) for x in counts)
  meta=freeze.get('arms',{}).get(arm);model=meta.get('path') if meta else None;sidecar=Path(model).with_suffix('.json') if model else None;info=read(sidecar) if sidecar and sidecar.exists() else {};config=info.get('config',{});coverage=config.get('coverage',{});bits=[coverage.get(f'blk.{i}.ffn_gate',{}).get('bits') for i in range(6)];down=[coverage.get(f'blk.{i}.ffn_down',{}).get('bits') for i in range(6)]
  family='G'+str(bits[0])+'/D'+str(down[0]) if bits and bits[0] else 'external' if arm in ('target','q4') else 'vanilla W1A8'
  memories=[]
  for cell in panel.glob('*-'+arm):
   mp=cell/'memory.json'
   if mp.exists():memories.extend(x['MiB'] for x in read(mp))
  trace_rounds=sum(x['native'].get('rounds',0) for x in traced);full=sum(x.get('draft_full_ms',0) for x in traced)/trace_rounds if trace_rounds else None
  prefix=[]
  for n in counts:
   for i,v in enumerate(n.get('prefix_survival_counts',[])):
    if i==len(prefix):prefix.append(0)
    prefix[i]+=v
  point=dict(id=panel.name+'::'+arm,panel=str(panel),arm=arm,cohort=cohort,scope='confirmation' if len({x['rep'] for x in plain})>=5 else 'diagnostic smoke',family=family,config=config,model=model,model_bytes=meta.get('bytes') if meta else (Path(model).stat().st_size if model else 0),correct=all(x['correct'] for x in plain+traced),requests=len(plain),prompts=len({x['prompt'] for x in plain}),repetitions=len({x['rep'] for x in plain}),accepted=accepted,proposed=proposed,rounds=rounds,acceptance=accepted/proposed if proposed else None,prefix_counts=prefix,prefix_survival=[x/rounds for x in prefix] if rounds else [],emitted_per_round=tokens/rounds if rounds else None,request_tps=tokens/wall,decode_tps=tokens/decode,full_draft_ms=full,narrow_generate_ms=sum(x.get('draft_ms',0) for x in counts)/rounds if rounds else None,total_peak_device_MiB=max(memories) if memories else None,training_updates=info.get('progress',{}).get('updates'),training_seconds=info.get('progress',{}).get('update_seconds'),lineage=info.get('lineage'),runtime=freeze.get('runtime'),instrumented_requests=len(traced),verified_paired_counters=result.get('paired_trace_counters_equal',False),target_verify_and_sample_ms_per_round=sum(x.get('target_verify_and_sample_ms',0) for x in traced)/trace_rounds if trace_rounds else None)
  baseline=[x for x in rows if x['arm']=='q4' and not x['warmup'] and not x.get('instrumented')]
  if baseline:
   point['request_vs_q4']=point['request_tps']/(sum(x['tokens'] for x in baseline)/sum(x['wall_seconds'] for x in baseline));point['decode_vs_q4']=point['decode_tps']/(sum(x['tokens'] for x in baseline)/(sum(x['timings']['predicted_ms'] for x in baseline)/1000))
  if baseline and point['repetitions']>=5:
   lookup={(x['rep'],x['prompt']):x for x in baseline};pairs=[(x,lookup[(x['rep'],x['prompt'])]) for x in plain];rng=np.random.default_rng(811);ratios=[]
   reps=sorted({x['rep'] for x in plain});prompts=sorted({x['prompt'] for x in plain});by={(x['rep'],x['prompt']):(x,y) for x,y in pairs}
   for _ in range(2000):
    chosen=[by[(r,p)] for r in rng.choice(reps,len(reps)) for p in rng.choice(prompts,len(prompts))];ca=[x for x,y in chosen];ba=[y for x,y in chosen];ratios.append((sum(x['tokens'] for x in ca)/sum(x['wall_seconds'] for x in ca))/(sum(x['tokens'] for x in ba)/sum(x['wall_seconds'] for x in ba)))
   point['request_vs_q4_bootstrap95']=np.quantile(ratios,[.025,.975]).tolist()
  points.append(point)
 return points

def build(root,out):
 out.mkdir(parents=True,exist_ok=True);points=[]
 for p in sorted(root.rglob('result.json')):
  if p.parent==out:continue
  try:
   data=read(p)
   if isinstance(data.get('rows'),list) and data['rows'] and 'arm' in data['rows'][0]:points+=summarize(p.parent)
  except (KeyError,TypeError):continue
 payload=dict(status='MEASURED_CHECKPOINT_NOT_CAMPAIGN_COMPLETION',points=points,limitations=['RTX5080 / modern Qwen3.5-4B Q4_K_M only.','Smokes and repeated confirmations are separated by panel/cohort.','A1/A2 execute INT8-code DP4A fallback, not hypothetical binary kernels.','Full draft cost uses disjoint instrumented host spans including catchup, generation and accept; asynchronous work may be charged at a later synchronization.','Sampled device-memory peak includes display/unrelated baseline and is not exact allocator peak.','Confidence intervals resample reused prompts/repetitions; no independent-training variance estimate.'])
 (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n');fields=['id','cohort','scope','family','training_updates','accepted','proposed','rounds','acceptance','emitted_per_round','full_draft_ms','request_tps','decode_tps','request_vs_q4','model_bytes','total_peak_device_MiB','correct']
 with (out/'comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(points)
 def fmt(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
 table='| '+' | '.join(fields)+' |\n|'+'|'.join(['---']*len(fields))+'|\n'+''.join('| '+' | '.join(fmt(p.get(k)) for k in fields)+' |\n' for p in points)
 (out/'full-table.md').write_text('Measured native points; scope and cohort remain separate.\n\n'+table)
 colors={'G8':'#3269a8','G4':'#38976b','G2':'#d98c22','G1':'#bd4d6c','external':'#676767'}
 def color(p):return next((v for k,v in colors.items() if p['family'].startswith(k)),'#7564ab')
 groups=sorted({p['cohort'] for p in points}) or ['none'];fig,axes=plt.subplots(1,len(groups),figsize=(7*len(groups),5),squeeze=False)
 for ax,group in zip(axes[0],groups):
  for p in points:
   if p['cohort']!=group or p['full_draft_ms'] is None or p['acceptance'] is None:continue
   size=25+max(0,(p['model_bytes'] or 0)/1e7);ax.scatter(p['full_draft_ms'],100*p['acceptance'],s=size,color=color(p),alpha=.7);ax.annotate(p['arm'],(p['full_draft_ms'],100*p['acceptance']),fontsize=7)
  ax.set(xlabel='Complete native draft ms/round (traced)',ylabel='Accepted / proposed (%)',title=group);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'acceptance-latency.png',dpi=160);fig.savefig(out/'acceptance-latency.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(max(10,len(points)*.18),5));selected=[p for p in points if p.get('request_vs_q4') is not None];ax.bar(range(len(selected)),[p['request_vs_q4'] for p in selected],color=[color(p) for p in selected]);ax.axhline(1,color='black',lw=1);ax.set_xticks(range(len(selected)),[p['panel'].split('/')[-1]+' / '+p['arm'] for p in selected],rotation=90,fontsize=6);ax.set(ylabel='Full request throughput / matched Q4',title='Native throughput; each bar uses its own matched panel control');fig.tight_layout();fig.savefig(out/'throughput.png',dpi=160);fig.savefig(out/'throughput.svg');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for p in points:
  if p['acceptance'] is None:continue
  axes[0].scatter((p['model_bytes'] or 0)/2**20,100*p['acceptance'],color=color(p),alpha=.6)
  if p['total_peak_device_MiB']:axes[1].scatter(p['total_peak_device_MiB'],p['request_tps'],color=color(p),alpha=.6)
 axes[0].set(xlabel='Actual drafter GGUF MiB (private tensors included)',ylabel='Acceptance (%)');axes[1].set(xlabel='Sampled total device peak MiB',ylabel='Full request tokens/s');fig.tight_layout();fig.savefig(out/'size-memory.png',dpi=160);fig.savefig(out/'size-memory.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5))
 for p in points:
  if p['training_updates'] is not None and p['acceptance'] is not None:ax.scatter(p['training_updates'],100*p['acceptance'],color=color(p));ax.annotate(p['arm'],(p['training_updates'],100*p['acceptance']),fontsize=6)
 ax.set(xlabel='Additional segment updates (ancestry in table)',ylabel='Native acceptance (%)',title='Measured recovery points; distinct recipes are not connected');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'recovery.png',dpi=160);fig.savefig(out/'recovery.svg');plt.close(fig)
 # Standalone interactive SVG and sortable evidence table: no network/CDN/server.
 data=json.dumps(points).replace('</','<\\/');columns=json.dumps(fields)
 body='''<!doctype html><html><head><meta charset="utf-8"><title>DFlash precision comparison</title><style>body{font:14px system-ui;margin:24px;color:#172232;background:#f6f8fb}h1{font-size:24px}select,input{padding:7px;margin:5px}svg{width:100%;max-width:1000px;background:white;border:1px solid #ddd}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:7px;border-bottom:1px solid #d6dce6;text-align:right}th{cursor:pointer;position:sticky;top:0;background:#e5ebf3}.scroll{overflow:auto;max-height:650px}#tip{position:fixed;display:none;background:#172232;color:white;padding:10px;max-width:400px;pointer-events:none}img{max-width:100%}</style></head><body><h1>DFlash precision campaign — measured evidence</h1><p id="status">A checkpoint chart is incomplete until the campaign record marks full comparison complete.</p><p>RTX 5080 · Qwen3.5 4B Q4_K_M target · depth 2 · greedy target-sample-and-match. A1/A2 use measured INT8-code DP4A execution. Traced host spans include draft generation, catch-up and acceptance; timings are diagnostic.</p><label>Panel <select id="panel"></select></label><label>X <select id="x"><option value="full_draft_ms">Complete draft ms/round</option value="model_bytes">Actual model bytes</option><option value="total_peak_device_MiB">Sampled total peak MiB</option><option value="training_updates">Segment updates</option></select></label><label>Y <select id="y"><option value="acceptance">Acceptance</option><option value="request_vs_q4">Request throughput / Q4</option><option value="decode_vs_q4">Decode throughput / Q4</option><option value="request_tps">Request tokens/s</option></select></label><svg id="plot" viewBox="0 0 1000 540"></svg><div id="tip"></div><p>Missing measurements are omitted from the selected plot, never plotted as zero. Bubble radius indicates actual GGUF bytes. Hover for counts, lineage and execution details. Click a column to sort.</p><input id="filter" placeholder="Filter rows"><div class="scroll"><table id="table"></table></div>'''
 body+='<script>const points='+data+',columns='+columns+''';let sort="id";const panel=document.querySelector('#panel'),sx=document.querySelector('#x'),sy=document.querySelector('#y'),tip=document.querySelector('#tip');panel.innerHTML='<option value="">All panels (different cohorts remain distinct)</option>'+[...new Set(points.map(p=>p.panel))].map(p=>'<option>'+p+'</option>').join('');function draw(){let rows=points.filter(p=>(!panel.value||p.panel===panel.value)&&JSON.stringify(p).toLowerCase().includes(document.querySelector('#filter').value.toLowerCase()));const valid=rows.filter(p=>p[sx.value]!=null&&p[sy.value]!=null);let maxX=Math.max(1,...valid.map(p=>p[sx.value]))*1.08,maxY=Math.max(.01,...valid.map(p=>p[sy.value]))*1.1;const svg=document.querySelector('#plot');svg.innerHTML='<path d="M70 20V470H970" stroke="#789" fill="none"/><text x="500" y="528" text-anchor="middle">'+sx.selectedOptions[0].text+'</text><text x="20" y="240" transform="rotate(-90 20 240)" text-anchor="middle">'+sy.selectedOptions[0].text+'</text>';for(let i=0;i<=5;i++){let x=70+i*180,y=470-i*85;svg.innerHTML+='<text x="'+x+'" y="491" text-anchor="middle">'+(maxX*i/5).toPrecision(3)+'</text><text x="62" y="'+y+'" text-anchor="end">'+(maxY*i/5).toPrecision(3)+'</text>';}valid.forEach(p=>{let c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',70+900*p[sx.value]/maxX);c.setAttribute('cy',470-425*p[sy.value]/maxY);c.setAttribute('r',Math.max(5,Math.sqrt((p.model_bytes||1)/1e7)));c.setAttribute('fill',p.family.startsWith('G1')?'#bd4d6c':p.family.startsWith('G2')?'#d98c22':p.family.startsWith('G4')?'#38976b':'#3269a8');c.setAttribute('opacity','.65');c.onmousemove=e=>{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=p.id+' | '+p.scope+' | '+p.family+' | '+p.accepted+'/'+p.proposed+' accepted | '+p.request_tps.toFixed(2)+' request tok/s | '+JSON.stringify(p.config.components||{})};c.onmouseleave=()=>tip.style.display='none';svg.append(c)});rows.sort((a,b)=>typeof a[sort]==='number'?(a[sort]||0)-(b[sort]||0):String(a[sort]).localeCompare(String(b[sort])));const esc=x=>String(x??'—').replaceAll('&','&amp;').replaceAll('<','&lt;');document.querySelector('#table').innerHTML='<thead><tr>'+columns.map(k=>'<th data-key="'+k+'">'+k+'</th>').join('')+'</tr></thead><tbody>'+rows.map(p=>'<tr>'+columns.map(k=>'<td>'+esc(typeof p[k]==='number'?Number(p[k].toPrecision(6)):p[k])+'</td>').join('')+'</tr>').join('')+'</tbody>';document.querySelectorAll('th').forEach(t=>t.onclick=()=>{sort=t.dataset.key;draw()})}for(const s of [panel,sx,sy,document.querySelector('#filter')])s.oninput=draw;draw();</script>'''
 for name in ('acceptance-latency','throughput','size-memory','recovery'):body+='<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'">'
 (out/'comparison.html').write_text(body+'</body></html>');return payload
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();result=build(a.root,a.out);print('Native points:',len(result['points']))
