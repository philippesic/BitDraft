"""Publish measured readouts, explicit missing fields and standalone interactive charts."""
import csv,json,math,statistics
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
 for directory in sorted((R/'arms').glob('*')):
  if directory.name=='H0-pilot':continue
  path=directory/'readouts.json'
  if not path.exists():continue
  result=json.load(open(directory/'result.json')) if (directory/'result.json').exists() else {}
  spec=json.load(open(directory/'spec.json'));records=json.load(open(path))
  if directory.name=='H0':records=json.load(open(R/'arms/H0-pilot/readouts.json'))[:1]+records
  logs=[directory/'updates.jsonl']
  if directory.name in ('H6','E6'):logs=[R/'arms/H0-pilot/updates.jsonl',R/'arms/H0/updates.jsonl']
  elif directory.name=='H0':logs=[R/'arms/H0-pilot/updates.jsonl',*logs]
  resolved_path=directory/'resolved.json'
  if resolved_path.exists():
   lineage=json.load(open(resolved_path)).get('lineage',{})
   logs=[*[Path(p)/'updates.jsonl' for p in lineage.get('resume_source_arms',[])],*logs]
  logs=list(dict.fromkeys(logs))
  events=sorted([json.loads(line) for log in logs if log.exists() for line in log.read_text().splitlines()],key=lambda x:x['update']);seen=set();cursor=0
  for point in records:
   while cursor<len(events) and events[cursor]['update']<=point['update']:
    event=events[cursor];seen.update((event['sequence'],event['anchor']+j-1) for j in range(1,16));cursor+=1
   step=point['update'];sidecar=Path(point['model']['path']).with_suffix('.component.json')
   component=json.load(open(sidecar));nm=native_metrics(R/f'smoke-{directory.name}-{step}',directory.name+'-'+str(step))
   row=dict(arm=directory.name,updates=step,component_spec=spec,head=spec.get('head','target-shared Q6_K'),embedding=spec.get('embedding','target-shared Q6_K'),tied=bool(spec.get('tied')),k=spec.get('kv',['f16','f16'])[0],v=spec.get('kv',['f16','f16'])[1],scalar_f16_groups=','.join(spec.get('scalar',{})),ce=point['dev']['ce'],agreement=point['dev']['agreement'],mean_rank=point['dev']['mean_rank'],mean_margin=point['dev']['mean_margin'],tensor_bytes=component['tensor_bytes'],training_peak_allocator_bytes=point['peak_training_bytes'],training_update_seconds=point['progress']['update_seconds'],supervised_positions=point['progress']['supervised_tokens'],unique_label_positions=result.get('unique_label_positions'),native=nm,status='recovering' if result.get('status')!='COMPLETE' else 'window complete; frontier decision pending',source_readout=str(path),model=point['model'],clean_gpu_forward_ms=None,full_draft_host_ms=None,final_request_tok_s=None,final_accepted=None,final_proposed=None,actual_gpu_model_bytes=None,actual_cpu_model_bytes=None,draft_cache_bytes=None)
   row.update(physical_updates=0 if directory.name in ('H6','E6') else step,training_peak_provenance='Measured H0 identical frozen-forward reuse' if directory.name in ('H6','E6') else 'Measured current trainer',unique_supervised_positions_exposed=len(seen),repeated_supervised_positions=max(0,point['progress']['supervised_tokens']-len(seen)),admitted_train_label_pool=1762,unique_coverage_source=[str(p) for p in logs])
   row.update({k:v for k,v in nm.items() if k not in ('prefix_counts',)})
   row.update(full_vocabulary_rows=248320,vocabulary_width=2560,additional_trainable_vocabulary_scalars=2483200 if spec.get('r2_vocab_scale') else 0,private_tying=bool(spec.get('tied')),vocabulary_tying='One identical private quantized matrix/device allocation' if spec.get('tied') else 'Frozen source-equivalent tables; target-sharing/private copies per head and embedding fields',ffn_weights='W1 all18',gate_up_activation='A4 learned clipping',down_activation='A8 learned clipping',fusion_weights='Q4_0',attention_qk_weights='Q4_0',attention_vo_weights='Q4_0',row_scale_storage='F16' if spec.get('scalar',{}).get('scale') else 'F32',midpoint_storage='F16' if spec.get('scalar',{}).get('midpoint') else 'F32',positive_gain_storage='F16' if spec.get('scalar',{}).get('gain') else 'F32',norm_gain_storage='F16' if spec.get('scalar',{}).get('norm') else 'F32',clip_storage='F16-rounded value in F32 GGUF metadata' if spec.get('scalar',{}).get('clip') else 'F32',trainable_policy='R2 native vocabulary block scales plus R1 groups' if spec.get('r2_vocab_scale') else 'R1 FFN sign/scale/midpoint/shared gate-up gain/separate down gain/learned clip/norm',model_path=point['model']['path'],model_sha256=point['model']['sha256'])
   recovery=json.load(open(directory/'recovery.json')) if (directory/'recovery.json').exists() else result.get('recovery',{})
   row['full_recovery_state_status']='retained' if recovery.get('archive') and Path(recovery['archive']).exists() else 'retired under NEW rolling policy; B0/native states retained'
   rows.append(row)
 control={r['updates']:r for r in rows if r['arm']=='H0'}
 for r in rows:
  c=control.get(r['updates']);r['ce_gap_matched_control']=r['ce']-c['ce'] if c else None
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
 groups={'Head':['H0','H6','H4','H3'],'Embedding and tying':['H0','E6','E4','HE4'],'Draft K/V':['H0',*[n for n in names if n.startswith('K')]],'Scalar storage':['H0',*[n for n in names if n.startswith('S')]]}
 for field,label,filename in [('ce_gap_matched_control','DEV cross entropy gap','component-recovery'),('acceptance','Native accepted / proposed','component-native-acceptance')]:
  fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
  for ax,(title,members) in zip(axes.flat,groups.items()):
   for name in members:
    points=sorted([r for r in rows if r['arm']==name and r.get(field) is not None],key=lambda x:x['updates'])
    if points:ax.plot([r['updates'] for r in points],[r[field] for r in points],marker='o',ms=3,label=name,color=colors[name])
   ax.set(title=title,xlabel='Additional updates from B0',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
  fig.suptitle('Matched recovery on consumed DEV; native readouts are short diagnostics');fig.tight_layout();fig.savefig(out/(filename+'.png'));fig.savefig(out/(filename+'.svg'));plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.4))
 final=[max([r for r in rows if r['arm']==n],key=lambda x:x['updates']) for n in names]
 acceptance_field='final_acceptance' if any(r.get('final_acceptance') is not None for r in final) else 'acceptance'
 acceptance_scope=next((r.get('final_request_cohort') for r in final if r.get('final_acceptance') is not None),'short consumed DEV smoke')
 for r in final:
  n=r['arm'];mem=r['tensor_bytes']/2**20
  for ax,x,y in [(axes[0],mem,r.get(acceptance_field)),(axes[1],r.get('clean_gpu_forward_ms'),r.get(acceptance_field)),(axes[2],mem,r.get('final_request_tok_s'))]:
   if x is not None and y is not None:ax.scatter(x,y,color=colors[n]);ax.annotate(n,(x,y),fontsize=7)
 axes[0].set(xlabel='Private drafter tensor MiB (shared target excluded)',ylabel='Native accepted / proposed',title='Storage / acceptance')
 axes[1].set(xlabel='Clean GPU forward ms',ylabel='Native accepted / proposed',title='Latency / acceptance')
 axes[2].set(xlabel='Private drafter tensor MiB',ylabel='Formal request tokens / second',title='Full-request frontier')
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Measured component tradeoffs; acceptance cohort: '+acceptance_scope+'; unmeasured fields blank');fig.tight_layout();fig.savefig(out/'tradeoffs.png');fig.savefig(out/'tradeoffs.svg');plt.close(fig)

HTML='''<!doctype html><meta charset="utf-8"><title>DFlash component campaign</title>
<style>body{font:15px system-ui;margin:24px;color:#182438;background:#f6f8fb}h1{font-size:24px}.note{max-width:1100px;color:#47566b}#filters{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}label{background:white;padding:5px;border-radius:4px}svg{background:white;border:1px solid #ccd5e3;width:100%;height:480px}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:7px;border:1px solid #dce3ef}#tip{position:fixed;background:#182438;color:white;padding:8px;display:none;pointer-events:none;border-radius:4px}select{padding:6px}</style>
<h1>Remaining precision components</h1><p class="note">Exact learned-small2048 B0. Frozen target Qwen3.5-4B Q4_K_M, RTX5080, consistent decode32. Recovery is compared with equally trained H0. Native short readouts are consumed-DEV diagnostics; formal warm repeated request measurements are labeled separately. Training histories, repeated labels and timing repetitions are different sources of uncertainty. Empty fields mean unmeasured.</p>
<select id="metric"><option value="ce_gap_matched_control">DEV loss gap to matched control</option><option value="ce">Dense DEV loss</option><option value="acceptance">Native acceptance (short readout)</option><option value="request_tok_s">Native short-readout request tok/s</option><option value="clean_gpu_forward_ms">Clean GPU forward ms</option><option value="final_request_tok_s">Formal request tok/s</option><option value="tensor_bytes">Private drafter tensor bytes</option></select><div id="filters"></div><svg id="plot" viewBox="0 0 1100 480"></svg><div id="tip"></div><p class="note" id="status"></p><div style="overflow:auto"><table id="table"></table></div>
<script>const data=__DATA__;const names=[...new Set(data.map(x=>x.arm))];const selected=new Set(names);const palette=['#2166ac','#b2182b','#1b7837','#762a83','#b35806','#008b8b','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'];const colors=Object.fromEntries(names.map((n,i)=>[n,palette[i%palette.length]]));const ns='http://www.w3.org/2000/svg';const plot=document.querySelector('#plot');function el(tag,attrs,text){let e=document.createElementNS(ns,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;plot.appendChild(e);return e}names.forEach(n=>{let l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.checked=true;c.onchange=()=>{c.checked?selected.add(n):selected.delete(n);draw()};l.append(c,document.createTextNode(n));l.style.color=colors[n];document.querySelector('#filters').append(l)});function draw(){let field=document.querySelector('#metric').value,rows=data.filter(x=>selected.has(x.arm)&&x[field]!==null&&x[field]!==undefined);plot.replaceChildren();let vals=rows.map(x=>x[field]);let lo=Math.min(...vals,0),hi=Math.max(...vals,1);if(hi===lo)hi=lo+1;let maxUpdates=Math.max(4096,...data.map(x=>x.updates));let X=x=>75+x/maxUpdates*970,Y=y=>415-(y-lo)/(hi-lo)*350;el('line',{x1:75,y1:415,x2:1045,y2:415,stroke:'#71829b'});el('line',{x1:75,y1:65,x2:75,y2:415,stroke:'#71829b'});[...new Set([0,256,1024,4096,maxUpdates])].forEach(v=>el('text',{x:X(v),y:440,'text-anchor':'middle'},v));el('text',{x:560,y:468,'text-anchor':'middle'},'Additional updates from frozen B0');for(let i=0;i<=5;i++){let v=lo+(hi-lo)*i/5;el('line',{x1:75,y1:Y(v),x2:1045,y2:Y(v),stroke:'#e7ecf3'});el('text',{x:68,y:Y(v)+4,'text-anchor':'end'},v.toPrecision(3))}names.filter(n=>selected.has(n)).forEach(n=>{let points=rows.filter(x=>x.arm===n).sort((a,b)=>a.updates-b.updates);if(!points.length)return;el('polyline',{points:points.map(x=>[X(x.updates),Y(x[field])].join(',')).join(' '),stroke:colors[n],fill:'none','stroke-width':2});points.forEach(r=>{let c=el('circle',{cx:X(r.updates),cy:Y(r[field]),r:5,fill:colors[n]});c.onmousemove=e=>{let t=document.querySelector('#tip');t.style.display='block';t.style.left=e.clientX+12+'px';t.style.top=e.clientY+12+'px';t.textContent=r.arm+' · '+r.updates+' updates · '+field+' '+r[field]};c.onmouseleave=()=>document.querySelector('#tip').style.display='none'})});document.querySelector('#status').textContent=data.length+' measured readouts. '+(rows.length?'Hover points for values.':'No measurements yet for this field.');let columns=['arm','updates','ce','ce_gap_matched_control','agreement','accepted','proposed','acceptance','request_tok_s','tensor_bytes','status'];let table=document.querySelector('#table');table.replaceChildren();let head=document.createElement('tr');columns.forEach(k=>{let c=document.createElement('th');c.textContent=k;head.append(c)});table.append(head);data.filter(r=>selected.has(r.arm)).forEach(r=>{let tr=document.createElement('tr');columns.forEach(k=>{let td=document.createElement('td');let v=r[k];td.textContent=v===undefined||v===null?'—':typeof v==='number'?Number.isInteger(v)?v:v.toFixed(5):v;tr.append(td)});table.append(tr)})}document.querySelector('#metric').onchange=draw;draw();</script>'''

def main():
 out=R/'charts';out.mkdir(exist_ok=True);rows=collect();index=json.load(open(R/'publication-index.json')) if (R/'publication-index.json').exists() else {}
 save(out/'comparison.json',dict(status=index.get('status','IN_PROGRESS'),rows=rows,freeze=pin(R/'freeze.json'),budget=pin(R/'budget.json'),measurement_note='Null means unmeasured. Short native readouts are not formal final repeated comparisons.'))
 keys=sorted({k for r in rows for k in r if k not in ('native','component_spec','model')})
 with (out/'full-table.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r.get(k) for k in keys} for r in rows)
 charts(rows,out);(out/'comparison.html').write_text(HTML.replace('__DATA__',json.dumps(rows).replace('</','<\\/')))
 print('PUBLISHED',len(rows),'readouts',out,flush=True)

if __name__=='__main__':main()
