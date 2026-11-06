"""Standalone recovery and measured Pareto views, without external libraries."""
import json

TEMPLATE=r'''<!doctype html><meta charset="utf-8"><title>DFlash component recovery</title>
<style>
body{font:14px system-ui;margin:24px;color:#17243a;background:#f4f7fb}h1{font-size:25px;margin-bottom:8px}.note{max-width:1200px;color:#46566d;line-height:1.5}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:16px 0}select{padding:7px;max-width:370px}#filters{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}#filters label{background:white;padding:5px 8px;border:1px solid #dae2ee;border-radius:5px}button{padding:6px 12px;background:white;border:1px solid #aebdd0;border-radius:5px;cursor:pointer}svg{background:white;border:1px solid #ccd6e4;width:100%;height:500px;border-radius:7px}table{border-collapse:collapse;background:white;font-size:12px}td,th{padding:7px;border:1px solid #dce3ee;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f8}#tip{position:fixed;background:#17243a;color:white;padding:10px;display:none;pointer-events:none;border-radius:5px;max-width:400px;font-size:12px;white-space:pre-line}#status{color:#46566d}
</style>
<h1>Remaining precision components</h1>
<p class="note">Frozen learned-small2048 G4/D8 B0 · Qwen3.5-4B Q4_K_M target · RTX5080 · consistent decode32 · target F16 KV. Recovery gaps use the named equally trained control. Native short readouts, natural12 selection, and final heldout are separate fields. Timing repetitions do not measure training-seed uncertainty.</p>
<div class="controls"><label>Horizontal <select id="x"></select></label><label>Vertical <select id="y"></select></label><label><input type="checkbox" id="latest"> Latest readout only</label><button id="all">All arms</button><button id="none">Clear</button></div>
<div id="filters"></div><svg id="plot" viewBox="0 0 1150 500" role="img" aria-label="Interactive component recovery comparison"></svg><div id="tip"></div><p id="status"></p>
<p class="note">Memory payload = private model tensor bytes + actually allocated draft cache and rotation bytes at4096context, with the immutable shared target excluded. It includes CPU/GPU model payload and is distinct from mapped CPU residency, conversion workspace and sampled device peak. Blank measurements remain blank. All numbers and source pins are retained in the companion JSON/CSV.</p>
<div style="overflow:auto;max-height:650px"><table id="table"></table></div>
<script>
const data=__DATA__;
const fields={updates:'Additional updates from B0',ce_gap_matched_control:'Consumed DEV CE gap to matched control',ce:'Consumed DEV cross entropy',acceptance:'Short DEV accepted / proposed',selection_acceptance:'Selection natural12 acceptance',selection_request_tok_s:'Selection request tok/s · one repetition',selection_clean_gpu_forward_ms:'Selection clean GPU ms · one repetition',final_acceptance:'Final heldout accepted / proposed',final_request_tok_s:'Final heldout request tok/s',clean_gpu_forward_ms:'Formal clean GPU forward ms',full_draft_host_ms:'Formal generation draft host ms / round',full_draft_all_phases_host_ms_per_round:'Formal all-phase draft host ms / round',tensor_bytes:'Private model tensor payload MiB',model_cache_payload_bytes:'Private model + cache/rotation payload MiB'};
const xFields=['updates','model_cache_payload_bytes','tensor_bytes','clean_gpu_forward_ms','selection_clean_gpu_forward_ms','full_draft_host_ms','final_request_tok_s'];
const yFields=Object.keys(fields).filter(k=>k!=='updates');
for(const [selector,keys,initial] of [['#x',xFields,'updates'],['#y',yFields,'ce_gap_matched_control']]){let s=document.querySelector(selector);for(const k of keys){let o=document.createElement('option');o.value=k;o.textContent=fields[k];s.append(o)}s.value=initial;s.onchange=draw}
const names=[...new Set(data.map(x=>x.arm))],selected=new Set(names),palette=['#2166ac','#b2182b','#1b7837','#762a83','#b35806','#008b8b','#d95f02','#7570b3','#e7298a','#66a61e','#a6761d','#65758b'];
const colors=Object.fromEntries(names.map((n,i)=>[n,palette[i%palette.length]])),plot=document.querySelector('#plot'),ns='http://www.w3.org/2000/svg';
const maxUpdate=Object.fromEntries(names.map(n=>[n,Math.max(...data.filter(r=>r.arm===n).map(r=>r.updates))]));
const number=(r,f)=>r[f]===null||r[f]===undefined?null:Number(r[f])/(f.endsWith('_bytes')?1048576:1);
function element(tag,attrs,text){let e=document.createElementNS(ns,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;plot.append(e);return e}
function pretty(v){return Math.abs(v)>=1000?v.toFixed(0):Math.abs(v)>=10?v.toFixed(2):v.toPrecision(4)}
for(const n of names){let l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.checked=true;c.dataset.arm=n;c.onchange=()=>{c.checked?selected.add(n):selected.delete(n);draw()};l.style.color=colors[n];l.append(c,document.createTextNode(n));document.querySelector('#filters').append(l)}
for(const [id,on] of [['all',true],['none',false]])document.querySelector('#'+id).onclick=()=>{document.querySelectorAll('#filters input').forEach(c=>{c.checked=on;on?selected.add(c.dataset.arm):selected.delete(c.dataset.arm)});draw()};
document.querySelector('#latest').onchange=draw;
function draw(){
 let xf=document.querySelector('#x').value,yf=document.querySelector('#y').value,scatter=xf!=='updates',latest=scatter||document.querySelector('#latest').checked;
 let all=data.filter(r=>selected.has(r.arm)&&(!latest||r.updates===maxUpdate[r.arm]));
 let rows=all.filter(r=>number(r,xf)!==null&&number(r,yf)!==null);plot.replaceChildren();
 let xs=rows.map(r=>number(r,xf)),ys=rows.map(r=>number(r,yf));
 let xlo=scatter&&xs.length?Math.min(...xs)*.94:0,xhi=Math.max(...xs,scatter?1:4096),ylo=Math.min(...ys,0),yhi=Math.max(...ys,1e-6);
 if(xhi===xlo)xhi=xlo+1;if(yhi===ylo)yhi=ylo+1;yhi+=(yhi-ylo)*.06;if(scatter)xhi+=(xhi-xlo)*.06;
 let X=x=>85+(x-xlo)/(xhi-xlo)*1010,Y=y=>425-(y-ylo)/(yhi-ylo)*350;
 element('line',{x1:85,y1:425,x2:1095,y2:425,stroke:'#6c7d94'});element('line',{x1:85,y1:75,x2:85,y2:425,stroke:'#6c7d94'});
 let ticks=scatter?Array.from({length:6},(_,i)=>xlo+(xhi-xlo)*i/5):[...new Set([0,256,1024,4096,Math.max(4096,...xs)])];
 for(const v of ticks)element('text',{x:X(v),y:450,'text-anchor':'middle','font-size':12},pretty(v));
 for(let i=0;i<=5;i++){let v=ylo+(yhi-ylo)*i/5; element('line',{x1:85,y1:Y(v),x2:1095,y2:Y(v),stroke:'#e5ebf3'});element('text',{x:77,y:Y(v)+4,'text-anchor':'end','font-size':12},pretty(v))}
 element('text',{x:590,y:484,'text-anchor':'middle','font-size':14},fields[xf]);element('text',{x:85,y:30,'font-size':14,fill:'#263750'},fields[yf]+(scatter?' · latest readouts':' · recovery history'));
 for(const n of names.filter(n=>selected.has(n))){let points=rows.filter(r=>r.arm===n).sort((a,b)=>a.updates-b.updates);if(!scatter&&points.length>1)element('polyline',{points:points.map(r=>[X(number(r,xf)),Y(number(r,yf))].join(',')).join(' '),fill:'none',stroke:colors[n],'stroke-width':2});for(const r of points){let circle=element('circle',{cx:X(number(r,xf)),cy:Y(number(r,yf)),r:5,fill:colors[n],stroke:'white','stroke-width':1});if(scatter)element('text',{x:X(number(r,xf))+7,y:Y(number(r,yf))-7,fill:colors[n],'font-size':11},n);circle.onmousemove=e=>{let tip=document.querySelector('#tip'),q=r.final_vs_q4;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,window.innerWidth-420)+'px';tip.style.top=e.clientY+12+'px';tip.textContent=n+' · '+r.updates+' updates\n'+fields[xf]+': '+pretty(number(r,xf))+'\n'+fields[yf]+': '+pretty(number(r,yf))+'\n'+(r.final_request_cohort||r.selection_request_cohort||'Consumed DEV readout')+'\n'+r.status+(q?'\nRequest ratio to Q4: '+q.ratio.toFixed(4)+' ['+q.paired_95_interval.map(x=>x.toFixed(4)).join(', ')+']':'')};circle.onmouseleave=()=>document.querySelector('#tip').style.display='none'}}
 document.querySelector('#status').textContent=data.length+' measured readouts; '+rows.length+' points shown. '+(rows.length?'Hover a point for values and cohort.':'This measurement has not been recorded for the selected arms.');
 const columns=['arm','updates','matched_recovery_control','ce','ce_gap_matched_control','agreement','accepted','proposed','unique_supervised_positions_exposed','tensor_bytes','draft_cache_bytes','clean_gpu_forward_ms','full_draft_host_ms','final_request_tok_s','final_request_cohort','status'];let table=document.querySelector('#table');table.replaceChildren();let tr=document.createElement('tr');for(const k of columns){let th=document.createElement('th');th.textContent=k;tr.append(th)}table.append(tr);
 for(const r of all){let tr=document.createElement('tr');for(const k of columns){let td=document.createElement('td'),v=r[k];td.textContent=v===null||v===undefined?'—':typeof v==='number'&&!Number.isInteger(v)?v.toFixed(5):v;tr.append(td)}table.append(tr)}
}
draw();
</script>'''

def render(rows):
 return TEMPLATE.replace('__DATA__',json.dumps(rows).replace('</','<\\/'))
