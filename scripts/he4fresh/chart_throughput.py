"""Plot completed formal request-throughput panels against actual update time."""
import hashlib
import json
from pathlib import Path

from .common import REPORT


def main():
    source=REPORT/'results.json';raw=source.read_bytes();results=json.loads(raw)
    readouts={r['label']:r for r in results['training']['readouts']}
    points=[]
    for panel,summary in results['panels'].items():
        if not panel.startswith('formal-'):continue
        row=readouts[panel.removeprefix('formal-')]
        assert all(arm['correct'] for arm in summary['arms'].values())
        points.append(dict(panel=panel,update_seconds=row['additional_update_seconds'],updates=row['updates'],
            he4=summary['arms']['HE4broad']['request_tok_s'],q4=summary['arms']['q4']['request_tok_s'],
            target=summary['arms']['target']['request_tok_s']))
    points.sort(key=lambda p:p['update_seconds'])
    data=dict(metric='Output tokens / complete-request wall seconds, including prompt processing',
        source_results_sha256=hashlib.sha256(raw).hexdigest(),points=points,
        limitations=results['limitations'],protocol='Consumed DEV3; five measured repetitions per prompt/arm, two warmups, 256-token output reserve. Formal panels only.')
    (REPORT/'throughput-progress.json').write_text(json.dumps(data,indent=2)+'\n')
    template='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--font-sans:system-ui,sans-serif;--radius:8px}body{margin:0;font-family:var(--font-sans);color:var(--foreground)}h3{font-size:19px;margin:10px 0 6px}p{font-size:12px;line-height:1.5;color:var(--muted-foreground);margin:7px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 4px}.metric b{display:block;font-size:25px;font-variant-numeric:tabular-nums}.metric small{font-size:12px}i{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px}.he4{color:var(--chart-1)}.q4{color:var(--chart-2)}.target{color:var(--chart-3)}#plot{height:320px;width:100%;display:block;touch-action:pan-y}select{font:inherit;font-size:12px;color:var(--foreground);background:var(--secondary);border:1px solid var(--border);border-radius:6px;padding:6px;max-width:100%}.selection{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0}.selection label{font-size:12px}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 5px;border-bottom:1px solid var(--border)}th:first-child,td:first-child{text-align:left}details{margin-top:16px}summary{font-size:12px;cursor:pointer}.note{font-size:11px}#selected{min-height:18px}svg text{font-family:var(--font-sans);font-size:11px;fill:var(--muted-foreground)}@media(max-width:420px){.metric b{font-size:21px}.metrics{gap:7px}}
</style></head><body><h3>End-to-end model throughput so far</h3><p>Output tokens per complete-request second · higher is faster</p>
<div class="metrics"><div class="metric he4"><small><i style="background:var(--chart-1)"></i>HE4broad</small><b id="he4"></b></div><div class="metric q4"><small><i style="background:var(--chart-2)"></i>DFlash Q4</small><b id="q4"></b></div><div class="metric target"><small><i style="background:var(--chart-3)"></i>Target only</small><b id="target"></b></div></div>
<p id="selected"></p><svg id="plot" role="img" aria-label="Complete-request throughput across cumulative optimizer update hours"></svg>
<div class="selection"><label for="point">Inspect checkpoint</label><select id="point"></select></div>
<p id="gap"></p><p class="note">X-axis is cumulative synchronized optimizer-update time, including the 47.9 minutes before local-data resume. It excludes loading, setup, checkpoints and evaluations. Only completed formal evaluations are plotted; no extrapolation to the eight-hour goal.</p>
<p class="note">Three consumed DEV prompts × five measured repetitions per arm; two warmups; 256-token output reserve. Complete-request wall time includes prompt processing. All formal requests match target output. The separate 64-token smoke is omitted. Single training trajectory; these are not heldout results.</p>
<details><summary>All measured points</summary><table><thead><tr><th>Update hours</th><th>HE4broad</th><th>DFlash Q4</th><th>Target only</th></tr></thead><tbody id="rows"></tbody></table></details>
<script>
const data=__DATA__,points=data.points,svg=document.getElementById('plot'),select=document.getElementById('point');let selected=points.length-1;
const colors={he4:'var(--chart-1)',q4:'var(--chart-2)',target:'var(--chart-3)'};
points.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(p.update_seconds/3600).toFixed(3)+' h · '+p.updates.toLocaleString()+' updates';select.appendChild(o);const row=document.createElement('tr');row.innerHTML='<td>'+(p.update_seconds/3600).toFixed(3)+'</td>'+['he4','q4','target'].map(k=>'<td>'+p[k].toFixed(1)+'</td>').join('');document.getElementById('rows').appendChild(row)});
const end=points.at(-1);document.getElementById('gap').textContent='Latest completed evaluation: '+(end.update_seconds/3600).toFixed(2)+' update hours. HE4broad is '+((end.he4/end.target-1)*100).toFixed(1)+'% faster than target-only and '+((1-end.he4/end.q4)*100).toFixed(1)+'% below DFlash Q4 in this panel.';
let scaleX;
function draw(){const w=svg.clientWidth,H=320,L=45,R=14,T=18,B=45,maxX=Math.max(.1,end.update_seconds/3600)*1.05,maxY=Math.ceil(Math.max(...points.flatMap(p=>[p.he4,p.q4,p.target]))/50)*50+25;scaleX=x=>L+x/maxX*(w-L-R);const y=v=>H-B-v/maxY*(H-T-B);let out='';for(let v=0;v<=maxY;v+=50){out+='<line x1="'+L+'" x2="'+(w-R)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="var(--border)"/><text x="'+(L-8)+'" y="'+(y(v)+4)+'" text-anchor="end">'+v+'</text>'}let step=w<450?1:.5;for(let v=0;v<=end.update_seconds/3600+.001;v+=step){out+='<text x="'+scaleX(v)+'" y="'+(H-B+21)+'" text-anchor="middle">'+v.toFixed(v%1?1:0)+'</text>'}out+='<text x="'+((L+w-R)/2)+'" y="'+(H-3)+'" text-anchor="middle">Cumulative optimizer update hours</text>';
for(const k of ['target','q4','he4']){out+='<path d="'+points.map((p,i)=>(i?'L':'M')+scaleX(p.update_seconds/3600)+','+y(p[k])).join(' ')+'" stroke="'+colors[k]+'" stroke-width="'+(k==='he4'?3:2)+'" fill="none"/>';for(let i=0;i<points.length;i++){const p=points[i];out+='<circle cx="'+scaleX(p.update_seconds/3600)+'" cy="'+y(p[k])+'" r="'+(i===selected?5:3)+'" fill="'+colors[k]+'"/>'}}const p=points[selected];out+='<line x1="'+scaleX(p.update_seconds/3600)+'" x2="'+scaleX(p.update_seconds/3600)+'" y1="'+T+'" y2="'+(H-B)+'" stroke="var(--muted-foreground)" stroke-dasharray="3 4" opacity=".5"/>';svg.setAttribute('viewBox','0 0 '+w+' '+H);svg.innerHTML=out;
select.value=selected;['he4','q4','target'].forEach(k=>document.getElementById(k).textContent=p[k].toFixed(1));document.getElementById('selected').textContent=(p.update_seconds/3600).toFixed(3)+' update hours · '+p.updates.toLocaleString()+' updates · tokens/s';}
select.onchange=()=>{selected=Number(select.value);draw()};svg.onpointermove=e=>{const x=e.clientX-svg.getBoundingClientRect().left;selected=points.reduce((best,p,i)=>Math.abs(scaleX(p.update_seconds/3600)-x)<Math.abs(scaleX(points[best].update_seconds/3600)-x)?i:best,0);draw()};svg.onpointerleave=()=>{selected=points.length-1;draw()};new ResizeObserver(draw).observe(svg);draw();
</script></body></html>'''
    (REPORT/'throughput-progress.html').write_text(template.replace('__DATA__',json.dumps(data)))
    print(json.dumps(dict(points=len(points),latest=points[-1])))


if __name__=='__main__':main()
