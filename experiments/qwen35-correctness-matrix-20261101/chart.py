"""Render the audited matrix as a self-contained interactive HTML artifact."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = list(csv.DictReader(args.matrix.open()))
    payload = json.dumps({'targets': summary['targets'], 'rows': rows})
    html = r'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
html,body{margin:0;background:transparent;color:var(--foreground);font-family:var(--font-sans)}
h2{font-size:17px;margin:0 0 5px;font-weight:650}.sub,.note{font-size:12px;line-height:1.5;color:var(--muted-foreground)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin:13px 0 12px}.dot{width:9px;height:9px;display:inline-block;border-radius:50%;margin-right:5px}
.group{margin:0 0 13px}.grouphead{display:flex;justify-content:space-between;gap:12px;font-size:12px;margin-bottom:5px}.bestname{color:var(--muted-foreground);font-size:11px;text-align:right}
.overview{height:360px}.r{display:grid;grid-template-columns:119px minmax(0,1fr) 48px;gap:8px;align-items:center;height:18px;font-size:11px}.track{height:11px;background:var(--secondary);border-radius:3px}.bar{height:100%;border-radius:3px}.v{text-align:right;font-variant-numeric:tabular-nums}.muted{color:var(--muted-foreground)}
.toolbar{display:flex;gap:12px;align-items:center;justify-content:space-between;margin:19px 0 9px;font-size:13px}select{font:inherit;color:var(--foreground);background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:6px 8px;max-width:55%}
.detail{height:472px}.detail .r{grid-template-columns:190px minmax(0,1fr) 48px;height:28px;font-size:11px}.detail .track{height:17px}.detail .label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.detail .primary .label{font-weight:650}.unit{text-align:right;font-size:11px;color:var(--muted-foreground);margin:4px 0}
@media(max-width:450px){.legend{gap:8px}.grouphead{font-size:11px}.bestname{max-width:55%}.r{grid-template-columns:96px minmax(0,1fr) 43px;gap:5px}.detail .r{grid-template-columns:151px minmax(0,1fr) 43px;gap:5px;font-size:10px}.toolbar{font-size:12px;gap:6px}select{padding:5px 3px}.overview{height:380px}}
</style></head><body><h2>Qwen3.5 correctness mode · RTX 5080</h2><div class="sub">Held-out full-request throughput · five repetitions per configuration · higher is faster</div>
<div class="legend"><span><i class="dot" style="background:var(--chart-3)"></i>Target only</span><span><i class="dot" style="background:var(--chart-1)"></i>Q4_0 EAGLE</span><span><i class="dot" style="background:var(--chart-2)"></i>Fastest observed</span></div>
<div id="overview" class="overview"></div><div class="note">Close rates can be unresolved on this prompt panel. The linked report includes paired prompt-cluster intervals.</div>
<div class="toolbar"><strong>Full 58-row matrix</strong><select id="target" aria-label="Target model and precision"></select></div><div class="unit">generated tokens / full HTTP request second</div><div id="detail" class="detail"></div>
<div class="note">Draft weight format and selected draft length appear in each label. Weight-only formats are not W4A4 execution. All rows use the same correctness runtime, target-specific reference, F16 KV, context 4096, concurrency one, thinking off and a 256-token performance limit. This measures complete requests, not concurrent serving capacity. No W1A8 model was trained or evaluated.</div>
<script>const DATA=__PAYLOAD__;
const families={target_only:'Target only',mtp:'MTP',eagle3:'EAGLE3',dflash:'DFlash',dspark:'DSpark',simple_0p8b:'0.8B draft'};
const color={target:'var(--chart-3)',primary:'var(--chart-1)',best:'var(--chart-2)',other:'var(--chart-4)'};
function title(id){return id.replace('-',' · ')}
function name(r){return families[r.family]+(r.family==='target_only'?'':r.family==='mtp'?' · d'+r.draft_depth:' '+r.draft_precision+' · d'+r.draft_depth)}
function row(label,value,c,kind='',hint=''){const e=document.createElement('div');e.className='r '+kind;e.title=hint;const l=document.createElement('span');l.className='label';l.textContent=label;const tr=document.createElement('div');tr.className='track';const b=document.createElement('div');b.className='bar';b.style.width=(value/300*100)+'%';b.style.background=c;tr.append(b);const v=document.createElement('span');v.className='v';v.textContent=Number(value).toFixed(1);e.append(l,tr,v);return e}
const select=document.getElementById('target');
for(const [target,t] of Object.entries(DATA.targets)){const group=document.createElement('div');group.className='group';const head=document.createElement('div');head.className='grouphead';const label=document.createElement('strong');label.textContent=title(target);const best=document.createElement('span');best.className='bestname';best.textContent=name(t.fastest_observed);head.append(label,best);group.append(head,row('Target only',t.target_only.tokens_per_second,color.target),row('Q4_0 EAGLE',t.primary_q4_0_eagle.tokens_per_second,color.primary),row('Fastest observed',t.fastest_observed.tokens_per_second,color.best));document.getElementById('overview').append(group);const option=document.createElement('option');option.value=target;option.textContent=title(target);select.append(option)}
function detail(){const t=DATA.targets[select.value];const rows=DATA.rows.filter(r=>r.target===select.value).sort((a,b)=>Number(b.tokens_per_second)-Number(a.tokens_per_second));const box=document.getElementById('detail');box.replaceChildren();for(const r of rows){const primary=r.arm_id===t.primary_q4_0_eagle.arm_id;const best=r.arm_id===t.fastest_observed.arm_id;const c=primary?color.primary:best?color.best:r.family==='target_only'?color.target:color.other;const hint=name(r)+'; '+Number(r.speedup_vs_target_only).toFixed(3)+'× target-only; '+Number(r.speedup_vs_primary_q4_0).toFixed(3)+'× Q4_0 EAGLE; accepted '+r.accepted+'/'+r.proposed+'; quality '+r.quality_passed+'/'+r.quality_cases; repetition wall CV '+Number(r.repetition_wall_cv_percent).toFixed(2)+'%';box.append(row(name(r)+(primary?' (primary)':''),Number(r.tokens_per_second),c,primary?'primary':'',hint))}}
select.addEventListener('change',detail);detail();</script></body></html>'''
    args.output.write_text(html.replace('__PAYLOAD__', payload))


if __name__ == '__main__':
    main()
