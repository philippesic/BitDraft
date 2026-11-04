from pathlib import Path
import json
P=Path('/home/philip/binary-eagle-decoding');s=json.loads((P/'runs/binary-screen-20261104-01/summary.json').read_text());q=s['quality_variants'];panels=[]
rows=[]
for key,label in [('A8_control','A8 control'),('A4_mlp_only','MLP A4'),('A1_mlp_only','MLP A1'),('A4_all','All A4'),('A1_all','All A1')]:
 n=q[key]['gold_top1'];rows.append(f'<div class="row"><span>{label}</span><div class="track"><i style="width:{n/24*100}%"></i></div><b>{n}/24</b></div>')
panels.append('<section><h3>Small TRAIN probe</h3><div class="caption">Target top1 matches · higher is better</div>'+''.join(rows)+'<p>12 groups,24 teacher-forced positions. A8-trained parameters, zero adaptation. These are not native acceptance rates or generalization results.</p></section>')
rows=[]
for key,label in [('W1A8_DP4A','A8 DP4A'),('W1A8_bitserial','A8 bit-serial'),('W1A4_bitserial','A4 bit-serial'),('W1A1_XOR_POPC','A1 popcount')]:
 x=next(r for r in s['kernel_timings']['kernel-screen-v2'] if r['m']==50000 and r['n']==1 and r['mode']==key);v=x['us'];rows.append(f'<div class="row"><span>{label}</span><div class="track"><i class="cost" style="width:{v/30*100:.3f}%"></i></div><b>{v:.1f}µs</b></div>')
panels.append('<section><h3>One-row head kernel</h3><div class="caption">Packing + dot · lower is better</div>'+''.join(rows)+'<p>50,000×2,560 matrix, warm graph timing on RTX5080. Independent prototypes; these costs do not predict a policy’s complete decoding speed.</p></section>')
html='''<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;color:var(--foreground);font-family:var(--font-sans)}.panels{display:grid;grid-template-columns:1fr 1fr;gap:30px}h3{font-size:16px;margin:0 0 7px}.caption{font-size:12px;color:var(--muted-foreground);margin-bottom:12px}.row{display:grid;grid-template-columns:82px minmax(40px,1fr) 46px;gap:8px;align-items:center;height:34px;font-size:12px}.track{height:19px;background:var(--secondary);border-radius:3px;overflow:hidden}.track i{display:block;height:19px;background:var(--chart-1)}.track .cost{background:var(--chart-2)}b{font-weight:600;font-variant-numeric:tabular-nums}p{font-size:12px;line-height:1.5;color:var(--muted-foreground);margin:14px 0 0}@media(max-width:550px){.panels{grid-template-columns:1fr;gap:24px}}</style></head><body><div class="panels">'''+''.join(panels)+'</div></body></html>'
out=Path(__file__).resolve().parents[2]/'experiments/binary-screen-20261104/results.html';out.write_text(html)
