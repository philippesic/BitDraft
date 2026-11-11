"""Read-only comparison of existing update-hour panels; no new GPU acquisition."""
import argparse
import json
import numpy as np
from .common import R,OUT,REPORT,pin,save
from scripts.component_campaign.metrics import summarize


def compare(older=5,newer=6):
    assert 0<older<newer
    hours=[older,newer];panels={};summaries={};rep_seconds={};quality={}
    for hour in hours:
        path=R/f'formal-update-{hour*3600}/result.json';raw=json.load(open(path))
        assert raw['status']=='COMPLETE' and all(row['correct'] for row in raw['rows'])
        rows=[row for row in raw['rows'] if not row.get('warmup')]
        panels[hour]=pin(path);summaries[hour]=summarize(path);rep_seconds[hour]={};quality[hour]={}
        for arm in ['HE4broad','q4','target']:
            rep_seconds[hour][arm]=np.array([sum(x['wall_seconds'] for x in rows if x['arm']==arm and x['rep']==j) for j in range(5)])
        for prompt in sorted({x['prompt'] for x in rows}):
            selected=[x for x in rows if x['arm']=='HE4broad' and x['prompt']==prompt]
            signatures={(x['tokens'],x['native']['accepted'],x['native']['proposed'],x['native']['rounds']) for x in selected}
            assert len(signatures)==1 and len(selected)==5
            tokens,accepted,proposed,rounds=next(iter(signatures))
            quality[hour][prompt]=dict(tokens=tokens,accepted=accepted,proposed=proposed,rounds=rounds,
                acceptance=accepted/proposed,request_tok_s=sum(x['tokens'] for x in selected)/sum(x['wall_seconds'] for x in selected))
    tokens_per_pass=summaries[older]['HE4broad']['output_tokens']/5
    assert all(summary['output_tokens']/5==tokens_per_pass for hour in hours for summary in summaries[hour].values())
    # Independent resampling across hours; bundle all fixed prompts and arms
    # within each repetition. This cannot measure between-hour hardware drift
    # or generalization to new prompts. Do not bootstrap deterministic tokens.
    rng=np.random.default_rng(61110506);i=rng.integers(0,5,(20000,5));j=rng.integers(0,5,(20000,5))
    ratio=rep_seconds[older]['HE4broad'][i].sum(1)/rep_seconds[newer]['HE4broad'][j].sum(1)
    normalized=(rep_seconds[newer]['q4'][j].sum(1)/rep_seconds[newer]['HE4broad'][j].sum(1))/(rep_seconds[older]['q4'][i].sum(1)/rep_seconds[older]['HE4broad'][i].sum(1))
    changes={}
    for arm in ['HE4broad','q4','target']:
        a=summaries[older][arm]['request_tok_s'];b=summaries[newer][arm]['request_tok_s']
        changes[arm]=dict(old=a,new=b,delta=b-a,relative_change=b/a-1,
            old_repetition_rates=(tokens_per_pass/rep_seconds[older][arm]).tolist(),new_repetition_rates=(tokens_per_pass/rep_seconds[newer][arm]).tolist())
    readouts={x['label']:x for x in json.load(open(OUT/'readouts.json'))}
    audit=json.load(open(REPORT/'update-plateau-audit.json'))
    result=dict(older_hour=older,newer_hour=newer,panel_sources=panels,summaries=summaries,changes=changes,per_prompt=quality,
        fixed_three_prompt_token_total=tokens_per_pass,
        acceptance_percentage_point_change=100*(summaries[newer]['HE4broad']['acceptance']-summaries[older]['HE4broad']['acceptance']),
        extra_accepted_tokens_per_three_prompt_pass=sum(x['accepted'] for x in quality[newer].values())-sum(x['accepted'] for x in quality[older].values()),
        repeat_only_bootstrap=dict(draws=20000,seed=61110506,throughput_ratio=summaries[newer]['HE4broad']['request_tok_s']/summaries[older]['HE4broad']['request_tok_s'],
            percentile_95_interval=np.quantile(ratio,[.025,.975]).tolist(),normalized_to_q4_percentile_95_interval=np.quantile(normalized,[.025,.975]).tolist(),
            limitations='Fixed three prompts; independently resampled five repetition bundles per hour; arms paired only within the same hour/repetition. Does not capture between-hour drift, new prompts or training seeds.'),
        dev={hour:dict(ce=readouts[f'update-{hour*3600}']['dev']['ce'],agreement=readouts[f'update-{hour*3600}']['dev']['agreement'],positions=135) for hour in hours},
        training_curve=audit['learning_curve'],exposure=audit['exposure'],integrity_audit=pin(REPORT/'update-plateau-audit.json'))
    save(REPORT/'update-plateau-comparison.json',result)
    template='''<!doctype html><html><head><meta charset="utf-8"><style>:root{--font-sans:system-ui,sans-serif}body{margin:0;font-family:var(--font-sans);color:var(--foreground)}h3{font-size:18px;margin:12px 0 8px}p{font-size:12px;color:var(--muted-foreground);line-height:1.5}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}th,td{padding:8px 5px;text-align:right;border-bottom:1px solid var(--border)}th:first-child,td:first-child{text-align:left}svg{width:100%;height:240px;display:block}svg text{font:11px var(--font-sans);fill:var(--muted-foreground)}i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.legend{display:flex;gap:20px;font-size:12px}.note{font-size:11px}</style></head><body>
<h3>__OLDER__ versus __NEWER__ update hours</h3><div class="legend"><span><i style="background:var(--chart-1)"></i>5 hours</span><span><i style="background:var(--chart-2)"></i>6 hours</span></div><p>Complete-request throughput (tokens/s). Small dots: individual repetition bundles; diamonds: aggregate. Shared axis is zoomed to 170–260.</p><svg id="plot" role="img" aria-label="Throughput repetitions and aggregate at five and six hours"></svg>
<table><tr><th>End-to-end tokens/s</th><th>__OLDER__ h</th><th>__NEWER__ h</th><th>Change</th></tr>__RATE_ROWS__</table>
<h3>Accepted draft tokens per three-prompt pass</h3><table><tr><th>Prompt</th><th>__OLDER__ h</th><th>__NEWER__ h</th><th>Change</th></tr>__QUALITY_ROWS__</table><p>The totals are __OLD_ACCEPTED__ → __NEW_ACCEPTED__ accepted tokens over the same __TOKENS__ output tokens. Each timing repetition repeats these same deterministic quality results.</p>
<p>Overall draft acceptance: __OLD_ACC__% → __NEW_ACC__% (+__DELTA_PP__ percentage points). Fixed teacher-forced DEV loss: __OLD_CE__ → __NEW_CE__ (lower is better).</p><p class="note">Candidate __NEWER__ h / __OLDER__ h throughput ratio: __RATIO__; repeat-only 95% bootstrap interval __LO__–__HI__. __INTERPRET__ Fixed three prompts, five repetitions per hour; this does not account for between-hour hardware drift or establish generalization. The slow Q4 repetition is retained.</p>
<script>const hours=__HOURS__,rows=__ROWS__,svg=document.getElementById('plot');function draw(){let w=svg.clientWidth,L=w<420?80:100,R=15;const x=v=>L+(v-170)/90*(w-L-R);let s='';for(let v=180;v<=260;v+=20)s+='<line x1="'+x(v)+'" x2="'+x(v)+'" y1="10" y2="208" stroke="var(--border)"/><text x="'+x(v)+'" y="230" text-anchor="middle">'+v+'</text>';rows.forEach((r,i)=>{const y=40+i*70;s+='<text x="4" y="'+(y+3)+'">'+r.name+'</text>';for(let h=0;h<2;h++){const color='var(--chart-'+(h+1)+')',yy=y+(h?9:-9);for(const [j,v] of r.reps[h].entries())s+='<circle cx="'+x(v)+'" cy="'+yy+'" r="3.5" fill="'+color+'" opacity=".45"><title>'+r.name+', '+hours[h]+' h, repetition '+j+': '+v.toFixed(3)+' tokens/s</title></circle>';const xx=x(r.aggregate[h]);s+='<path d="M'+xx+','+(yy-6)+' l6,6 l-6,6 l-6,-6 Z" fill="'+color+'"><title>Aggregate: '+r.aggregate[h].toFixed(3)+'</title></path>'}});svg.setAttribute('viewBox','0 0 '+w+' 240');svg.innerHTML=s}new ResizeObserver(draw).observe(svg);draw();</script></body></html>'''
    names={'HE4broad':'HE4broad','q4':'DFlash Q4','target':'Target only'}
    replacements={'__OLDER__':str(older),'__NEWER__':str(newer),'__TOKENS__':str(int(tokens_per_pass)),'__HOURS__':json.dumps(hours),'__INTERPRET__':'The interval includes 1 (no change).' if np.quantile(ratio,.025)<=1<=np.quantile(ratio,.975) else 'The interval excludes 1 on this conditional repeat-only analysis.',
        '__RATE_ROWS__':''.join(f'<tr><td>{names[a]}</td><td>{c["old"]:.2f}</td><td>{c["new"]:.2f}</td><td>{100*c["relative_change"]:+.2f}%</td></tr>' for a,c in changes.items()),
        '__QUALITY_ROWS__':''.join(f'<tr><td>{p.removeprefix("development-").removesuffix("-00").title()}</td><td>{quality[older][p]["accepted"]}</td><td>{quality[newer][p]["accepted"]}</td><td>{quality[newer][p]["accepted"]-quality[older][p]["accepted"]:+d}</td></tr>' for p in quality[older]),
        '__OLD_ACCEPTED__':str(sum(x['accepted'] for x in quality[older].values())), '__NEW_ACCEPTED__':str(sum(x['accepted'] for x in quality[newer].values())),
        '__OLD_ACC__':f'{100*summaries[older]["HE4broad"]["acceptance"]:.4f}','__NEW_ACC__':f'{100*summaries[newer]["HE4broad"]["acceptance"]:.4f}',
        '__DELTA_PP__':f'{result["acceptance_percentage_point_change"]:.4f}', '__OLD_CE__':f'{result["dev"][older]["ce"]:.4f}', '__NEW_CE__':f'{result["dev"][newer]["ce"]:.4f}',
        '__RATIO__':f'{result["repeat_only_bootstrap"]["throughput_ratio"]:.4f}', '__LO__':f'{np.quantile(ratio,.025):.4f}', '__HI__':f'{np.quantile(ratio,.975):.4f}',
        '__ROWS__':json.dumps([dict(name=names[a],reps=[c['old_repetition_rates'],c['new_repetition_rates']],aggregate=[c['old'],c['new']]) for a,c in changes.items()])}
    for k,v in replacements.items():template=template.replace(k,v)
    (REPORT/'update-plateau-comparison.html').write_text(template)
    print(json.dumps(dict(changes=changes,acceptance_percentage_point_change=result['acceptance_percentage_point_change'],repeat_only_bootstrap=result['repeat_only_bootstrap']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--older',type=int,default=5);p.add_argument('--newer',type=int,default=6);a=p.parse_args();compare(a.older,a.newer)
