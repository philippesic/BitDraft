"""Publish measured matched-loader diagnostics without modifying run state."""
import argparse
import html
import json
from pathlib import Path


def report(source, destination):
    result=json.loads((source/'result.json').read_text())
    assert result['status']=='MATCHED_LOCAL_LOADER_SMOKE_PASS'
    old=json.loads((source/'legacy-result.json').read_text())
    new=json.loads((source/'local-result.json').read_text())
    categories=[('file_reads','File reads'),('sha256','SHA-256'),
        ('array_parse_and_copy','Array parsing / copying'),('finite_feature_check','Full feature finite checks'),
        ('metadata_and_python','Metadata / Python')]
    def overall(row):
        return [row['loader_seconds'],row['update_seconds'],max(0,row['training_loop_seconds']-row['loader_seconds']-row['update_seconds'])]
    def steady(row):
        return sum(x['loader_seconds']+x['update_seconds'] for x in row['records'][1:])
    data=dict(qualification=result,old=old,new=new,
        steady_steps_2_to_32=dict(old_seconds=steady(old),new_seconds=steady(new),speedup=steady(old)/steady(new)),
        note='One matched 32-update diagnostic, serial old then new; same full saved state, windows, optimizer and RNG. Model setup/checkpoint restore excluded. Warm OS caches may affect local reads; no sustained-run or inference-speed claim.')
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'local-loader-comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    colors=['var(--chart-1)','var(--chart-2)','var(--chart-3)']
    limit=max(old['training_loop_seconds'],new['training_loop_seconds'])
    def bar(row):
        segments=''.join(f'<span style="width:{v/limit*100:.4f}%;background:{c}" title="{v:.4f} seconds"></span>' for v,c in zip(overall(row),colors))
        return f'<div class="track">{segments}</div><b>{row["training_loop_seconds"]:.2f} s</b>'
    overview=''.join(f'<div class="bar"><label>{name}</label>{bar(row)}</div>' for name,row in [('Old · SSHFS',old),('New · local',new)])
    legend=''.join(f'<span><i style="background:{c}"></i>{n}</span>' for n,c in zip(['Data loader','GPU/model update','Loop bookkeeping'],colors))
    maximum=max(old['categories'].get('file_reads',0),new['categories'].get('file_reads',0),.0001)
    rows=''
    for key,label in categories:
        a=old['categories'].get(key,0);b=new['categories'].get(key,0)
        rows+=f'<tr><td>{label}<div class="mini"><span style="width:{a/maximum*100:.4f}%;background:var(--chart-1)"></span></div><div class="mini"><span style="width:{b/maximum*100:.4f}%;background:var(--chart-2)"></span></div></td><td>{a:.4f}</td><td>{b:.4f}</td></tr>'
    document=f'''<!doctype html><html><head><meta charset="utf-8"><style>
    :root{{--font-sans:system-ui,sans-serif;--radius:10px}}body{{font-family:var(--font-sans);color:var(--foreground);margin:0}}h2{{font-size:20px;margin:12px 0}}h3{{font-size:16px;margin:20px 0 10px}}p{{font-size:13px;line-height:1.5;color:var(--muted-foreground)}}.metrics{{display:flex;gap:24px;flex-wrap:wrap;margin:18px 0}}.metrics b{{font-size:24px;display:block;color:var(--foreground)}}.metrics small{{font-size:12px;color:var(--muted-foreground)}}.bar{{display:grid;grid-template-columns:94px 1fr 78px;gap:10px;align-items:center;margin:16px 0;font-size:13px}}.track{{height:30px;display:flex;background:var(--secondary);border-radius:4px;overflow:hidden}}.track span{{height:30px}}.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}}i{{display:inline-block;width:10px;height:10px;margin-right:5px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:9px 6px;border-bottom:1px solid var(--border);text-align:right}}td:first-child,th:first-child{{text-align:left;width:64%}}.mini{{height:5px;margin-top:4px;background:var(--secondary)}}.mini span{{height:5px;display:block}}.note{{font-size:12px}}@media(max-width:450px){{.bar{{grid-template-columns:80px 1fr 62px;gap:5px}}}}
    </style></head><body><h2>Local data loading: matched before / after</h2>
    <p>{old['steps']} actual training updates per arm, starting from the same {old['parent_updates']:,}-update checkpoint. Production training stays paused.</p>
    <div class="metrics"><div><b>{result['loop_speedup']:.2f}×</b><small>all 32 updates, including startup</small></div><div><b>{result['loader_speedup']:.2f}×</b><small>data-loader speedup</small></div><div><b>{steady(old)/steady(new):.2f}×</b><small>steps 2–32, loader + update</small></div></div>
    <h3>Overall training-loop time</h3>{overview}<div class="legend">{legend}</div>
    <table><tr><th>Seconds, same updates</th><th>Old</th><th>New</th></tr>{''.join(f'<tr><td>{label}</td><td>{a:.4f}</td><td>{b:.4f}</td></tr>' for label,a,b in zip(['Data loader','GPU/model update','Loop bookkeeping'],overall(old),overall(new)))}</table>
    <p class="note">First-update GPU/model time: old {old['records'][0]['update_seconds']:.2f} s; new {new['records'][0]['update_seconds']:.2f} s. For steps 2–32, loader + update totals are {steady(old):.2f} → {steady(new):.2f} s. This excludes loop bookkeeping and does not project a full 12-hour result.</p>
    <h3>Data loader by category</h3><p class="note">Seconds across all {old['steps']} updates. Paired bars: old (first), new (second); shared linear scale.</p>
    <table><tr><th>Category</th><th>Old seconds</th><th>New seconds</th></tr>{rows}</table>
    <p class="note">The small difference between category totals and loader time is window/cache lookup overhead. One-time complete dataset validation is outside this diagnostic; the local runtime omits repeated hashes and full-array validation.</p>
    <p class="note">{html.escape(data['note'])} Setup / full-state restore: old {old['setup_seconds']:.1f} s; new {new['setup_seconds']:.1f} s, outside the bars. Matching losses: {result['losses_match']}; exact final parameters / Adam / RNG digest: {result['exact_final_state_equal']}.</p></body></html>'''
    (destination/'local-loader-comparison.html').write_text(document)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();report(args.source,args.destination)
