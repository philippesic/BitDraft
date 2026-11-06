"""Compact, cell-specific tables/charts from preserved native evidence."""
import csv
import hashlib
import html
import io
import json
import re
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.component_campaign.common import ROOT, pin, save
from scripts.component_campaign.metrics import allocations, summarize
from scripts.dflash2_smoke.train import R, OUT

DEST=ROOT/'experiments/dflash2-smoke-20261106'
METHODS=['target','q4','dflash2','before','mid300','mid900','after','HE4']
LABELS=dict(target='Target only',q4='DFlash Q4_K_M',dflash2='DFlash2 Q4_K_M',
            before='Csmallbroad before',mid300='Csmallbroad +5min',mid900='Csmallbroad +15min',
            after='Csmallbroad +30min',HE4='HE4 unchanged')


def measured(directory):
    data=json.load(open(directory/'result.json'))
    assert data['status']=='COMPLETE'
    rows=[r for r in data['rows'] if not r['warmup']]
    assert all(r['correct'] for r in rows)
    return data, rows, summarize(directory/'result.json')


def cell_rows(directory):
    plan=json.load(open(directory/'plan.json'));cell=plan['registry']['target']
    _,native,stats=measured(directory/'native')
    _,traces,host=measured(directory/'trace')
    gpu=json.load(open(directory/'gpu/result.json'))
    out=[]
    for method in METHODS:
        row=dict(cell=cell,method=method,label=LABELS[method],depth=plan['registry']['draft_depth'],
                 status='MEASURED' if method in stats else 'UNSUPPORTED_NO_MATCHED_9B_RELEASE' if method=='dflash2' else 'UNTESTED_NO_9B_CANDIDATE_WEIGHTS')
        if method not in stats:
            out.append(row);continue
        if method=='dflash2' and plan.get('dflash2_variant',{}).get('f32_down_fallback'):
            row['label']='DFlash2 Q4 + F32 down'
        row.update(stats[method]);selected=[x for x in native if x['arm']==method]
        assert row['timing_repetitions']==2 and row['prompts']==3 and row['responses']==6
        rates=[]
        for rep in range(2):
            chosen=[x for x in selected if x['rep']==rep]
            rates.append(sum(x['tokens'] for x in chosen)/sum(x['wall_seconds'] for x in chosen))
        row.update(request_rep_rates=rates,request_rep_min=min(rates),request_rep_max=max(rates),
            mean_request_seconds=statistics.mean(x['wall_seconds'] for x in selected),
            max_request_seconds=max(x['wall_seconds'] for x in selected),
            ratio_to_q4=row['request_tok_s']/stats['q4']['request_tok_s'])
        trace_rows=[x for x in traces if x['arm']==method]
        for t in trace_rows:
            plain=next(x for x in selected if x['rep']==0 and x['prompt']==t['prompt'])
            assert all(plain['native'].get(k)==t['native'].get(k) for k in ('accepted','proposed','rounds'))
        row['trace_counters_match_plain']=True
        for key in ('full_draft_host_ms_per_round','full_draft_all_phases_host_ms_per_round','draft_prefill_host_ms_per_request'):
            row[key]=host[method][key]
        prefix=[x['native']['prefix_survival_counts'] for x in selected if 'prefix_survival_counts' in x['native']]
        row['prefix_survival_counts']=[sum(p[i] if i<len(p) else 0 for p in prefix) for i in range(max(map(len,prefix),default=0))]
        row['prefix_count_method']='Summed native per-request rounded rate times exact rounds; rounding may differ by<0.5/request.'
        residency=[];alloc=[]
        for rep in range(2):
            d=directory/'native'/f'{rep:02d}-{method}'
            alloc.append(allocations(d))
            chosen=[x for x in selected if x['rep']==rep]
            lo=min(x['bounds_ns'][0] for x in chosen)/1e9;hi=max(x['bounds_ns'][1] for x in chosen)/1e9
            residency += [x for x in json.load(open(d/'memory.json')) if lo<=x['monotonic']<=hi]
        assert residency
        row['native_allocations']=alloc[0]
        for key in ('MiB','Rss_bytes','Pss_bytes','Private_Clean_bytes','Private_Dirty_bytes','Shared_Clean_bytes','Shared_Dirty_bytes'):
            values=[x[key] for x in residency if key in x]
            row['sampled_peak_'+key]=max(values) if values else None
        row['sampled_private_resident_bytes']=max((x.get('Private_Clean_bytes',0)+x.get('Private_Dirty_bytes',0) for x in residency),default=None)
        row['sampled_shared_resident_bytes']=max((x.get('Shared_Clean_bytes',0)+x.get('Shared_Dirty_bytes',0) for x in residency),default=None)
        row['target_cache_bytes']=sum(x[k] for x in alloc[0]['cache_buffers'][:1] for k in ('k_bytes','v_bytes'))
        log=(directory/'native'/f'00-{method}'/'server.log').read_text(errors='replace')
        row['target_recurrent_state_bytes']=sum(float(v)*2**20 for v in
            re.findall(r'llama_memory_recurrent:\s+\S+ RS buffer size =\s+([\d.]+) MiB',log))
        row['draft_cache_bytes']=sum(x[k] for x in alloc[0]['cache_buffers'][1:] for k in ('k_bytes','v_bytes'))
        row['rotation_bytes']=sum(x['reported_bytes'] for x in alloc[0]['rotation_buffers'])
        row['combined_gpu_model_bytes']=sum(v for k,v in alloc[0]['model_buffers'].items() if k.startswith('CUDA') and 'Host' not in k)
        points=[x['graph_event_ms'] for rec in gpu['records'] if rec['arm']==method for x in rec['result']['rows'] if not x['warmup']]
        row['clean_gpu_forward_ms']=statistics.mean(points) if points else None
        row['clean_gpu_forward_samples']=len(points)
        row['gpu_scope']='Full native draft graph, depth+1 slots, fresh native target taps/history; excludes host selector walk/catchup;2process reps×3domains×3neighbor contexts.'
        row['precision']='F16 draft KV' if method in ('q4','dflash2','HE4') else 'Q4_0 Hadamard K128/V64 draft KV' if method in ('before','mid300','mid900','after') else None
        row['runtime_format_fallback']=method=='dflash2' and plan.get('dflash2_variant',{}).get('f32_down_fallback',False)
        if row['runtime_format_fallback']:
            row['precision']='Mixed Q4_K_M weights with five original BF16 FFN-down matrices stored/executed F32; F16 draft KV; unchanged runtime04.'
        if method=='dflash2':
            proof=json.load(open(directory/'trace/00-dflash2-traced/selector-proof.json'))
            assert proof['status']=='ACTUAL_DFLASH2_SELECTOR_LATTICE_READ_PASS' and proof['calls']>0
            row['selector_proof']=proof
            audit=json.load(open(directory/'audit/result.json'))
            assert audit['status']=='COMPLETE' and audit['diagnostic_audit']
            row['actual_head_selector_audit']=dict(status='PASS',
                source=pin(directory/'audit/result.json'),
                scope='All248320native head logits finite and unordered selector candidate set contains unary argmax at each draft position; diagnostic warmups+three neighboring consumed-code histories; timing excluded.')
        out.append(row)
    target=next(x for x in out if x['method']=='target')
    for row in out:
        if row['status']=='MEASURED':
            row['private_gpu_model_bytes']=row['combined_gpu_model_bytes']-target['combined_gpu_model_bytes']
            row['immutable_target_gpu_model_bytes']=target['combined_gpu_model_bytes']
    return out


def figures(rows,training):
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    colors=dict(target='#64748b',q4='#2563eb',dflash2='#8b5cf6',before='#f59e0b',
                mid300='#d97706',mid900='#4d7c0f',after='#16a34a',HE4='#db2777')
    cells=list(dict.fromkeys(x['cell'] for x in rows))
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,cell in zip(axs.flat,cells):
        group=[x for x in rows if x['cell']==cell and x['status']=='MEASURED']
        for i,r in enumerate(group):
            ax.barh(i,r['request_tok_s'],color=colors[r['method']],alpha=.86)
            ax.plot(r['request_rep_rates'],[i]*2,'k|',markersize=10)
            ax.text(r['request_tok_s']+3,i,f"{r['request_tok_s']:.1f}",va='center',fontsize=9)
        ax.set_yticks(range(len(group)),[x['label'] for x in group]);ax.invert_yaxis()
        ax.set_title(cell+f" · depth{group[0]['depth']}");ax.set_xlabel('Full request output tokens/s')
        ax.set_xlim(0,max(x['request_tok_s'] for x in group)*1.22);ax.grid(axis='x',alpha=.15)
        if cell.startswith('9B'):ax.text(.02,.08,'DFlash2: no matched release found\nCandidate: no9B weights',transform=ax.transAxes,fontsize=9)
    fig.suptitle('RTX5080 · consumed DEV smoke · 2warmups +2alternating reps · exact target outputs')
    for ext in ('png','svg'):fig.savefig(DEST/f'comparison.{ext}',dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for cell in cells:
        group=[x for x in rows if x['cell']==cell and x['status']=='MEASURED' and x['method']!='target']
        for r in group:
            axs[0].scatter(r['acceptance'],r['request_tok_s'],c=colors[r['method']],s=32)
            axs[1].scatter(r['clean_gpu_forward_ms'],r['full_draft_all_phases_host_ms_per_round'],c=colors[r['method']],s=32)
            for ax,x,y in [(axs[0],r['acceptance'],r['request_tok_s']),(axs[1],r['clean_gpu_forward_ms'],r['full_draft_all_phases_host_ms_per_round'])]:
                ax.annotate(cell+' '+r['method'],(x,y),xytext=(3,3),textcoords='offset points',fontsize=6)
    axs[0].set(xlabel='Accepted/proposed',ylabel='Request tokens/s',title='Distinct target cohorts')
    axs[1].set(xlabel='Clean GPU full draft graph ms',ylabel='All-phase draft host ms/round',title='Separate GPU and instrumented host work')
    points=training['readouts'];xs=[x['additional_update_seconds']/60 for x in points]
    axs[2].plot(xs,[x['dev']['ce'] for x in points],'o-',color=colors['after'])
    axs[2].set(xlabel='Actual additional update minutes',ylabel='Consumed DEV cross entropy',title='Csmallbroad continuation · lower is better')
    for ax in axs:ax.grid(alpha=.2)
    for ext in ('png','svg'):fig.savefig(DEST/f'diagnostics.{ext}',dpi=160)
    plt.close(fig)


def main():
    rows=[]
    for directory in sorted((R/'matrix').iterdir()):
        if directory.is_dir() and (directory/'plan.json').exists():rows+=cell_rows(directory)
    assert len({x['cell'] for x in rows})==4
    training=json.load(open(OUT/'result.json'));resume=json.load(open(OUT/'final-recovery/result.json'))
    assert training['status']=='COMPLETE' and resume['status']=='FULL_FINAL_STATE_NATIVE_BYTE_IDENTITY_AND_GPU_RESUME_PASS'
    assert resume['gpu_resume']['caller_optimizer_unchanged']
    failed=json.load(open(R/'candidate-attempt01/failure-summary.json'))
    invalid=json.load(open(R/'candidate-attempt02-invalid-step/invalid-resume-step.json'))
    invalid_training=json.load(open(R/'candidate-attempt02-invalid-step/result.json'))
    invalid.update(actual_update_seconds=invalid_training['actual_update_seconds'],
        additional_updates=invalid_training['additional_updates'],result=pin(R/'candidate-attempt02-invalid-step/result.json'))
    failed_seconds=failed['actual_update_seconds']+invalid['actual_update_seconds']
    resolved=json.load(open(OUT/'resolved.json'))
    compact=dict(status='COMPLETE',rows=rows,training=training,final_resume=resume,
        failed_attempts=[failed,invalid],failed_invalid_update_seconds=failed_seconds,
        recipe={key:resolved[key] for key in ('config','spec','lineage','windows','unique_labels','sequences','parameters')},
        dflash2_original_q4_status='INVALID_NONFINITE_NATIVE_LOGITS_AND_SELECTOR_SCORES',
        dflash2_fallback=json.load(open(R/'models/dflash2-4b-Q4-F32down.fallback.json')),
        dflash2_failed_path_evidence=[pin(R/'diagnostics/selector01/result.json'),pin(R/'diagnostics/selector02/result.json')],
        metadata=json.load(open(R/'release-audit/native-metadata.json')),
        no_9b_release=json.load(open(R/'release-audit/bounded-9b-search-summary.json')),
        scope='Consumed DEV smoke; two alternating process repetitions, no deployment promotion or statistical ranking. Methods/cells remain distinct. Before/after is additional training, not isolated precision attribution; no extra matched30-minute control.',
        memory_scope='Model/cache allocations from native logs (0.01MiB rounding); process resident/private/shared/PSS sampled during measured requests; device sampled total includes desktop/target/workspace. These are separate quantities, not one exact allocator peak.',
        provenance=[pin(p) for p in (R/'release-audit/adaptation.json',R/'diagnostic-build.json',OUT/'zero-resume-proof.json',OUT/'resolved.json',R/'cpu-validation.json')],
        source=pin(__file__))
    save(DEST/'results.json',compact)
    fields=['cell','method','status','depth','request_tok_s','request_rep_min','request_rep_max','ratio_to_q4','mean_request_seconds','output_tokens','accepted','proposed','rounds','acceptance','accepted_per_round','emitted_per_round','clean_gpu_forward_ms','full_draft_host_ms_per_round','full_draft_all_phases_host_ms_per_round','immutable_target_gpu_model_bytes','private_gpu_model_bytes','target_cache_bytes','target_recurrent_state_bytes','draft_cache_bytes','rotation_bytes','sampled_peak_MiB','sampled_private_resident_bytes','sampled_shared_resident_bytes','sampled_peak_Pss_bytes','correct']
    with (DEST/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    figures(rows,training)
    table='<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'
    for row in rows:
        table+='<tr>'+''.join('<td>'+html.escape(f'{row[k]:.5g}' if isinstance(row.get(k),float) else str(row.get(k,'')))+'</td>' for k in fields)+'</tr>'
    table+='</tbody></table>'
    document='''<!doctype html><html><meta charset="utf-8"><title>DFlash2 native smoke</title>
<style>body{font:14px var(--font-sans,system-ui);margin:0;color:var(--foreground,#17243a)}p{line-height:1.5;max-width:1100px}svg{width:100%;height:520px}svg:nth-of-type(2){height:300px}table{border-collapse:collapse;font-size:11px}td,th{padding:6px;border:1px solid var(--border,#ddd);white-space:nowrap}th{position:sticky;top:0;background:var(--secondary,#eef2ff)}select{padding:8px}tr.hidden{display:none}</style>
<p>RTX5080 · immutable Qwen3.5 targets · consistent decode32 · target F16 KV · context4096 · greedy output cap256 · three consumed DEV prompts. Two warmups and two alternating process repetitions. Tick marks show repetition rates; they are not confidence intervals. No sealed heldout reuse claim.</p>'''
    document+='<svg'+(DEST/'comparison.svg').read_text().split('<svg',1)[1]
    document+='<p>DFlash2 uses the authenticated released five-layer conv/selector model with a declared precision fallback: five FFN down projections use original BF16 values stored/executed F32; remaining weights use Q4_K_M with the normal F32 exceptions and F16 KV. Original Q4 and F16 native paths produced nonfinite logits and are invalid, excluded from these bars. Csmallbroad is ordinary binary DFlash with its original recipe and Q4_0 Hadamard KV. HE4 remains unchanged. No compatible9B DFlash2 release was found in the queried inventories; no9B candidate weights exist.</p>'
    document+='<svg'+(DEST/'diagnostics.svg').read_text().split('<svg',1)[1]
    document+=f'<p>Successful continuation: {training["actual_update_seconds"]:.3f}s synchronized updates, {training["additional_updates"]}updates; final full-state recovery and next GPU Adam update passed. Earlier failed/invalid attempts spent {failed_seconds:.3f}s and are excluded from the accepted trajectory. Setup, readout and checkpoint costs are separate in JSON. Before/after does not isolate a precision or architecture benefit. HF/native graph parity remains an inherited unresolved limitation for QAT quality diagnostics.</p>'
    document+='<label>Target cell <select id="cell"><option>All</option>'+''.join('<option>'+html.escape(c)+'</option>' for c in sorted({x['cell'] for x in rows}))+'</select></label><div style="overflow:auto;max-height:650px">'+table+'</div><script>document.querySelector("#cell").onchange=e=>{for(const r of document.querySelectorAll("tbody tr"))r.classList.toggle("hidden",e.target.value!=="All"&&r.firstChild.textContent!==e.target.value)}</script></html>'
    (DEST/'comparison.html').write_text(document)
    save(R/'published-results.json',dict(files=[pin(DEST/n) for n in ('results.json','comparison.csv','comparison.html','comparison.svg','comparison.png','diagnostics.svg','diagnostics.png')]))
    print('COMPACT_RESULTS_PUBLISHED',len(rows),flush=True)


if __name__=='__main__': main()
