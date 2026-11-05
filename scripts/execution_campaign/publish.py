"""Publish measured execution data; requires completed native and replay receipts."""
from pathlib import Path
import json,csv,statistics,collections,sqlite3,subprocess,time,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01';C=R/'charts-final';C.mkdir(exist_ok=True)
NAMES={'target':'Target only','w1_original':'Original W1','w1_optimized':'Optimized W1','q4':'Q4_K_M','q4_optimized':'Q4 + eligible changes'}
COLORS={'target':'#8c929f','w1_original':'#e39044','w1_optimized':'#3c9bd8','q4':'#7865c4','q4_optimized':'#46a389'}
def load(p):return json.load(open(p))
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def pin(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
raw=load(R/'native-final/result.json');allrows=raw['rows'];rows=[r for r in allrows if not r['warmup']];plain=[r for r in rows if not r['instrumented']];traced=[r for r in rows if r['instrumented']];assert len(plain)==len(traced)==300 and all(r['correct'] for r in allrows)
arms=list(NAMES);prompts=list(dict.fromkeys(x['prompt'] for x in plain));assert len(prompts)==12
# Exact proposal-token and counter checks, independent of final target-output correctness.
proposal_rows=[];proposal_checks=[]
audit_rows=load(R/'native-proposal-audit/result.json')['rows']
for arm in arms:
 if arm=='target':continue
 cell=R/'native-proposal-audit'/f'00-{arm}-traced';events=[json.loads(l) for l in (cell/'proposals.jsonl').read_text().splitlines()]
 for row in [r for r in audit_rows if r['arm']==arm]:
  lo,hi=row['bounds_ns'];e=[dict(index=x['index'],token=x['token']) for x in events if lo<=x['time_ns']<=hi];assert len(e)==2*row['native']['rounds'] and 0<=len(e)-row['native']['proposed']<=2
  proposal_rows.append(dict(arm=arm,prompt=row['prompt'],proposals=e,warmup=row['warmup']))
for base,opt in [('w1_original','w1_optimized'),('q4','q4_optimized')]:
 for prompt in prompts:
  a=next(r for r in proposal_rows if r['arm']==base and r['prompt']==prompt);b=next(r for r in proposal_rows if r['arm']==opt and r['prompt']==prompt);assert a['proposals']==b['proposals'],(base,opt,prompt)
  for rep in range(5):
   aa=next(r for r in plain if r['rep']==rep and r['arm']==base and r['prompt']==prompt);bb=next(r for r in plain if r['rep']==rep and r['arm']==opt and r['prompt']==prompt)
   for key in ['rounds','proposed','accepted','prefix_survival_counts']:assert aa['native'][key]==bb['native'][key]
   diag=next(r for r in audit_rows if r['arm']==base and r['prompt']==prompt)
   for key in ['rounds','proposed','accepted','prefix_survival_counts']:assert aa['native'][key]==diag['native'][key]
   proposal_checks.append(dict(rep=rep,prompt=prompt,base=base,optimized=opt,live_selected_proposal_tokens_exact=True,counters_exact_across_all_timing_reps=True))
save(C/'proposal-audit.json',dict(checks=proposal_checks,total_proposals=sum(len(r['proposals']) for r in proposal_rows),source='Actual selected token arguments to common_sampler_accept inside draft_generate: the token DFlash appends to result. One deterministic diagnostic natural12 panel; all5timing-repetition counters agree. Final request-budget trimming can leave0..2unsubmitted tail samples. Earlier ignored-return-value traces are superseded, preserved and not used as proposal evidence.'))
# Crossed prompt/repetition bootstrap. Each resampled prompt and repetition remains paired.
rng=np.random.default_rng(20261105);weights=[]
for _ in range(4000):weights.append(np.bincount(rng.integers(12,size=12),minlength=12)[:,None]*np.bincount(rng.integers(5,size=5),minlength=5)[None,:])
weights=np.asarray(weights)
def grid(data,arm,key):return np.array([[next(r for r in data if r['arm']==arm and r['prompt']==p and r['rep']==rep)[key] for rep in range(5)] for p in prompts])
def interval(values):return [float(x) for x in np.quantile(values,[.025,.975])]
def request_ratio(a,b):
 at=grid(plain,a,'wall_seconds');bt=grid(plain,b,'wall_seconds');tok=grid(plain,a,'tokens');tb=grid(plain,b,'tokens');vals=(np.sum(weights*tok,(1,2))/np.sum(weights*at,(1,2)))/(np.sum(weights*tb,(1,2))/np.sum(weights*bt,(1,2)))
 return dict(value=float(tok.sum()/at.sum()/(tb.sum()/bt.sum())),ci95=interval(vals))
summary=[]
for arm in arms:
 rs=[r for r in plain if r['arm']==arm];ts=[r for r in traced if r['arm']==arm];rounds=sum(r['native'].get('rounds',0) for r in rs);accepted=sum(r['native'].get('accepted',0) for r in rs);proposed=sum(r['native'].get('proposed',0) for r in rs);cells=[R/'native-final'/f'{rep:02d}-{arm}' for rep in range(5)];mem=[m['MiB'] for cell in cells for m in load(cell/'memory.json')]
 decode_tokens=sum(r['timings']['predicted_n'] for r in rs);decode_ms=sum(r['timings']['predicted_ms'] for r in rs)
 summary.append(dict(arm=arm,name=NAMES[arm],requests=len(rs),tokens=sum(r['tokens'] for r in rs),request_seconds=sum(r['wall_seconds'] for r in rs),request_tps=sum(r['tokens'] for r in rs)/sum(r['wall_seconds'] for r in rs),decode_tps=decode_tokens/(decode_ms/1000),rounds=rounds,accepted=accepted,proposed=proposed,acceptance=accepted/proposed if proposed else None,host_full_draft_ms_per_round=sum(r.get('draft_full_ms',0) for r in ts)/rounds if rounds else None,device_sample_peak_MiB=max(mem),drafter_MiB=0 if arm=='target' else (183441120 if arm.startswith('w1') else 381456064)/2**20,request_ratio_q4=request_ratio(arm,'q4')))
# Fixed external-work replay: actual clean graph events, all recorded host outliers retained.
rr=[r for r in load(R/'replay-final/result.json')['rows'] if not r['warmup']];replay=[]
for arm in arms[1:]:
 rs=[r for r in rr if r['arm']==arm];replay.append(dict(arm=arm,gpu_forward_ms=statistics.mean(r['graph_event_ms'] for r in rs),gpu_median_ms=statistics.median(r['graph_event_ms'] for r in rs),host_forward_ms=statistics.mean(r['forward_host_ms'] for r in rs),host_median_ms=statistics.median(r['forward_host_ms'] for r in rs),n=len(rs),catchup_rebuild_host_ms=statistics.mean(r['catchup_host_ms'] for r in rs)))
# Bootstrap full captured domains and repetitions together; retain each domain's3 changing contexts.
rg=np.random.default_rng(4197);domains=['code','reasoning','prose'];rw=[]
for _ in range(4000):rw.append(np.bincount(rg.integers(3,size=3),minlength=3)[:,None]*np.bincount(rg.integers(5,size=5),minlength=5)[None,:])
rw=np.asarray(rw)
def rgrid(arm,key):return np.array([[statistics.mean(r[key] for r in rr if r['arm']==arm and r['domain']==d and r['rep']==rep) for rep in range(5)] for d in domains])
def replay_ratio(a,b,key='graph_event_ms'):
 aa=rgrid(a,key);bb=rgrid(b,key);return dict(value=float(aa.sum()/bb.sum()),ci95=interval(np.sum(rw*aa,(1,2))/np.sum(rw*bb,(1,2))))
comparisons=dict(request_optimized_original=request_ratio('w1_optimized','w1_original'),request_optimized_q4=request_ratio('w1_optimized','q4'),request_fairq4_original=request_ratio('q4_optimized','q4'),gpu_optimized_original=replay_ratio('w1_optimized','w1_original'),gpu_optimized_q4=replay_ratio('w1_optimized','q4'),gpu_optimized_fairq4=replay_ratio('w1_optimized','q4_optimized'))
def draft_ratio(a,b):
 aa=grid(traced,a,'draft_full_ms');bb=grid(traced,b,'draft_full_ms')
 ar=np.array([[next(r for r in traced if r['arm']==a and r['prompt']==p and r['rep']==rep)['native']['rounds'] for rep in range(5)] for p in prompts]);br=np.array([[next(r for r in traced if r['arm']==b and r['prompt']==p and r['rep']==rep)['native']['rounds'] for rep in range(5)] for p in prompts])
 vals=(np.sum(weights*aa,(1,2))/np.sum(weights*ar,(1,2)))/(np.sum(weights*bb,(1,2))/np.sum(weights*br,(1,2)))
 return dict(value=float(aa.sum()/ar.sum()/(bb.sum()/br.sum())),ci95=interval(vals))
comparisons['host_full_draft_optimized_original']=draft_ratio('w1_optimized','w1_original');comparisons['host_full_draft_optimized_q4']=draft_ratio('w1_optimized','q4')
# Trace attribution uses complete root draft spans; never per-kernel device synchronization.
stages=[];epoch_offset=time.time_ns()-time.monotonic_ns()
for arm in arms[1:]:
 cell=R/'profile-final'/f'00-{arm}-traced';db=cell/'profile.sqlite'
 if not db.exists():subprocess.run(['nsys','export','--type','sqlite','--output',str(db),str(cell/'profile.nsys-rep')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
 c=sqlite3.connect(db);epoch=c.execute('select utcEpochNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0];result=next(r for r in load(R/'profile-final/result.json')['rows'] if r['arm']==arm and not r['warmup']);lo,hi=result['bounds_ns'];spans=[json.loads(l) for l in (cell/'cost-trace.jsonl').read_text().splitlines()];spans=[s for s in spans if s['depth']==0 and s['name']=='draft_generate' and lo<=s['start_ns']<=s['end_ns']<=hi];assert len(spans)==result['native']['rounds']
 groups=collections.defaultdict(float);counts=collections.Counter();kernel_rows=[]
 for s in spans:
  a=s['start_ns']+epoch_offset-epoch;b=s['end_ns']+epoch_offset-epoch
  ks=c.execute('select z.value,k.start,k.end,k.gridX,k.gridY,k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds z on z.id=k.demangledName where k.start>=? and k.end<=? order by k.start',(a,b)).fetchall();assert ks
  for name,start,end,gx,gy,gid in ks:
   if 'mul_mat' in name and gx>=100000:key='Vocabulary head'
   elif any(x in name for x in ['exec_tiled','w1ax_quantize','quantize_q8_1']):key='Quantization'
   elif any(x in name for x in ['w1ax_integer_dot','exec_joint_dot','mul_mat']):key='FFN / attention projections'
   elif 'flash_attn' in name:key='Attention'
   elif 'top_k' in name:key='Backend sampler'
   else:key='Norm / elementwise / copies'
   groups[key]+=(end-start)/1000;counts[key]+=1;kernel_rows.append(dict(name=name,start=start,end=end,grid=[gx,gy],graph_id=gid,stage=key))
 n=len(spans);stages.append(dict(arm=arm,rounds=n,us_per_round={k:v/n for k,v in groups.items()},kernel_counts=dict(counts),kernel_sum_ms_per_round=sum(groups.values())/n/1000,scope='instrumented short consumed code request,32output reserve; summed actual CUDA kernel durations inside complete draft-generate host ranges; excludes catch-up; no DRAM/occupancy measurement'))
 save(C/(arm+'-profile-kernels.json'),kernel_rows)
# Ablations are sequential native replay screens, explicitly distinct from final paired confirmation.
ablations=[]
for folder,names in [('replay-tranche1',['original','copy','bound256','both256','both32','cross','mma']),('replay-tranche4',['tiled','silu03','residual03','joint','narrow03','prenorm','head02','lastffn02']),('replay-tranche6',['combined02','combined-last02','lut02','silu04'])]:
 for name in names:
  rs=[x for x in load(R/folder/(name+'-clean.json'))['rows'] if not x['warmup']];ablations.append(dict(variant=name,gpu_median_ms=statistics.median(x['graph_event_ms'] for x in rs),evidence=str(R/folder/(name+'-clean.json')),scope='sequential screening; no paired confidence claim'))
rs=[x for x in load(R/'runtime-planes4/replay.json')['rows'] if not x['warmup']];ablations.append(dict(variant='A4_planes4',gpu_median_ms=statistics.median(x['graph_event_ms'] for x in rs),evidence=str(R/'runtime-planes4/replay.json'),scope='late bounded screen; see paired methods for uncertainty'))
data=dict(native=summary,replay=replay,comparisons=comparisons,stages=stages,ablations=ablations,proposal_audit=pin(C/'proposal-audit.json'),limitations=['Primary checkpoint725c3c31... is frozen learned-small2048, not fixed-small1024.','RTX5080, Qwen3.5-4B Q4_K_M target, F16KV, depth2, consistent32, greedy, concurrency1.','Natural12 is already-consumed DEV, not sealed final validation.','Replay uses3separate consumed native captureDEV domains and model-specific rebuilt caches; backend sampler absent.','Replay host outlier28.61ms is retained; graph event remained1.171ms.','Final fixed-work graph events, live host diagnostic spans and Nsight kernel sums are different metrics.','Hardware-counter permissions were not changed; traffic estimates are not measured DRAM.','Full-model HF/native floating parity remains historically unresolved; exact native execution/cache/proposals qualify this change.'])
method=load(R/'replay-methods/result.json');data['bounded_method_followup']={k:v for k,v in method.items() if k not in ('rows','checks')};save(C/'comparison.json',data)
for name,records in [('native',summary),('replay',replay),('ablations',ablations)]:
 with (C/(name+'.csv')).open('w') as f:
  flat=[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in records];w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def emit(name):plt.tight_layout();plt.savefig(C/(name+'.png'),dpi=180,bbox_inches='tight');plt.savefig(C/(name+'.svg'),bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(9,4));ax.bar([NAMES[r['arm']] for r in replay],[r['gpu_forward_ms'] for r in replay],color=[COLORS[r['arm']] for r in replay]);ax.set_ylabel('Mean CUDA graph forward (ms)');ax.set_title('Fixed external work · 3 captured DEV domains · 5 alternating reps');ax.tick_params(axis='x',rotation=12);emit('forward-latency')
fig,ax=plt.subplots(figsize=(10,4));ax.bar([r['name'] for r in summary],[r['request_tps'] for r in summary],color=[COLORS[r['arm']] for r in summary]);ax.set_ylabel('Full-request output tokens / second');ax.set_title('Consumed natural DEV12 · plain requests · 5 alternating reps');ax.tick_params(axis='x',rotation=12);emit('request-throughput')
fig,ax=plt.subplots(figsize=(10,4));bottom=np.zeros(4)
for key in ['Quantization','FFN / attention projections','Vocabulary head','Attention','Backend sampler','Norm / elementwise / copies']:
 vals=[s['us_per_round'].get(key,0)/1000 for s in stages];ax.bar([NAMES[s['arm']] for s in stages],vals,bottom=bottom,label=key);bottom+=vals
ax.set_ylabel('Profiled kernel sum / draft round (ms)');ax.set_title('Short live code trace · diagnostic stage costs');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1));ax.tick_params(axis='x',rotation=12);emit('stage-costs')
fig,ax=plt.subplots(figsize=(10,7));ax.barh([r['variant'] for r in ablations],[r['gpu_median_ms'] for r in ablations],color='#3c9bd8');ax.invert_yaxis();ax.set_xlabel('CUDA graph forward median (ms)');ax.set_title('Ablation screens · sequential, not paired confidence estimates');emit('ablations')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,keys,title in [(axes[0],['gpu_optimized_original','gpu_optimized_q4','gpu_optimized_fairq4'],'GPU latency ratio (lower is faster)'),(axes[1],['request_optimized_original','request_optimized_q4','request_fairq4_original'],'Request throughput ratio (higher is faster)')]:
 labels=['Optimized W1 / original W1','Optimized W1 / Q4','Optimized W1 / fair Q4'] if ax==axes[0] else ['Optimized W1 / original W1','Optimized W1 / Q4','Fair Q4 / original Q4']
 for i,key in enumerate(keys):
  z=comparisons[key];lo,hi=z['ci95'];v=z['value'];ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=5,color='#247ba4');ax.text(hi+.006,i,f'{v:.3f}',va='center',fontsize=8)
 ax.axvline(1,color='#777',ls='--',lw=1);ax.set_yticks(range(3),labels,fontsize=8);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
emit('paired-effects')
# Small offline interactive artifact, with embedded data and no CDN dependencies.
payload=json.dumps(data).replace('</','<\\/');html='''<!doctype html><meta charset="utf-8"><title>DFlash execution comparison</title><style>body{font:16px system-ui;margin:32px;background:#10151e;color:#e5eaf3}h1{font-size:25px}select,button{font:inherit;padding:7px;margin:6px;background:#1f2a3c;color:inherit;border:1px solid #596779;border-radius:6px}svg{max-width:100%;background:#182131;border-radius:12px}td,th{padding:8px 14px;text-align:right;border-bottom:1px solid #394457}td:first-child,th:first-child{text-align:left}p{max-width:1000px;line-height:1.5}.muted{color:#a6b6cd}</style><h1>DFlash execution · frozen learned-small2048</h1><p id="headline"></p><select id="metric"><option value="forward">Fixed-work GPU forward latency</option><option value="request">Full request throughput</option><option value="draft">Live full-draft host span (diagnostic)</option><option value="ablation">All ablation screens</option></select><button onclick="download()">Download exact JSON</button><div id="view"></div><p class="muted">RTX 5080 · Qwen3.5-4B Q4_K_M target · F16 KV · depth2 · consistent decode32. All original/optimized proposal tokens and native counters match. Rates use plain execution; host spans and profiled kernels are diagnostics.</p><div id="table"></div><details><summary>Scope, error gates and limitations</summary><ul id="limits"></ul></details><script>const DATA=PAYLOAD;const names=NAMES;function draw(){let metric=document.querySelector('#metric').value;let rows=metric==='forward'?DATA.replay.map(r=>[names[r.arm],r.gpu_forward_ms]):metric==='request'?DATA.native.map(r=>[r.name,r.request_tps]):metric==='draft'?DATA.native.filter(r=>r.arm!=='target').map(r=>[r.name,r.host_full_draft_ms_per_round]):DATA.ablations.map(r=>[r.variant,r.gpu_median_ms]);let max=Math.max(...rows.map(r=>r[1]))*1.1,h=60+rows.length*42;document.querySelector('#view').innerHTML='<svg viewBox="0 0 950 '+h+'">'+rows.map((r,i)=>'<text x="15" y="'+(42+i*42)+'" fill="#dfe9fa" font-size="14">'+r[0]+'</text><rect x="250" y="'+(24+i*42)+'" height="26" width="'+(r[1]/max*580)+'" rx="4" fill="#58afe0"><title>'+r[0]+': '+r[1]+'</title></rect><text x="'+(260+r[1]/max*580)+'" y="'+(42+i*42)+'" fill="#dfe9fa" font-size="14">'+r[1].toFixed(3)+'</text>').join('')+'</svg>';}document.querySelector('#metric').onchange=draw;let cmp=DATA.comparisons;document.querySelector('#headline').textContent='Optimized W1 GPU latency / original: '+cmp.gpu_optimized_original.value.toFixed(3)+'× (95% paired '+cmp.gpu_optimized_original.ci95.map(x=>x.toFixed(3)).join('–')+'). Request throughput / Q4: '+cmp.request_optimized_q4.value.toFixed(3)+'×.';document.querySelector('#table').innerHTML='<table><tr><th>Arm</th><th>Request tok/s</th><th>Draft host ms/round</th><th>Acceptance</th><th>Sample peak MiB</th></tr>'+DATA.native.map(r=>'<tr><td>'+r.name+'</td><td>'+r.request_tps.toFixed(2)+'</td><td>'+(r.host_full_draft_ms_per_round===null?'—':r.host_full_draft_ms_per_round.toFixed(3))+'</td><td>'+(r.acceptance===null?'—':(100*r.acceptance).toFixed(2)+'%')+'</td><td>'+r.device_sample_peak_MiB+'</td></tr>').join('')+'</table>';document.querySelector('#limits').innerHTML=DATA.limitations.map(x=>'<li>'+x+'</li>').join('');function download(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(DATA,null,2)],{type:'application/json'}));a.download='comparison.json';a.click();}draw();</script>'''.replace('PAYLOAD',payload).replace('NAMES',json.dumps(NAMES));(C/'comparison.html').write_text(html)
print(json.dumps(dict(native=summary,replay=replay,comparisons=comparisons),indent=2))
