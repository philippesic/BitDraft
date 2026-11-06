"""Publish separate formal/context panels and the full measured component ledger."""
import csv,json
from .common import *

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 summary=json.load(open(R/'final-summary.json'));comparison=json.load(open(R/'charts/comparison.json'))
 out=R/'charts';panels=[]
 for panel,data in summary['native'].items():
  for arm,entry in data['arms'].items():
   panels.append(dict(panel=panel,cohort=data['cohort'],arm=arm,**entry,evidence=data['source']))
 save(out/'all-panels.json',dict(panels=panels,summary=pin(R/'final-summary.json'),policy='Every cohort, context length and instrumented panel remains separate. Counts include measured requests only. Repeated greedy outcomes are correlated.'))
 keys=sorted({k for row in panels for k in row})
 with (out/'all-panels.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  w.writerows({k:json.dumps(row.get(k),sort_keys=True) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in keys} for row in panels)
 def finish(fig,name):
  fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);fig.savefig(out/(name+'.svg'));plt.close(fig)
 # Each context length is its own fixed workload and own Q4 comparator.
 contexts=[(n,summary['native']['final-context-'+str(n)]) for n in (128,1536,3840)]
 names=list(contexts[0][1]['arms']);colors={n:plt.cm.tab20(i%20) for i,n in enumerate(names)}
 fig,axes=plt.subplots(1,3,figsize=(15,4.5))
 for ax,(length,panel) in zip(axes,contexts):
  points=list(panel['arms'].items())
  ax.bar(range(len(points)),[v['request_tok_s'] for n,v in points],color=[colors[n] for n,v in points])
  ax.set_xticks(range(len(points)),[n for n,v in points],rotation=55,ha='right');ax.set(title=f'{length} input positions +256 output reserve',ylabel='Full request tokens / second');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Separate synthetic cache-history stress panels · 2 warmups +5 alternating repetitions · target-identical outputs')
 finish(fig,'context-request-panels')
 # Publish all components even when they are not selected for fresh heldout.
 rows=comparison['rows'];latest=[]
 for name in dict.fromkeys(r['arm'] for r in rows):
  latest.append(max((r for r in rows if r['arm']==name),key=lambda r:r['updates']))
 groups=[('Head',['H0','H6','H4','H3','H4R2','H6R2']),('Embedding / tied vocabulary',['H0','E6','E4','HE4','HE4R2','HE6R2']),('Draft cache',['H0','K8F','KF8','K88','K48','K84','K44']),('Scalar storage',['H0','Sscale','Smidpoint','Sgain','Sclip','Snorm','Sall']),('Combined ladder',['H0','H4','HE4','CtieKV','Ctied','Chead','Csmall']),('Broader recipe',['H0broad','Csmallbroad'])]
 lookup={r['arm']:r for r in latest}
 for field,ylabel,file in [('selection_request_tok_s','Consumed natural12 request tokens / second','all-component-requests'),('selection_acceptance','Consumed natural12 accepted / proposed','all-component-acceptance'),('tensor_bytes','Private draft tensor payload MiB','all-component-model-memory'),('training_update_seconds','Actual update seconds · setup/export/checkpoints separate','all-component-training-cost')]:
  fig,axes=plt.subplots(2,3,figsize=(15,8))
  for ax,(label,members) in zip(axes.flat,groups):
   points=[lookup[n] for n in members if n in lookup and lookup[n].get(field) is not None]
   ax.bar(range(len(points)),[r[field]/2**20 if field.endswith('_bytes') else r[field] for r in points],color=[plt.cm.tab20(i%20) for i in range(len(points))])
   ax.set_xticks(range(len(points)),[r['arm'] for r in points],rotation=55,ha='right');ax.set(title=label,ylabel=ylabel);ax.grid(axis='y',alpha=.2)
  fig.suptitle('Full component ledger · native selection is one repetition; formal finalists are reported separately')
  finish(fig,file)
 # Explicit incremental precision ladder, all trained directly from B0.
 stages=[n for n in ['B0','H0','H4','HE4','CtieKV','Ctied','Chead','Csmall'] if n in lookup]
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,field,label in [(axes[0],'selection_clean_gpu_forward_ms','Selection clean GPU forward ms'),(axes[1],'selection_request_tok_s','Selection natural12 request tokens / second'),(axes[2],'model_cache_payload_bytes','Private model +allocated draft cache / rotation MiB')]:
  points=[lookup[n] for n in stages if lookup[n].get(field) is not None]
  ax.bar(range(len(points)),[r[field]/2**20 if field.endswith('_bytes') else r[field] for r in points]);ax.set_xticks(range(len(points)),[r['arm'] for r in points],rotation=55,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Head → tied embedding → K/V → half gain/clip/norm · matched4096-update common-B0 recipes; B0 frozen')
 finish(fig,'incremental-ladder')
 # The formal comparison is a distinct post-lock cohort.
 final=summary['native']['final-heldout24'];points=list(final['arms'].items())
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,key,label in [(axes[0],'request_tok_s','Full request tokens / second'),(axes[1],'acceptance','Accepted / proposed'),(axes[2],'emitted_per_round','Emitted tokens / speculative round')]:
  ax.bar(range(len(points)),[v[key] if v[key] is not None else np.nan for n,v in points]);ax.set_xticks(range(len(points)),[n for n,v in points],rotation=55,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
 fig.suptitle('Fresh24 heldout · locked recipes · 2 warmups +5 alternating process repetitions · no post-heldout tuning')
 finish(fig,'formal-finalists')
 from .extended_charts import generate
 generate(summary,comparison,out)
 from .interactive import render
 finalists=[{k:v for k,v in row.items() if isinstance(v,(str,int,float,bool,type(None))) or k=='final_vs_q4'} for row in comparison['rows'] if row.get('final_request_tok_s') is not None]
 html=render(finalists).replace("['#x',xFields,'updates']","['#x',xFields,'model_cache_payload_bytes']").replace("['#y',yFields,'ce_gap_matched_control']","['#y',yFields,'final_request_tok_s']").replace('id="latest">','id="latest" checked>')
 html=html.replace('ylo=Math.min(...ys,0)','ylo=ys.length?Math.min(...ys)-.06*(Math.max(...ys)-Math.min(...ys)||1):0')
 html=html.replace("'agreement','accepted','proposed'","'agreement','final_accepted','final_proposed'")
 html=html.replace('</style>','body{margin:0;background:transparent;color:var(--foreground,#17243a);font-family:var(--font-sans,system-ui)}h1{display:none}.note,#status{color:var(--muted-foreground,#46566d)}#filters label,select,button,table,svg{background:var(--card,white);color:var(--foreground,#17243a);border-color:var(--border,#ccd6e4)}th{background:var(--secondary,#eaf0f8)}svg text{fill:var(--foreground,#17243a)}</style>',1)
 (out/'finalists.html').write_text(html)
 save(R/'final-publication.json',dict(status='MEASURED_PUBLICATION_READY_FOR_REVIEW',comparison=pin(out/'comparison.json'),panels=pin(out/'all-panels.json'),figures=[pin(p) for p in sorted(out.glob('*.png'))]+[pin(p) for p in sorted(out.glob('*.svg'))],interactive=pin(out/'comparison.html'),finalists_interactive=pin(out/'finalists.html'),csv=[pin(out/'full-table.csv'),pin(out/'all-panels.csv')],source=pin(__file__),limits='Timing prompt/process resampling does not establish independent training-seed uncertainty. Native head numerical bounds are not full graph bit equivalence. Private vocabulary adds to the immutable target. Target cache/workspace and desktop are not private model payload.'))
 print('FINAL_PUBLICATION_READY',len(panels),'separate panel/arm rows',flush=True)

if __name__=='__main__':main()
