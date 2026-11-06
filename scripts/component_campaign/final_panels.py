"""Execute the frozen frontier's recovery and actual formal comparison protocol."""
import argparse,json
from .common import *
from .post_primary import run
from .measure import completed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidates',nargs='+',required=True);ap.add_argument('--controls',nargs='+',required=True);ap.add_argument('--retain',nargs=2,required=True);a=ap.parse_args()
 decision=R/'final-recipe-decision.json';selected=json.load(open(decision))
 if any(selected[k]!=getattr(a,k) for k in ('candidates','controls','retain')):raise ValueError('Formal recipe selection differs from recorded decision')
 if not (R/'post-recovery-result.json').exists():raise ValueError('Recovery selection must actually finish before final lock')
 if len(set(a.candidates+a.controls))!=len(a.candidates+a.controls) or not 1<=len(a.candidates)<=4:raise ValueError('Distinct bounded candidates/controls required')
 save(R/'final-panels-owner.json',dict(owner=identity(),decision=pin(decision),source=pin(__file__),policy='One GPU lane; full-state retention/recovery, common-runtime formal DEV/GPU/host/context/profiles then post-lock heldout. No tuning after heldout.'))
 for ordinal,name in enumerate(a.retain,1):run('retain_frontier',name,'--ordinal',str(ordinal))
 run('verify_frontier_recovery');run('head_forward_qualify')
 names=a.controls+a.candidates;registry=completed();models=dict(target=None,q4=str(Q4),B0=str(B/'model-final.gguf'),**{n:registry[n]['model'] for n in names});cache={n:registry[n]['spec'].get('kv',['f16','f16']) for n in names}
 save(R/'panels/final-common-arms.json',models);save(R/'panels/final-common-cache.json',cache)
 dev=R/'measurements/final-dev';trace=R/'measurements/final-host';replay=R/'measurements/final-replay'
 if not (dev/'result.json').exists():run('measure',dev.name,'--arms',*names,'--reps','5','--cohort','natural12','--runtime',R/'runtime04')
 save(R/'panels/final-replay-arms.json',{n:p for n,p in models.items() if p is not None})
 if not (replay/'result.json').exists():run('replay',R/'panels/final-replay-arms.json','--cache-map',R/'panels/final-common-cache.json','--out',replay,'--runtime',R/'runtime04','--reps','5')
 paired=R/'qualification/final-trace'
 if not (paired/'result.json').exists():run('native','--arms',R/'panels/final-common-arms.json','--draft-cache-map',R/'panels/final-common-cache.json','--runtime',R/'runtime04','--out',paired,'--paired-trace','--cohort','capture3','--output-cap','64','--reps','1','--warmups','2')
 if not (trace/'result.json').exists():run('measure',trace.name,'--arms',*names,'--reps','5','--cohort','capture3','--trace','--runtime',R/'runtime04')
 run('final_context',*names)
 profile_names=[n for n in registry if not n.endswith(('-confirm','-pilot'))]
 if not (R/'format-profiles/dispatch.json').exists():run('profile_formats',*profile_names)
 run('final_heldout',*a.candidates,'--controls',*a.controls)
 panels=[R/'measurements'/n for n in ('selection-primary','selection-primary-host','selection-recovered','selection-recovered-host','final-dev','final-host','final-context-128','final-context-1536','final-context-3840','final-heldout24')]
 summaries=R/'final-summary.json';run('formal_summary','--native',*panels,'--replay',R/'measurements/replay-primary-selection',R/'measurements/replay-recovered-selection',replay,'--out',summaries)
 save(R/'publication-index.json',dict(status='IN_PROGRESS',summary=str(summaries),dev_panel=dev.name,request_panel='final-heldout24',selection_panel='selection-recovered',selection_panel_fallbacks=['selection-primary'],gpu_panel=replay.name,gpu_panel_fallbacks=['replay-recovered-selection','replay-primary-selection'],trace_panel=trace.name,trace_panel_fallbacks=['selection-recovered-host','selection-primary-host'],matched_controls={'H4R2':'H6R2','H6R2':'H6R2','HE4R2':'HE6R2','HE6R2':'HE6R2','Csmallbroad':'H0broad','H0broad':'H0broad'},arm_status={n:'Locked final recipe; formal panels complete, frontier report pending' for n in names}))
 run('publish');run('campaign_accounting')
 save(R/'final-panels-result.json',dict(status='FORMAL_MEASUREMENTS_COMPLETE_FRONTIER_REPORT_AND_RELEASE_REQUIRED',owner=identity(),decision=pin(decision),summary=pin(summaries),source=pin(__file__)))
 print('FINAL_PANELS_COMPLETE_REPORT_AND_RELEASE_REQUIRED',flush=True)

if __name__=='__main__':main()
