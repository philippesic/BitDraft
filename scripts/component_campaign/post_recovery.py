"""Measure recovered recipes on consumed DEV before selecting locked finalists."""
import argparse,json
from .common import *
from .post_primary import wait_release,run
from .measure import completed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after',type=Path,required=True);a=ap.parse_args();wait_release(a.after)
 names=['H0','H4','HE4','K44','H4R2','H6R2','HE4R2','HE6R2','Csmall','Chead','Ctied','H0broad','Csmallbroad'];registry=completed()
 save(R/'post-recovery-owner.json',dict(owner=identity(),predecessor=pin(a.after),decision=pin(R/'selection-evidence.json'),source=pin(__file__)))
 for n in names:
  if n not in registry:raise ValueError('Selected recovery not actually complete '+n)
  if not (R/'arms'/n/'native-tensor-audit.json').exists():run('audit',n)
 request=R/'measurements/selection-recovered';trace=R/'measurements/selection-recovered-host';replay=R/'measurements/replay-recovered-selection'
 if not (request/'result.json').exists():run('measure',request.name,'--arms',*names,'--reps','1','--cohort','natural12')
 save(R/'panels/replay-recovered-arms.json',dict(q4=str(Q4),B0=str(B/'model-final.gguf'),**{n:registry[n]['model'] for n in names}));save(R/'panels/replay-recovered-cache.json',{n:registry[n]['spec'].get('kv',['f16','f16']) for n in names})
 if not (replay/'result.json').exists():run('replay',R/'panels/replay-recovered-arms.json','--cache-map',R/'panels/replay-recovered-cache.json','--runtime',R/'runtime04','--out',replay,'--reps','1')
 if not (trace/'result.json').exists():run('measure',trace.name,'--arms',*names,'--reps','1','--cohort','capture3','--trace')
 summary=R/'selection-recovered-summary.json'
 run('formal_summary','--native',R/'measurements/selection-primary',R/'measurements/selection-primary-host',request,trace,'--replay',R/'measurements/replay-primary-selection',replay,'--out',summary)
 save(R/'publication-index.json',dict(status='IN_PROGRESS',summary=str(summary),selection_panel=request.name,selection_panel_fallbacks=['selection-primary'],gpu_panel=replay.name,gpu_panel_fallbacks=['replay-primary-selection'],trace_panel=trace.name,trace_panel_fallbacks=['selection-primary-host'],matched_controls={'H4R2':'H6R2','HE4R2':'HE6R2','H0broad':'H0broad','Csmallbroad':'H0broad'},arm_status={n:'Recovery and consumed-DEV selection complete; final confirmation pending' for n in names}))
 run('publish');run('campaign_accounting')
 save(R/'post-recovery-result.json',dict(status='RECOVERED_SELECTION_COMPLETE_FINAL_LOCK_REQUIRED',owner=identity(),summary=pin(summary),source=pin(__file__)))
 print('POST_RECOVERY_SELECTION_COMPLETE_NOT_CAMPAIGN_COMPLETION',flush=True)

if __name__=='__main__':main()
