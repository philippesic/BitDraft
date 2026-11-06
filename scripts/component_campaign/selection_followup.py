"""Finish actual scalar/host evidence in the released serial GPU lane."""
import argparse,json
from .common import *
from .post_primary import wait_release,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after',type=Path,required=True);a=ap.parse_args();wait_release(a.after)
 save(R/'selection-followup-owner.json',dict(owner=identity(),predecessor=pin(a.after),source=pin(__file__)))
 run('scalar_import_audit')
 if not (R/'arms/Sall/native-tensor-audit.json').exists():run('audit','Sall')
 trace=R/'measurements/selection-primary-host'
 if not (trace/'result.json').exists():run('measure',trace.name,'--reps','1','--cohort','capture3','--trace')
 primary=R/'measurements/selection-primary';replay=R/'measurements/replay-primary-selection'
 run('formal_summary','--native',primary,trace,'--replay',replay,'--out',R/'selection-primary-summary.json')
 save(R/'publication-index.json',dict(status='IN_PROGRESS',summary=str(R/'selection-primary-summary.json'),selection_panel=primary.name,gpu_panel=replay.name,trace_panel=trace.name,arm_status={n:'4096 R1 window complete; measured selection, final frontier pending' for n in json.load(open(R/'panels/replay-primary-arms.json'))}))
 run('publish')
 save(R/'selection-followup-result.json',dict(status='SELECTION_EVIDENCE_COMPLETE_RECOVERY_AND_COMBINATIONS_REQUIRED',owner=identity(),summary=pin(R/'selection-primary-summary.json'),scalar_import=pin(R/'qualification/scalar-import/verified.json'),physical_tie=pin(R/'qualification/physical-tie/verified.json') if (R/'qualification/physical-tie/verified.json').exists() else None))
 print('SELECTION_FOLLOWUP_COMPLETE_NOT_CAMPAIGN_COMPLETION',flush=True)

if __name__=='__main__':main()
