"""Execute selected recoveries/combinations, with all GPU work strictly serial."""
import argparse,json
from .common import *
from .post_primary import wait_release,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--after',type=Path,required=True);a=ap.parse_args();wait_release(a.after)
 save(R/'recovery-combinations-owner.json',dict(owner=identity(),predecessor=pin(a.after),decision=pin(R/'selection-evidence.json'),source=pin(__file__),scope='One GPU lane; four bounded R2 windows, three original-history R1 combinations, native TRAIN capture then two explicitly matched broader-data windows. Final selection/retention/formal measurements remain.'))
 run('tied_q6_qualify')
 run('recover','H4R2','H6R2','HE4R2','HE6R2','--pilot-first')
 run('queue','Csmall','Chead','Ctied')
 run('broader_train_capture')
 manifest=pin(R/'broader-train-capture/manifest.json')
 specs={'H0broad':dict(name='H0broad',training_data=manifest),'Csmallbroad':dict(json.load(open(R/'specs/Csmall.json')),name='Csmallbroad',training_data=manifest)}
 for name,spec in specs.items():
  path=R/'specs'/(name+'.json')
  if path.exists() and json.load(open(path))!=spec:raise ValueError('broader recipe already differs')
  save(path,spec)
 save(R/'broader-recipe-decision.json',dict(control='H0broad',candidate='Csmallbroad',manifest=manifest,decision=pin(R/'selection-evidence.json'),policy='Both train directly from original B0 with identical seed8101 new-manifest order/cursor0,4096updates and unchanged R1optimizer/hard-label objective. Primary results remain separate.'))
 run('queue','H0broad','Csmallbroad')
 run('campaign_accounting')
 save(R/'recovery-combinations-result.json',dict(status='RECOVERY_COMBINATIONS_AND_BROADER_RECIPE_COMPLETE_FINAL_CONFIRMATION_REQUIRED',owner=identity(),decision=pin(R/'selection-evidence.json'),accounting=pin(R/'accounting.json')))
 print('RECOVERY_COMBINATIONS_COMPLETE_NOT_CAMPAIGN_COMPLETION',flush=True)

if __name__=='__main__':main()
