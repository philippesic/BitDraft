"""Lock declared finalists before opening a new24-cluster natural cohort."""
import argparse,json,subprocess,sys
from .common import *
from .measure import completed
from .native import pin as native_pin

def main():
 ap=argparse.ArgumentParser();ap.add_argument('candidates',nargs='+');ap.add_argument('--controls',nargs='*',default=['H0']);a=ap.parse_args()
 if not 1<=len(a.candidates)<=4 or len(set(a.candidates+a.controls))!=len(a.candidates+a.controls):raise ValueError('bounded distinct candidates and controls required')
 registry=completed();selection=R/'final-selection.json';runtime=R/'runtime04'
 files=('libllama.so.0.8.0','libggml-base.so.0.28.0','libggml-cuda.so.0.28.0','libggml-cpu.so.0.28.0','libllama-common.so.0.8.0','libllama-server-impl.so','libggml.so.0.28.0','llama-server')
 from scripts.dflash_baseline import prepare_holdout as holdout
 prior=[P/'runs/precision-campaign-20261104-01/heldout24/consumption.json',P/'runs/precision-campaign-20261104-01/clip-control-followup/heldout24/consumption.json']
 controls={'B0':native_pin(B/'model-final.gguf'),**{n:native_pin(registry[n]['model']) for n in a.controls}}
 packet=dict(status='SHORTLIST_LOCKED_BEFORE_HELDOUT',heldout_opened=False,target=pin(TARGET),baseline=pin(Q4),candidates={n:native_pin(registry[n]['model']) for n in a.candidates},controls=controls,runtime={n:pin(runtime/'bin'/n) for n in files},training_data=json.load(open(R/'freeze.json'))['B0_config']['data'],prior_consumptions=[pin(p) for p in prior],context_policy=holdout.CONTEXT_POLICY,component_recipes={n:registry[n] for n in a.candidates+a.controls},selection_evidence=pin(R/'selection-evidence.json'),source=pin(__file__),policy='Recipes/configs fixed from B0 matched histories. This cohort is final confirmation only; no tuning/selection after opening. Earlier consumed DEV diagnostics are separate.')
 if selection.exists():
  old=json.load(open(selection))
  if old!=packet:raise ValueError('final selection already locked differently')
 else:save(selection,packet)
 for ref in controls.values():
  if native_pin(ref['path'])['sha256']!=ref['sha256']:raise ValueError('locked control changed before heldout access')
 # The historical helper's artifact pin is extended to authenticated slice archives.
 # No native model hydration and no source prompt access precedes shortlist lock.
 holdout.pin=native_pin
 folder=R/'heldout24'
 prompts=folder/'prompts.json'
 if not prompts.exists():prompts=holdout.prepare(selection,folder)
 arms=dict(target=None,q4=str(Q4),**{n:r['path'] for n,r in controls.items()},**{n:registry[n]['model'] for n in a.candidates})
 cache={n:registry[n]['spec'].get('kv',['f16','f16']) for n in a.candidates+a.controls}
 save(R/'panels/final-heldout-arms.json',arms);save(R/'panels/final-heldout-cache.json',cache)
 out=R/'measurements/final-heldout24'
 if not (out/'result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.native','--arms',str(R/'panels/final-heldout-arms.json'),'--draft-cache-map',str(R/'panels/final-heldout-cache.json'),'--runtime',str(runtime),'--out',str(out),'--prompts',str(prompts),'--cohort-label','final-heldout24','--warmups','2','--reps','5'],cwd=ROOT,check=True)
 print('FINAL_HELDOUT_COMPLETE',flush=True)

if __name__=='__main__':main()
