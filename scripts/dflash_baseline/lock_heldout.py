"""Freeze predeclared2048 family models after DEV, before any fresh payload."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.precision_benchmark import pin,save,auth
from scripts.dflash_baseline.prepare_holdout import CONTEXT_POLICY
from scripts.dflash_baseline.native_tokenize import OUT as TOKENIZER_OUT
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/precision-campaign-20261104-01';runtime=P/'runs/precision-native-20261104-04'
def main():
 panel=R/'confirmation-dev12';result=json.load(open(panel/'result.json'));freeze=json.load(open(panel/'freeze.json'));training=json.load(open(R/'finalist-training-confirmation.json'));plan=auth(training['plan'])
 if result['status']!='COMPLETE' or not result['paired_trace_counters_equal'] or not all(x['correct'] for x in result['rows']):raise ValueError('DEV confirmation incomplete')
 if freeze['warmup_policy']!=dict(per_server_cell=2,measured_repetitions=5,same_process=True,source='first consumed cohort prompts, repeated only for excluded warmup'):raise ValueError('final DEV protocol differs')
 if json.load(open(TOKENIZER_OUT/'qualification.json'))['status']!='NATIVE_VOCAB_ADMISSION_QUALIFIED':raise ValueError('native tokenizer unqualified')
 if json.load(open(R/'slot16-qualification/result.json'))['status']!='GAINED_STREAM_CUDA36_N16_BYTEEXACT':raise ValueError('actual16-slot operations unqualified')
 candidates={}
 for name,record in training['records'].items():
  label=name+'-2048';ref=record['model']
  if record['updates']!=2048 or freeze['arms'].get(label)!=ref or pin(ref['path'])!=ref:raise ValueError('predeclared finalist changed')
  measured=[x for x in result['rows'] if x['arm']==label and not x['warmup'] and not x['instrumented']]
  if len(measured)!=60 or {x['rep'] for x in measured}!=set(range(5)) or len({x['prompt'] for x in measured})!=12:raise ValueError('finalist repeated DEV coverage')
  candidates[label]=ref
 if len(candidates)!=4 or training['status']!='FOUR_MATCHED_2048_RECIPES_COMPLETE':raise ValueError('four matched predeclared recipes required')
 for name,ref in freeze['runtime'].items():
  if pin(runtime/'bin'/name)!=ref:raise ValueError('runtime changed after confirmation')
 selected=dict(status='SHORTLIST_LOCKED_BEFORE_HELDOUT',heldout_opened=False,created_at_utc=datetime.now(timezone.utc).isoformat(),candidates=candidates,runtime=freeze['runtime'],target=freeze['target'],baseline=freeze['arms']['q4'],training_data=plan['data'],training_plan=training['plan'],dev_confirmation=pin(panel/'result.json'),context_policy=CONTEXT_POLICY,selection_policy='The four family recipes/configurations were locked before2048 training confirmation; retain the four2048 points for heldout coverage. No heldout-derived selection or epoch tuning.',sources=dict(lock=pin(__file__),tokenizer_qualification=pin(TOKENIZER_OUT/'qualification.json'),slot16_qualification=pin(R/'slot16-qualification/result.json')))
 path=R/'heldout-selection.json'
 if path.exists():raise FileExistsError('shortlist is already frozen')
 save(path,selected);save(R/'heldout-arms.json',{'target':None,'q4':selected['baseline']['path'],**{name:ref['path'] for name,ref in candidates.items()}})
if __name__=='__main__':main()
