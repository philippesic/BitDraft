"""Post-shortlist natural heldout selection with all historical and192-group exclusions.

This module never opens candidate prompt rows until a frozen shortlist has been
validated. It does not evaluate quality, open old sealed payloads or tune models.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/w1a8_generalization'),str(ROOT/'scripts/contender_pipeline')]
import modern_natural_confirmation as natural
import modern_fresh as fresh
from scripts.dflash_baseline.precision_benchmark import pin,auth,save
from scripts.dflash_baseline.native_tokenize import Tokenizer,OUT as TOKENIZER_OUT
CONTEXT_POLICY=dict(context_tokens=4096,output_reserve=256,max_prompt_tokens=3840,eligibility='Native vocabulary-only full-prompt token count before ranked cohort selection; no quality outcomes')

def prepare(selection_path,out):
 selection_path=Path(selection_path);out=Path(out)
 if out.exists():raise FileExistsError(out)
 selected=json.load(open(selection_path));selection_pin=pin(selection_path)
 if selected.get('status')!='SHORTLIST_LOCKED_BEFORE_HELDOUT' or selected.get('heldout_opened') is not False:raise ValueError('shortlist must lock before heldout')
 if not 1<=len(selected['candidates'])<=4:raise ValueError('bounded finalist shortlist')
 if selected.get('context_policy')!=CONTEXT_POLICY:raise ValueError('native context eligibility must freeze before heldout')
 if selected['target']!=pin(P/'runs/spec-shortlist-20261030/models/Qwen3.5-4B-Q4_K_M.gguf') or selected['baseline']!=pin(P/'runs/spec-shortlist-20261030/models/qwen35-4b-dflash-Q4_K_M.gguf'):raise ValueError('heldout lane differs from frozen4B Q4 lane')
 tokenizer_qualification=auth(pin(TOKENIZER_OUT/'qualification.json'))
 if tokenizer_qualification['status']!='NATIVE_VOCAB_ADMISSION_QUALIFIED':raise ValueError('native tokenizer admission unqualified')
 for ref in [*selected['candidates'].values(),*selected['runtime'].values(),selected['target'],selected['baseline'],selected['training_data']]:
  if pin(ref['path'])['sha256']!=ref['sha256']:raise ValueError('shortlist artifact changed')
 # All source metadata pins are fixed in the post-selection receipt before source-row access.
 old=P/'runs/w1a8-modern-protected-20261103-01/natural-replay-confirmation01';base=json.load(open(old/'selected.json'));audit=natural.Metadata();evidence=natural.exclusion_evidence(base,audit)
 exposure_ref=pin(P/'runs/data-cost-transfer-20261104-01/train-exposure.json');exposure=auth(exposure_ref)
 if exposure['schema']!='opaque_train_exposure_v1' or exposure['groups']!=192:raise ValueError('missing192TRAIN exposure')
 for row in exposure['records']:
  evidence['hashes'].update(row['identities'].values());evidence['forbidden_ids'].add(row['original_prompt_id']);evidence['groups'].update([row['group_id'],row['original_group']]);evidence['topics'].update(row['original_topics'])
 consumed_ref=pin(old/'inputs01/manifest.json');consumed=auth(consumed_ref);consumed_inputs=json.load(open(P/'runs/data-cost-transfer-20261104-01/consumed-natural-inputs.json'));marker=auth(consumed_inputs['marker'])
 if not marker['evaluated']:raise ValueError('prior natural cohort consumption not authenticated')
 for row in consumed['groups']:
  evidence['hashes'].update(row['identities'].values());evidence['groups'].add(row['group_id'])
  if 'original_prompt_id' in row:evidence['forbidden_ids'].add(row['original_prompt_id'])
  if 'original_group' in row:evidence['groups'].add(row['original_group'])
  evidence['topics'].update(row.get('original_topics',[]))
 # No new prompt payload has been opened above. Preserve freeze first.
 out.mkdir();save(out/'selection-freeze.json',dict(selection=selection_pin,natural_sources=base['natural_sources'],train192=exposure_ref,consumed_natural=consumed_ref,consumption_marker=consumed_inputs['marker'],native_tokenizer=pin(TOKENIZER_OUT/'build.json'),tokenizer_qualification=pin(TOKENIZER_OUT/'qualification.json'),sources={n:pin(p) for n,p in {'adapter':__file__,'natural':natural.__file__,'identity':fresh.__file__}.items()},settings=dict(output_limit=256,domains={'code':8,'reasoning':8,'prose':8},context_policy=selected['context_policy']),scope='24 unused native-context-admissible natural singleton source clusters; all historical TRAIN/DEV/pilot/matrix/sealed opaque indices and192TRAIN exposure remain excluded; no tuning after this freeze.'))
 geometry=[]
 with Tokenizer(out/'native-tokenizer') as tokenizer:
  def eligible(row):
   text=natural.payload(natural.user_text(row['messages']))['prompt'];count=len(tokenizer.tokens(text));accepted=0<count<=3840;geometry.append(dict(id=row['prompt_id'],domain=row['domain'],tokens=count,context_eligible=accepted));return accepted
  groups,report=natural.select_natural(base,selection_pin,audit,evidence,eligibility=eligible)
 audit.recheck();save(out/'eligibility-geometry.json',dict(policy=selected['context_policy'],records=geometry,scope='Geometry-only before ranked24-case roster; no completion, labels, outcomes, truncation or post-selection replacement.'))
 if len(groups)!=24:raise ValueError('heldout cluster coverage')
 rows=[dict(id=x['id'],domain=x['domain'],group_id=x['group_id'],original_prompt_id=x['original_prompt_id'],original_group=x['original_group'],original_topics=x['original_topics'],payload=natural.payload(natural.user_text(x['messages'])),identities=x['identities'],source_ancestry=x['source_ancestry']) for x in groups]
 # Consumed before first request, including failed future serving attempts.
 save(out/'consumption.json',dict(status='CONSUMED_FOR_LOCKED_COMPARISON',selection=selection_pin,records=[{k:x[k] for k in ('group_id','original_prompt_id','original_group','original_topics','identities')} for x in rows],failed_attempts_also_consume=True))
 save(out/'prompts.json',rows);save(out/'audit.json',dict(selection=selection_pin,selection_report=report,exclusion_sources=list(audit.pins.values()),groups=24,role='HELDOUT_AFTER_SHORTLIST_LOCK',no_training_or_tuning=True));return out/'prompts.json'
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('selection',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();prepare(a.selection,a.out)
