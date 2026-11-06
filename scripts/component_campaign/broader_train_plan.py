"""Prepare geometry-only inputs from the already admitted192-group TRAIN bank.

No GPU capture or training runs here. Native DFlash features/labels would be new,
and any use requires a separately labeled matched-control recipe.
"""
import hashlib,json
from .common import *
from scripts.dflash_baseline.native_tokenize import Tokenizer,OUT

def auth(ref):
 if pin(ref['path'])!=ref:raise ValueError('TRAIN source changed')
 return json.load(open(ref['path']))

def main():
 out=R/'broader-train-plan'
 if (out/'result.json').exists():return
 out.mkdir(exist_ok=False)
 bankpath=P/'runs/data-cost-transfer-20261104-01/bank-inputs/bank.json';bank=json.load(open(bankpath));exposure_path=P/'runs/data-cost-transfer-20261104-01/train-exposure.json';exposure=json.load(open(exposure_path))
 if bank['status']!='FROZEN_TRAIN_INPUTS' or bank['groups']!=192 or exposure['role']!='TRAIN':raise ValueError('not an admitted TRAIN bank')
 groups={r['group_id'] for r in exposure['records']};rows=[]
 for chunk in bank['chunks']:
  value=auth(chunk)
  if value['status']!='FROZEN_INPUTS_NO_LABELS':raise ValueError('unexpected bank role')
  for row in value['rows']:
   if row['phase']!='training' or row['group_id'] not in groups:raise ValueError('TRAIN group ancestry missing')
   rows.append(dict(row,chunk=chunk))
 if len(rows)!=192 or len({r['group_id'] for r in rows})!=192:raise ValueError('bank cluster identity changed')
 if json.load(open(OUT/'qualification.json'))['status']!='NATIVE_VOCAB_ADMISSION_QUALIFIED':raise ValueError('native tokenizer unqualified')
 selected=[];geometry=[]
 with Tokenizer(out/'tokenizer') as tokenizer:
  for domain,count in [('code',6),('reasoning',5),('prose',5)]:
   for row in sorted([r for r in rows if r['domain']==domain],key=lambda r:r['group_id']):
    textpath=Path(row['rendered_text_path'])
    if pin(textpath)['sha256']!=row['rendered_prompt_sha256']:raise ValueError('rendered TRAIN prompt changed')
    tokens=tokenizer.tokens(textpath.read_text());eligible=0<len(tokens)<=512
    geometry.append(dict(id=row['id'],domain=domain,prompt_tokens=len(tokens),eligible=eligible))
    if not eligible:continue
    dest=out/(row['id']+'.ids');dest.write_text(' '.join(map(str,tokens))+'\n')
    selected.append(dict(id=row['id'],domain=domain,group_id=row['group_id'],tokens=pin(dest),prompt_tokens=len(tokens),native_greedy_steps=128,source_chunk=row['chunk'],rendered_prompt=pin(textpath),identities=row['identities'],source_ancestry=row['source_ancestry'],role='TRAIN'))
    count-=1
    if count==0:break
   if count:raise ValueError('insufficient geometry-admitted TRAIN inputs')
 features=sum((x['prompt_tokens']+128)*20480*4 for x in selected)
 save(out/'result.json',dict(status='INPUTS_PREPARED_NO_CAPTURE_OR_TRAINING',records=selected,geometry=geometry,bank=pin(bankpath),TRAIN_exposure=pin(exposure_path),native_tokenizer=pin(OUT/'build.json'),planned_feature_bytes=features,planned_one_cell_logit_transient_bytes=129*248320*4,retention_policy='All PRIOR paid captures remain protected. If this plan is selected, retain new8-tap features, exact native greedy labels, inputs and source/row-hash receipts. The new full-logit file is a disposable argmax-verification scratch for this hard-label objective; verify every row against captured greedy IDs before removing it. No KL/distillation objective. Never use earlier EAGLE feature files as DFlash captures.',eligibility='Already frozen TRAIN192clusters permanently excluded from evaluation. Geometry/domain/group order only; no quality-outcome selection. New native capture and new manifest require a distinct matched-control recipe before use.',source=pin(__file__)))
 print('BROADER_TRAIN_INPUTS_READY',len(selected),features,'feature bytes; no GPU work',flush=True)

if __name__=='__main__':main()
