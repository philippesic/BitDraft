"""Bounded CPU qualification and group-balanced historical capture selection."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';A=ROOT/'runs/w1a8-algorithms-20261020'
def pin(p):
 with Path(p).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h}
def main():
 import torch
 sp=json.loads((RUN/'historical-original-spec.json').read_text());cp=json.loads((RUN/'historical16-sidecar.json').read_text());initial=sp['original_initializer']['checkpoint'];base=Path(cp['path']).parent
 anchors={'initializer':initial,'8hour':json.loads(next(base.glob('step-000000029866-*.receipt.json')).read_text()),'16hour':cp}
 source_roots=sorted((ROOT/'runs').glob('unattended-runtime-*/source'));qualified={}
 for name,rec in anchors.items():
  assert pin(rec['path'])['sha256']==rec['sha256'];saved=torch.load(rec['path'],map_location='cpu',mmap=True,weights_only=False);runtime=saved['runtime'];matches=[]
  for s in source_roots:
   if all((s/'src/w1a1_eagle'/f).exists() and pin(s/'src/w1a1_eagle'/f)['sha256']==h for f,h in runtime['source'].items()):matches.append(str(s))
  qualified[name]={'checkpoint':rec,'contract':saved['contract'],'source':saved['source'],'runtime':runtime,'matching_source_roots':matches,'seen_groups':list((saved['cursor'].get('telemetry') or {}).get('unique_groups',[])),'seen_chains':list((saved['cursor'].get('telemetry') or {}).get('unique_chains',[]))}
  del saved
 # Preserved native model identity joined to paid checkpoint via original receipt/export audit.
 binding=Path(initial['path']).parents[2];ir=json.loads((binding/'initial-export-receipt.json').read_text());qualified['initializer']['model']=ir['model'];qualified['initializer']['export_receipt']=pin(binding/'initial-export-receipt.json')
 # Match every retained evaluation to its exact checkpoint hash; no use of latest pointer.
 for exp in base.parent.glob('unattended/*/evaluations/*/export/trained.gguf'):
  for p in exp.parent.parent.glob('*.json'):
   try:j=json.loads(p.read_text())
   except (ValueError,OSError):continue
   checkpoint=j.get('checkpoint',{})
   if isinstance(checkpoint,dict):
    for name,q in qualified.items():
     if checkpoint.get('sha256')==q['checkpoint']['sha256']:
      audit=exp.parent/'export-audit.json';ar=json.loads(audit.read_text())
      if ar.get('output',{}).get('path')==str(exp):q['model']=ar['output'];q['export_audit']=pin(audit);q['evaluation_receipt']=pin(p)
 for name,q in qualified.items():
  if 'model' in q:assert pin(q['model']['path'])['sha256']==q['model']['sha256']
 # Initializer has no fit history; use original authenticated explicit initializer transition if source absent.
 qualified['initializer']['adapter_if_needed']='original restore_original_scale_only_initializer; zero-history transition only'
 inputs=A/'loop2-answer-broad-data01/inputs.json';groups={}
 for c in json.loads(inputs.read_text())['fit']['answer']:groups.setdefault(c['group_id'],c)
 existing={w['group_id'] for w in json.loads((RUN/'matrix.json').read_text())['windows']};seen=set(qualified['16hour']['seen_groups']);cases=[];counts={}
 for domain in ('code','prose','reasoning'):
  for exposed in (True,False):
   available=sorted((c for g,c in groups.items() if g not in existing and c['domain']==domain and (g in seen)==exposed),key=lambda c:hashlib.sha256(c['group_id'].encode()).hexdigest());counts[domain+'/'+str(exposed)]=len(available)
   assert len(available)>=4,('insufficient16h exposure strata',counts)
   for c in available[:4]:
    rec=json.loads(Path(c['native_receipt']['path']).read_text());assert pin(c['native_receipt']['path'])['sha256']==c['native_receipt']['sha256']
    cases.append({'index':len(cases),'role':'historical_seen' if exposed else 'historical_not_seen','historical16_seen':exposed,'group_id':c['group_id'],'domain':domain,'chain_id':c['chain_id'],'ancestry':c['native_receipt'],'prompt_tokens':rec['tokens'][:rec['prompt_length']],'prompt_length':rec['prompt_length'],'source_split':'TRAIN'})
 (RUN/'historical-anchors.json').write_text(json.dumps(qualified,indent=2));(RUN/'historical-split.json').write_text(json.dumps({'schema':'historical_exposure_groups_v1','source':pin(inputs),'policy':'four16h-seen +four16h-not-seen groups/domain, deterministic sha256 group ordering excluding36current groups. Each checkpoint exposure joined separately. All reused projectTRAIN.','available_strata_counts':counts,'cases':cases},indent=2));print(json.dumps({'status':'HISTORICAL_PREPARED','anchors':{n:{'step':q['checkpoint']['cursor']['step'],'matching_sources':len(q['matching_source_roots']),'native_model_available':'model' in q,'seen_groups':len(q['seen_groups'])} for n,q in qualified.items()},'new_groups':len(cases),'strata':counts}),flush=True)
if __name__=='__main__':main()
