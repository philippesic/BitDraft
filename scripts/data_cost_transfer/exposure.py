"""Opaque TRAIN exposure exclusion, including selected groups whose capture failed.

Future source selection must apply this after its historical exclusion evidence.
Never reinterpret a paid/tried TRAIN source as an untouched evaluation input.
"""
from pathlib import Path
import argparse,json,sys
P=Path('/home/philip/binary-eagle-decoding')
sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,require,atomic

def build(bank_path,out):
 bank=read(bank_path);require(bank['schema']=='native_public_train_bank_v1','bank schema differs')
 rows=[];seen=set()
 for ref in bank['chunks']:
  require(pin(ref['path'])==ref,'bank roster changed')
  for r in read(ref['path'])['rows']:
   require(r['role']=='fit' and r['group_id'] not in seen,'duplicate or heldout source');seen.add(r['group_id'])
   rows.append(dict(group_id=r['group_id'],original_prompt_id=r['original_prompt_id'],original_group=r['original_group'],original_topics=r['original_topics'],identities=r['identities']))
 require(len(rows)==bank['groups'],'bank exposure incomplete')
 atomic(out,dict(schema='opaque_train_exposure_v1',role='TRAIN',source_bank=pin(bank_path),source=pin(__file__),groups=len(rows),records=rows,exclude_even_if_capture_or_training_failed=True))

def apply(evidence,ref,audit):
 value=audit.read(ref);require(value['schema']=='opaque_train_exposure_v1' and value['role']=='TRAIN' and value['exclude_even_if_capture_or_training_failed'],'invalid TRAIN exposure')
 require(len(value['records'])==value['groups'] and len({r['group_id'] for r in value['records']})==value['groups'],'incomplete exposure groups')
 for row in value['records']:
  require(set(row['identities'])=={'raw_sha256','normalized_raw_sha256','rendered_sha256','normalized_rendered_sha256'} and all(len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in row['identities'].values()),'incomplete exposure identity')
  evidence['hashes'].update(row['identities'].values());evidence['forbidden_ids'].add(row['original_prompt_id']);evidence['groups'].update([row['group_id'],row['original_group']]);evidence['topics'].update(row['original_topics'])
 return evidence
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.bank,a.out)
