import copy,unittest
from pathlib import Path
import bank
P=bank.P;R=P/'runs/data-cost-transfer-20261104-01'
class BankTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows,cls.evidence,_,_=bank.select(R/'scope.json')
 def test_reproducible_frozen_selection(self):
  allrows=[]
  for ref in bank.read(R/'bank-inputs/bank.json')['chunks']:
   roster=bank.read(ref['path']);bank.validate(roster)
   allrows.extend({k:v for k,v in r.items() if k not in ('rendered_text_path','rendered_prompt_sha256')} for r in roster['rows'])
  self.assertEqual({r['id']:r for r in self.rows},{r['id']:r for r in allrows})
 def test_consumed_aliases_excluded(self):
  scope=bank.read(R/'scope.json');used=bank.read(scope['latest_consumed']['path'])['groups']
  hashes=set().union(*(set(r['identities'].values()) for r in used))
  ids={r.get('original_prompt_id') for r in used};groups={r.get('original_group') for r in used};topics=set().union(*(set(r.get('original_topics',[])) for r in used))
  for r in self.rows:
   self.assertFalse(set(r['identities'].values())&hashes);self.assertNotIn(r['original_prompt_id'],ids);self.assertNotIn(r['original_group'],groups);self.assertFalse(set(r['original_topics'])&topics)
 def test_balanced_unique_bounded(self):
  self.assertEqual(len(self.rows),192);self.assertEqual(len({r['group_id'] for r in self.rows}),192)
  for d in ('code','reasoning','prose'):self.assertEqual(sum(r['domain']==d for r in self.rows),64)
  self.assertTrue(all(len(r['messages'][0]['content'].encode())<=280 and r['phase']=='training' for r in self.rows))
  self.assertEqual(self.evidence['partition_indices'],25)
 def test_bad_partition_or_prompt_rejected(self):
  ref=bank.read(R/'bank-inputs/bank.json')['chunks'][0];roster=bank.read(ref['path'])
  for field,value in [('role','development'),('phase','performance'),('message_identity_sha256','0'*64)]:
   bad=copy.deepcopy(roster);bad['rows'][0][field]=value
   with self.assertRaises(ValueError):bank.validate(bad)
  bad=copy.deepcopy(roster);bad['rows'][0]['messages'][0]['content']+=' tampered'
  with self.assertRaises(ValueError):bank.validate(bad)
if __name__=='__main__':unittest.main()
