import json,tempfile,unittest
from pathlib import Path
from bundle import create,verify
from runner import pin

class BundleTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  source=self.root/'model';source.write_bytes(b'GGUF-fixture')
  ref=pin(source);self.selection=dict(schema='contender_release_selection_v1',candidate='fixture',supported_workloads=['synthetic test'],model=ref,target=ref,runtime_dependencies=[ref],evidence={'smoke':ref},correctness='greedy_exact_target',claim='reference_rehearsal',settings={'context':4096})
  self.path=self.root/'selection.json';self.path.write_text(json.dumps(self.selection))
 def test_bundle_copy_relocation_and_corruption(self):
  create(self.path,self.root/'bundle');(self.root/'bundle').rename(self.root/'moved');verify(self.root/'moved')
  (self.root/'moved/draft.gguf').write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'changed'):verify(self.root/'moved')
 def test_no_winner_from_rehearsal(self):
  self.selection['claim']='confirmed_winner';self.path.write_text(json.dumps(self.selection))
  with self.assertRaisesRegex(ValueError,'confirmation panel missing'):create(self.path,self.root/'bundle')
 def test_changed_dependency(self):
  (self.root/'model').write_bytes(b'other')
  with self.assertRaisesRegex(ValueError,'dependency changed'):create(self.path,self.root/'bundle')
 def test_confirmed_bundle_requires_matched_frozen_candidate_and_dflash(self):
  from test_comparison import ComparisonTests
  from comparison import reserve_confirmation
  fixture=ComparisonTests('test_known_ratio');fixture.setUp();self.addCleanup(fixture.doCleanups)
  panel=fixture.panel;model=self.selection['model'];panel['model_pins']['candidate']=model
  choice=fixture.put('choice',{'candidate':'candidate','model':model})
  ledger=fixture.root/'ledger';ledger.write_text(json.dumps(dict(schema='evaluation_consumption_v1',consumed=[])))
  receipt=fixture.root/'receipt';reserve_confirmation(ledger,receipt,panel['cases'],choice)
  panel.update(tier='confirmation',partition='confirmation',selection_pin=choice,consumption_receipt=pin(receipt))
  panel_ref=fixture.put('panel',panel)
  policy=fixture.put('policy',dict(primary_comparison_baseline='DFlash Q4_K_M',rows=[dict(target_pin=panel['target_pin'],draft_pin=panel['model_pins']['base'])]))
  self.selection.update(candidate='candidate',claim='confirmed_winner',target=panel['target_pin'],baseline_policy=policy,frozen_candidate_selection=choice['path'],evidence={'confirmation_panel':panel_ref})
  self.path.write_text(json.dumps(self.selection));create(self.path,self.root/'confirmed');verify(self.root/'confirmed')
  self.selection['model']=panel['model_pins']['base'];self.path.write_text(json.dumps(self.selection))
  with self.assertRaises(ValueError):create(self.path,self.root/'wrong-model')
if __name__=='__main__':unittest.main()
