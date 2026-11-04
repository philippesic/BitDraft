import copy,json,tempfile,unittest
from pathlib import Path
from comparison import summarize,reserve_confirmation
from runner import pin,read

class ComparisonTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  def put(name,v):
   p=self.root/name;p.write_text(json.dumps(v));return pin(p)
  self.put=put
  artifact=put('artifact',{'test':True});req=put('request',{'temperature':0,'cache_prompt':False});res=put('response',dict(tokens=[1,2],content='a',stop_type='limit',stopping_word='',truncated=False))
  self.panel=dict(schema='contender_comparison_v1',tier='contender',repetitions=5,cases=['x','y'],arms=['base','candidate'],baseline='base',model_pins={'base':artifact,'candidate':artifact},runtime_pins={'base':artifact,'candidate':artifact},target_pin=artifact,workload_pin=artifact,admission_pin=artifact,semantics='greedy_exact_target',records=[])
  for a in self.panel['arms']:
   for c in self.panel['cases']:
    for r in range(5):self.panel['records'].append(dict(arm=a,case=c,repetition=r,wall_seconds=2. if a=='base' else 1.,tokens=2,response=res,reference=res,request=req))
 def test_known_ratio(self):
  r=summarize(self.panel)['results'][1];self.assertEqual(r['speed_ratio'],2);self.assertEqual(r['interval95'],[2.,2.])
 def test_incomplete_duplicate_and_nonfinite(self):
  for kind in ['missing','duplicate','nan','bad_tokens','rep','mismatched_request']:
   p=copy.deepcopy(self.panel)
   if kind=='missing':p['records'].pop()
   if kind=='duplicate':p['records'].append(copy.deepcopy(p['records'][0]))
   if kind=='nan':p['records'][0]['wall_seconds']=float('nan')
   if kind=='bad_tokens':p['records'][0]['tokens']=3
   if kind=='rep':p['records'][0]['repetition']=5
   if kind=='mismatched_request':p['records'][-1]['request']=self.put('otherrequest',{'temperature':0,'cache_prompt':False,'seed':5})
   with self.subTest(kind=kind),self.assertRaises(ValueError):summarize(p)
 def test_parity_and_pin_corruption(self):
  p=copy.deepcopy(self.panel);p['records'][0]['response']=self.put('badresponse',dict(tokens=[3,4],content='b',stop_type='limit',stopping_word='',truncated=False))
  with self.assertRaisesRegex(ValueError,'parity'):summarize(p)
  (self.root/'response').write_text('{}')
  with self.assertRaisesRegex(ValueError,'changed'):summarize(self.panel)
 def test_smoke_has_no_rank(self):
  self.panel['tier']='smoke';self.assertNotIn('finding',summarize(self.panel)['results'][1])
 def test_confirmation_consumes_before_use(self):
  ledger=self.root/'ledger.json';ledger.write_text(json.dumps(dict(schema='evaluation_consumption_v1',consumed=['used'])))
  selection=self.put('selection',{'candidate':'candidate'})
  with self.assertRaises(ValueError):reserve_confirmation(ledger,self.root/'bad.json',['used'],selection)
  receipt=self.root/'reservation.json';reserve_confirmation(ledger,receipt,['x','y'],selection)
  self.assertEqual(read(ledger)['consumed'],['used','x','y'])
  with self.assertRaises(ValueError):reserve_confirmation(ledger,self.root/'again.json',['x'],selection)
  self.panel.update(tier='confirmation',partition='confirmation',selection_pin=selection,consumption_receipt=pin(receipt))
  self.assertEqual(summarize(self.panel)['status'],'PAIRED_PANEL_PASS')
  self.panel['cases']=['z','y']
  with self.assertRaisesRegex(ValueError,'reservation mismatch'):summarize(self.panel)
if __name__=='__main__':unittest.main()
