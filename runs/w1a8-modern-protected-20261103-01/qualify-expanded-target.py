from pathlib import Path
import argparse,json,sys,signal,time
P=Path('/home/philip/binary-eagle-decoding');W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');R=P/'runs/w1a8-modern-protected-20261103-01';C=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01')
sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
from modern_requests import Requests,atomic,pin,signature,read
from modern_expand_capture import validate_roster,protocol
state=read(P/'runs/local-w1a8-modern-fit-expansion-20261103-01/state.json');assert state['status']=='finished' and state['exit_code']==0
roster=read(C/'inputs01/roster.json');validate_roster(roster);summary=read(C/'capture01/summary.json');assert len(summary['groups'])==24 and all(g['status']=='ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS' for g in summary['groups'])
old=read(P/'runs/w1a8-modern-actual-training-20261103-01/inputs480.json')
inputs={'schema':'new24FIT_frozen_target_runtime_label_crosscheck_v1','roster':pin(C/'inputs01/roster.json'),'source_pins':[pin(C/'capture01/summary.json'),old['roster']],'rows':[{'id':g['id'],'request':pin(C/'capture01'/g['id']/'request.json'),'reference':pin(C/'capture01'/g['id']/'response.json')} for g in summary['groups']],'warmups':old['warmups']}
file=C/'frozen-target-inputs.json';assert not file.exists();atomic(file,inputs)
class ExpandedTargetRequests(Requests):
 def authenticate_inputs(self):
  self.output_limit=80;self.request_schema='new24FIT_frozen_target_runtime_label_crosscheck_v1';self.request_scope='newFIT capture label agreement with frozen target runtime only; no repeatedthroughput/qualityclaim';self.input_source_refs=[]
  roster=read(self.checked(self.inputs['roster']));validate_roster(roster);oldroster=read(self.checked(old['roster']));rows={r['id']:r for r in roster['rows']+oldroster['rows']};self.rows=self.inputs['rows'];self.warmups=self.inputs['warmups'];assert len(self.rows)==24 and {r['id'] for r in self.rows}=={r['id'] for r in roster['rows']} and len(self.warmups)==2
  self.payloads={};self.gold={}
  for item in self.rows+self.warmups:
   payload=read(self.checked(item['request']));assert payload=={'prompt':Path(rows[item['id']]['rendered_text_path']).read_text(),**protocol()};self.payloads[item['id']]=payload;self.gold[item['id']]=signature(read(self.checked(item['reference'])))
x=ExpandedTargetRequests(argparse.Namespace(models=R/'selected-lr-models-verbose.json',inputs=file,output=C/'frozen-target-check01',repetitions=5,allow_cuda=True));x.stopping=False;x.records=[];signal.signal(signal.SIGTERM,x.request_stop);signal.signal(signal.SIGINT,x.request_stop);x.authenticate();arm=next(a for a in x.arms if a['kind']=='target');x.cell(0,arm);assert len(x.records)==26
atomic(x.output/'result.json',{'status':'NEW24FIT_LABELS_EXACT_FROZEN_TARGET_RUNTIME','records':x.records,'inputs':pin(file),'models':pin(R/'selected-lr-models-verbose.json'),'wall_seconds':time.perf_counter()-x.started,'scope':'all24 captured parent labels agree token/text/stopping with independent frozen correctness target-only runtime; no repeatedperformanceclaim'})
