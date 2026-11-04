"""Reuse consumed natural performance inputs as DEV; never claim fresh confirmation."""
from pathlib import Path
import argparse,sys
import native_log_text
native_log_text.install()
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/design_readiness'),str(P/'scripts/w1a8_generalization'),str(P/'scripts/contender_pipeline')]
from native_panel import Panel
from modern_requests import signature
from modern_confirm import validate_payload,validate_response
from runner import pin,read,require,atomic
from audit_panel import audit

class ConsumedPanel(Panel):
 def authenticate_inputs(self):
  refs=self.inputs;require(refs['schema']=='consumed_natural_performance_v1','explicit consumed schema required')
  manifest=read(self.checked(refs['manifest']));marker=read(self.checked(refs['marker']));prior=read(self.checked(refs['prior_result']))
  require(marker['evaluated'] and marker['input_manifest']==refs['manifest'] and marker['result']==refs['prior_result'],'consumption ancestry differs')
  originals={r['id']:r for r in read(self.checked(refs['original_requests']))}
  groups={g['id']:g for g in manifest['groups'] if g.get('phase')=='performance'}
  require(len(originals)==len(groups)==24,'all consumed natural24 required')
  previous={r['id']:r for r in prior['records'] if r['arm']=='target_only' and r['repetition']==0 and not r['warmup'] and r['stage']=='performance'}
  self.rows=refs['rows'];self.warmups=refs['warmups'];self.payloads={};self.gold={}
  require(len(self.rows)==24 and {r['id'] for r in self.rows}==set(groups),'complete consumed cohort required')
  warm=read(self.checked(refs['warmup_source']));require(self.warmups==warm['warmups'],'warmup inputs changed')
  for item in self.rows+self.warmups:
   payload=read(self.checked(item['request']));gold=read(self.checked(item['reference']))
   if item['id'] in originals:
    require(item['request']==previous[item['id']]['request'] and item['reference']==previous[item['id']]['response'],'prior target reference differs')
    require(payload==originals[item['id']]['request'],'consumed request changed');validate_payload(payload,256);validate_response(gold,payload)
   else:validate_payload(payload,80)
   self.payloads[item['id']]=payload;self.gold[item['id']]=signature(gold)
  self.output_limit=256;self.request_schema='consumed_natural_performance_v1';self.request_scope='Previously consumed natural24 DEV only; no fresh generalization confirmation, final pool, or quality requests'
  self.input_source_refs=[pin(__file__)]
 def prepare_cell(self,port,cell,arm,rep):
  geometry=[]
  for item in self.rows:
   payload=self.payloads[item['id']];tokens=self.request(port,'tokenize',dict(content=payload['prompt'],add_special=True,parse_special=True))['tokens']
   require(0<len(tokens)+256<=4096,'native context exceeded; no replacement')
   geometry.append(dict(id=item['id'],prompt_tokens=len(tokens),output_reserve=256))
  atomic(cell/'geometry.json',geometry)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();require(a.allow_cuda,'explicit GPU required');a.repetitions=5
 ConsumedPanel(a).run();result=audit(a.output,'paid5760','paid5760');atomic(a.output/'audit.json',result)
