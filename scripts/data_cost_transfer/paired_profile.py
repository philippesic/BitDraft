"""Interleave traced/untraced native blocks and require identical decode decisions."""
from pathlib import Path
import argparse,signal,sys
import host_profile as cost_profile
P=cost_profile.P
from native_panel import Panel
from audit_panel import audit
from runner import atomic,pin,read,require

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-cuda',action='store_true');a=p.parse_args();require(a.allow_cuda,'explicit GPU required');require(not a.out.exists(),'unique profile output');a.out.mkdir(parents=True)
 models=read(a.models);models['arms']=[x for x in models['arms'] if x['id'] in ('target_only','dflash_Q4KM','w1a8_initial','paid5760')];require(len(models['arms'])==4,'four controls required');atomic(a.out/'models.json',models)
 panels=[]
 for name,cls in [('plain',Panel),('traced',cost_profile.Profile)]:
  panel=cls(argparse.Namespace(models=a.out/'models.json',inputs=a.inputs,output=a.out/name,repetitions=5,allow_cuda=True));panel.records=[];panel.stopping=False
  if name=='traced':panel.library=a.library.resolve()
  panel.authenticate();panels.append(panel)
 def stop(sig,frame):
  for panel in panels:panel.request_stop(sig,frame)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 atomic(a.out/'freeze.json',dict(source=pin(__file__),trace_adapter=pin(cost_profile.__file__),library=pin(a.library),models=pin(a.models),inputs=pin(a.inputs),order='arms forward/reverse each repetition; paired plain/traced block order reversed each repetition',scope='Instrumentation overhead and complete native host costs; no architecture or runtime optimization'))
 for rep in range(5):
  ids=[x['id'] for x in panels[0].arms]
  if rep%2:ids.reverse()
  for ident in ids:
   for panel in panels if rep%2==0 else list(reversed(panels)):
    panel.cell(rep,next(x for x in panel.arms if x['id']==ident))
 plain,traced=panels
 atomic(plain.output/'result.json',dict(status='DEVELOPMENT_NATIVE_REQUESTS_PASS',records=plain.records,measured_requests=240))
 atomic(plain.output/'audit.json',audit(plain.output,'paid5760','paid5760'))
 lookup={(r['arm'],r['repetition'],r['id']):r for r in plain.records};pairs=[]
 for row in traced.records:
  key=(row['arm'],row['repetition'],row['id']);original=lookup.pop(key)
  require(all(row[k]==original[k] for k in ('completion_tokens','draft_proposed','draft_accepted','verification_rounds')),'instrumentation changed decode decisions')
  pairs.append(dict(arm=row['arm'],repetition=row['repetition'],id=row['id'],warmup=row['warmup'],plain_ms=original['full_http_wall_seconds']*1000,traced_ms=row['full_http_wall_seconds']*1000,rounds=row['verification_rounds']))
 require(not lookup,'incomplete instrumentation pairs')
 atomic(traced.output/'result.json',dict(status='DIAGNOSTIC_NATIVE_COST_PANEL_PASS',records=traced.records,repetitions=5,exact_decode_counters_match_uninstrumented=True))
 atomic(a.out/'result.json',dict(status='PAIRED_NATIVE_COST_AND_OVERHEAD_PASS',pairs=pairs,plain_audit=pin(plain.output/'audit.json'),traced_result=pin(traced.output/'result.json'),responses=len(plain.records)+len(traced.records),scope='Instrumented component costs with interleaved uninstrumented reference; nested times are non-additive'))
if __name__=='__main__':main()
