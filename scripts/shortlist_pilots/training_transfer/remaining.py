"""Finite serial remainder; no controllers, training or automatic successors."""
from pathlib import Path
import json,subprocess,time
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def contexts():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=15).strip()
def main():
 assert json.loads((RUN/'historical-actor/16hour/result.json').read_text())['status']=='HISTORICAL_ACTOR_COMPLETE'
 jobs=[('actor-8hour','historical_actor.py',['8hour']),('actor-initializer','historical_actor.py',['initializer']),('native-historical','historical_native.py',[]),('actor-current-historical','historical_current_actor.py',[]),('head-control','head_control.py',[]),('precision','precision_inventory.py',[]),('analysis-historical','analyze_historical.py',[])]
 receipts=[];started=time.monotonic()
 for label,script,args in jobs:
  assert contexts()=='','original released-context gate before subsequent launch'
  log=RUN/(label+'-remainder.log');entry={'stage':label,'script':script,'args':args,'started_unix':time.time(),'log':str(log)};receipts.append(entry);(RUN/'remaining-progress.json').write_text(json.dumps({'stages':receipts,'wall_seconds':time.monotonic()-started},indent=2))
  with log.open('xb') as f:p=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(RUN/'source'/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=750)
  entry.update(returncode=p.returncode,ended_unix=time.time(),contexts_after=contexts());(RUN/'remaining-progress.json').write_text(json.dumps({'stages':receipts,'wall_seconds':time.monotonic()-started},indent=2));assert not entry['contexts_after']
  if p.returncode:print(log.read_text(errors='replace')[-7000:],flush=True);raise RuntimeError('Finite remainder stopped for producer/consumer diagnosis at '+label)
  print(json.dumps({'stage':label,'status':'COMPLETE'}),flush=True)
 (RUN/'remaining-result.json').write_text(json.dumps({'status':'FINITE_REMAINDER_COMPLETE','stages':receipts,'wall_seconds':time.monotonic()-started},indent=2))
if __name__=='__main__':main()
