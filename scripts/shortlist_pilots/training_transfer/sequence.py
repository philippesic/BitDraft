"""Finite serial native/CPU stages within one already-held supervised GPU custody."""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def main():
 jobs={'capture':[('historical_capture.py',[]),('historical_data.py',[]),('audit_labels.py',[])],'analyze-current':[('analyze_current.py',[])]}[sys.argv[1]];started=time.monotonic();receipts=[]
 for name,args in jobs:
  dest=RUN/(name.removesuffix('.py')+'-stage.log')
  with dest.open('xb') as log:p=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(RUN/'source'/name),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  row={'script':name,'returncode':p.returncode,'log':str(dest)};receipts.append(row);(RUN/('sequence-'+sys.argv[1]+'.json')).write_text(json.dumps({'stages':receipts,'wall_seconds':time.monotonic()-started},indent=2))
  if p.returncode:print(dest.read_text(errors='replace')[-6000:],flush=True);raise SystemExit(p.returncode)
 print(json.dumps({'status':'FINITE_SEQUENCE_COMPLETE','stages':receipts}),flush=True)
if __name__=='__main__':main()
