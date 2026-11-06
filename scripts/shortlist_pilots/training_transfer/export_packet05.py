"""Serialize compact closed evidence into a private, hash-inventoried transfer packet."""
from pathlib import Path
import json,hashlib,shutil,csv
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 stage=RUN/'evidence-staged';stage.mkdir(exist_ok=False);joins={}
 def add(src,name):
  src=Path(src);dest=stage/name;assert src.is_file() and not dest.exists();shutil.copyfile(src,dest);assert sha(src)==sha(dest);joins[name]={'source':str(src),'source_sha256':sha(src),'staged':str(dest),'bytes':dest.stat().st_size}
 for rel,name in [('preflight.json','preflight.json'),('source-pins.json','source-pins.json'),('matrix.json','matrix-current.json'),('historical-matrix.json','matrix-historical.json'),('historical-split.json','split-historical.json'),('label-audit.json','label-audit.json'),('precision-inventory.json','precision-inventory.json'),('unchanged-controls.json','unchanged-controls.json'),('head-controls/result.json','head-controls.json'),('analysis-current/summary.json','summary-current.json'),('analysis-current/disagreements.json','disagreements-current.json'),('analysis-historical02/summary.json','summary-historical.json'),('analysis-historical02/first-disagreements.json','disagreements-historical.json'),('cleanup-result.json','cleanup-result.json'),('capture05/complete.json','capture-complete.json'),('capture05/release.json','capture-release.json'),('occupancy.json','occupancy-current.json'),('fidelity-extra.json','fidelity-extra.json'),('remainder-bound-repair.json','remainder-bound-repair.json'),('native/ready.json','native-ready.json'),('remaining-r2-progress.json','remaining-r2-progress.json'),('native-work-accounting.json','native-work-accounting.json'),('new-capture-command.json','new-capture-command.json'),('new-capture-requests.jsonl','new-capture-requests.jsonl')]:
  if (RUN/rel).exists():add(RUN/rel,name)
 anchors=json.loads((RUN/'historical-anchors-qualified.json').read_text());compact={}
 for name,q in anchors.items():
  cp=q['checkpoint'];compact[name]={**{k:v for k,v in q.items() if k!='checkpoint'},'checkpoint':{k:v for k,v in cp.items() if k!='cursor'},'step':cp['cursor']['step'],'TRAIN_seconds':cp['cursor']['elapsed_seconds'],'paid_checkpoint_group_count':len(q['seen_groups'])}
 p=RUN/'anchors-compact.json';p.write_text(json.dumps(compact,indent=2));add(p,p.name)
 # Preserve every launched command and every terminal supervisor/guard result, including failures.
 states={p.parent.name:json.loads(p.read_text()) for p in (RUN/'supervision').glob('*/state.json')};assert all(v['status'] in ('finished','interrupted') for v in states.values())
 p=RUN/'closed-supervision.json';p.write_text(json.dumps({'supervision':states,'guards':{p.parent.name:json.loads(p.read_text()) for p in RUN.glob('*-guard/result.json')},'launches':{p.name:json.loads(p.read_text()) for p in RUN.glob('*-launch.json')},'guard_receipt_absent_for_interrupted':[p.parent.name for p in RUN.glob('*-guard/child.log') if not (p.parent/'result.json').exists()],'keeper':json.loads((RUN/'keeper/keeper.json').read_text())},indent=2));add(p,p.name)
 for p in sorted((RUN/'analysis-historical02').glob('*-rows.csv')):add(p,p.name)
 # Current full-vocabulary arrays stay on PC; every window/depth scalar is transferred.
 p=RUN/'current-rows.csv';rr=[json.loads(l) for l in (RUN/'analysis-current/rows.jsonl').read_text().splitlines()]
 with p.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
 add(p,p.name)
 for base,prefix in [('current-actor','current'),('historical-actor','historical')]:
  for p in (RUN/base).glob('*/qualification.json'):add(p,prefix+'-'+p.parent.name+'-qualification.json')
  for p in (RUN/base).glob('*/failure.json'):add(p,prefix+'-'+p.parent.name+'-failure.json')
 for p,name in [(RUN/'current-actor/EAL2160/zero-update-export.json','current-EAL-zero-export.json'),(RUN/'current-actor/setup/parent-loader.json','current-parent-restoration.json')]:add(p,name)
 # Compact per-stage failure text is evidence; raw failed logs remain on PC too.
 failures={p.parent.name:p.read_text(errors='replace')[-16000:] for p in RUN.glob('*-guard/child.log') if not (p.parent/'result.json').exists() or json.loads((p.parent/'result.json').read_text()).get('returncode')}
 p=RUN/'failed-receipts.json';p.write_text(json.dumps(failures,indent=2));add(p,p.name)
 p=RUN/'source-staged-join.json';p.write_text(json.dumps(joins,indent=2));shutil.copyfile(p,stage/p.name)
 files={p.name:sha(p) for p in stage.iterdir()};assert len(files)<=64 and all(p.stat().st_size<=8*1024**2 for p in stage.iterdir()) and sum(p.stat().st_size for p in stage.iterdir())<=64*1024**2
 (RUN/'evidence-expected.json').write_text(json.dumps(files,indent=2));print(json.dumps({'status':'EVIDENCE_STAGED','files':len(files),'bytes':sum(p.stat().st_size for p in stage.iterdir()),'expected':files}),flush=True)
if __name__=='__main__':main()
