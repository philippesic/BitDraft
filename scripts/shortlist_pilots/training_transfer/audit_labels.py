"""Audit fixed ordinary/enriched labels against actual captured target logits."""
from pathlib import Path
import json,numpy as np
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
def main():
 results={}
 for name,matrix,cap in [('current',RUN/'matrix.json',OLD/'capture05'),('historical',RUN/'historical-matrix.json',RUN/'capture05')]:
  if not matrix.exists():continue
  windows=json.loads(matrix.read_text())['windows'];led=[json.loads(l) for l in (cap/'capture.target_features.jsonl').read_text().splitlines()];logits={r['feature_row']:r for r in led if r['event']=='quality_logits'};raw=np.memmap(cap/'capture.quality_logits.f32',dtype='<f4',mode='r');wrong=[]
  for w in windows:
   for depth,key in enumerate(w['teacher_feature_rows']):
    r=logits[key];row=raw[r['byte_offset']//4:r['byte_offset']//4+r['vocab_size']];assert np.isfinite(row).all()
    if int(row.argmax())!=w['labels'][depth]:wrong.append({'window':w['case_index'],'depth':depth,'actual_label':w['labels'][depth],'raw_argmax':int(row.argmax())})
  results[name]={'windows':len(windows),'positions':7*len(windows),'argmax_label_disagreements':wrong};assert not wrong,(name,wrong)
 (RUN/'label-audit.json').write_text(json.dumps({'status':'ACTUAL_CAPTURED_TARGET_LABEL_AUDIT_PASS','matrices':results},indent=2));print(json.dumps(results))
if __name__=='__main__':main()
