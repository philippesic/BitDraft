"""Positive seven-predecessor and known-wrong-predecessor native head controls."""
from pathlib import Path
import json,subprocess,os,numpy as np
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def main():
 out=RUN/'head-controls';out.mkdir(exist_ok=False);helper=json.loads((RUN/'native/ready.json').read_text())['binary']['path'];window=json.loads((RUN/'matrix.json').read_text())['windows'][0];parent=json.loads((RUN/'current-native/parent3504/result.json').read_text());original=parent['rows'][0];labels=[r['argmax'] for r in original['native']['rows']];cases=[]
 for wrong in (False,True):
  ys=list(labels)
  if wrong:ys[0]=(ys[0]+1)%151936
  cases.append({**window,'case_index':len(cases),'labels':ys})
 plan=out/'plan.json';plan.write_text(json.dumps({'schema':'group_transfer12_v1','cases':cases}));dest=out/'native'
 with (out/'native.log').open('wb') as f:subprocess.run([helper,parent['model']['path'],str(plan),str(dest),'on'],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 greedy=[np.fromfile(dest/f'case-{i}.f32',dtype='<f4').reshape(7,151936) for i in range(2)];tf=[np.fromfile(dest/f'teacher-{i}.f32',dtype='<f4').reshape(7,151936) for i in range(2)]
 assert np.array_equal(greedy[0],greedy[1]),'teacher labels altered deployed greedy path'
 assert np.array_equal(greedy[0].argmax(-1),tf[0].argmax(-1)),'all7 same predecessor decisions must match'
 assert np.array_equal(tf[0][0],tf[1][0]) and np.max(np.abs(tf[0][1]-tf[1][1]))>0,'known-wrong predecessor must alter only conditioned later rows'
 result={'status':'NATIVE_TEACHER_HEAD_CONTROLS_PASS','same7predecessor_all_predictions_match':True,'same7predecessor_max_abs_logit_difference':float(np.abs(greedy[0]-tf[0]).max()),'known_wrong_predecessor_rejected':True,'wrong_predecessor_row1_logit_change':float(np.abs(tf[0][1]-tf[1][1]).max()),'labels_do_not_change_greedy_path':True,'control_labels_are_artificial_no_quality_score':True};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
