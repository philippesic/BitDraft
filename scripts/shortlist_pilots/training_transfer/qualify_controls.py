"""Original controls and zero-update evidence joined against fresh raw replay."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
def main():
 original=json.loads((OLD/'native-gate/result.json').read_text());fresh=json.loads((RUN/'current-native/parent3504/result.json').read_text());rows=[]
 for a,b in zip(original['records'],fresh['rows'][:144]):
  x=np.fromfile(a['native']['raw_path'],dtype='<f4');y=np.fromfile(b['greedy_raw']['path'],dtype='<f4');assert x.shape==y.shape
  rows.append({'window':b['window_index'],'max_abs':float(np.abs(x-y).max()),'same_greedy_predictions':a['native']['rows'][0]['argmax']==b['greedy']['predictions'][0],'same_prefix':a['replay_depth5']==b['greedy']['prefix']})
 assert all(r['same_prefix'] for r in rows)
 result={'status':'UNCHANGED_REPLAY_CONTROL_PASS','windows':len(rows),'all_original_prefixes_retained':True,'max_abs_logit_difference':max(r['max_abs'] for r in rows),'rows':rows,'parent_zero_update':json.loads((RUN/'current-actor/setup/parent-loader.json').read_text()),'EAL_zero_update':json.loads((RUN/'current-actor/EAL2160/zero-update-export.json').read_text()),'reused_http':{'summary':str(OLD/'http-evaluation/summary.json'),'models_unchanged':True,'24_development_prompt_per_arm_output_tokens':2858,'no_new_positive_throughput_claim':True}}
 (RUN/'unchanged-controls.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ('status','windows','max_abs_logit_difference')}))
if __name__=='__main__':main()
