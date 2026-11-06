"""CPU-only freeze of error-enriched and ordinary serving states before new scores."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
sys.path[:0]=[str(ROOT/'scripts'),str(RUN/'source')]
from shortlist_pilots.training_transfer.metrics import ordinary_indices
from shortlist_pilots.overnight_quality.data import ledger,committed_rows,select_windows,pin

def main():
 import numpy as np
 old=json.loads((OLD/'data/manifest.json').read_text());capture=OLD/'capture05';complete=json.loads((capture/'complete.json').read_text());split=json.loads((OLD/'split.json').read_text());out=RUN/'data';out.mkdir(exist_ok=False)
 for ref in complete['files'].values():assert pin(ref['path'])['sha256']==ref['sha256']
 decoded,disp,logits=ledger([json.loads(l) for l in (capture/'capture.target_features.jsonl').read_text().splitlines()]);rounds=[json.loads(l) for l in (capture/'rounds.jsonl').read_text().splitlines()]
 features=np.memmap(capture/'capture.target_features.f32',dtype='<f4',mode='r').reshape(-1,12800)
 windows=[{**w,'panel':'error_enriched','source_window_index':w['case_index']} for w in old['windows']]
 occupancy=[]
 for case,record in zip(split['cases'],complete['records']):
  committed=committed_rows(decoded,disp,record['task_id'])
  eligible=[]
  for rr in rounds:
   if rr['task_id']!=record['task_id']:continue
   occupancy.append({'group_id':case['group_id'],'domain':case['domain'],'role':case['role'],'round_index':rr['round_index'],'accepted':rr['accepted_drafts'],'proposed':len(rr['draft_token_ids']),'pos0':rr['pos0']})
   # Single-round consumer validates prefix and completeness without choosing on errors.
   candidates=select_windows([rr],committed,logits,record['task_id'])
   eligible.extend(candidates)
  selected=[eligible[i] for i in ordinary_indices(len(eligible))]
  for w in selected:
   i=len(windows);dest=out/f'window-{i:03d}';dest.mkdir();anchor=w['anchor'];chunks=[];last=None
   for pos in range(anchor):
    ordinal=committed[pos]['decode_ordinal']
    if ordinal!=last:chunks.append(0);last=ordinal
    chunks[-1]+=1
   np.asarray(features[[committed[i]['feature_row'] for i in range(anchor)]],dtype='<f4').tofile(dest/'features.f32')
   windows.append({**w,'case_index':i,'panel':'ordinary','feature_chunks':chunks,'group_id':case['group_id'],'domain':case['domain'],'role':case['role'],'chain_id':case['chain_id'],'block_index':-1,'anchor_token':w['prefix_tokens'][-1],'features':pin(dest/'features.f32'),'source_case_index':case['index'],'source_task_id':record['task_id'],'source_split':'TRAIN'})
 assert len(windows)==288
 for i,w in enumerate(windows):w['case_index']=i
 plan={'schema':'fixed_training_transfer_v1','selection':'All144 retained enriched windows + four ordinary eligible rounds/group at floor(k*N/4), k=0..3. No candidate scores. Ordinary excludes terminal/incomplete7-label rows; occupancy ledger includes all rounds.','windows':windows,'source':pin(OLD/'data/manifest.json'),'capture':pin(capture/'complete.json'),'protocol':{'target':'05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6','target_kv':'F16','target_mask':3,'depth':5,'body':7,'markov':'full','seed':42},'historical_membership':'pending checkpoint-specific consumed-ledger join; current fit/development never means historical train/test','not_pristine_holdout':True}
 (RUN/'matrix.json').write_text(json.dumps(plan,indent=2));(RUN/'occupancy.json').write_text(json.dumps(occupancy,indent=2));print(json.dumps({'status':'MATRIX_FROZEN','windows':288,'groups':36,'panels':['error_enriched','ordinary'],'matrix':pin(RUN/'matrix.json')}),flush=True)
if __name__=='__main__':main()
