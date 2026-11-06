"""Freeze historical states and checkpoint-specific consumed-group exposure."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027';OLD=ROOT/'runs/overnight-quality-20261027'
sys.path.insert(0,str(OLD/'source'))
from data import ledger,committed_rows,select_windows,pin

def main():
 import numpy as np
 cap=RUN/'capture05';complete=json.loads((cap/'complete.json').read_text());split=json.loads((RUN/'historical-split.json').read_text());anchors=json.loads((RUN/'historical-anchors.json').read_text());assert len(complete['records'])==24
 decoded,disp,logits=ledger([json.loads(l) for l in (cap/'capture.target_features.jsonl').read_text().splitlines()]);rounds=[json.loads(l) for l in (cap/'rounds.jsonl').read_text().splitlines()];features=np.memmap(cap/'capture.target_features.f32',dtype='<f4',mode='r').reshape(-1,12800)
 out=RUN/'historical-data';out.mkdir(exist_ok=False);windows=[]
 for case,record in zip(split['cases'],complete['records']):
  committed=committed_rows(decoded,disp,record['task_id']);rr=[r for r in rounds if r['task_id']==record['task_id']];enriched=select_windows(rr,committed,logits,record['task_id']);eligible=[c for r in rr for c in select_windows([r],committed,logits,record['task_id'])];assert len(enriched)==4 and len(eligible)>=4
  ordinary=[eligible[k*len(eligible)//4] for k in range(4)]
  for panel,ww in [('error_enriched',enriched),('ordinary',ordinary)]:
   for w in ww:
    i=len(windows);dest=out/f'{i:03d}';dest.mkdir();anchor=w['anchor'];chunks=[];last=None
    for p in range(anchor):
     ordinal=committed[p]['decode_ordinal']
     if ordinal!=last:chunks.append(0);last=ordinal
     chunks[-1]+=1
    np.asarray(features[[committed[p]['feature_row'] for p in range(anchor)]],dtype='<f4').tofile(dest/'features.f32')
    windows.append({**w,'case_index':i,'panel':panel,'role':case['role'],'group_id':case['group_id'],'domain':case['domain'],'chain_id':case['chain_id'],'block_index':-1,'anchor_token':w['prefix_tokens'][-1],'features':pin(dest/'features.f32'),'feature_chunks':chunks,'checkpoint_group_exposure':{name:case['group_id'] in set(q['seen_groups']) for name,q in anchors.items()},'source_task_id':record['task_id']})
 current=json.loads((RUN/'matrix.json').read_text());joins={name:{role:len({w['group_id'] for w in current['windows'] if w['role']==role and w['group_id'] in set(q['seen_groups'])}) for role in ('fit','development')} for name,q in anchors.items()}
 plan={'schema':'fixed_historical_transfer_v1','windows':windows,'selection':split['policy']+'; per group earliest4enriched and four ordinary round-order quartiles, fixed before candidate scores. D5common of original7body; historicalD7 separately.','anchors':pin(RUN/'historical-anchors.json'),'capture':pin(cap/'complete.json'),'not_pristine_heldout':True,'current_matrix_membership_join':joins}
 (RUN/'historical-matrix.json').write_text(json.dumps(plan,indent=2));print(json.dumps({'status':'HISTORICAL_MATRIX_FROZEN','windows':len(windows),'groups':24,'current_membership_join':joins}),flush=True)
if __name__=='__main__':main()
