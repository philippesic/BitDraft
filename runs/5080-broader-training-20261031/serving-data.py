"""Join actual serving ledgers. Reject wrong-prefix, rejected or cross-task rows."""
from pathlib import Path
import json,hashlib

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def ledger(rows):
 decoded={};dispositions={};logits={}
 for row in rows:
  event=row['event'];key=row['feature_row']
  dest={'decoded_row':decoded,'disposition':dispositions,'quality_logits':logits}.get(event)
  if dest is None or key in dest:raise ValueError('unknown or duplicate ledger row')
  dest[key]=row
 if set(decoded)!=set(dispositions) or not set(logits)<=set(decoded):raise ValueError('incomplete disposition ledger')
 return decoded,dispositions,logits

def committed_rows(decoded,dispositions,task_id):
 result={}
 for key,row in decoded.items():
  disp=dispositions[key]
  if row['task_id']!=disp['task_id']:raise ValueError('task disposition mismatch')
  if row['task_id']!=task_id or not disp['retained_input']:continue
  pos=row['position']
  if pos in result:raise ValueError('duplicate committed position')
  result[pos]=row
 if sorted(result)!=list(range(len(result))):raise ValueError('committed position gap')
 for pos,row in result.items():
  if row['prefix_token_ids']!=[result[i]['token_id'] for i in range(pos+1)]:raise ValueError('committed prefix mismatch')
 return result

def select_windows(rounds,committed,logits,task_id):
 candidates=[]
 for round in rounds:
  if round['task_id']!=task_id or len(round['draft_token_ids'])!=5:continue
  prefix=round['prefix_token_ids']+[round['seed_token_id']];anchor=len(prefix)-1
  # Native pos0 is independently checked by the actual fixture consumer.
  if prefix!=[committed[i]['token_id'] for i in range(anchor+1)]:raise ValueError('round prefix does not match committed history')
  if anchor+7>=len(committed):continue
  labels=[committed[i]['token_id'] for i in range(anchor+1,anchor+8)]
  teacher=[committed[i]['feature_row'] for i in range(anchor,anchor+7)]
  if not all(i in logits for i in teacher):continue
  accepted=round['accepted_drafts']
  if round['verifier_token_ids']!=labels[:accepted+1]:raise ValueError('actual verifier labels differ from committed continuation')
  if round['draft_token_ids'][:accepted]!=labels[:accepted]:raise ValueError('accepted proposal mismatch')
  if accepted<5 and round['draft_token_ids'][accepted]==labels[accepted]:raise ValueError('first rejection mismatch')
  candidates.append({'anchor':anchor,'prefix_tokens':prefix,'labels':labels,'accepted_drafts':accepted,'proposed':round['draft_token_ids'],'round_index':round['round_index'],'teacher_feature_rows':teacher,'native_pos0':round['pos0']})
 # Fixed policy: earliest four unsaturated starts, then earliest saturated fillers.
 errors=[c for c in candidates if c['accepted_drafts']<5];sat=[c for c in candidates if c['accepted_drafts']==5]
 return (errors+sat)[:4]

def build(run):
 import numpy as np
 run=Path(run);out=run/'data';out.mkdir(exist_ok=False);capture=run/'capture05';complete=json.loads((capture/'complete.json').read_text());split=json.loads((run/'split.json').read_text());assert complete['status']=='SERVING_CAPTURE_COMPLETE'
 for ref in complete['files'].values():assert sha(ref['path'])==ref['sha256']
 rows=[json.loads(l) for l in (capture/'capture.target_features.jsonl').read_text().splitlines()];decoded,disp,logits=ledger(rows)
 features=np.memmap(capture/'capture.target_features.f32',dtype='<f4',mode='r').reshape(-1,12800);raw=np.memmap(capture/'capture.quality_logits.f32',dtype='<f4',mode='r')
 assert len(features)==len(decoded)
 rounds=[json.loads(l) for l in (capture/'rounds.jsonl').read_text().splitlines()];windows=[]
 for case,record in zip(split['cases'],complete['records']):
  assert case['index']==record['case_index'];committed=committed_rows(decoded,disp,record['task_id']);selected=select_windows(rounds,committed,logits,record['task_id'])
  for window in selected:
   n=len(windows);dest=out/f'window-{n:03d}';dest.mkdir();anchor=window['anchor'];chunks=[];last_ordinal=None
   for pos in range(anchor):
    ordinal=committed[pos]['decode_ordinal']
    if ordinal!=last_ordinal:chunks.append(0);last_ordinal=ordinal
    chunks[-1]+=1
   indices=[committed[i]['feature_row'] for i in range(anchor)];x=np.asarray(features[indices],dtype='<f4');assert np.isfinite(x).all();x.tofile(dest/'features.f32')
   ys=[]
   for i,k in enumerate(window['teacher_feature_rows']):
    row=logits[k];assert row['vocab_size']==151936 and row['byte_offset']%4==0
    y=np.array(raw[row['byte_offset']//4:row['byte_offset']//4+151936]);assert np.isfinite(y).all() and int(y.argmax())==window['labels'][i],('target raw argmax/actual verifier disagreement',n,i)
    ys.append(y)
   np.stack(ys).tofile(dest/'teacher.f32')
   windows.append({**window,'case_index':n,'feature_chunks':chunks,'group_id':case['group_id'],'domain':case['domain'],'role':case['role'],'chain_id':case['chain_id'],'block_index':-1,'anchor_token':window['prefix_tokens'][-1],'features':pin(dest/'features.f32'),'teacher':pin(dest/'teacher.f32'),'source_case_index':case['index'],'source_task_id':record['task_id'],'source_split':'TRAIN'})
 coverage={}
 for role in ('fit','development'):
  selected=[w for w in windows if w['role']==role];coverage[role]={'windows':len(selected),'groups':len({w['group_id'] for w in selected}),'unsaturated':sum(w['accepted_drafts']<5 for w in selected),'by_domain':{d:sum(w['domain']==d for w in selected) for d in ('code','prose','reasoning')}}
 assert not {w['group_id'] for w in windows if w['role']=='fit'}&{w['group_id'] for w in windows if w['role']=='development'}
 value={'schema':'overnight_actual_serving_windows_v1','split':pin(run/'split.json'),'capture':pin(capture/'complete.json'),'windows':windows,'coverage':coverage,'reused_TRAIN':True,'request_panels_used':False}
 with (out/'manifest.json').open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({'status':'SERVING_DATA_JOINED','coverage':coverage,'manifest':pin(out/'manifest.json')}),flush=True)
 return value
if __name__=='__main__':build('/home/philip/binary-eagle-decoding/runs/overnight-quality-20261027')


def training_pairs(windows):
 groups={}
 for i,w in enumerate(windows):groups.setdefault(w['group_id'],[]).append(i)
 rows=list(groups.values())
 if len(rows)!=24 or any(len(r)!=4 for r in rows):raise ValueError('exact24groups/fourwindows required')
 pairs=[]
 for g in range(0,24,2):
  for j in range(4):pairs.append([rows[g][j],rows[g+1][j]])
 if sorted(i for pair in pairs for i in pair)!=list(range(96)):raise ValueError('exposure coverage')
 return pairs
