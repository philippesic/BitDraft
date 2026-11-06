from pathlib import Path
import json,hashlib,subprocess,collections,shutil
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/5080-broader-training-20261031';W=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/5080-broader-training-20261031');A=P/'runs/w1a8-algorithms-20261020'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
W.mkdir(exist_ok=False)
inputs=A/'loop2-answer-broad-data01/inputs.json';d=json.loads(inputs.read_text());old=json.loads((P/'runs/overnight-quality-20261027/split.json').read_text());excluded={c['group_id'] for c in old['cases']};groups={}
for c in d['fit']['answer']:groups.setdefault(c['group_id'],c)
cases=[];dropped=[]
for domain in ('code','prose','reasoning'):
 eligible=sorted([c for c in groups.values() if c['domain']==domain and c['group_id'] not in excluded],key=lambda c:hashlib.sha256(('broader20261031:'+c['group_id']).encode()).hexdigest());selected=[]
 for c in eligible:
  rec=json.loads(Path(c['native_receipt']['path']).read_text());assert sha(c['native_receipt']['path'])==c['native_receipt']['sha256']
  if rec['prompt_length']>384:dropped.append({'group_id':c['group_id'],'reason':'predeclared prompt cap384'});continue
  j=len(selected);cases.append({'index':len(cases),'role':'fit' if j<80 else 'development','group_id':c['group_id'],'domain':domain,'chain_id':c['chain_id'],'ancestry':c['native_receipt'],'prompt_tokens':rec['tokens'][:rec['prompt_length']],'prompt_length':rec['prompt_length'],'source_split':'TRAIN'});selected.append(c)
  if len(selected)==96:break
 assert len(selected)==96,(domain,len(selected))
(W/'split.json').write_text(json.dumps({'schema':'overnight_serving_groups_v1','source':pin(inputs),'policy':'salted sha group order; exclude prior36 serving groups; first80fit next16dev per domain; prompt<=384; all inherited TRAIN, no final ancestry claim','cases':cases,'request_panels_used':False,'excluded_prior_groups':sorted(excluded),'drops':dropped},indent=2))
# Only the existing capture executable and its unchanged libraries are reused.
ready=P/'runs/overnight-quality-20261027/native02/ready.json';(W/'native02').symlink_to(ready.parent,target_is_directory=True)
print(json.dumps({'status':'CPU_PREPARED','groups':len(cases),'fit':240,'dev':48,'output':str(W),'disk':{str(q):shutil.disk_usage(q).free for q in [P,Path('/mnt/c')]}}),flush=True)
