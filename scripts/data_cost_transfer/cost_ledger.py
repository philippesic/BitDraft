"""Account paid updates and nonoverlapping supervised job wall, including failures."""
from pathlib import Path
import argparse,json,sys
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require

def ledger(root,artifacts):
 runners=[];intervals=[]
 for d in sorted(root.iterdir()):
  f=d/'state.json'
  if not f.is_file():continue
  s=read(f)
  if 'jobs' not in s or 'charged_seconds' not in s:continue
  require(s['status'] in ('COMPLETE','FAILED'),'runner still active: '+str(f))
  jobs=[]
  for ident,j in s['jobs'].items():
   require(j['status'] not in ('RUNNING','PENDING'),'unfinished job')
   if 'started_unix' in j and 'ended_unix' in j:intervals.append((j['started_unix'],j['ended_unix'],str(f),ident))
   jobs.append(dict(id=ident,status=j['status'],seconds=j.get('charged_seconds',0),group_released=j.get('group_released')))
  runners.append(dict(state=pin(f),status=s['status'],charged_seconds=s['charged_seconds'],jobs=jobs))
 ordered=sorted(intervals)
 require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'overlapping job charges need explicit reconciliation')
 updates=[]
 for name in ('old-qualification48','broader-qualification48','broader-split24','broader-resumed48','old-final1920','broader-final1920'):
  directory=artifacts/name;rows=[json.loads(x) for x in (directory/'updates.jsonl').read_text().splitlines()];result=read(directory/'result.json')
  require(rows and len({x['cursor'] for x in rows})==len(rows),'duplicate/missing update entries')
  updates.append(dict(name=name,updates=len(rows),first_cursor=rows[0]['cursor'],last_cursor=rows[-1]['cursor'],update_seconds=sum(x['update_seconds'] for x in rows),result_wall_seconds=result['wall_seconds'],log=pin(directory/'updates.jsonl'),result=pin(directory/'result.json')))
 require(sum(x['updates'] for x in updates)==3888,'physical update count changed')
 return dict(status='CAMPAIGN_COSTS_ACCOUNTED',runners=runners,supervised_job_wall_seconds=sum(x['charged_seconds'] for x in runners),training=updates,physical_updates=3888,unique_comparison_updates=3840,update_wall_seconds=sum(x['update_seconds'] for x in updates),scope='Supervised job walls include failures, preparation, serving and cleanup; update wall is nested within them and includes materialization. Neither is pure GPU time. Unsupervised source work, builds, CPU admission/analysis and gaps are not included; do not add nested clocks.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();v=ledger(a.campaign,a.artifacts);atomic(a.out,v);print({k:v[k] for k in ('physical_updates','unique_comparison_updates','update_wall_seconds','supervised_job_wall_seconds')})
