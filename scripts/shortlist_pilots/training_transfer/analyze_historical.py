"""Raw recomputation, paired checkpoint gains and exposure-specific interactions."""
from pathlib import Path
import json,sys,csv,math,hashlib
import numpy as np
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
sys.path.insert(0,str(RUN/'source'))
from analyze_current02 import raw_score

def cell(rows):
 n=len(rows)
 if not n:return {'windows':0,'groups':0}
 return {'windows':n,'groups':len({r['group_id'] for r in rows}),'positions':5*n,'prefix5':sum(r['greedy']['prefix'] for r in rows),'prefix7':sum(r['greedy_D7']['prefix'] for r in rows),'TF_CE':float(np.mean([x for r in rows for x in r['teacher_forced']['ce']])),'TF_top1':sum(sum(r['teacher_forced']['top1']) for r in rows),'TF_margin':float(np.mean([x for r in rows for x in r['teacher_forced']['margins']])),'prefix_survival':[sum(r['greedy']['prefix']>d for r in rows) for d in range(5)]}
def main():
 out=RUN/'analysis-historical';out.mkdir(exist_ok=False);windows=json.loads((RUN/'historical-matrix.json').read_text())['windows'];allrows={};checks=0;missing=[];disagreements=[]
 for engine,root,arms in [('native',RUN/'historical-native',['initializer','8hour','16hour','parent3504','EAL2160','q4_k_m']),('actor',RUN/'historical-actor',['initializer','8hour','16hour']),('actor',RUN/'historical-current-actor',['parent3504','EAL2160'])]:
  for arm in arms:
   p=root/arm/'result.json'
   if not p.exists():missing.append({'engine':engine,'arm':arm,'path':str(p),'failure':json.loads((root/arm/'failure.json').read_text()) if (root/arm/'failure.json').exists() else None});continue
   rows=json.loads(p.read_text())['rows'];assert len(rows)==len(windows);allrows[engine+'/'+arm]=rows;flat=[]
   for w,r in zip(windows,rows):
    assert r['window_index']==w['case_index']
    for mode in ('teacher_forced','greedy'):
     ref=r['teacher_raw' if mode=='teacher_forced' else 'greedy_raw'] if engine=='native' else r['raw'][mode]
     with Path(ref['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref['sha256']
     actual=raw_score(ref['path'],w['labels']);assert actual['predictions']==r[mode]['predictions'] and actual['prefix']==r[mode]['prefix'];assert np.allclose(actual['ce'],r[mode]['ce'],atol=1e-10,rtol=0) and np.allclose(actual['margins'],r[mode]['margins'],atol=1e-10,rtol=0);checks+=1
     full=np.fromfile(ref['path'],dtype='<f4').reshape(7,151936);pred=full.argmax(-1).tolist();prefix7=next((i for i,(x,y) in enumerate(zip(pred,w['labels'])) if x!=y),7);assert prefix7==r[mode+'_D7']['prefix']
     for d in range(5):flat.append({'window':w['case_index'],'panel':w['panel'],'stratum16':w['role'],'own_seen':w['checkpoint_group_exposure'].get(arm),'group_id':w['group_id'],'domain':w['domain'],'conditioning':mode,'depth':d,'label':w['labels'][d],'prediction':actual['predictions'][d],'ce':actual['ce'][d],'margin':actual['margins'][d],'prefix5':actual['prefix'],'prefix7':prefix7,'round_index':w['round_index'],'anchor':w['anchor']})
   with (out/f'{engine}-{arm}-rows.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 summaries={}
 for key,rr in allrows.items():
  arm=key.split('/')[1];cells={}
  for p in ('error_enriched','ordinary'):
   for role in ('historical_seen','historical_not_seen'):
    for domain in ('all','code','prose','reasoning'):cells[p+'/'+role+'/'+domain]=cell([r for r,w in zip(rr,windows) if w['panel']==p and w['role']==role and (domain=='all' or w['domain']==domain)])
   for exposed in (True,False):cells[p+'/own_seen_'+str(exposed)]=cell([r for r,w in zip(rr,windows) if w['panel']==p and w['checkpoint_group_exposure'].get(arm)==exposed])
  summaries[key]=cells
 paired={}
 for newer,older in [('8hour','initializer'),('16hour','initializer'),('16hour','8hour'),('EAL2160','parent3504')]:
  for panel in ('error_enriched','ordinary'):
   for role in ('historical_seen','historical_not_seen'):
    ids=[i for i,w in enumerate(windows) if w['panel']==panel and w['role']==role];result={}
    for engine in ('native','actor'):
     if engine+'/'+newer not in allrows or engine+'/'+older not in allrows:continue
     a=allrows[engine+'/'+newer];b=allrows[engine+'/'+older];deltas=[a[i]['greedy']['prefix']-b[i]['greedy']['prefix'] for i in ids];groups={g:sum(delta for i,delta in zip(ids,deltas) if windows[i]['group_id']==g) for g in {windows[i]['group_id'] for i in ids}}
     result[engine]={'prefix5_delta':sum(deltas),'TF_CE_delta':float(np.mean([sum(a[i]['teacher_forced']['ce'])-sum(b[i]['teacher_forced']['ce']) for i in ids]))/5,'TF_top1_delta':sum(sum(a[i]['teacher_forced']['top1'])-sum(b[i]['teacher_forced']['top1']) for i in ids),'gaining_groups':sum(x>0 for x in groups.values()),'losing_groups':sum(x<0 for x in groups.values()),'group_deltas':groups}
    if len(result)==2:result['native_minus_actor_gain']=result['native']['prefix5_delta']-result['actor']['prefix5_delta']
    paired[newer+'-'+older+'/'+panel+'/'+role]=result
 interactions={}
 for arm in ('initializer','8hour','16hour','parent3504','EAL2160'):
  if 'actor/'+arm not in allrows:continue
  n=allrows['native/'+arm];a=allrows['actor/'+arm];cells={}
  for p in ('error_enriched','ordinary'):
   for role in ('historical_seen','historical_not_seen'):
    ids=[i for i,w in enumerate(windows) if w['panel']==p and w['role']==role];delta=[n[i]['greedy']['prefix']-a[i]['greedy']['prefix'] for i in ids];cells[p+'/'+role]={'windows':len(ids),'prefix_disagreements':sum(x!=0 for x in delta),'native_minus_actor_prefix':sum(delta),'mean_abs_prefix_error':sum(abs(x) for x in delta)/len(ids),'original_fidelity_thresholds_pass':sum(x!=0 for x in delta)/len(ids)<=5/96 and sum(abs(x) for x in delta)/len(ids)<=.1}
  for w,x,y in zip(windows,n,a):
   for mode in ('teacher_forced','greedy'):
    dif=[d for d,(j,k) in enumerate(zip(x[mode]['predictions'],y[mode]['predictions'])) if j!=k]
    if dif:
     d=dif[0];disagreements.append({'arm':arm,'window':w['case_index'],'panel':w['panel'],'stratum16':w['role'],'domain':w['domain'],'conditioning':mode,'depth':d,'native':x[mode]['predictions'][d],'actor':y[mode]['predictions'][d],'label':w['labels'][d],'native_margin':x[mode]['margins'][d],'actor_margin':y[mode]['margins'][d],'decision_relevant':mode=='teacher_forced' or d<=min(x['greedy']['prefix'],y['greedy']['prefix'])})
  interactions[arm]=cells
 result={'status':'HISTORICAL_ANALYSIS_COMPLETE','raw_arrays_recomputed':checks,'missing':missing,'summaries':summaries,'paired':paired,'actor_native_interactions':interactions,'group_counts_by_checkpoint':{arm:{p:len({w['group_id'] for w in windows if w['checkpoint_group_exposure'].get(arm)==p}) for p in (True,False)} for arm in ('initializer','8hour','16hour')}}
 (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'first-disagreements.json').write_text(json.dumps(disagreements,indent=2));print(json.dumps({'status':result['status'],'raw_arrays_recomputed':checks,'missing':missing,'group_counts':result['group_counts_by_checkpoint'],'paired':paired,'interactions':interactions}),flush=True)
if __name__=='__main__':main()
