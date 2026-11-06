"""Independent raw-byte recomputation and paired current-checkpoint interactions."""
from pathlib import Path
import json,math,hashlib
import numpy as np
ROOT=Path('/home/philip/binary-eagle-decoding');RUN=ROOT/'runs/training-transfer-diagnosis-20261027'
def raw_score(path,labels):
 x=np.fromfile(path,dtype='<f4').reshape(7,151936)[:5].astype('f8');y=np.asarray(labels[:5]);p=np.argmax(x,axis=1);correct=p==y;first=next((i for i,v in enumerate(correct) if not v),5);ce=[];margin=[]
 for v,t in zip(x,y):
  m=float(v.max());ce.append(m+math.log(np.exp(v-m).sum(dtype=np.float64))-v[t]);wrong=np.concatenate((v[:t],v[t+1:]));margin.append(float(v[t]-wrong.max()))
 return {'prefix':first,'top1':correct.tolist(),'predictions':p.tolist(),'ce':ce,'margins':margin,'first_error':first if first<5 else None}
def main():
 data=json.loads((RUN/'matrix.json').read_text());windows=data['windows'];allrows={};checks=0;disagreements=[];flat=[]
 for engine,arms in [('native',['parent3504','EAL2160','q4_k_m']),('actor',['parent3504','EAL2160'])]:
  for arm in arms:
   records=json.loads((RUN/('current-'+engine)/arm/'result.json').read_text())['rows'];key=engine+'/'+arm;allrows[key]=[]
   for w,r in zip(windows,records):
    assert w['case_index']==r['window_index'];rec={k:r[k] for k in ('window_index','panel','role','domain','group_id')}
    for mode in ('teacher_forced','greedy'):
     ref=(r['teacher_raw'] if mode=='teacher_forced' else r['greedy_raw']) if engine=='native' else r['raw'][mode]
     with Path(ref['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==ref['sha256']
     actual=raw_score(ref['path'],w['labels']);assert actual['predictions']==r[mode]['predictions'] and actual['prefix']==r[mode]['prefix'];assert np.allclose(actual['ce'],r[mode]['ce'],rtol=0,atol=1e-10) and np.allclose(actual['margins'],r[mode]['margins'],rtol=0,atol=1e-10);checks+=1;rec[mode]=actual
     for depth in range(5):flat.append({**{k:rec[k] for k in ('window_index','panel','role','domain','group_id')},'engine':engine,'checkpoint':arm,'conditioning':mode,'depth':depth,'label':w['labels'][depth],'prediction':actual['predictions'][depth],'ce':actual['ce'][depth],'top1':actual['top1'][depth],'margin':actual['margins'][depth],'correct_prefix':actual['prefix'],'round_index':w['round_index'],'anchor':w['anchor']})
    allrows[key].append(rec)
 summaries={}
 for key,rows in allrows.items():
  cells={}
  for panel in ('error_enriched','ordinary'):
   for role in ('fit','development'):
    for domain in ('all','code','prose','reasoning'):
     rr=[r for r in rows if r['panel']==panel and r['role']==role and (domain=='all' or r['domain']==domain)];n=len(rr)
     cells['/'.join((panel,role,domain))]={'windows':n,'positions':5*n,'groups':len({r['group_id'] for r in rr}),'prefix5':sum(r['greedy']['prefix'] for r in rr),'TF_CE':float(np.mean([v for r in rr for v in r['teacher_forced']['ce']])),'TF_top1':sum(sum(r['teacher_forced']['top1']) for r in rr),'TF_margin_mean':float(np.mean([v for r in rr for v in r['teacher_forced']['margins']])),'prefix_survival':[sum(r['greedy']['prefix']>d for r in rr) for d in range(5)]}
  summaries[key]=cells
 paired={}
 for panel in ('error_enriched','ordinary'):
  for role in ('fit','development'):
   indices=[i for i,w in enumerate(windows) if w['panel']==panel and w['role']==role];cell={}
   for arm in ('parent3504','EAL2160'):
    n=[allrows['native/'+arm][i] for i in indices];a=[allrows['actor/'+arm][i] for i in indices]
    dd=[x['greedy']['prefix']-y['greedy']['prefix'] for x,y in zip(n,a)]
    cell[arm]={'actor_native_prefix_disagreements':sum(x!=0 for x in dd),'native_minus_actor_prefix':sum(dd),'mean_absolute_prefix_error':sum(abs(x) for x in dd)/len(dd),'native_minus_actor_TF_CE':float(np.mean([sum(x['teacher_forced']['ce'])-sum(y['teacher_forced']['ce']) for x,y in zip(n,a)])/5)}
    for i,x,y in zip(indices,n,a):
     if x['greedy']['predictions']!=y['greedy']['predictions']:
      first=next(j for j,(p,q) in enumerate(zip(x['greedy']['predictions'],y['greedy']['predictions'])) if p!=q);disagreements.append({'window_index':i,'checkpoint':arm,'panel':panel,'role':role,'domain':windows[i]['domain'],'first_disagreement':first,'native_prediction':x['greedy']['predictions'][first],'actor_prediction':y['greedy']['predictions'][first],'label':windows[i]['labels'][first],'native_margin':x['greedy']['margins'][first],'actor_margin':y['greedy']['margins'][first]})
   for engine in ('native','actor'):
    b=[allrows[engine+'/parent3504'][i] for i in indices];e=[allrows[engine+'/EAL2160'][i] for i in indices];delta=[x['greedy']['prefix']-y['greedy']['prefix'] for x,y in zip(e,b)];bygroup={g:sum(d for i,d in zip(indices,delta) if windows[i]['group_id']==g) for g in sorted({windows[i]['group_id'] for i in indices})}
    cell[engine+'_gain']={'prefix5_delta':sum(delta),'TF_CE_delta':float(np.mean([sum(x['teacher_forced']['ce'])-sum(y['teacher_forced']['ce']) for x,y in zip(e,b)])/5),'TF_top1_delta':sum(sum(x['teacher_forced']['top1'])-sum(y['teacher_forced']['top1']) for x,y in zip(e,b)),'gaining_windows':sum(d>0 for d in delta),'losing_windows':sum(d<0 for d in delta),'group_deltas':bygroup}
   cell['interaction_native_minus_actor_gain']=cell['native_gain']['prefix5_delta']-cell['actor_gain']['prefix5_delta'];paired[panel+'/'+role]=cell
 # Same state may occur in both separately declared panels; report it explicitly.
 enriched={(w['group_id'],w['anchor']) for w in windows if w['panel']=='error_enriched'};overlap=sum((w['group_id'],w['anchor']) in enriched for w in windows if w['panel']=='ordinary')
 occupancy=json.loads((RUN/'occupancy.json').read_text());occ={}
 for role in ('fit','development'):
  for d in ('all','code','prose','reasoning'):
   rr=[r for r in occupancy if r['role']==role and (d=='all' or r['domain']==d)];occ[role+'/'+d]={'rounds':len(rr),'accepted':sum(r['accepted'] for r in rr),'proposed':sum(r['proposed'] for r in rr),'depth_rounds':[sum(r['accepted']>=j for r in rr) for j in range(1,6)]}
 out=RUN/'analysis-current';out.mkdir(exist_ok=False);(out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in flat));(out/'disagreements.json').write_text(json.dumps(disagreements,indent=2));result={'status':'CURRENT_ANALYSIS_COMPLETE','raw_arrays_independently_recomputed':checks,'summaries':summaries,'paired':paired,'ordinary_enriched_overlap_states':overlap,'all_round_occupancy':occ,'native_same_predecessor_head_max_abs':max(r['same_predecessor_head_max_abs'] for arm in ('parent3504','EAL2160','q4_k_m') for r in json.loads((RUN/'current-native'/arm/'result.json').read_text())['rows'])};(out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'checks':checks,'paired':paired,'overlap':overlap}),flush=True)
if __name__=='__main__':main()
