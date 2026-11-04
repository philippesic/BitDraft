"""Paired prompt/domain/repetition uncertainty for complete draft host cost per round."""
from pathlib import Path
import argparse,sys
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require

def analyze(root):
 receipt=read(root/'result.json');require(receipt['status']=='PAIRED_NATIVE_COST_AND_OVERHEAD_PASS','profile not qualified')
 rows=[];sources=[]
 for f in sorted((root/'traced').glob('rep-*/cost-trace.summary.json')):
  sources.append(pin(f));rows += [r for r in read(f)['rows'] if not r['warmup'] and r['arm'] in ('paid5760','dflash_Q4KM')]
 ids=sorted({r['id'] for r in rows});rng=np.random.default_rng(611043);weights=np.zeros((20000,len(ids)),dtype=np.int64)
 for domain in ('code','reasoning','prose'):
  ix=[i for i,k in enumerate(ids) if domain in k];require(ix,'missing domain');weights[:,ix]=rng.multinomial(len(ix),[1/len(ix)]*len(ix),size=len(weights))
 reps=rng.multinomial(5,[.2]*5,size=len(weights));out={}
 for component,names in [('narrow_generate',('draft_generate',)),('complete_draft',('draft_generate','draft_catchup','draft_accept'))]:
  sampled={};point={}
  for arm in ('paid5760','dflash_Q4KM'):
   times=np.full((len(ids),5),np.nan);rounds=times.copy()
   for r in rows:
    if r['arm']!=arm:continue
    i,j=ids.index(r['id']),r['repetition'];require(np.isnan(times[i,j]),'duplicate profile row');times[i,j]=sum(r['root_ms']['generation'].get(n,0) for n in names);rounds[i,j]=r['verification_rounds']
   require(np.isfinite(times).all() and (rounds>0).all(),'incomplete profile')
   sampled[arm]=np.einsum('bi,ij,bj->b',weights,times,reps)/np.einsum('bi,ij,bj->b',weights,rounds,reps);point[arm]=float(times.sum()/rounds.sum())
  ratios=sampled['paid5760']/sampled['dflash_Q4KM'];out[component]=dict(ms_per_round=point,current_cost_percent=float((point['paid5760']/point['dflash_Q4KM']-1)*100),bootstrap95_percent=((np.quantile(ratios,[.025,.975])-1)*100).tolist())
 return dict(status='PAIRED_DRAFT_HOST_COST_CONTRAST',components=out,source_panel=pin(root/'result.json'),summaries=sources,scope='Same consumed DEV12 and five repetitions; paired prompt-within-domain × repetition bootstrap,20000 draws. Different serving histories/round counts are preserved. Descriptive host-span costs, not isolated kernel latency or causal attribution.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();v=analyze(a.panel);atomic(a.out,v);print(v['components'])
