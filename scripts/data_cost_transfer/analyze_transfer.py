"""Paired, domain-stratified native data-transfer contrasts; no early-pruning rule."""
from pathlib import Path
import argparse,json,sys
import numpy as np
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import read,pin,atomic,require

def contrast(root,old,new,label):
 proof=read(root/'audit.json') if (root/'audit.json').exists() else read(root.parent/'audit.json')
 require(proof['status']=='RAW_REQUEST_MODEL_RUNTIME_METRICS_AND_RELEASE_AUDIT_PASS','panel not audited')
 value=read(root/'result.json');inputs=read(root/'inputs.json');models=read(root/'models.json');ids=[r['id'] for r in inputs['rows']];groups={r['id']:r.get('domain') or next(d for d in ('code','reasoning','prose') if d in r['id']) for r in inputs['rows']}
 tables={};summaries={};tokens={}
 for arm in (old,new):
  rows=[r for r in value['records'] if r['arm']==arm and not r['warmup']];require(len(rows)==len(ids)*5,'incomplete panel')
  wall=np.full((len(ids),5),np.nan);tok=np.zeros_like(wall)
  for r in rows:
   i,j=ids.index(r['id']),r['repetition'];require(np.isnan(wall[i,j]),'duplicate request');wall[i,j]=r['full_http_wall_seconds'];tok[i,j]=r['completion_tokens']
  require(np.isfinite(wall).all() and (wall>0).all(),'invalid times');tables[arm]=wall;tokens[arm]=tok
  proposed=sum(r['draft_proposed'] for r in rows);accepted=sum(r['draft_accepted'] for r in rows);rounds=sum(r['verification_rounds'] for r in rows)
  summaries[arm]=dict(tps=float(tok.sum()/wall.sum()),tokens=int(tok.sum()),wall_seconds=float(wall.sum()),draft_proposed=proposed,draft_accepted=accepted,verification_rounds=rounds,acceptance=accepted/proposed,tokens_per_round=float(tok.sum()/rounds))
 require(np.array_equal(tokens[old],tokens[new]),'output counts differ')
 rng=np.random.default_rng(611041);pw=np.zeros((20000,len(ids)),dtype=np.int64)
 for domain in ('code','reasoning','prose'):
  ix=[i for i,k in enumerate(ids) if groups[k]==domain];require(ix,'domain missing');pw[:,ix]=rng.multinomial(len(ix),[1/len(ix)]*len(ix),size=len(pw))
 rw=rng.multinomial(5,[.2]*5,size=len(pw));ratio=np.einsum('bi,ij,bj->b',pw,tables[old],rw)/np.einsum('bi,ij,bj->b',pw,tables[new],rw);ci=(np.quantile(ratio,[.025,.975])-1)*100
 domains={}
 for domain in ('code','reasoning','prose'):
  ix=[i for i,k in enumerate(ids) if groups[k]==domain];domains[domain]=float((tables[old][ix].sum()/tables[new][ix].sum()-1)*100)
 return dict(label=label,old_arm=old,broader_arm=new,summary=summaries,broader_percent=float((tables[old].sum()/tables[new].sum()-1)*100),bootstrap95_percent=ci.tolist(),ranking='broader faster' if ci[0]>0 else 'broader slower' if ci[1]<0 else 'unresolved',domain_percent=domains,independent_prompts=len(ids),repetitions=5,panel=pin(root/'result.json'),audit=pin(root/'audit.json') if (root/'audit.json').exists() else pin(root.parent/'audit.json'),method='20000 paired prompt-within-domain × repetition bootstrap draws; unadjusted intervals, consumed DEV')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.campaign
 values=[contrast(r/'early-checkpoint-panel/serving','candidate_0','candidate_1','480 additional updates / DEV12'),contrast(r/'early-checkpoint-panel/serving','candidate_2','candidate_3','960 additional updates / DEV12'),contrast(r/'final-checkpoint-panel/serving','candidate_0','candidate_1','1920 additional updates / DEV12'),contrast(r/'natural-final-panel','candidate_0','candidate_1','1920 additional updates / consumed natural24')]
 atomic(a.out,dict(status='BOUNDED_DATA_TRANSFER_NATIVE_COMPARISON_AUDITED',comparisons=values,scope='Fixed row-W1A8 actor,1old+1new CE vs2old CE, unchanged replay. Equal optimizer updates, separately accounted seconds. Previously consumed evaluation inputs; no fresh confirmation, architecture ranking, universal predictor, or permanent pruning claim.'))
 print(json.dumps([dict(label=v['label'],change=v['broader_percent'],ci=v['bootstrap95_percent'],ranking=v['ranking']) for v in values]))
