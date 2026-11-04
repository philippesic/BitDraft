"""Exact reference equivalence and new-bank split-resume verification."""
from pathlib import Path
import argparse,sys
import torch
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,atomic,require

def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return type(a) is type(b) and a==b

def load(path):
 r=read(path)
 for key in ('checkpoint','model'):require(pin(r[key]['path'])==r[key],'result artifact changed')
 return r,torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--continuous',type=Path,required=True);p.add_argument('--resumed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 reports=[]
 for left,right,label,contract in [(a.old,a.reference,'old_reference_equivalence',False),(a.continuous,a.resumed,'broader_split_resume',True)]:
  lr,ls=load(left);rr,rs=load(right);fields=['cursor','scales','optimizer','rng','cuda_rng','python_rng','local_exposure']+(['contract'] if contract else [])
  for key in fields:require(equal(ls[key],rs[key]),label+' differs in '+key)
  require(lr['model']['sha256']==rr['model']['sha256'],label+' export differs')
  reports.append(dict(comparison=label,left=pin(left),right=pin(right),fields=fields,exact_state=True,exact_export=True))
 atomic(a.out,dict(status='REFERENCE_AND_BROADER_RESUME_EXACT',comparisons=reports,source=pin(__file__)))
 print('REFERENCE_AND_BROADER_RESUME_EXACT',flush=True)
