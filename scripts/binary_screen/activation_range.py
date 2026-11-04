"""Descriptive range/zero-code analysis of the already captured TRAIN activations."""
from pathlib import Path
import torch,numpy as np,sys
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,atomic
R=P/'runs/binary-screen-20261104-01';torch.set_num_threads(2);values=torch.load(R/'quality-screen/activations.pt',map_location='cpu',weights_only=True);rows={}
for name,t in values.items():
 x=t.numpy();mx=np.max(np.abs(x),axis=1,keepdims=True);rms=np.sqrt(np.mean(x.astype('f8')**2,axis=1));assert np.all(mx>0) and np.isfinite(x).all();out={}
 for b in [4,8]:
  qmax=np.float32((1<<(b-1))-1);scale=mx/qmax;q=np.rint(x*(qmax/mx)).clip(-qmax,qmax);deq=q*scale;out[str(b)]=dict(zero_fraction=float((q==0).mean()),input_nmse=float(np.sum((deq.astype('f8')-x)**2)/np.sum(x.astype('f8')**2)))
 rows[name]=dict(rows=len(x),median_absmax_over_rms=float(np.median(mx[:,0]/rms)),precision=out)
atomic(R/'activation-range.json',dict(source=pin(__file__),capture=pin(R/'quality-screen/activations.pt'),rows=rows,scope='Descriptive current-actor TRAIN input range with fixed per-token absmax reciprocal/RNE quantization. Not a clipping/rotation experiment or proof of an optimal quantizer.'))
