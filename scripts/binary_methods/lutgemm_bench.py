from pathlib import Path
import torch,numpy as np,ctypes,json
R=Path('/home/philip/binary-eagle-decoding/runs/binary-methods-20261104-01');torch.set_num_threads(2);rng=np.random.default_rng(61104)
f=ctypes.CDLL(str(R/'lutgemm.so')).lut_launch;f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*5+[ctypes.c_void_p];f.restype=ctypes.c_int;rows=[]
for m,k in [(9728,2560),(2560,9728)]:
 for n in [1,7]:
  w=rng.integers(0,2**32,(m,k//32),dtype='u4');alpha=rng.uniform(.001,.03,m).astype('f2');x=rng.normal(size=(n,k)).astype('f2');W=torch.from_numpy(w.T.copy()).cuda();A=torch.from_numpy(alpha).cuda();X=torch.from_numpy(x).cuda();Y=torch.empty((n,m),device='cuda',dtype=torch.float16)
  idx=np.unique(np.r_[0,m//2,m-1,rng.choice(m,13,replace=False)]);signs=(((w[idx,:,None]>>np.arange(32,dtype='u4'))&1).reshape(len(idx),k).astype('f4')*2-1);ref=(x.astype('f4')@signs.T)*alpha[idx].astype('f4')
  for kt,mt in [(64,2048),(128,2048),(256,2048),(64,512),(128,512),(256,512)]:
   def call():assert f(W.data_ptr(),A.data_ptr(),X.data_ptr(),Y.data_ptr(),m,k,n,kt,mt,torch.cuda.current_stream().cuda_stream)==0
   call();torch.cuda.synchronize();got=Y.cpu().float().numpy()[:,idx];err=float(np.sqrt(np.mean((got-ref)**2)/np.mean(ref**2)));assert np.isfinite(got).all() and err<.05,(m,k,n,kt,mt,err)
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
    for _ in range(32):call()
   times=[]
   for rep in range(3):
    g.replay();torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();g.replay();b.record();b.synchronize();times.append(a.elapsed_time(b)*1000/32)
   row=dict(m=m,k=k,n=n,k_tile=kt,m_tile=mt,us=times,relative_rms_error=err);rows.append(row);print(row,flush=True)
(R/'lutgemm.json').write_text(json.dumps(dict(rows=rows,scope='Official FP16 kernel,1 binary basis, FP16 input/scales/output, shared LUT construction and output clear included; weight transpose once outside timing; NO A8/A1 quantization; native integration not claimed;16 sampled rows checked'),indent=2))
