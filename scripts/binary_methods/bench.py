from pathlib import Path
import torch,numpy as np,ctypes,json,time
R=Path('/home/philip/binary-eagle-decoding/runs/binary-methods-20261104-01');torch.set_num_threads(2)
fn=ctypes.CDLL(str(R/'kernels.so')).method_launch;fn.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]*7;fn.restype=ctypes.c_int
modes={0:'A8_tensor',1:'A8_dp4a',2:'A8_planes1',3:'A8_planes7',4:'A8_lut4',10:'A1_xor',11:'A1_bmma',12:'A1_tensor',5:'A8_sharedlut',13:'A1_sharedlut'}
rng=np.random.default_rng(61104);rows=[];checks=[]
for m,k in [(18,256),(9728,2560),(2560,9728)]:
 for n in [1,7]:
  w=rng.integers(0,2**32,(m,k//32),dtype=np.uint32);a=rng.uniform(.001,.03,m).astype('f4');x=rng.normal(size=(n,k)).astype('f4')
  W=torch.from_numpy(w).cuda();A=torch.from_numpy(a).cuda();X=torch.from_numpy(x).cuda()
  q=torch.empty((n,k),device='cuda',dtype=torch.int8);p=torch.empty((n*k+m*(k//32),),device='cuda',dtype=torch.int32);b=torch.empty(n,device='cuda');s=torch.empty((n,k),device='cuda',dtype=torch.int32);lut=torch.empty((n,k*4),device='cuda',dtype=torch.int16);y=torch.empty((n,m),device='cuda')
  def call(mode,phase=0):
   err=fn(mode,phase,X.data_ptr(),W.data_ptr(),A.data_ptr(),m,k,n,q.data_ptr(),p.data_ptr(),b.data_ptr(),s.data_ptr(),lut.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream);assert err==0,err
  idx=np.arange(m) if m<20 else np.unique(np.r_[0,m//2,m-1,rng.choice(m,13,replace=False)]);signs=(((w[idx,:,None]>>np.arange(32,dtype='u4'))&1).reshape(len(idx),k).astype('i4')*2-1)
  for mode,name in modes.items():
   if mode in (0,12) and n!=7:continue
   call(mode);torch.cuda.synchronize()
   if mode>=10:codes=np.where(x>=0,1,-1).astype('i4');sc=np.abs(x).astype('f8').mean(1).astype('f4')
   else:mx=np.abs(x).max(1);sc=(mx/np.float32(127)).astype('f4');codes=np.rint(x*(np.float32(127)/mx)[:,None]).astype('i4')
   integer=codes@signs.T;ref=(integer.astype('f4')*a[idx])*sc[:,None];actual=y.cpu().numpy()[:,idx]
   assert np.array_equal(actual,ref),(name,m,k,n,float(np.max(np.abs(actual-ref))))
   checks.append(dict(mode=name,m=m,k=k,n=n,rows=len(idx),scaled_exact=True))
   if m<20:continue
   timing={}
   for phase,label in [(0,'full'),(1,'prep'),(2,'dot')]:
    call(mode);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     for _ in range(32):call(mode,phase)
    values=[]
    for rep in range(3):
     g.replay();torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();g.replay();end.record();end.synchronize();values.append(start.elapsed_time(end)*1000/32)
    timing[label]=values
   rows.append(dict(mode=name,m=m,k=k,n=n,timing_us=timing));print(name,m,k,n,{key:round(float(np.median(v)),2) for key,v in timing.items()},flush=True)
  (R/'kernels.json').write_text(json.dumps(dict(rows=rows,checks=checks,scope='resident actual FFN shapes; three graph repetitions of32; FP32 scaling exact on sampled rows; no occupancy counters'),indent=2))
