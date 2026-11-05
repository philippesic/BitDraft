from pathlib import Path
import torch,ctypes
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/execution-campaign-20261105-01';fn=ctypes.CDLL(str(R/'micro2/kernels.so')).exec_launch;fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5+[ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_void_p]*8;fn.restype=ctypes.c_int
for m,k in [(9216,2560),(2560,9216)]:
 n=3;x=torch.randn((n,k),device='cuda');g=torch.ones(k,device='cuda');w=torch.randint(0,2**31-1,(m,k//32),device='cuda',dtype=torch.int32);a=torch.ones(m,device='cuda');mid=torch.ones_like(a)*.01;q=torch.empty((n,k),device='cuda',dtype=torch.int8);b=torch.empty(n,device='cuda');s=torch.empty_like(b);y=torch.empty((n,m),device='cuda');ex=torch.empty((m,k),device='cuda',dtype=torch.int8);hw=torch.empty((m,k),device='cuda',dtype=torch.float16);hq=torch.empty((n,k),device='cuda',dtype=torch.float16)
 def call(phase):
  rc=fn(1,4,phase,x.data_ptr(),g.data_ptr(),w.data_ptr(),a.data_ptr(),mid.data_ptr(),m,k,n,4 if k==2560 else 8,2,1.005,q.data_ptr(),b.data_ptr(),s.data_ptr(),y.data_ptr(),ex.data_ptr(),hw.data_ptr(),hq.data_ptr(),torch.cuda.current_stream().cuda_stream);assert rc==0
 call(3);call(0)
 for _ in range(5):call(0)
 torch.cuda.synchronize()
