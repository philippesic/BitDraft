"""One launch per stage and precision for hardware-counter diagnosis."""
import ctypes, torch
from pathlib import Path
import json
assert not json.loads((Path.home()/'.config/binary-eagle-decoding/gpu-control.json').read_text())['rtx5080']['pause_requested']
torch.set_num_threads(2)
lib=ctypes.CDLL('/home/philip/binary-eagle-decoding/runs/binary-screen-20261104-01/kernels-v2.so')
f=lib.screen_launch
f.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_void_p]
f.restype=ctypes.c_int
for m,k in [(9728,2560),(2560,9728)]:
 n=7
 X=torch.randn(n,k,device='cuda');W=torch.randint(0,2**31-1,(m,k//32),device='cuda',dtype=torch.int32);A=torch.ones(m,device='cuda')
 C=torch.empty(n,k,device='cuda',dtype=torch.int8);P=torch.empty(n,8,k//32,device='cuda',dtype=torch.int32)
 B=torch.empty(n,device='cuda');Q=torch.empty(n,device='cuda',dtype=torch.int32);Y=torch.empty(n,m,device='cuda');I=torch.empty(n,m,device='cuda',dtype=torch.int32)
 for mode in [0,1,8]:
  assert f(mode,0,X.data_ptr(),W.data_ptr(),A.data_ptr(),m,k,n,C.data_ptr(),P.data_ptr(),B.data_ptr(),Q.data_ptr(),Y.data_ptr(),I.data_ptr(),0,torch.cuda.current_stream().cuda_stream)==0
  torch.cuda.synchronize()
