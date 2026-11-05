"""Zero-training actual-shape kernel screen; no model-serving speed inference."""
from pathlib import Path
import argparse,ctypes,json,sys,time
import numpy as np
import torch
P=Path('/home/philip/binary-eagle-decoding');sys.path.insert(0,str(P/'scripts/contender_pipeline'))
from runner import pin,read,atomic,require
MODES={0:'W1A8_DP4A',1:'W1A1_XOR_POPC',4:'W1A4_bitserial',8:'W1A8_bitserial'}
def main(a):
 require(not a.out.exists(),'unique output');a.out.mkdir();require(torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 5080','wrong GPU');torch.set_num_threads(2)
 lib=ctypes.CDLL(str(a.library));fn=lib.screen_launch;fn.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]*6+[ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
 rng=np.random.default_rng(61104);records=[];checks=[];start=time.monotonic()
 shapes=[(7,64)]+[(m,k) for m,k in [(9728,2560),(2560,9728)]]
 for m,k in shapes:
  for n in ([5] if m==7 else [1,7]):
   weights=rng.integers(0,2**32,size=(m,k//32),dtype=np.uint32);alpha=rng.uniform(.001,.03,size=m).astype('f4');x=rng.normal(size=(n,k)).astype('f4')
   if m==7:x[0]=0;x[1]=np.tile(np.array([-1,1],dtype='f4'),k//2);x[2]*=np.float32(1e-39);x[3]*=np.float32(1e20)
   W=torch.from_numpy(weights).cuda();A=torch.from_numpy(alpha).cuda();X=torch.from_numpy(x).cuda();codes=torch.empty((n,k),dtype=torch.int8,device='cuda');planes=torch.empty((n,8,k//32),dtype=torch.int32,device='cuda');beta=torch.empty(n,device='cuda');qs=torch.empty(n,dtype=torch.int32,device='cuda');Y=torch.empty((n,m),device='cuda');I=torch.empty((n,m),dtype=torch.int32,device='cuda');stream=torch.cuda.current_stream().cuda_stream
   def call(mode,phase=0,check=0):
    error=fn(mode,phase,X.data_ptr(),W.data_ptr(),A.data_ptr(),m,k,n,codes.data_ptr(),planes.data_ptr(),beta.data_ptr(),qs.data_ptr(),Y.data_ptr(),I.data_ptr(),check,torch.cuda.current_stream().cuda_stream);require(error==0,'CUDA launch error '+str(error))
   selected=np.arange(m) if m==7 else np.unique(np.r_[0,m//2,m-1,rng.choice(m,5,replace=False)]);bits=((weights[selected,:,None]>>np.arange(32,dtype=np.uint32))&1).reshape(len(selected),k).astype('i4');signs=bits*2-1
   for mode in MODES:
    call(mode,check=1);torch.cuda.synchronize();b=1 if mode==1 else 4 if mode==4 else 8
    if b==1:q=np.where(x>=0,1,-1).astype('i4');scale=np.abs(x).astype('f8').mean(axis=1).astype('f4')
    else:
     mx=np.abs(x).max(axis=1);scale=(mx/np.float32((1<<(b-1))-1)).astype('f4')
     with np.errstate(over='ignore',invalid='ignore',divide='ignore'):inv=np.float32((1<<(b-1))-1)/mx;normal=x*inv[:,None]
     for i in range(n):
      if mx[i]==0:normal[i]=0
      elif not np.isfinite(inv[i]):normal[i]=(x[i].astype('f8')/float(mx[i])*((1<<(b-1))-1)).astype('f4')
     q=np.rint(normal).clip(-((1<<(b-1))-1),(1<<(b-1))-1).astype('i4')
    expected=q@signs.T;actual=I.cpu().numpy()[:,selected];require(np.array_equal(expected,actual),'integer mismatch '+str((mode,m,k,n)))
    require(np.array_equal(scale,beta.cpu().numpy()),'quantizer scale mismatch')
    if mode in (4,8):require(np.array_equal(q.sum(axis=1),qs.cpu().numpy()),'quantizer sum mismatch')
    if mode==0:actual_codes=codes.cpu().numpy().astype('i4')
    else:
     words=planes.cpu().numpy().reshape(-1)[:n*b*(k//32)].reshape(n,b,k//32).astype('u4');expanded=((words[:,:,:,None]>>np.arange(32,dtype='u4'))&1).reshape(n,b,k).astype('i4')
     if b==1:actual_codes=2*expanded[:,0]-1
     else:actual_codes=sum(expanded[:,j]*(1<<j) for j in range(b-1))-expanded[:,b-1]*(1<<(b-1))
    require(np.array_equal(q,actual_codes),'packed activation codes mismatch')
    target=(expected.astype('f4')*alpha[selected][None,:])*scale[:,None];got=Y.cpu().numpy()[:,selected];require(np.array_equal(target,got),'scaled output mismatch')
    checks.append(dict(mode=MODES[mode],m=m,k=k,n=n,checked_rows=len(selected),integer_and_scaled_exact=True))
   if m==7:continue
   graphs={}
   # Each graph contains32 repeated operations, amortizing Python/host launch.
   # Buffers and weights are resident; these are optimistic device costs.
   for mode in MODES:
    for phase in [0,1,2]:
     call(mode);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):
      for _ in range(32):call(mode,phase)
     graphs[mode,phase]=g
   timings={key:[] for key in graphs}
   for rep in range(3):
    order=list(graphs)
    if rep%2:order.reverse()
    for key in order:
     mode,phase=key;call(mode);graphs[key].replay();torch.cuda.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();graphs[key].replay();end.record();end.synchronize();timings[key].append(begin.elapsed_time(end)*1000/32)
   for mode in MODES:records.append(dict(mode=MODES[mode],m=m,k=k,n=n,pack_dot_us=timings[mode,0],pack_us=timings[mode,1],dot_us=timings[mode,2],median_pack_dot_us=float(np.median(timings[mode,0]))))
   atomic(a.out/'progress.json',dict(completed_shapes=len(records)//4,last=[m,k,n],wall_seconds=time.monotonic()-start));print(json.dumps(dict(shape=[m,k,n],us={MODES[b]:round(float(np.median(timings[b,0])),3) for b in MODES})),flush=True)
   del graphs;torch.cuda.empty_cache()
 atomic(a.out/'result.json',dict(status='BINARY_KERNEL_SCREEN_INTEGER_AND_SCALE_PASS',training_updates=0,source=pin(__file__),library=pin(a.library),hardware=torch.cuda.get_device_name(0),records=records,checks=checks,wall_seconds=time.monotonic()-start,scope='Standalone prototypes, warm resident random weights/inputs at historical DSpark binary FFN shapes. Three alternating graph timing repetitions,32 operations per graph. Includes quantization/packing; no native graph/export/acceptance/end-to-end qualification. Correctness checks all7rows of edge fixture and8 selected rows of each actual matrix shape.'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
