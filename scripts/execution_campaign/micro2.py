"""Actual N3/1 CUDA-event prep+dot timing, independent integer/affine references."""
from pathlib import Path
import ctypes,json,os,sys,subprocess
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.execution_campaign.build import P,F,B,pin
R=P/'runs/execution-campaign-20261105-01';r=R/'micro2';r.mkdir(exist_ok=True);S=P/'runs/w1a8-modern-packed-runtime-20261103-01/source'
lib=r/'kernels.so'
if not lib.exists():
 manifest=json.load(open(F/'build.json'));cmd=next(c.copy() for c in manifest['commands'] if '-c' in c and str(F/'w1a1.cu.o') in c)
 for flag in ['-MD','-MT','-MF']:
  if flag in cmd:
   i=cmd.index(flag);del cmd[i:i+(1 if flag=='-MD' else 2)]
 cmd[cmd.index('-c')+1]=str(ROOT/'scripts/execution_campaign/micro2.cu');cmd.remove('-c');cmd[cmd.index('-o')+1]=str(lib);cmd+=['-DEXEC_IMPL="'+str(R/'runtime-both32/w1a1.cu')+'"','-I'+str(R/'runtime-both32'),'-shared','-L'+str(F/'bin'),'-lggml-cuda','-lggml-base','-L/usr/local/cuda/lib64','-lcublas','-Wl,-rpath,'+str(F/'bin')]
 with (r/'build.log').open('w') as log:subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
 (r/'build.json').write_text(json.dumps(dict(command=cmd,source=pin(ROOT/'scripts/execution_campaign/micro2.cu'),library=pin(lib)),indent=2))
fn=ctypes.CDLL(str(lib)).exec_launch;fn.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*5+[ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_void_p]*8;fn.restype=ctypes.c_int
rng=np.random.default_rng(20261105);torch.set_num_threads(2);rows=[];checks=[]
for m,k in [(9216,2560),(2560,9216)]:
 for n in [1,3]:
  w=rng.integers(0,2**32,(m,k//32),dtype='u4');x=rng.normal(size=(n,k)).astype('f4');g=rng.uniform(.25,2,k).astype('f4');a=rng.uniform(.001,.04,m).astype('f4');mid=rng.uniform(-.01,.01,m).astype('f4')
  W=torch.from_numpy(w).cuda();X=torch.from_numpy(x).cuda();G=torch.from_numpy(g).cuda();A=torch.from_numpy(a).cuda();M=torch.from_numpy(mid).cuda()
  q=torch.empty((n,k),dtype=torch.int8,device='cuda');b=torch.empty(n,device='cuda');s=torch.empty_like(b);y=torch.empty((n,m),device='cuda');ex=torch.empty((m,k),dtype=torch.int8,device='cuda');hw=torch.empty((m,k),dtype=torch.float16,device='cuda');hq=torch.empty((n,k),dtype=torch.float16,device='cuda')
  bits=4 if k==2560 else 8;qm=2**(bits-1)-1;clip=np.float32(1.005);beta=np.float32(clip/np.float32(qm));codes=np.rint(np.clip(np.float32(x*g)*np.float32(qm/clip),-qm,qm)).astype('i4');qsum=codes.sum(1).astype('f4')*beta
  idx=np.unique(np.r_[0,m-1,m//2,rng.choice(m,21,replace=False)]);sign=(((w[idx,:,None]>>np.arange(32,dtype='u4'))&1).reshape(len(idx),k).astype('i4')*2-1);integer=codes@sign.T;ref=np.float32(np.float32(integer.astype('f4')*a[idx])*beta)+np.float32(mid[idx]*qsum[:,None])
  def call(prep,dot,phase=0):
   err=fn(prep,dot,phase,X.data_ptr(),G.data_ptr(),W.data_ptr(),A.data_ptr(),M.data_ptr(),m,k,n,bits,2,float(clip),q.data_ptr(),b.data_ptr(),s.data_ptr(),y.data_ptr(),ex.data_ptr(),hw.data_ptr(),hq.data_ptr(),torch.cuda.current_stream().cuda_stream);assert err==0,err
  # The final void* count includes stream: q,b,s,y,expanded,hw,hq,stream =8.
  call(0,0);call(0,0,3);torch.cuda.synchronize()
  for prep,dot,label in [(0,0,'original'),(1,0,'bound256'),(2,0,'bound32'),(3,0,'bound128'),(1,1,'cross'),(1,2,'signed_mma'),(1,3,'resident_int8_mma'),(1,4,'resident_fp16_codes'),(1,5,'converted_fp16_codes'),(1,6,'planes_N1'),(1,7,'planes_N3'),(1,8,'lut4_N3')]:
   call(prep,dot);torch.cuda.synchronize();observed=y.cpu().numpy()[:,idx]
   assert np.array_equal(observed,ref),(label,m,k,n,float(np.max(np.abs(observed-ref))))
   assert np.array_equal(q.cpu().numpy(),codes.astype('i1')) and np.array_equal(s.cpu().numpy(),qsum),label
   checks.append(dict(label=label,m=m,k=k,n=n,sampled_rows=len(idx),codes_sums_affine_exact=True))
   timing={}
   for phase,stage in [(0,'pack_dot'),(1,'prep'),(2,'dot')]:
    for _ in range(2):call(prep,dot,phase)
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(32):call(prep,dot,phase)
    values=[]
    for rep in range(7):
     graph.replay();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();graph.replay();end.record();end.synchronize();values.append(start.elapsed_time(end)*1000/32)
    timing[stage]=values
   rows.append(dict(label=label,m=m,k=k,n=n,bits=bits,timing_us=timing,resident_sign_bytes=m*k if dot==3 else m*k*2 if dot==4 else m*k//8));print(label,m,k,n,{s:round(float(np.median(v)),3) for s,v in timing.items()},flush=True)
   (r/'result.json').write_text(json.dumps(dict(rows=rows,checks=checks,scope='projection; resident operands; 2warmup/7event reps of32 graph calls; no mutations/cache; random actual FFN geometries; FP16 includes per-call activation conversion, excludes weight initialization'),indent=2)+'\n')
