"""External CPU-only budget around the fixed capacity binding/export commands."""
import os,time,resource,subprocess,json,sys,signal
from pathlib import Path
start=time.monotonic();cpu0=time.process_time();child0=resource.getrusage(resource.RUSAGE_CHILDREN)
HERE=Path(__file__).resolve().parent;cfg=json.loads((HERE/'capacity-operation-config01.json').read_text());D=Path(cfg['receipt_directory']);D.mkdir(exist_ok=False)
resource.setrlimit(resource.RLIMIT_CPU,(360,365));ticks=os.sysconf('SC_CLK_TCK');records=[]
def cpu_used():
 r=resource.getrusage(resource.RUSAGE_CHILDREN)
 return time.process_time()-cpu0+r.ru_utime+r.ru_stime-child0.ru_utime-child0.ru_stime
def run(name,args):
 with (D/(name+'.log')).open('xb') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
  peak=0;failure=None
  try:
   while p.poll() is None:
    live=0
    try:
     st=Path('/proc',str(p.pid),'stat').read_text();fields=st[st.rfind(')')+2:].split();live=(int(fields[11])+int(fields[12]))/ticks
     status=Path('/proc',str(p.pid),'status').read_text();peak=max(peak,max([int(l.split()[1])*1024 for l in status.splitlines() if l.startswith('VmHWM:')]+[0]))
    except FileNotFoundError:pass
    assert time.monotonic()-start<=600 and cpu_used()+live<=360 and peak<=1024**3,'external CPU/wall/RSS cap'
    assert not any(Path(s).exists() for s in cfg['stop_paths']),'Root STOP'
    time.sleep(.2)
  except BaseException as e:
   failure=str(e);p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
  records.append({'name':name,'returncode':p.returncode,'peak_RSS_bytes':peak,'CPU_total':cpu_used(),'whole_total':time.monotonic()-start,'failure':failure})
  assert failure is None and p.returncode==0,records[-1]
try:
 for command in cfg['commands']:run(command['name'],command['argv'])
 result=json.loads(Path(cfg['expected_result']).read_text());assert result['status']=='EXPORTED_NATIVE_UNQUALIFIED'
 assert result['GPU_calls']==result['optimizer_updates']==0 and len(result['projections'])==13
 assert time.monotonic()-start<=600 and cpu_used()<=360
 record={'status':'CPU_EXPORT_COMPLETE_NATIVE_UNQUALIFIED','at':time.time(),'commands':records,'CPU_seconds':cpu_used(),'whole_seconds':time.monotonic()-start,'result_path':cfg['expected_result'],'GPU_calls':0,'optimizer_updates':0}
 (D/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
except BaseException as e:
 (D/'failure.json').write_text(json.dumps({'status':'FAILED_RETAINED','error':str(e),'commands':records,'CPU_seconds':cpu_used(),'whole_seconds':time.monotonic()-start},indent=2));raise
