"""Actual exercised CUDA format dispatch; instrumented timings are not throughput claims."""
import argparse,csv,json,subprocess,sys
from .common import *
from .measure import completed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');a=ap.parse_args();registry=completed();out=R/'format-profiles';out.mkdir(exist_ok=True)
 models=dict(q4=str(Q4),B0=str(B/'model-final.gguf'),**{n:registry[n]['model'] for n in a.arms});cache={n:registry[n]['spec'].get('kv',['f16','f16']) for n in a.arms}
 save(out/'arms.json',models);save(out/'cache.json',cache)
 native=out/'draft-only-replay'
 if not (native/'result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.replay',str(out/'arms.json'),'--cache-map',str(out/'cache.json'),'--runtime',str(R/'runtime04'),'--out',str(native),'--profile','--domains','code','--reps','1'],cwd=ROOT,check=True)
 records={}
 for report in sorted(native.glob('00-*-code.nsys-rep')):
  cell=report.parent;arm=report.stem.removeprefix('00-').removesuffix('-code');dest=report.with_suffix('.kernels.csv')
  if not dest.exists():
   with dest.open('w') as stream:subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(report)],stdout=stream,stderr=subprocess.PIPE,text=True,check=True)
  text=dest.read_text();lines=text.splitlines();header=next((i for i,l in enumerate(lines) if 'Name' in l and 'Instances' in l),None)
  if header is None:raise ValueError('Nsight kernel CSV header missing')
  rows=[r for r in csv.DictReader(lines[header:]) if r.get('Name') and r.get('Instances','').replace(',','').isdigit()]
  if not rows:raise ValueError('Nsight kernel CSV has no actual exercised instances')
  records[arm]=dict(report=pin(report),kernel_csv=pin(dest),kernels=rows,command=pin(report.with_suffix('.command.json')),model=registry.get(arm,{}))
  # Nsight's exported SQLite duplicates the retained trace. Keep the paid raw
  # .nsys-rep and parsed CSV; retire only this derived, fully re-creatable file.
  sql=report.with_suffix('.sqlite')
  if sql.exists():
   if sql.is_symlink() or sql.parent!=cell:raise ValueError('unexpected profile SQLite ownership')
   save(report.with_suffix('.sqlite-retirement.json'),dict(retired=pin(sql),recoverable_from=pin(report),verified_kernel_csv=pin(dest),reason='Derived NEW campaign database; nsys stats re-exports it from retained trace; process reaped.'))
   sql.unlink()
 if set(records)!=set(models):raise ValueError('Missing actual draft-only format profile')
 save(out/'dispatch.json',dict(status='ACTUAL_NATIVE_CUDA_FORMAT_PROFILE_PASS',records=records,source=pin(__file__),replay=pin(native/'result.json'),scope='Actual native DFlash-only captured-DEV replay: target initialized for immutable sharing, never decoded. Two same-process warmup cycles and3changing contexts. Kernels therefore belong to draft/cache work, not mixed target decoding. Exact model/cache command and CUDA/source pins retained. Nsight timing is diagnostic; separate clean replay/request panels provide performance.'))
 print('FORMAT_PROFILES_COMPLETE',flush=True)

if __name__=='__main__':main()
