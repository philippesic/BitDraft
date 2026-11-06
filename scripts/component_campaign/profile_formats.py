"""Actual exercised CUDA format dispatch; instrumented timings are not throughput claims."""
import argparse,csv,json,subprocess,sys
from .common import *
from .measure import completed

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arms',nargs='+');a=ap.parse_args();registry=completed();out=R/'format-profiles';out.mkdir(exist_ok=True)
 models=dict(target=None,q4=str(Q4),**{n:registry[n]['model'] for n in a.arms});cache={n:registry[n]['spec'].get('kv',['f16','f16']) for n in a.arms}
 save(out/'arms.json',models);save(out/'cache.json',cache)
 native=out/'native'
 if not (native/'result.json').exists():subprocess.run([sys.executable,'-m','scripts.component_campaign.native','--arms',str(out/'arms.json'),'--draft-cache-map',str(out/'cache.json'),'--runtime',str(R/'runtime04'),'--out',str(native),'--profile','--cohort','capture3','--limit','1','--output-cap','64','--reps','1'],cwd=ROOT,check=True)
 records={}
 for cell in sorted(native.glob('00-*')):
  report=cell/'profile.nsys-rep'
  if not report.exists():continue
  dest=cell/'kernel-summary.csv'
  if not dest.exists():
   with dest.open('w') as stream:subprocess.run(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str(report)],stdout=stream,stderr=subprocess.PIPE,text=True,check=True)
  text=dest.read_text();lines=text.splitlines();header=next((i for i,l in enumerate(lines) if 'Name' in l and 'Instances' in l),None)
  if header is None:raise ValueError('Nsight kernel CSV header missing')
  rows=[r for r in csv.DictReader(lines[header:]) if r.get('Name') and r.get('Instances','').replace(',','').isdigit()]
  if not rows:raise ValueError('Nsight kernel CSV has no actual exercised instances')
  records[cell.name]=dict(report=pin(report),kernel_csv=pin(dest),kernels=rows,command=pin(cell/'command.json'),model=registry.get(cell.name[3:],{}))
  # Nsight's exported SQLite duplicates the retained trace. Keep the paid raw
  # .nsys-rep and parsed CSV; retire only this derived, fully re-creatable file.
  sql=report.with_suffix('.sqlite')
  if sql.exists():
   if sql.is_symlink() or sql.parent!=cell:raise ValueError('unexpected profile SQLite ownership')
   save(cell/'sqlite-retirement.json',dict(retired=pin(sql),recoverable_from=pin(report),verified_kernel_csv=pin(dest),reason='Derived NEW campaign database; nsys stats re-exports it from retained trace; process reaped.'))
   sql.unlink()
 save(out/'dispatch.json',dict(status='ACTUAL_NATIVE_CUDA_FORMAT_PROFILE_PASS',records=records,source=pin(__file__),scope='Instrumented first consumed-DEV request,64output cap. Actual kernel names/instances plus exact model/cache command and immutable CUDA binary/source pins. Nsight timing is diagnostic; clean GPU replay and uninstrumented request panels provide performance.'))
 print('FORMAT_PROFILES_COMPLETE',flush=True)

if __name__=='__main__':main()
