"""Link actual native/replay/allocator evidence without pooling different cohorts."""
import argparse,json,statistics
import numpy as np
from .common import *
from .metrics import summarize,paired_ratio,allocations

def native_panel(directory):
 data=json.load(open(directory/'result.json'));result=summarize(directory/'result.json')
 for name,entry in result.items():
  cells=sorted({directory/(f"{x['rep']:02d}-{name}"+('-traced' if x['instrumented'] else '')) for x in data['rows'] if x['arm']==name})
  measurements=[]
  for cell in cells:
   memory=json.load(open(cell/'memory.json'))
   measurements.append(dict(cell=str(cell),allocation=allocations(cell),sampled_device_peak_bytes=max([x['MiB']*2**20 for x in memory],default=None)))
  entry['allocations']=measurements
  entry['sampled_device_peak_bytes']=max([x['sampled_device_peak_bytes'] for x in measurements if x['sampled_device_peak_bytes'] is not None],default=None)
  if not entry['instrumented']:
   entry['vs_q4']=paired_ratio(directory/'result.json',name,'q4')
   if 'H0' in result:entry['vs_H0']=paired_ratio(directory/'result.json',name,'H0')
  # Repeated greedy outcomes are correlated, and are reported as actual totals.
  rows=[x for x in data['rows'] if x['arm']==name and not x.get('warmup')]
  entry['prefix_counts']=[sum(x['native'].get('prefix_survival_counts',[0,0])[i] for x in rows) for i in (0,1)]
 return dict(source=pin(directory/'result.json'),freeze=pin(directory/'freeze.json'),cohort=json.load(open(directory/'freeze.json'))['cohort'],arms=result)

def replay_panel(directory):
 data=json.load(open(directory/'result.json'))
 if data.get('audit'):raise ValueError('instrumented audit is not clean latency')
 result={}
 for arm in dict.fromkeys(x['arm'] for x in data['records']):
  records=[x for x in data['records'] if x['arm']==arm]
  rows=[x for rec in records for x in rec['result']['rows'] if not x['warmup']]
  # Each domain/context has the same weight; no timing rep acts as a new training seed.
  result[arm]=dict(clean_gpu_forward_ms=float(np.mean([x['graph_event_ms'] for x in rows])),fixed_replay_forward_host_ms=float(np.mean([x['forward_host_ms'] for x in rows])),fixed_replay_catchup_host_ms=float(np.mean([x['catchup_host_ms'] for x in rows])),cells=len(records),forwards=len(rows),source_models={x['model']['sha256'] for x in records})
  result[arm]['source_models']=sorted(result[arm]['source_models'])
 return dict(source=pin(directory/'result.json'),scope=data['scope'],arms=result)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--native',nargs='*',type=Path,default=[]);ap.add_argument('--replay',nargs='*',type=Path,default=[]);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 save(a.out,dict(native={p.name:native_panel(p) for p in a.native},replay={p.name:replay_panel(p) for p in a.replay},source=pin(__file__),limitations='Panels remain separate by cohort/context/trace. Timing prompt/process resampling cannot estimate training-seed uncertainty. GPU sampled peak includes desktop/target; persistent model and cache buffers separately reported.'))
 print('FORMAL_SUMMARY',a.out,flush=True)

if __name__=='__main__':main()
