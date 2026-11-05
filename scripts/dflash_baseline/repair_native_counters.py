"""Recover delayed log sampling from final immutable per-request native logs.

Never reruns inference or changes raw measurements. All cumulative log boundaries
must align one-for-one to requests, response counters and prefix integer counts.
"""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.dflash_baseline.precision_benchmark import pin,save
from scripts.dflash_baseline.native_counters import reconcile

def recover(panel,expected_cells):
 panel=Path(panel)
 if (panel/'result.json').exists():raise FileExistsError('completed panel already exists')
 original=pin(panel/'progress.json');rows=json.load(open(panel/'progress.json'));groups={};corrections=[];evidence=[]
 for row in rows:groups.setdefault((row['rep'],row['arm'],row['instrumented']),[]).append(row)
 freeze=json.load(open(panel/'freeze.json'));prompts=json.load(open(panel/'prompts.json'));expected_requests=len(prompts)+freeze['warmup_policy']['per_server_cell']
 if len(groups)!=expected_cells or len(rows)!=expected_cells*expected_requests or any(len(items)!=expected_requests for items in groups.values()):raise ValueError('all measured cells and requests must finish before recovery')
 for (rep,arm,traced),items in groups.items():
  cell=panel/(f'{rep:02d}-{arm}'+('-traced' if traced else ''))
  if not json.load(open(cell/'release.json'))['reaped']:raise ValueError('server still owned')
  if arm=='target':
   if any(x['native'] for x in items):raise ValueError('unexpected target speculative counters')
   continue
  corrections.extend(reconcile(cell,items))
  evidence.append(dict(log=pin(cell/'server.log'),requests=len(items),sequential_cumulative_boundaries=True,all_response_counters_match=True))
 pairs={}
 for row in rows:pairs.setdefault((row['rep'],row['arm'],row['prompt']),{})[row['instrumented']]=row
 if not all(len(x)==2 and all(x[False]['native'].get(k)==x[True]['native'].get(k) for k in ('accepted','proposed','rounds','prefix_survival_counts')) for x in pairs.values()):raise ValueError('actual paired native counters differ after final log recovery')
 if not all(x['correct'] for x in rows):raise ValueError('incorrect native output')
 if any(x['instrumented'] and 'host_spans_ms' not in x for x in rows):raise ValueError('traced cell processing incomplete')
 receipt=dict(status='FINAL_NATIVE_LOG_COUNTERS_RECOVERED',recovered_at_utc=datetime.now(timezone.utc).isoformat(),original_progress=original,expected_cells=expected_cells,rows=len(rows),corrections=corrections,evidence=evidence,method='Final reaped-server logs contain one sequential cumulative boundary and one per-request prefix line per HTTP response. Validate every recovered delta against independent response accepted/proposed counters. No inference/timing/raw request or response changed.',source=pin(__file__),parser=pin(ROOT/'scripts/dflash_baseline/native_counters.py'))
 save(panel/'counter-recovery.json',receipt)
 save(panel/'result.json',dict(status='COMPLETE',rows=rows,paired_trace_counters_equal=True,counter_recovery=pin(panel/'counter-recovery.json')))
 print('Recovered native counter attribution:',len(corrections),'rows;',len(rows),'responses unchanged',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('panel',type=Path);ap.add_argument('--expected-cells',type=int,required=True);a=ap.parse_args();recover(a.panel,a.expected_cells)
