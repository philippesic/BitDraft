"""Align complete final server logs to HTTP requests; no inference or timing edits."""
import json,re
from pathlib import Path
STAT=re.compile(r'#calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s*(\d+), #acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+).*?dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms')
def delta(now,previous):return {k:v-(previous or {}).get(k,0) for k,v in now.items()}
def reconcile(cell,items):
 cell=Path(cell);corrections=[]
 text=(cell/'server.log').read_text(errors='strict');matches=list(STAT.finditer(text));prefix=re.findall(r'acc per pos = \(([^)]+)\)',text)
 if len(matches)!=len(items) or len(prefix)!=len(items):raise ValueError('ambiguous per-request final log '+str(cell))
 previous=None
 for index,(row,match,rates) in enumerate(zip(items,matches,prefix),1):
  values=[int(match[i]) for i in range(1,8)]+[float(match[i]) for i in range(8,11)]
  now=dict(zip(('begin_calls','rounds','accept_calls','drafts','accepted_rounds','proposed','accepted','begin_ms','draft_ms','accept_ms'),values))
  if now['begin_calls']!=index:raise ValueError('nonsequential native request boundary')
  metrics=delta(now,previous);previous=now
  response=json.loads((cell/(row['prompt']+'-response.json')).read_text())
  if metrics['accepted']!=response['timings']['draft_n_accepted'] or metrics['proposed']!=response['timings']['draft_n']:raise ValueError('final log/response counter disagreement')
  if metrics['rounds']<=0 or metrics['rounds']>256:raise ValueError('round count outside exact prefix rounding bound')
  counts=[round(float(v)*metrics['rounds']) for v in rates.split(',')]
  if sum(counts)!=metrics['accepted'] or not all(0<=x<=metrics['rounds'] for x in counts) or not all(a>=b for a,b in zip(counts,counts[1:])):raise ValueError('prefix integer reconstruction invalid')
  metrics['prefix_survival_counts']=counts;metrics['prefix_count_method']='per-request rounded rate * exact rounds;rounding error<0.5 at<=256 rounds'
  if row['native']!=metrics:corrections.append(dict(rep=row['rep'],arm=row['arm'],instrumented=row['instrumented'],prompt=row['prompt'],old=row['native'],recovered=metrics))
  row['native']=metrics
 return corrections
