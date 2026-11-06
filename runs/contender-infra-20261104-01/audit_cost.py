import sys,time,json
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';sys.path[:0]=[str(R/'source-v2'),'/home/philip/.codex/worktrees/w1a8-actual-training-20261103/scripts/w1a8_generalization']
from runner import atomic,read,pin
from snapshot_audit import snapshot_audit
from modern_actual_union import Audit,build_union
m=read(read(R/'data-probe-v2.json')['union']['path']);paths=[v['manifest']['path'] for v in m['members']];rows=[]
for name,cls in [('snapshot',snapshot_audit(Audit)),('original',Audit)]:
 audit=cls(lambda:None);t=time.perf_counter();value=build_union(paths,audit=audit);elapsed=time.perf_counter()-t;assert value==m
 rows.append(dict(name=name,wall_seconds=elapsed,row_bytes=audit.row_bytes,hash_bytes=audit.hash_bytes,metadata_bytes=audit.metadata_bytes))
 atomic(R/'audit-cost-progress.json',dict(completed=rows));print(json.dumps(rows[-1]),flush=True)
atomic(R/'audit-cost.json',dict(status='SAME_UNION_AUDIT_BOTH_EXACT',order='snapshot then original, after shared prior reads; one CPU comparison, not a distribution',rows=rows,scope='Setup validation only; same row/hash/metadata byte accounting and exact rebuilt manifest. Not decode throughput or pure GPU time.'))
