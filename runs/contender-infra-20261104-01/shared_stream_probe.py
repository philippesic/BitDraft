import sys,time,resource,json
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';sys.path[:0]=[str(R/'source-v3'),str(R/'source-v2'),'/home/philip/.codex/worktrees/w1a8-actual-training-20261103/scripts/w1a8_generalization']
from runner import atomic,pin
from manifest_stream import ManifestStream
from sharded_rows import ShardRows,ExposureSampler
from modern_actual_data import ActualDataset
import torch;torch.set_num_threads(2)
t=time.perf_counter();rows=ShardRows(R/'shards/manifest.json',max_maps=2);stream=ManifestStream(R/'stream-registry.json',ActualDataset,max_manifests=1,row_reader=rows);init=time.perf_counter()-t
sampler=ExposureSampler(stream.ids,421104);keys=[sampler.next() for _ in range(24)];t=time.perf_counter()
for key in keys:
 data,window=stream.window(key);assert data._rows is rows
 a=stream.materialize(key);assert torch.isfinite(a.prefix_rawfeatures).all() and torch.isfinite(a.seed_rawfeature).all()
 # Actual materializer has authenticated each exact original row hash.
elapsed=time.perf_counter()-t
atomic(R/'shared-manifest-stream.json',dict(status='SHARED_READER_STREAM_PASS',init_seconds=init,materialize_seconds=elapsed,windows=24,metadata=stream.report(),rows=rows.report(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source=pin(R/'source-v3/manifest_stream.py')))
print('SHARED_STREAM_PASS')
