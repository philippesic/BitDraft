import sys,time,json
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';sys.path.insert(0,str(R/'source-v2'))
from runner import pin,read,atomic
from sharded_rows import ShardRows
import numpy as np
index=R/'shards/manifest.json';m=read(index);reader=ShardRows(index,max_maps=2);t=time.perf_counter()
for entry in m['entries'].values():
 spec=entry['source'];a=reader.matrix(spec)
 assert pin(spec['path'])=={k:spec[k] for k in ('path','bytes','sha256')}
 # Reader authenticates each segment, not just its enclosing shard.
 assert a.shape==tuple(spec['shape']) and a.dtype==np.dtype('<f4') and not a.flags.writeable
 assert reader.report()['mapped_bytes']<=2*m['shard_bytes']
atomic(R/'final-shard-reader.json',dict(status='ALL_PACKED_SOURCES_AUTHENTICATED',reader=reader.report(),sources=len(m['entries']),wall_seconds=time.perf_counter()-t,source=pin(R/'source-v2/sharded_rows.py')))
