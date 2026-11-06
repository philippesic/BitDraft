import sys,time,json,resource
from pathlib import Path
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/contender-infra-20261104-01';sys.path[:0]=[str(R/'source-v2'),'/home/philip/.codex/worktrees/w1a8-actual-training-20261103/scripts/w1a8_generalization','/home/philip/.codex/worktrees/w1a8-actual-training-20261103/src']
import torch
torch.set_num_threads(2)
from modern_resident_rows import ResidentRows
from runner import atomic,pin
from sharded_rows import pack,ShardRows,ExposureSampler
from modern_actual_union import UnionActualDataset,Audit,build_union
from snapshot_audit import snapshot_audit
parent=torch.load('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-added-replay-pair-20261103-01/true5760/checkpoint-00005760.pt',map_location='cpu',weights_only=False);m=parent['contract']['union_manifest'];data=UnionActualDataset(m['path']);other=UnionActualDataset(m['path']);specs=data.manifest['raw_files']+data.manifest['embedding_files']
t=time.perf_counter();packed=json.loads((R/'shards/manifest.json').read_text());packing=None;data._rows=ResidentRows(max_cache_bytes=1024**3)
reader=ShardRows(R/'shards/manifest.json',max_maps=2);other._rows=reader
t=time.perf_counter();windows=0
for w in data.windows:
 a=data.materialize(w);b=other.materialize(w)
 assert a.prefix_tokens==b.prefix_tokens and torch.equal(a.prefix_rawfeatures,b.prefix_rawfeatures) and torch.equal(a.seed_rawfeature,b.seed_rawfeature)
 windows+=1
embedding_ids=data.manifest['required_embedding_ids']
for token in embedding_ids:assert torch.equal(data.embedding_lookup(token),other.embedding_lookup(token))
materialization=time.perf_counter()-t
audit=snapshot_audit(Audit)(lambda:None);t=time.perf_counter();rebuilt=build_union([v['manifest']['path'] for v in data.manifest['members']],audit=audit);audit_seconds=time.perf_counter()-t;assert rebuilt==data.manifest
result=dict(status='PACKED_DATA_AND_SNAPSHOT_AUDIT_EXACT',union=m,shards=pin(R/'shards/manifest.json'),files=len(packed['entries']),bytes=sum(s['bytes'] for s in packed['shards']),shard_count=len(packed['shards']),windows=windows,embeddings=len(embedding_ids),packing_seconds=packing,paired_materialization_seconds=materialization,snapshot_union_seconds=audit_seconds,snapshot_hits=audit.snapshot_hits,snapshot_misses=audit.snapshot_misses,snapshot_bytes=audit.snapshot_bytes,reader=reader.report(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='All current union windows/required embeddings, existing consumed data only. Storage mechanism qualification, not larger-corpus diversity.')
atomic(R/'data-probe-v2.json',result);print(json.dumps(result))
