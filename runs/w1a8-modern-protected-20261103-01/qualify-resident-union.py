from pathlib import Path
import sys,json,time,resource,hashlib
W=Path('/home/philip/.codex/worktrees/w1a8-actual-training-20261103');sys.path.insert(0,str(W/'scripts/w1a8_generalization'))
import torch
from modern_actual_data import _Rows
from modern_actual_union import UnionActualDataset
def file_pin(p):
 p=Path(p).resolve()
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return {'path':str(p),'sha256':h,'bytes':p.stat().st_size}
from modern_resident_rows import ResidentRows
P=Path('/home/philip/binary-eagle-decoding');R=P/'runs/w1a8-modern-protected-20261103-01';M=Path('/mnt/c/Users/philip/binary-eagle-decoding-artifacts/w1a8-modern-fit-expansion-20261103-01/union01/manifest.json');start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1);a=UnionActualDataset(M);b=UnionActualDataset(M);cache=ResidentRows(capacity_mode='error');base=_Rows();a._rows=base;b._rows=cache
specs={}
for ref in a.manifest['raw_files']+a.manifest['embedding_files']:
 key=(str(Path(ref['path']).resolve()),ref['sha256'],ref['bytes'],tuple(ref['shape']));specs[key]=ref
bytes_=0;t=time.monotonic()
for i,spec in enumerate(specs.values()):
 old=base.matrix(spec);new=cache.matrix(spec);assert old.shape==new.shape and old.dtype==new.dtype and old.tobytes()==new.tobytes();assert not new.flags.writeable;bytes_+=spec['bytes']
 if i%512==0:print(json.dumps({'stage':'matrix_parity','matrices':i,'total':len(specs)}),flush=True)
matrix_seconds=time.monotonic()-t;selected=[]
for group in a.manifest['groups']:
 windows=[w for w in a.windows if w['group']==group['id']];assert windows;selected.append(max(windows,key=lambda w:(w['anchor'],w['id'])))
assert len(selected)==60;t=time.monotonic()
for i,w in enumerate(selected):
 old=a.materialize(w,'cpu');new=b.materialize(b._window_ids[w['id']],'cpu');assert old.prefix_tokens==new.prefix_tokens and torch.equal(old.prefix_rawfeatures,new.prefix_rawfeatures) and torch.equal(old.seed_rawfeature,new.seed_rawfeature)
 if i%12==0:print(json.dumps({'stage':'whole_window_parity','windows':i,'total':len(selected)}),flush=True)
window_seconds=time.monotonic()-t;t=time.monotonic();ids=a.manifest['required_embedding_ids'];assert len(ids)==2135
for token in ids:assert torch.equal(a.embedding_lookup(token,'cpu'),b.embedding_lookup(token,'cpu'))
embedding_seconds=time.monotonic()-t;report=cache.report(include_descriptors=True);assert report['fallbacks']==0 and report['cached_files']==len(specs)
proof={'schema':'w1a8_resident_union_parity_v1','status':'REAL_UNION_RESIDENT_ROWS_BYTE_AND_WINDOW_PARITY_PASS','manifest':file_pin(M),'helper':file_pin(W/'scripts/w1a8_generalization/modern_resident_rows.py'),'original_reader':file_pin(W/'scripts/w1a8_generalization/modern_actual_data.py'),'all_matrices_byte_equal':True,'all_sampled_materializations_byte_equal':True,'all_embeddings_byte_equal':True,'unique_descriptors':len(specs),'matrix_bytes_compared':bytes_,'sampled_windows':60,'sampled_window_ids':[w['id'] for w in selected],'embedding_rows':len(ids),'matrix_seconds':matrix_seconds,'window_seconds':window_seconds,'embedding_seconds':embedding_seconds,'wall_seconds':time.monotonic()-start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cache_report':report,'scope':'CPU-only every declared source matrix bit-identical, one max-anchor wholewindow/group and every2135embedding lookup; not all3603wholewindowmaterializations or native/model speed proof','overlap':'oldcontrolposttrainingreadoutcontinues; no fullrequesttiming quietwindow'}
(R/'resident-row-real-qualification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k not in ['cache_report','sampled_window_ids']},indent=2),flush=True)
