"""Versioned fixed-candidate data comparison; narrowly derive the pinned trainer."""
from pathlib import Path
import argparse,difflib,sys
P=Path('/home/philip/binary-eagle-decoding')
sys.path[:0]=[str(P/'scripts/contender_pipeline'),str(Path(__file__).parent)]
from runner import pin,read,require
from snapshot_audit import snapshot_audit
from sharded_rows import ShardRows
from training_bank import TrainingBank

def derive(source):
 changes=[
  ("SCHEMA = 'modern_fixed_objective_checkpoint_v1'","SCHEMA = 'modern_fixed_data_transfer_checkpoint_v1'"),
  ("embeddings = resident.load_embeddings(data, device, guard)","embeddings = resident.load_embeddings(data, device, guard)\n    extra = make_extra()\n    if extra is not None: extra.add_embeddings(embeddings, device)"),
  ("parent_horizon=19200, max_bounded_cursor=7680)","parent_horizon=19200, max_bounded_cursor=7680, data_arm=DATA_ARM, bank=extra.contract if extra else None)"),
  ("for w in resident.ce_at(ce_fit, order, i):","for w in (extra.windows(ce_fit, order, i, resident.ce_at, metadata=True) if extra else resident.ce_at(ce_fit, order, i)):"),
  ("batch = data.materialize(window, device)","batch = extra.materialize(window, device) if extra and window['id'] in extra.keys else data.materialize(window, device)"),
  ("for w in resident.ce_at(ce_fit, order, cursor):","for w in (extra.windows(ce_fit, order, cursor, resident.ce_at) if extra else resident.ce_at(ce_fit, order, cursor)):"),
  ("peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), resident=cache.report(),","peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), resident=cache.report(), bank=extra.report() if extra else None,"),
 ]
 for before,after in changes:
  require(source.count(before)==1,'trainer derivation target changed: '+before);source=source.replace(before,after)
 return source

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-arm',choices=['old','broader'],required=True);p.add_argument('--registry',type=Path);p.add_argument('--shards',type=Path);p.add_argument('--original-shards',type=Path,required=True);p.add_argument('--derived-source',type=Path,required=True);a,rest=p.parse_known_args()
 freeze=read(rest[rest.index('--freeze')+1]);source=P/'scripts/design_readiness/fixed_objective.py'
 for path in [source,Path(__file__),Path(__file__).with_name('training_bank.py'),a.original_shards]:require(pin(path) in freeze['sources'],'unfrozen trainer dependency')
 if a.data_arm=='broader':
  for path in [a.registry,a.shards]:require(pin(path) in freeze['sources'],'unfrozen bank')
 derived=derive(source.read_text());require(a.derived_source.read_text()==derived and pin(a.derived_source) in freeze['sources'],'derived source differs')
 namespace=dict(__name__='_fixed_data_trial',__file__=str(source),DATA_ARM=a.data_arm,make_extra=lambda:TrainingBank(a.registry,a.shards) if a.data_arm=='broader' else None)
 exec(compile(derived,str(a.derived_source),'exec'),namespace)
 namespace['Audit']=snapshot_audit(namespace['Audit'])
 resident=namespace['resident'];original=resident.bind_resident
 def bind(data,config):
  original(data,config);reader=ShardRows(a.original_shards,max_maps=2);data._rows=reader;return reader
 resident.bind_resident=bind
 sys.argv=[str(source)]+rest
 namespace['main']()
