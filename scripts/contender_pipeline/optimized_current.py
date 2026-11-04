"""Explicit current-reference workflow optimization; original trainer stays pinned.

Only audit file access and optional equivalent F32 storage change. Freeze must
pin this wrapper and its helpers; no objective, optimizer, order or cap changes.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
from runner import pin,read,require
from snapshot_audit import snapshot_audit


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--project',type=Path,required=True);parser.add_argument('--shards',type=Path)
    args,remaining=parser.parse_known_args()
    source=args.project/'scripts/design_readiness/fixed_objective.py'
    freeze=read(remaining[remaining.index('--freeze')+1])
    for path in [Path(__file__),Path(__file__).with_name('snapshot_audit.py'),Path(__file__).with_name('runner.py')]:
        require(pin(path) in freeze['sources'],'workflow source not pinned')
    spec=importlib.util.spec_from_file_location('_optimized_reference',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.Audit=snapshot_audit(module.Audit)
    if args.shards:
        from sharded_rows import ShardRows
        require(pin(Path(__file__).with_name('sharded_rows.py')) in freeze['sources'],'shard source not pinned')
        require(pin(args.shards) in freeze['sources'],'shard index not pinned')
        original=module.resident.bind_resident
        def bind(data,config):
            original(data,config) # Preserve original config/instance validation.
            reader=ShardRows(args.shards,max_maps=2);data._rows=reader;return reader
        module.resident.bind_resident=bind
    sys.argv=[str(source)]+remaining
    module.main()

if __name__=='__main__':main()
