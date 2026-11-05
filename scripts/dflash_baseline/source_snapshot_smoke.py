"""Verify the interpreter uses the owned frozen source and dependencies, not cwd."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.dflash_baseline import model,prototype,prototype_model,pipeline,export_precision
import export_recurrent_binary as recurrent_export
import w1a1_eagle.block_qat as block
import gguf
paths={m.__name__:str(Path(m.__file__).resolve()) for m in (model,prototype,prototype_model,pipeline,export_precision,recurrent_export,block,gguf)}
assert 'source-snapshots' in str(ROOT)
assert all(Path(p).is_relative_to(ROOT) for p in paths.values())
assert (ROOT/'experiments/w1a8-standing-20261102/baseline.json').exists()
print(json.dumps(dict(status='IMMUTABLE_SOURCE_IMPORT_PASS',root=str(ROOT),modules=paths)),flush=True)
