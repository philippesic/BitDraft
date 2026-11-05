"""Full-byte recovery and fail-closed dependency checks for paid GGUF exports."""
import random,tempfile
from pathlib import Path
from scripts.dflash_baseline.native_archive import archive,hydrate,release,reference,sha
def test_native_archive_recovery():
 with tempfile.TemporaryDirectory() as directory:
  d=Path(directory);basis=d/'basis.gguf';model=d/'model.gguf';data=random.Random(7).randbytes(2<<20);basis.write_bytes(data);model.write_bytes(data[:10000]+b'changed'+data[10000:]);expected=sha(model)
  record=archive(model,basis);assert record['verified_full_byte_roundtrip'] and not model.exists() and reference(model)['sha256']==expected
  assert hydrate(model) and sha(model)==expected
  assert not hydrate(model);release(model,True);assert not model.exists()
def test_native_archive_corrupt_dependency():
 with tempfile.TemporaryDirectory() as directory:
  d=Path(directory);basis=d/'basis.gguf';model=d/'model.gguf';basis.write_bytes(b'unchanged'*4096);model.write_bytes(basis.read_bytes()+b'mutated');archive(model,basis);basis.write_bytes(b'corrupted')
  try:hydrate(model)
  except ValueError:pass
  else:raise AssertionError('changed native recovery basis accepted')
  assert not model.exists() and not model.with_suffix('.gguf.hydrating').exists()
