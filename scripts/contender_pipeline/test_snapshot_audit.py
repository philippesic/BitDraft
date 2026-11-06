import hashlib,tempfile,unittest
from pathlib import Path
import sys
import numpy as np
from runner import pin
from snapshot_audit import snapshot_audit

# Use admitted original validator, not a test-shaped substitute.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'w1a8_generalization'))
from modern_actual_union import Audit

class SnapshotTests(unittest.TestCase):
 def test_original_checks_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'a.f32';a=np.arange(32,dtype='<f4').reshape(4,8);a.tofile(p);spec=dict(**pin(p),shape=[4,8]);audit=snapshot_audit(Audit,128)(lambda:None)
   digest=hashlib.sha256(a[2].tobytes()).hexdigest()
   for _ in range(3):audit.row(spec,2,8,digest)
   self.assertEqual(audit.row_bytes,96);self.assertEqual(audit.snapshot_hits,2)
   for row,width,h in [(4,8,digest),(2,9,digest),(2,8,'0'*64)]:
    with self.assertRaises(ValueError):audit.row(spec,row,width,h)
   # Subsequent filesystem admission still detects external change.
   p.write_bytes(b'0'*128)
   with self.assertRaises(ValueError):audit.file(spec)
 def test_eviction_fallback_and_corruption(self):
  with tempfile.TemporaryDirectory() as tmp:
   audit=snapshot_audit(Audit,32)(lambda:None)
   for i in range(4):
    p=Path(tmp)/str(i);a=np.arange(8,dtype='<f4').reshape(1,8);a.tofile(p);spec=dict(**pin(p),shape=[1,8]);audit.row(spec,0,8,hashlib.sha256(a.tobytes()).hexdigest())
    self.assertLessEqual(audit.snapshot_bytes,32)
   fallback=snapshot_audit(Audit,0)(lambda:None);fallback.row(spec,0,8,hashlib.sha256(a.tobytes()).hexdigest());self.assertEqual(fallback.snapshot_bytes,0)
if __name__=='__main__':unittest.main()
