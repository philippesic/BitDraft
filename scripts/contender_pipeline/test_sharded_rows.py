import tempfile,unittest
from pathlib import Path
import numpy as np
from runner import pin
from sharded_rows import pack,ShardRows,ExposureSampler

class ShardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.specs=[];self.arrays=[]
  for i in range(20):
   a=(np.arange(128,dtype='<f4')+i).reshape(8,16);p=self.root/f'{i}.f32';a.tofile(p);self.specs.append(dict(**pin(p),shape=list(a.shape)));self.arrays.append(a)
 def test_multiple_shards_eviction_exact_bytes(self):
  manifest=pack(self.specs,self.root/'packed',1024);reader=ShardRows(self.root/'packed/manifest.json',max_maps=2)
  self.assertEqual(len(manifest['shards']),10)
  for i in list(range(20))+list(reversed(range(20))):
   a=reader.matrix(self.specs[i]);self.assertEqual(a.tobytes(),self.arrays[i].tobytes());self.assertFalse(a.flags.writeable)
   self.assertLessEqual(reader.report()['mapped_bytes'],2048)
 def test_corruption_and_unknown_source(self):
  m=pack(self.specs,self.root/'packed',1024);reader=ShardRows(self.root/'packed/manifest.json')
  with open(m['shards'][0]['path'],'r+b') as f:f.write(b'xxxx')
  with self.assertRaisesRegex(ValueError,'shard changed'):reader.matrix(self.specs[0])
  changed=dict(self.specs[2],sha256='0'*64)
  with self.assertRaisesRegex(ValueError,'differs'):reader.matrix(changed)
 def test_pack_corruption_no_manifest(self):
  Path(self.specs[0]['path']).write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'source bytes changed'):pack(self.specs,self.root/'packed',1024)
  self.assertFalse((self.root/'packed/manifest.json').exists())
 def test_sampler_resume_epoch_boundary(self):
  ids=[str(i) for i in range(17)];a=ExposureSampler(ids,42)
  prefix=[a.next() for _ in range(16)];b=ExposureSampler(ids,42,a.state())
  tail=[a.next() for _ in range(80)];self.assertEqual(tail,[b.next() for _ in range(80)])
  self.assertEqual(set(prefix+tail[:1]),set(ids))
  with self.assertRaises(ValueError):ExposureSampler(list(reversed(ids)),42,a.state())
  with self.assertRaises(ValueError):ExposureSampler(ids,43,a.state())
if __name__=='__main__':unittest.main()
