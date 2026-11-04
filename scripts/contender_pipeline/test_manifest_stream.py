import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from runner import pin,read
from manifest_stream import ManifestStream
from sharded_rows import ExposureSampler

class StreamTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.registry=dict(schema='training_manifest_stream_v1',contract={'target':{'sha256':'fixed'}},shards=[])
  for i in range(6):
   p=self.root/f'{i}.json';p.write_text(json.dumps(dict(target={'sha256':'fixed'},windows=[dict(id=f'w{i}',role='fit',value=i)])))
   self.registry['shards'].append(dict(manifest=pin(p),admission=pin(p),contract=self.registry['contract'],windows=[f'w{i}']))
  self.path=self.root/'registry.json';self.path.write_text(json.dumps(self.registry))
 def factory(self,path):
  value=read(path);return SimpleNamespace(manifest=value,windows=value['windows'],materialize=lambda w,device:w['value'])
 def test_eviction_and_resume(self):
  stream=ManifestStream(self.path,self.factory,max_manifests=1);sampler=ExposureSampler(stream.ids,17)
  prefix=[stream.materialize(sampler.next()) for _ in range(5)];state=sampler.state()
  tail=[stream.materialize(sampler.next()) for _ in range(13)]
  resumed=ExposureSampler(stream.ids,17,state);self.assertEqual(tail,[stream.materialize(resumed.next()) for _ in range(13)])
  self.assertEqual(set(prefix+tail[:1]),set(range(6)));self.assertEqual(stream.report()['cached_manifests'],1)
 def test_heldout_and_oversize_rejected(self):
  p=self.root/'0.json';v=read(p);v['windows'][0]['role']='development';p.write_text(json.dumps(v));self.registry['shards'][0].update(manifest=pin(p),admission=pin(p));self.path.write_text(json.dumps(self.registry))
  stream=ManifestStream(self.path,self.factory)
  with self.assertRaisesRegex(ValueError,'held-out'):stream.materialize(stream.ids[0])
  with self.assertRaisesRegex(ValueError,'exceeds'):ManifestStream(self.path,self.factory,max_manifest_bytes=1)
 def test_reopened_changed_manifest_rejected(self):
  stream=ManifestStream(self.path,self.factory,max_manifests=1);stream.materialize(stream.ids[0]);stream.materialize(stream.ids[1])
  p=self.root/'0.json';p.write_text(p.read_text()+' ')
  with self.assertRaisesRegex(ValueError,'manifest changed'):stream.materialize(stream.ids[0])
if __name__=='__main__':unittest.main()
