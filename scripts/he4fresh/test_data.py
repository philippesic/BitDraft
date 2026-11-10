"""Guard teacher-window tails and fail-closed array authentication."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .data import TeacherCapture, LocalTeacherCapture, LazyTeachers
from .common import REMOTE_TEACHER


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        files=[]
        def put(name,raw):
            (self.root/name).write_bytes(raw)
            files.append(dict(path=str(REMOTE_TEACHER/name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        for tap in (2,6,10,14,18,22,26,30):
            put(f'layer-{tap}.f32',np.full((24,2560),tap,dtype='<f4').tobytes())
        def array(name,value):
            out=io.BytesIO();np.save(out,np.asarray(value,dtype='int32'),allow_pickle=False);put(name,out.getvalue())
        positions=list(range(8,24));labels=list(range(100,115))+[999]
        array('inputs.npy',list(range(24)));array('label_positions.npy',positions);array('labels.npy',labels)
        self.row=dict(receipt=dict(metadata=dict(path=str(REMOTE_TEACHER/'capture.json')),files=files,feature_shape=[24,2560]),
            metadata=dict(input_tokens=list(range(24)),logit_positions=positions,greedy_ids=labels),anchors=[8,9])
        self.patch=patch('scripts.he4fresh.data.TEACHER',self.root);self.patch.start()

    def tearDown(self): self.patch.stop();self.temp.cleanup()

    def test_final_window_includes_last_native_eog_label(self):
        capture=TeacherCapture(self.row)
        self.assertEqual([capture.labels[9+j-1] for j in range(1,16)],list(range(101,115))+[999])
        self.assertEqual(capture.anchors,[8,9])
        self.assertEqual(tuple(capture.features.shape),(24,20480))
        self.assertEqual(float(capture.features[0,2560]),6)

    def test_corrupt_feature_rejected_before_exposure(self):
        path=self.root/'layer-2.f32';raw=bytearray(path.read_bytes());raw[0]^=1;path.write_bytes(raw)
        with self.assertRaises(AssertionError): TeacherCapture(self.row)

    def test_consistent_hash_does_not_hide_label_metadata_mismatch(self):
        self.row['metadata']['greedy_ids'][0]=1000
        with self.assertRaises(AssertionError): TeacherCapture(self.row)

    def test_pool_bounds_materialized_sequences(self):
        teachers=LazyTeachers([self.row]*5)
        for i in range(5): teachers[i]
        self.assertEqual(list(teachers.cache),[3,4])

    def test_local_loader_preserves_exact_features_and_eog_labels(self):
        old=TeacherCapture(self.row);new=LocalTeacherCapture(self.row)
        np.testing.assert_array_equal(old.features.numpy(),new.features.numpy())
        self.assertEqual(old.tokens,new.tokens)
        self.assertEqual(old.labels,new.labels)
        self.assertEqual(old.anchors,new.anchors)

    def test_local_loader_does_not_rehash_and_rejects_truncation(self):
        with patch('scripts.he4fresh.data.hashlib.sha256',side_effect=AssertionError('Repeated hash')):
            LocalTeacherCapture(self.row)
        path=self.root/'layer-2.f32';path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError,'Truncated local feature'):
            LocalTeacherCapture(self.row)


if __name__=='__main__': unittest.main()
