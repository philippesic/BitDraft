import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from capture_probe import compare_arrays


class RecaptureTests(unittest.TestCase):
    def fixture(self, root, ids, values):
        root.mkdir()
        np.array([1., 2.], dtype='<f4').tofile(root/'input.f32')
        np.array(values, dtype='<f4').tofile(root/'logits.f32')
        row = dict(candidate_ids=ids, candidate_logits={'file':'logits.f32'},
                   logits_mode='backend_explicit_candidate_pairs',
                   argmax_target_token=3, backend_sampled_token=-1)
        (root/'calls.jsonl').write_text(json.dumps({'selected':[row]})+'\n')

    def test_aligned_permutation_is_exact(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b'
            self.fixture(a,[2,3],[1.,4.]);self.fixture(b,[3,2],[4.,1.])
            self.assertEqual(compare_arrays(a,b)['reordered_candidate_lists'],1)

    def test_changed_value_or_token_or_feature_rejected(self):
        for kind in ['value','token','feature']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
                a,b=Path(t)/'a',Path(t)/'b'
                self.fixture(a,[2,3],[1.,4.])
                self.fixture(b,[2,4] if kind=='token' else [2,3], [2.,4.] if kind=='value' else [1.,4.])
                if kind=='feature':np.array([1.,3.],dtype='<f4').tofile(b/'input.f32')
                with self.assertRaises(AssertionError):compare_arrays(a,b)
