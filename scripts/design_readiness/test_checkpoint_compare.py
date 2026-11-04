import unittest
import torch
from checkpoint_compare import exact,compare

class CheckpointComparisonTests(unittest.TestCase):
    def test_signed_zero_and_dtype_are_part_of_exact_state(self):
        self.assertFalse(exact(torch.tensor([0.]),torch.tensor([-0.])))
        self.assertFalse(exact(torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64)))
        self.assertTrue(exact(torch.tensor([2.]),torch.tensor([2.])))

    def test_state_keys_types_and_nested_rng(self):
        a={'cursor':8,'rng':(1,[3,4]),'scale':torch.tensor([2.])}
        self.assertTrue(exact(a,{'cursor':8,'rng':(1,[3,4]),'scale':torch.tensor([2.])}))
        self.assertFalse(exact(a,{**a,'cursor':8.0}))
        self.assertFalse(exact(a,{**a,'rng':(1,[3,5])}))
        self.assertFalse(compare(a,a,['missing'])['missing'])
