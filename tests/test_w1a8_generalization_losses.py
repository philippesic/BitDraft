import importlib.util
from pathlib import Path
import unittest

import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization/losses.py"
spec = importlib.util.spec_from_file_location("generalization_losses", PATH)
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


class RetentionTests(unittest.TestCase):
    def test_protects_reachable_prefix_without_preserving_wrong_tail(self):
        labels = torch.tensor([0, 0, 0])
        parent = torch.tensor([[3., 1.], [0., 1.], [3., 1.]])
        ref = losses.parent_reference(parent, labels)
        self.assertEqual(ref["protected"], [True, False, False])
        student = torch.tensor([[1., 2.], [9., 0.], [-4., 4.]], requires_grad=True)
        value = losses.retention_penalty(student, labels, ref)
        value.backward()
        self.assertLess(student.grad[0, 0], 0)
        self.assertGreater(student.grad[0, 1], 0)
        self.assertTrue(torch.equal(student.grad[1:], torch.zeros_like(student.grad[1:])))

    def test_improvement_and_no_parent_success_have_zero_penalty(self):
        labels = torch.tensor([0])
        ref = losses.parent_reference(torch.tensor([[2., 1.]]), labels)
        self.assertEqual(float(losses.retention_penalty(torch.tensor([[5., 1.]]), labels, ref)), 0.)
        failed = losses.parent_reference(torch.tensor([[0., 1.]]), labels)
        logits = torch.tensor([[1., 2.]], requires_grad=True)
        penalty = losses.retention_penalty(logits, labels, failed)
        penalty.backward()
        self.assertEqual(float(penalty.detach()), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rejects_nonfinite_teacher(self):
        with self.assertRaises(ValueError):
            losses.parent_reference(torch.tensor([[float("nan"), 1.]]), torch.tensor([0]))

    def test_actor_only_success_cannot_become_native_retention_target(self):
        logits = torch.tensor([[3., 0.], [3., 0.], [3., 0.]])
        ref = losses.parent_reference(logits, torch.tensor([0, 0, 0]), native_prefix=1)
        self.assertEqual(ref["protected"], [True, False, False])
        with self.assertRaises(ValueError):
            losses.parent_reference(logits, torch.tensor([0, 0, 0]), native_prefix=-1)


if __name__ == "__main__":
    unittest.main()
