"""Focused CPU contracts for detached current-prefix CE admission; no real data."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/w1a8_generalization"))
SPEC = importlib.util.spec_from_file_location(
    "modern_train_reachable",
    PROJECT / "scripts/w1a8_generalization/modern_train_reachable.py",
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def window(mask, role="fit"):
    return {"role": role, "native_supervision_reachable_mask": list(mask)}


def legacy_loss(values, classes, mask):
    ce = values.sum() * 0.0
    for depth, cls in enumerate(classes.tolist()):
        if cls >= 0 and mask[depth]:
            ce = ce + torch.nn.functional.cross_entropy(
                values[depth : depth + 1], classes[depth : depth + 1]
            )
    return ce


class CurrentPrefixReachabilityTests(unittest.TestCase):
    def test_static_is_exact_original_loss_and_gradient_with_unsupported_slots(self):
        initial = torch.tensor([[3.0, 0.5, -1.0], [-2.0, 1.0, 4.0], [1.0, 2.0, 3.0]])
        classes = torch.tensor([0, 1, -1])
        mask = [True, False, True]
        expected_values = initial.clone().requires_grad_()
        actual_values = initial.clone().requires_grad_()
        expected = legacy_loss(expected_values, classes, mask) / 4.0
        actual, stats = trainer.ce_reachable_loss(actual_values, classes, window(mask), "static")
        actual = actual / 4.0
        expected.backward()
        actual.backward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_values.grad, expected_values.grad))
        self.assertEqual(stats["static_supported_labels"], 1)
        self.assertEqual(stats["extra_supported_labels"], 0)
        self.assertEqual(stats["effective_supported_mass"], 0.25)

    def test_static_exact_inherited_adam_trajectory(self):
        initial = torch.tensor([[3.0, 0.5, -1.0], [-2.0, 1.0, 4.0]])
        expected_values = torch.nn.Parameter(initial.clone())
        actual_values = torch.nn.Parameter(initial.clone())
        expected_optimizer = torch.optim.Adam([expected_values], lr=1e-5)
        actual_optimizer = torch.optim.Adam([actual_values], lr=1e-5)
        # Create real inherited moments before the compared trajectory.
        warm_classes = torch.tensor([1, 2])
        (legacy_loss(expected_values, warm_classes, [True, True]) / 4.0).backward()
        expected_optimizer.step()
        actual_values.data.copy_(expected_values.data)
        actual_optimizer.load_state_dict(copy.deepcopy(expected_optimizer.state_dict()))
        for classes, mask in (
            (torch.tensor([0, 1]), [True, False]),
            (torch.tensor([-1, 2]), [True, True]),
            (torch.tensor([2, 1]), [True, True]),
        ):
            expected_optimizer.zero_grad(set_to_none=True)
            actual_optimizer.zero_grad(set_to_none=True)
            expected = legacy_loss(expected_values, classes, mask) / 4.0
            actual, _ = trainer.ce_reachable_loss(actual_values, classes, window(mask), "static")
            (actual / 4.0).backward()
            expected.backward()
            expected_optimizer.step()
            actual_optimizer.step()
            self.assertTrue(torch.equal(actual_values, expected_values))
            left = expected_optimizer.state[expected_values]
            right = actual_optimizer.state[actual_values]
            self.assertEqual(set(left), set(right))
            self.assertTrue(all(torch.equal(left[k], right[k]) for k in left))

    def test_wrong_current_depth_still_gets_ce_when_previous_gold_was_reached(self):
        values = torch.tensor([[4.0, 0.0, -1.0], [3.0, -2.0, 0.0]], requires_grad=True)
        classes = torch.tensor([0, 1])
        ce, stats = trainer.ce_reachable_loss(
            values, classes, window([True, False]), "current-prefix-union"
        )
        ce.backward()
        self.assertEqual(stats["extra_supported_labels"], 1)
        self.assertEqual(stats["effective_supported_labels"], 2)
        self.assertEqual(stats["extra_supported_mass"], 0.25)
        self.assertLess(float(values.grad[1, 1]), 0)
        self.assertGreater(float(values.grad[1, 0]), 0)

    def test_current_depth_success_cannot_rescue_wrong_predecessor(self):
        values = torch.tensor([[0.0, 4.0, -1.0], [0.0, 5.0, 1.0]], requires_grad=True)
        classes = torch.tensor([0, 1])
        ce, stats = trainer.ce_reachable_loss(
            values, classes, window([True, False]), "current-prefix-union"
        )
        ce.backward()
        self.assertEqual(stats["extra_supported_labels"], 0)
        self.assertTrue(torch.equal(values.grad[1], torch.zeros_like(values.grad[1])))

    def test_old_reachable_slots_remain_when_current_prefix_fails(self):
        values = torch.tensor([[0.0, 4.0, -1.0], [3.0, -2.0, 0.0]], requires_grad=True)
        classes = torch.tensor([0, 1])
        ce, stats = trainer.ce_reachable_loss(
            values, classes, window([True, True]), "current-prefix-union"
        )
        ce.backward()
        self.assertEqual(stats["static_supported_labels"], 2)
        self.assertEqual(stats["extra_supported_labels"], 0)
        self.assertLess(float(values.grad[1, 1]), 0)

    def test_unsupported_gold_blocks_current_prefix_without_dropping_old_slots(self):
        values = torch.tensor([[4.0, 0.0, -1.0], [3.0, -2.0, 0.0]], requires_grad=True)
        classes = torch.tensor([-1, 1])
        ce, stats = trainer.ce_reachable_loss(
            values, classes, window([True, False]), "current-prefix-union"
        )
        ce.backward()
        self.assertEqual(stats["effective_supported_labels"], 0)
        self.assertTrue(torch.equal(values.grad, torch.zeros_like(values)))
        preserved, counts = trainer.ce_reachable_loss(
            values.detach().clone().requires_grad_(), classes,
            window([True, True]), "current-prefix-union"
        )
        self.assertEqual(counts["static_supported_labels"], 1)
        self.assertGreater(float(preserved), 0)

    def test_prefix_gate_uses_every_earlier_depth_and_never_later_depth(self):
        values = torch.tensor([[4.0, 0.0, -1.0], [3.0, -2.0, 0.0], [0.0, 1.0, 5.0]])
        classes = torch.tensor([0, 1, 2])
        _, stats = trainer.ce_reachable_loss(
            values, classes, window([True, False, False]), "current-prefix-union"
        )
        self.assertEqual(stats["extra_supported_labels"], 1)
        self.assertEqual(stats["effective_supported_labels"], 2)

    def test_development_invalid_modes_and_corrupt_masks_are_rejected(self):
        values = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        classes = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "FIT windows only"):
            trainer.ce_reachable_loss(values, classes, window([True, False], "development"), "static")
        with self.assertRaisesRegex(ValueError, "unknown CE"):
            trainer.ce_reachable_loss(values, classes, window([True, False]), "other")
        with self.assertRaisesRegex(ValueError, "shapes/mask"):
            trainer.ce_reachable_loss(values, classes, window([True]), "static")
        with self.assertRaisesRegex(ValueError, "shapes/mask"):
            trainer.ce_reachable_loss(values, classes, window([1, 0]), "static")
        with self.assertRaisesRegex(ValueError, "invalid CE class"):
            trainer.ce_reachable_loss(values, torch.tensor([0, 2]), window([True, True]), "static")

    def test_fixed_normalization_reports_added_mass_instead_of_renormalizing(self):
        values = torch.tensor([[4.0, 0.0], [3.0, -2.0]], requires_grad=True)
        classes = torch.tensor([0, 1])
        _, static = trainer.ce_reachable_loss(values, classes, window([True, False]), "static")
        _, dynamic = trainer.ce_reachable_loss(
            values, classes, window([True, False]), "current-prefix-union"
        )
        self.assertEqual(static["effective_supported_mass"], 0.25)
        self.assertEqual(dynamic["effective_supported_mass"], 0.5)
        self.assertEqual(dynamic["extra_supported_mass"], 0.25)
        self.assertIn("fixed4", trainer.reachability_contract("current-prefix-union")["normalization"])
        self.assertNotEqual(
            trainer.reachability_contract("static"),
            trainer.reachability_contract("current-prefix-union"),
        )

    def test_stage_counter_resume_roundtrip_and_corruption(self):
        counts = trainer.reachability_totals()
        counts.update(updates=2, static_supported_labels=3, extra_supported_labels=2,
                      effective_supported_labels=5, static_supported_mass=0.75,
                      extra_supported_mass=0.5, effective_supported_mass=1.25)
        self.assertEqual(trainer.reachability_totals(counts), counts)
        self.assertIsNot(trainer.reachability_totals(counts), counts)
        for key, value in (("updates", -1), ("extra_supported_mass", 0.75),
                           ("effective_supported_labels", 4)):
            bad = {**counts, key: value}
            with self.assertRaisesRegex(ValueError, "invalid reachability"):
                trainer.reachability_totals(bad)
        with self.assertRaisesRegex(ValueError, "accounting fields"):
            trainer.reachability_totals({"updates": 1})

    def test_original_forward_and_native_replay_bank_sources_remain_identical(self):
        own = (PROJECT / "scripts/w1a8_generalization/modern_train_reachable.py").read_text()
        for name in ("modern_train.py", "modern_train_protected.py", "modern_train_native_replay.py"):
            source = (PROJECT / "scripts/w1a8_generalization" / name).read_text()
            self.assertEqual(trainer.forward_semantics(own), trainer.forward_semantics(source))
        native = (PROJECT / "scripts/w1a8_generalization/modern_train_native_replay.py").read_text()
        def definitions(source):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef)}
        own_defs, native_defs = definitions(own), definitions(native)
        for name in ("native_fit_references", "native_schedule", "validate_v1_warm_start",
                     "original_validation_packet", "stage_totals"):
            self.assertEqual(own_defs[name], native_defs[name])


if __name__ == "__main__":
    unittest.main()
