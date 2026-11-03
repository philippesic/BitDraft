"""CPU checks of fit replay fairness, actual gradient normalization and scope safety."""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/w1a8_generalization"))
SPEC = importlib.util.spec_from_file_location(
    "modern_train_protected", PROJECT / "scripts/w1a8_generalization/modern_train_protected.py"
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def fixture():
    windows, refs = [], {}
    for group, count in (("a", 1), ("b", 3), ("c", 2)):
        for index in range(count):
            window = {"id": f"{group}:{index}", "group_id": group, "role": "fit", "labels": [0, 1]}
            windows.append(window)
            refs[window["id"]] = {**window, "protected": [True, False], "margins": [1.0, -1.0]}
    return windows, refs


def actor_fixture():
    drafter = nn.Module()
    drafter.projections = nn.ModuleDict()
    linears = {}
    for name in ("fc", "q", "k", "v", "o", "gate", "up", "down", "lm_head"):
        module = nn.Module()
        module.scale_offset = nn.Parameter(torch.zeros(2))
        drafter.projections[name] = module
        linears[name] = module
    actor = SimpleNamespace(drafter=drafter, linears=linears)
    actor.scale_parameters = lambda: tuple(m.scale_offset for m in linears.values())
    return actor


class ProtectedTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_group_balance_unequal_group_sizes_and_resume_without_rng_consumption(self):
        windows, refs = fixture()
        initial_rng = random.getstate()
        schedule = trainer.ProtectedReplaySchedule(windows, refs)
        self.assertEqual(initial_rng, random.getstate())
        selected = [schedule.at(i) for i in range(600)]
        self.assertEqual(Counter(g for g, _ in selected), {"a": 200, "b": 200, "c": 200})
        for group in ("a", "b", "c"):
            counts = Counter(w["id"] for g, w in selected if g == group)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        restored = trainer.ProtectedReplaySchedule(list(reversed(windows)), refs)
        self.assertEqual(schedule.contract, restored.contract)
        self.assertEqual(
            [schedule.at(i)[1]["id"] for i in range(193, 240)],
            [restored.at(i)[1]["id"] for i in range(193, 240)],
        )

    def test_replay_excludes_unprotected_without_ever_selecting_development(self):
        windows, refs = fixture()
        refs["b:0"]["protected"] = [False, False]
        schedule = trainer.ProtectedReplaySchedule(windows, refs)
        self.assertNotIn("b:0", {schedule.at(i)[1]["id"] for i in range(300)})
        with self.assertRaisesRegex(ValueError, "fit windows only"):
            trainer.ProtectedReplaySchedule(
                windows + [{**windows[0], "id": "dev", "role": "development"}], refs
            )
        for row in refs.values():
            row["protected"] = [False, False]
        with self.assertRaisesRegex(ValueError, "no fit windows"):
            trainer.ProtectedReplaySchedule(windows, refs)

    def test_retention_normalization_actual_gradients_and_no_unprotected_dilution(self):
        values = torch.tensor([[1.5, 1.0, 0.0], [0.0, 2.0, 1.0]], requires_grad=True)
        reference = {"protected": [True, False], "margins": [1.5, 2.0]}
        loss, count = trainer.normalized_replay_loss(values, torch.tensor([0, 1]), reference)
        self.assertEqual(count, 1)
        self.assertEqual(float(loss.detach()), 1.0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[-2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]))
        # Two equally deficient protected tokens retain the same mean loss.
        reference = {"protected": [True, True], "margins": [1.5, 2.0]}
        loss, count = trainer.normalized_replay_loss(
            values.detach(), torch.tensor([0, 1]), reference
        )
        self.assertEqual(count, 2)
        self.assertEqual(float(loss), 1.0)

    def test_retention_cap_and_invalid_protected_labels_fail_closed(self):
        values = torch.tensor([[2.0, 1.0]], requires_grad=True)
        loss, _ = trainer.normalized_replay_loss(
            values, torch.tensor([0]), {"protected": [True], "margins": [5.0]}
        )
        self.assertEqual(float(loss.detach()), 1.0)  # frozen cap2 minus current margin1
        for cls, reference in (
            (-1, {"protected": [True], "margins": [1.0]}),
            (0, {"protected": [True], "margins": [None]}),
            (0, {"protected": [False], "margins": [1.0]}),
        ):
            with self.assertRaises(ValueError):
                trainer.normalized_replay_loss(values, torch.tensor([cls]), reference)

    def test_head_scope_only_steps_selected_scales_and_clears_entire_actor(self):
        actor = actor_fixture()
        paths, params, excluded = trainer.optimizer_scope(actor, "head")
        self.assertEqual(paths, ["lm_head"])
        self.assertEqual(len(actor.scale_parameters()), 9)
        self.assertTrue(all(p.requires_grad for p in actor.scale_parameters()))
        optimizer = torch.optim.Adam(params, lr=0.01)
        for _ in range(2):
            actor.drafter.zero_grad(set_to_none=True)
            self.assertTrue(all(p.grad is None for p in actor.scale_parameters()))
            loss = sum((m.scale_offset - 1).square().sum() for m in actor.linears.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            trainer.assert_excluded_unchanged(actor, excluded)
        self.assertTrue(bool((actor.linears["lm_head"].scale_offset > 0).all()))
        with torch.no_grad():
            actor.linears["fc"].scale_offset[0] = 1
        with self.assertRaisesRegex(ValueError, "excluded scale changed"):
            trainer.assert_excluded_unchanged(actor, excluded)

    def test_all_scope_and_invalid_scope(self):
        actor = actor_fixture()
        paths, params, excluded = trainer.optimizer_scope(actor, "all")
        self.assertEqual(len(paths), 9)
        self.assertEqual(len(params), 9)
        self.assertEqual(excluded, {})
        with self.assertRaises(ValueError):
            trainer.optimizer_scope(actor, "arbitrary")

    def test_head_checkpoint_roundtrip_resumes_optimizer_rng_and_replay_cursor(self):
        actor = actor_fixture()
        _, params, excluded = trainer.optimizer_scope(actor, "head")
        optimizer = torch.optim.Adam(params, lr=0.01)
        windows, refs = fixture()
        schedule = trainer.ProtectedReplaySchedule(windows, refs)
        saved_torch_rng, saved_python_rng = torch.get_rng_state(), random.getstate()
        self.addCleanup(torch.set_rng_state, saved_torch_rng)
        self.addCleanup(random.setstate, saved_python_rng)
        torch.manual_seed(71)
        random.seed(71)

        def update(cursor):
            actor.drafter.zero_grad(set_to_none=True)
            group, window = schedule.at(cursor)
            target = torch.rand(()) + random.random()
            loss = (params[0] - target).square().sum()
            loss.backward()
            optimizer.step()
            trainer.assert_excluded_unchanged(actor, excluded)
            return float(loss.detach()), group, window["id"], cursor + 1

        update(0)
        checkpoint = {
            "scales": {k: trainer.cpu(m.scale_offset) for k, m in actor.linears.items()},
            "optimizer": trainer.cpu(optimizer.state_dict()),
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cursor": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact.pt"
            torch.save(checkpoint, path)
            restored = torch.load(path, weights_only=False)
        self.assertTrue(trainer.equal(checkpoint, restored))
        expected = update(1)
        expected_parameter = params[0].detach().clone()
        expected_optimizer = trainer.cpu(optimizer.state_dict())
        with torch.no_grad():
            for path, value in restored["scales"].items():
                actor.linears[path].scale_offset.copy_(value)
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"])
        random.setstate(restored["python_rng"])
        actual = update(restored["cursor"])
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(params[0], expected_parameter))
        self.assertTrue(trainer.equal(optimizer.state_dict(), expected_optimizer))

    def test_original_numeric_semantics_unchanged_but_changed_functions_rejected(self):
        original = (PROJECT / "scripts/w1a8_generalization/modern_train.py").read_text()
        current = (PROJECT / "scripts/w1a8_generalization/modern_train_protected.py").read_text()
        self.assertEqual(trainer.forward_semantics(original), trainer.forward_semantics(current))
        changed = current.replace('token = window["labels"][d]', 'token = window["seed_token"]')
        self.assertNotEqual(trainer.forward_semantics(original), trainer.forward_semantics(changed))

    def test_reference_ancestry_mismatch_cannot_be_silently_rebased(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "original.py"
            p.write_text("def forward(x): return x\ndef inspect(x): return x\n")
            current = Path(temporary) / "current.py"
            current.write_bytes(p.read_bytes())
            windows = [
                {"id": "fit", "role": "fit", "labels": [1], "native_prefix": 1, "max_depth": 1}
            ]
            reference = {
                "rows": [
                    {
                        "id": "fit",
                        "role": "fit",
                        "labels": [1],
                        "native_prefix": 1,
                        "actor_prefix": 1,
                        "protected": [True],
                        "margins": [1.0],
                    }
                ]
            }
            keys = (
                "manifest",
                "actor",
                "view_source",
                "optimizer",
                "seed",
                "batch_windows",
                "margin_cap",
                "horizon",
                "context",
                "runtime",
                "precision",
            )
            contract = {key: "same" for key in keys}
            old_contract = {**contract, "trainer": trainer.file_pin(p)}
            got = trainer.validate_original_reference(
                reference, old_contract, contract, windows, p, current
            )
            self.assertIn("fit", got)
            for key in ("manifest", "actor", "view_source"):
                changed = copy.deepcopy(contract)
                changed[key] = "different"
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    trainer.validate_original_reference(
                        reference, old_contract, changed, windows, p, current
                    )
            p.write_text(p.read_text() + "# changed pinned source\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                trainer.validate_original_reference(
                    reference, old_contract, contract, windows, p, current
                )


if __name__ == "__main__":
    unittest.main()
