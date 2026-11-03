"""Synthetic qualification protocol fixtures; no real checkpoint/model or GPU reads."""

from __future__ import annotations

# ruff: noqa: E402
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/w1a8_generalization"))
import modern_selected_head_qualify as qualify
import modern_train_selected_head as trainer


class HeadQualificationTests(unittest.TestCase):
    def test_failed_or_stopped_qualification_preserves_checkpoint_and_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "synthetic-state.pt"
            checkpoint.write_bytes(b"synthetic checkpoint fixture")
            context = {
                "save": Mock(return_value=checkpoint),
                "write": Mock(),
                "args": SimpleNamespace(out=root),
            }
            with (
                patch.object(
                    qualify, "_qualify_loaded", side_effect=RuntimeError("stop requested")
                ),
                self.assertRaisesRegex(RuntimeError, "stop requested"),
            ):
                qualify.qualify_loaded(context)
            context["save"].assert_called_once_with("INTERRUPTED")
            self.assertEqual(context["write"].call_args.kwargs, {"recovery": True})
            self.assertFalse(context["write"].call_args.args[1]["native_admission_passed"])

    def test_bit_comparison_distinguishes_signed_zero_and_ignores_no_tensor_bits(self):
        positive = {"tensor": torch.tensor([0.0, 1.0]), "tuple": (37, True)}
        negative = copy.deepcopy(positive)
        negative["tensor"][0] = -0.0
        self.assertFalse(qualify.bit_equal(positive, negative))
        self.assertTrue(qualify.bit_equal(positive, copy.deepcopy(positive)))
        self.assertFalse(qualify.bit_equal(positive, {**positive, "tuple": [37, True]}))

    def test_qualifier_rejects_dev_or_replay_dev_disabled_head_and_wrong_parent(self):
        context = {
            "ce_fit": [{"role": "fit"}],
            "replay_windows": [{"role": "fit"}],
            "selected": SimpleNamespace(learn_signs=True),
            "snapshot": lambda: {"cursor": 3840},
        }
        qualify.fit_only_context(context)
        for kind in ("ce_fit", "replay_windows", "disabled", "cursor"):
            bad = copy.copy(context)
            if kind in ("ce_fit", "replay_windows"):
                bad[kind] = [{"role": "development"}]
            elif kind == "disabled":
                bad["selected"] = SimpleNamespace(learn_signs=False)
            else:
                bad["snapshot"] = lambda: {"cursor": 3841}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                qualify.fit_only_context(bad)

    def test_actual_gradient_evidence_requires_full_selected_row_count_finite_nonzero(self):
        good = {
            "shape": [512, 2560],
            "finite": True,
            "nonzero_elements": 100,
            "nonzero_rows": 20,
            "norm": 0.2,
        }
        qualify.validate_gradient(good)
        for key, value in (
            ("shape", [511, 2560]),
            ("finite", False),
            ("nonzero_elements", 0),
            ("norm", 0.0),
            ("norm", float("nan")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualify.validate_gradient({**good, key: value})

    def test_zero_export_is_parent_byte_identity_not_initializer_identity(self):
        expected = {"path": "/parent/model.gguf", "bytes": 40, "sha256": "a" * 64}
        actual = {**expected, "path": "/qualification/zero.gguf"}
        self.assertTrue(qualify.validate_zero_export(actual, expected)["byte_identity"])
        with self.assertRaises(ValueError):
            qualify.validate_zero_export({**actual, "sha256": "b" * 64}, expected)

    def test_two_unique_updates_repeat_exact_compact_adam_rng_without_forced_flips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latent = torch.nn.Parameter(torch.full((512, 3), 0.5))
            scales = [torch.nn.Parameter(torch.full((1,), 0.02)) for _ in range(9)]
            selected = SimpleNamespace(
                latent=latent,
                learn_signs=True,
                rows=torch.arange(512),
                head=SimpleNamespace(latent_sign=latent.detach().clone()),
            )
            optimizer = torch.optim.Adam(
                [{"params": scales, "lr": 1e-5}, {"params": [latent], "lr": 1e-3}]
            )
            cursor = 3840
            physical = 0

            def snapshot():
                updates = cursor - 3840
                return {
                    "selected_state": {
                        "selected_latent": latent.detach().clone(),
                        "scale_offsets": {str(i): p.detach().clone() for i, p in enumerate(scales)},
                    },
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "cursor": cursor,
                    "rng": torch.get_rng_state().clone(),
                    "cuda_rng": torch.zeros(8, dtype=torch.uint8),
                    "python_rng": random.getstate(),
                    "exposure": {"ce_windows": 2 * cursor},
                    "ce_stage_exposure": {"old": updates, "new": updates},
                    "head_stage": {
                        "global_start_cursor": 3840,
                        "stage_updates": updates,
                        "stage_exposure": {"ce_windows": 2 * updates},
                        "ce_stage_exposure": {"old": updates, "new": updates},
                        "stage_trainer_seconds": physical / 100,
                    },
                }

            def update(*, capture_gradient):
                nonlocal cursor, physical
                optimizer.zero_grad(set_to_none=True)
                x = torch.randn(2, 3)
                target = torch.randint(0, 512, (2,))
                logits = torch.nn.functional.linear(x, latent)
                loss = torch.nn.functional.cross_entropy(logits, target) * (1 + random.random())
                loss = loss + sum(p.square().sum() for p in scales)
                loss.backward()
                gradients = trainer.gradient_metrics(selected)
                torch.nn.utils.clip_grad_norm_(scales + [latent], 1.0)
                optimizer.step()
                with torch.no_grad():
                    latent.clamp_(-1, 1)
                cursor += 1
                physical += 1
                return {
                    "head_gradient": gradients,
                    "loss": float(loss.detach()),
                    "update_seconds": 0.01,
                    "ce_forward_seconds": 0.003,
                    "replay_forward_seconds": 0.002,
                }

            def restore(state):
                nonlocal cursor
                with torch.no_grad():
                    latent.copy_(state["selected_state"]["selected_latent"])
                    for i, p in enumerate(scales):
                        p.copy_(state["selected_state"]["scale_offsets"][str(i)])
                optimizer.load_state_dict(state["optimizer"])
                cursor = state["cursor"]
                torch.set_rng_state(state["rng"])
                random.setstate(state["python_rng"])

            def save(status):
                path = root / f"checkpoint-{cursor}.pt"
                trainer.guarded_checkpoint(path, snapshot(), Mock())
                return path

            context = {
                "selected": selected,
                "snapshot": snapshot,
                "update": update,
                "save": save,
                "restore": restore,
            }
            result = qualify.resume_two_updates(context)
            self.assertTrue(result["exact_compact_Adam_RNG_resume"])
            self.assertEqual(result["unique_updates"], 2)
            self.assertEqual(result["physical_updates"], physical)
            self.assertEqual(cursor, 3842)
            self.assertEqual(trainer.sign_metrics(selected)["hard_sign_flips"], 0)
            self.assertGreater(trainer.sign_metrics(selected)["changed_latents"], 0)
            self.assertEqual(result["physical_update_seconds"], 0.03)


if __name__ == "__main__":
    unittest.main()
