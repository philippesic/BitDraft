"""CPU contracts for declared native-bank migration; no real cases or CUDA."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/w1a8_generalization"))
SPEC = importlib.util.spec_from_file_location(
    "modern_train_native_replay",
    PROJECT / "scripts/w1a8_generalization/modern_train_native_replay.py",
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def fit_fixture():
    windows, refs = [], {}
    for identifier, group, prefix, margins, actor_prefix in (
        ("a:0", "a", 1, [-0.4, -0.5], 0),
        ("b:0", "b", 2, [0.02, 5.0], 2),
        ("b:1", "b", 0, [-1.0, -1.0], 0),
    ):
        labels = [11, 13]
        protections = [
            {"proposal_depth": d, "token": labels[d], "response_index": 3 + d}
            for d in range(prefix)
        ]
        windows.append(
            {
                "id": identifier,
                "group_id": group,
                "role": "fit",
                "labels": labels,
                "native_prefix": prefix,
                "max_depth": 2,
                "response_label_start": 3,
                "native_protections": protections,
            }
        )
        refs[identifier] = {
            "id": identifier,
            "role": "fit",
            "labels": labels,
            "native_prefix": prefix,
            "actor_prefix": actor_prefix,
            "protected": [d < min(prefix, actor_prefix) for d in range(2)],
            "margins": margins,
            "supported": [True, True],
        }
    return windows, refs


def warm_fixture():
    keys = (
        "manifest",
        "actor",
        "view_source",
        "optimizer",
        "retention",
        "seed",
        "horizon",
        "batch_windows",
        "margin_cap",
        "optimizer_scope",
        "replay_windows_per_update",
        "context",
        "runtime",
        "precision",
        "reference",
        "reference_freeze",
        "original_trainer",
        "numeric_forward_semantics_sha256",
        "original_reference_objective",
        "ce_schedule",
        "optimized_scale_paths",
        "native_evaluation_masks",
        "readout",
    )
    expected = {key: "same-" + key for key in keys}
    expected.update(retention=10.0, optimizer_scope="all", horizon=19200)
    helper = {"sha256": "a" * 64, "bytes": 123}
    schedule = {"original": "frozen actor-intersection bank"}
    old = {
        **expected,
        "trainer": helper,
        "objective": trainer.V1_OBJECTIVE,
        "protected_schedule": schedule,
    }
    packet = {
        "cursor": 1920,
        "status": "RECOVERY",
        "contract": old,
        "optimizer": {
            "param_groups": [
                {"params": list(range(9)), "lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8}
            ]
        },
        "exposure": {"ce_windows": 3840, "replay_windows": 1920},
        "costs": {"ce": 1.0, "replay": 1.0},
        "trainer_seconds": 2.0,
        "scales": "same scales",
        "rng": "same CPU RNG",
        "cuda_rng": "same CUDA RNG",
        "python_rng": "same Python RNG",
    }
    return packet, expected, schedule, helper


class NativeReplayMigrationTests(unittest.TestCase):
    def test_all_native_fit_successes_enter_bank_with_floor_and_cap(self):
        windows, refs = fit_fixture()
        before = copy.deepcopy(refs)
        expanded = trainer.native_fit_references(windows, refs, {11, 13})
        self.assertEqual(sum(sum(x["protected"]) for x in expanded.values()), 3)
        self.assertEqual(expanded["a:0"]["protected"], [True, False])
        self.assertEqual(expanded["a:0"]["margins"][0], 0.1)
        self.assertEqual(expanded["b:0"]["margins"], [0.1, 2.0])
        self.assertEqual(refs, before)
        self.assertTrue(all(x["role"] == "fit" for x in expanded.values()))

    def test_dev_unsupported_and_wrong_response_ancestry_are_rejected(self):
        windows, refs = fit_fixture()
        bad = copy.deepcopy(windows)
        bad[0]["role"] = "development"
        with self.assertRaisesRegex(ValueError, "fit windows only"):
            trainer.native_fit_references(bad, refs, {11, 13})
        for change in ("token", "response_index", "proposal_depth"):
            bad = copy.deepcopy(windows)
            bad[0]["native_protections"][0][change] = 999
            with self.assertRaises(ValueError):
                trainer.native_fit_references(bad, refs, {11, 13})
        with self.assertRaisesRegex(ValueError, "vocabulary support"):
            trainer.native_fit_references(windows, refs, {11})
        refs["a:0"]["supported"][0] = False
        with self.assertRaisesRegex(ValueError, "vocabulary support"):
            trainer.native_fit_references(windows, refs, {11, 13})

    def test_native_schedule_includes_previously_unmatched_group_and_resumes_global_cursor(self):
        windows, refs = fit_fixture()
        expanded = trainer.native_fit_references(windows, refs, {11, 13})
        bank = trainer.native_schedule(windows, expanded)
        self.assertEqual({bank.at(i)[0] for i in range(1920, 1922)}, {"a", "b"})
        self.assertEqual({bank.at(i)[1]["id"] for i in range(1920, 1940)}, {"a:0", "b:0"})
        restored = trainer.native_schedule(list(reversed(windows)), expanded)
        self.assertEqual(bank.contract, restored.contract)
        self.assertEqual(bank.at(1920), restored.at(1920))
        self.assertNotIn("groups_without_actor_matched_protections", bank.contract)

    def test_exact_v1_all_scale_checkpoint_is_accepted_without_rewriting_state(self):
        packet, expected, schedule, helper = warm_fixture()
        unchanged = copy.deepcopy(packet)
        trainer.validate_v1_warm_start(packet, expected, schedule, helper)
        self.assertEqual(packet, unchanged)

    def test_warm_start_rejects_changed_scope_source_contract_optimizer_and_cursor(self):
        packet, expected, schedule, helper = warm_fixture()
        for key in ("manifest", "actor", "view_source", "ce_schedule", "precision", "context"):
            with self.subTest(key=key):
                bad = copy.deepcopy(packet)
                bad["contract"][key] = "different"
                with self.assertRaisesRegex(ValueError, "contract differs"):
                    trainer.validate_v1_warm_start(bad, expected, schedule, helper)
        for kind in ("cursor", "scope", "lambda", "helper", "optimizer", "bank"):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(packet)
                if kind == "cursor":
                    bad["cursor"] = 1919
                elif kind == "scope":
                    bad["contract"]["optimizer_scope"] = "head"
                elif kind == "lambda":
                    bad["contract"]["retention"] = 0.0
                elif kind == "helper":
                    bad["contract"]["trainer"]["sha256"] = "0" * 64
                elif kind == "optimizer":
                    bad["optimizer"]["param_groups"][0]["lr"] = 0.1
                else:
                    bad["contract"]["protected_schedule"] = {"modified": True}
                with self.assertRaises(ValueError):
                    trainer.validate_v1_warm_start(bad, expected, schedule, helper)

    def test_interrupted_new_stage_validates_original_status_without_rewinding_loaded_state(self):
        original, expected, schedule, helper = warm_fixture()
        origin = {
            "original_contract": original["contract"],
            "global_start_cursor": 1920,
            "initial_costs": original["costs"],
            "initial_exposure": original["exposure"],
            "initial_trainer_seconds": original["trainer_seconds"],
            "original_checkpoint_status": original["status"],
        }
        current = copy.deepcopy(original)
        current.update(
            cursor=3000, status="INTERRUPTED", scales="new-stage scales", rng="new-stage RNG"
        )
        validation = trainer.original_validation_packet(current, origin)
        trainer.validate_v1_warm_start(validation, expected, schedule, helper)
        self.assertEqual(current["cursor"], 3000)
        self.assertEqual(current["status"], "INTERRUPTED")
        self.assertEqual(current["scales"], "new-stage scales")
        self.assertEqual(current["rng"], "new-stage RNG")
        self.assertEqual(validation["status"], "RECOVERY")
        self.assertIs(validation["optimizer"], current["optimizer"])

    def test_stage_exposure_is_incremental_while_cumulative_state_stays_intact(self):
        origin = {
            "global_start_cursor": 1920,
            "initial_trainer_seconds": 100.0,
            "initial_costs": {"ce": 60.0, "replay": 40.0},
            "initial_exposure": {"ce_windows": 3840, "replay_windows": 1920},
        }
        totals = trainer.stage_totals(
            3840,
            {"ce": 120.0, "replay": 80.0},
            {"ce_windows": 7680, "replay_windows": 3840},
            200.0,
            origin,
        )
        self.assertEqual(totals["stage_updates"], 1920)
        self.assertEqual(totals["stage_exposure"], {"ce_windows": 3840, "replay_windows": 1920})
        self.assertEqual(totals["stage_trainer_seconds"], 100.0)
        with self.assertRaises(ValueError):
            trainer.stage_totals(
                1920,
                {"ce": 59.0, "replay": 40.0},
                {"ce_windows": 3840, "replay_windows": 1920},
                99.0,
                origin,
            )

    def test_forward_semantics_match_original_and_pinned_v1_helper(self):
        source = (PROJECT / "scripts/w1a8_generalization/modern_train_native_replay.py").read_text()
        for name in ("modern_train.py", "modern_train_protected.py"):
            original = (PROJECT / "scripts/w1a8_generalization" / name).read_text()
            self.assertEqual(trainer.forward_semantics(source), trainer.forward_semantics(original))


if __name__ == "__main__":
    unittest.main()
