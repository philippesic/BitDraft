"""Small CPU migration/clip/resume contracts; no real model, data or CUDA."""

from __future__ import annotations

# ruff: noqa: E402
import ast
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(PROJECT / "src"),
    str(PROJECT / "tests"),
    str(PROJECT / "scripts/w1a8_generalization"),
    "/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py",
]
import modern_actor
import modern_selected_sign as signs
import modern_train_selected_sign as trainer
from test_w1a8_modern_selected_sign import small_actor


def optimizer_options(optimizer):
    return [
        {k: v for k, v in group.items() if k != "params"}
        for group in optimizer.state_dict()["param_groups"]
    ]


def toy_loss(actor):
    values = torch.linspace(-1.0, 2.0, 35).unsqueeze(0)
    return sum(
        (index + 1) * module(values).square().sum() * 100
        for index, module in enumerate(actor.linears.values())
    )


class SelectedSignTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        (root / "old").mkdir()
        (root / "new").mkdir()
        self.old = small_actor(root / "old")
        modern_actor.enable_scale_only(self.old)
        self.old_optimizer = torch.optim.Adam(
            self.old.scale_parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8
        )
        sum((index + 1) * p.sum() for index, p in enumerate(self.old.scale_parameters())).backward()
        torch.nn.utils.clip_grad_norm_(self.old.scale_parameters(), 1.0)
        self.old_optimizer.step()
        for module in self.old.linears.values():
            module.project_scales_()
        # Synthetic mature moments exercise the inherited Adam bias counter.
        for entry in self.old_optimizer.state.values():
            entry["step"].fill_(trainer.WARM_CURSOR)
        self.saved = {
            "cursor": trainer.WARM_CURSOR,
            "scales": modern_actor.collect_scale_state(self.old),
            "optimizer": copy.deepcopy(self.old_optimizer.state_dict()),
            "contract": {"optimized_scale_paths": list(trainer.CANDIDATE_D_BASE_TO_PATH.values())},
        }
        self.actor = small_actor(root / "new")
        signs.enable_selected_sign(self.actor)
        self.optimizer = torch.optim.Adam(
            signs.selected_parameter_groups(self.actor, sign_lr=1e-3, scale_lr=1e-5),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def migrate(self):
        return trainer.migrate_v1_scale_state_and_optimizer(self.actor, self.optimizer, self.saved)

    def test_migration_joins_arbitrary_source_parameter_ids_by_declared_path(self):
        old_group = self.saved["optimizer"]["param_groups"][0]
        old_ids = list(old_group["params"])
        new_ids = [713, 41, 900, 2, 65, 18, 222, 87, 5]
        self.saved["optimizer"]["state"] = {
            new: self.saved["optimizer"]["state"][old]
            for old, new in zip(old_ids, new_ids, strict=True)
        }
        old_group["params"] = new_ids
        report = self.migrate()
        self.assertEqual([r["source_parameter"] for r in report["path_mapping"]], new_ids)
        for base, path in trainer.CANDIDATE_D_BASE_TO_PATH.items():
            self.assertTrue(
                torch.equal(
                    self.actor.linears[path].scale_offset,
                    self.saved["scales"]["scale_offsets"][base],
                )
            )
            old_param = self.old.linears[path].scale_offset
            new_param = self.actor.linears[path].scale_offset
            self.assertTrue(
                trainer.checkpoint_equal(
                    self.old_optimizer.state[old_param], self.optimizer.state[new_param]
                )
            )
        self.assertNotIn(self.actor.linears[signs.SELECTED_PATH].latent_sign, self.optimizer.state)

    def test_late_corruption_and_changed_options_reject_without_partial_migration(self):
        for kind in (
            "ancestry",
            "scale_nan",
            "duplicate_path",
            "missing_moment",
            "moment_nan",
            "moment_shape",
            "step",
            "lr",
        ):
            with self.subTest(kind=kind):
                saved = copy.deepcopy(self.saved)
                saved["scales"]["scale_offsets"]["fc"].fill_(1)
                last = saved["optimizer"]["param_groups"][0]["params"][-1]
                if kind == "ancestry":
                    saved["scales"]["ancestry"]["initializer"]["sha256"] = "0" * 64
                elif kind == "scale_nan":
                    saved["scales"]["scale_offsets"]["output"][0] = float("nan")
                elif kind == "duplicate_path":
                    saved["contract"]["optimized_scale_paths"][-1] = "fc"
                elif kind == "missing_moment":
                    saved["optimizer"]["state"].pop(last)
                elif kind == "moment_nan":
                    saved["optimizer"]["state"][last]["exp_avg"][0] = float("nan")
                elif kind == "moment_shape":
                    saved["optimizer"]["state"][last]["exp_avg_sq"] = torch.zeros(3)
                elif kind == "step":
                    saved["optimizer"]["state"][last]["step"].fill_(1)
                else:
                    saved["optimizer"]["param_groups"][0]["lr"] = 0.01
                with self.assertRaises(ValueError):
                    trainer.migrate_v1_scale_state_and_optimizer(self.actor, self.optimizer, saved)
                self.assertTrue(torch.equal(self.actor.linears["fc"].scale_offset, torch.zeros(4)))
                self.assertFalse(self.optimizer.state)

    def test_false_mode_matches_original_scale_clip_and_adam_trajectory_exactly(self):
        self.migrate()
        initial = self.actor.linears[signs.SELECTED_PATH].latent_sign.detach().clone()
        parameters = tuple(p for group in self.optimizer.param_groups for p in group["params"])
        for _ in range(3):
            self.old.drafter.zero_grad(set_to_none=True)
            self.actor.drafter.zero_grad(set_to_none=True)
            expected_loss, actual_loss = toy_loss(self.old), toy_loss(self.actor)
            self.assertTrue(torch.equal(expected_loss, actual_loss))
            expected_loss.backward()
            actual_loss.backward()
            diagnostic = trainer.prepare_sign_gradient(self.actor, False)
            self.assertGreater(diagnostic["norm_before_clip"], 0)
            old_norm = torch.nn.utils.clip_grad_norm_(self.old.scale_parameters(), 1.0)
            new_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            self.assertTrue(torch.equal(old_norm, new_norm))
            self.assertGreater(float(old_norm), 1)
            self.old_optimizer.step()
            self.optimizer.step()
            signs.clamp_selected_latents_(self.actor)
            for actor in (self.old, self.actor):
                for module in actor.linears.values():
                    module.project_scales_()
            trainer.assert_sign_mode(self.actor, self.optimizer, False, full=True)
            self.assertTrue(
                torch.equal(self.actor.linears[signs.SELECTED_PATH].latent_sign, initial)
            )
            for path in trainer.CANDIDATE_D_BASE_TO_PATH.values():
                old = self.old.linears[path].scale_offset
                new = self.actor.linears[path].scale_offset
                self.assertTrue(torch.equal(old, new))
                self.assertTrue(
                    trainer.checkpoint_equal(
                        self.old_optimizer.state[old], self.optimizer.state[new]
                    )
                )

    def test_true_mode_keeps_actual_ste_gradient_and_can_flip_at_explicit_toy_lr(self):
        # LR=1 is a toy STE test, not this trainer's production LR contract.
        module = self.actor.linears[signs.SELECTED_PATH]
        optimizer = torch.optim.Adam(
            signs.selected_parameter_groups(self.actor, sign_lr=1.0, scale_lr=1e-5)
        )
        before = module.latent_sign.detach().clone()
        module(torch.ones(1, 35)).sum().backward()
        diagnostic = trainer.prepare_sign_gradient(self.actor, True)
        self.assertTrue(diagnostic["applied"])
        self.assertGreater(diagnostic["norm_before_clip"], 0)
        optimizer.step()
        signs.clamp_selected_latents_(self.actor)
        self.assertGreater(int(before.ge(0).ne(module.latent_sign.detach().ge(0)).sum()), 0)
        self.assertIn(module.latent_sign, optimizer.state)

    def resume_fixture(self):
        self.migrate()
        initial_costs = {
            "ce_forward_seconds": 1.0,
            "replay_forward_seconds": 2.0,
            "shared_backward_optimizer_seconds": 3.0,
        }
        origin = {
            "global_start_cursor": trainer.WARM_CURSOR,
            "initial_costs": initial_costs,
            "initial_exposure": {
                "ce_windows": 3840,
                "replay_windows": 1920,
                "ce_supported_labels": 5000,
                "replay_protected_tokens": 3000,
            },
            "initial_trainer_seconds": 6.0,
        }
        contract = {
            "selected_optimizer": {
                "learn_signs": False,
                "param_group_options": optimizer_options(self.optimizer),
            },
            "warm_start": origin,
        }
        saved = {
            "schema": trainer.CHECKPOINT_SCHEMA,
            "contract": contract,
            "status": "RECOVERY",
            "selected_state": signs.collect_selected_state(self.actor),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "cursor": trainer.WARM_CURSOR,
            "selected_stage": trainer.new_selected_stage(),
            "costs": initial_costs,
            "exposure": copy.deepcopy(origin["initial_exposure"]),
            "trainer_seconds": 6.0,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.zeros(16, dtype=torch.uint8),
            "python_rng": random.getstate(),
        }
        return saved, contract

    def test_strict_resume_preserves_all_state_without_mutation(self):
        saved, contract = self.resume_fixture()
        before = copy.deepcopy(saved)
        trainer.validate_selected_resume(self.actor, saved, contract, False)
        self.assertTrue(trainer.checkpoint_equal(saved, before))

    def test_true_mode_resume_retains_first_sign_moments_and_does_not_require_flips(self):
        saved, contract = self.resume_fixture()
        contract["selected_optimizer"]["learn_signs"] = True
        self.actor.drafter.zero_grad(set_to_none=True)
        toy_loss(self.actor).backward()
        trainer.prepare_sign_gradient(self.actor, True)
        parameters = tuple(p for group in self.optimizer.param_groups for p in group["params"])
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        signs.clamp_selected_latents_(self.actor)
        for module in self.actor.linears.values():
            module.project_scales_()
        saved["cursor"] += 1
        saved["selected_stage"].update(updates=1, sign_optimizer_updates=1)
        saved["selected_state"] = signs.collect_selected_state(self.actor)
        saved["optimizer"] = copy.deepcopy(self.optimizer.state_dict())
        saved["exposure"]["ce_windows"] += 2
        saved["exposure"]["replay_windows"] += 1
        trainer.validate_selected_resume(self.actor, saved, contract, True)
        self.assertEqual(signs.selected_sign_metrics(self.actor)["hard_sign_flips"], 0)
        sign_id = saved["optimizer"]["param_groups"][0]["params"][0]
        self.assertEqual(float(saved["optimizer"]["state"][sign_id]["step"]), 1)

    def test_checkpoint_comparison_is_byte_exact_including_signed_zero(self):
        first = {"offset": torch.tensor([0.0, 1.0]), "step": torch.tensor(1920.0)}
        self.assertTrue(trainer.checkpoint_equal(first, copy.deepcopy(first)))
        changed = copy.deepcopy(first)
        changed["offset"][0] = -0.0
        self.assertTrue(torch.equal(first["offset"], changed["offset"]))
        self.assertFalse(trainer.checkpoint_equal(first, changed))

    def test_resume_rejects_schema_mode_sources_moments_rng_and_accounting_changes(self):
        saved, contract = self.resume_fixture()
        for kind in (
            "schema",
            "source",
            "mode",
            "lr",
            "moment",
            "sign_moment",
            "counter",
            "cost",
            "exposure",
            "rng",
            "sign_value",
        ):
            with self.subTest(kind=kind):
                bad = copy.deepcopy(saved)
                if kind == "schema":
                    bad["schema"] = "other"
                elif kind == "source":
                    bad["contract"]["different_source"] = True
                elif kind == "mode":
                    bad["contract"]["selected_optimizer"]["learn_signs"] = True
                elif kind == "lr":
                    bad["optimizer"]["param_groups"][0]["lr"] = 0.1
                elif kind == "moment":
                    index = bad["optimizer"]["param_groups"][1]["params"][-1]
                    bad["optimizer"]["state"][index]["exp_avg"][0] = float("nan")
                elif kind == "sign_moment":
                    index = bad["optimizer"]["param_groups"][0]["params"][0]
                    bad["optimizer"]["state"][index] = {}
                elif kind == "counter":
                    bad["selected_stage"]["updates"] = 1
                elif kind == "cost":
                    bad["selected_stage"]["optimizer_step_seconds"] = -1
                elif kind == "exposure":
                    bad["exposure"]["ce_windows"] -= 1
                elif kind == "rng":
                    bad["rng"] = torch.zeros(3, dtype=torch.float32)
                else:
                    bad["selected_state"]["latent_signs"][signs.SELECTED_BASES[0]][0, 0] *= -1
                with self.assertRaises(ValueError):
                    trainer.validate_selected_resume(self.actor, bad, contract, False)

    def test_stage_detail_does_not_replace_inherited_physical_totals(self):
        stage = trainer.new_selected_stage()
        stage.update(updates=12, sign_optimizer_updates=12, optimizer_step_seconds=0.5)
        trainer.validate_selected_stage(stage, 1932, True)
        with self.assertRaises(ValueError):
            trainer.validate_selected_stage(stage, 1932, False)
        with self.assertRaises(ValueError):
            trainer.validate_selected_stage({**stage, "backward_seconds": float("nan")}, 1932, True)
        origin = {
            "global_start_cursor": 1920,
            "initial_costs": {"ce": 1.0},
            "initial_exposure": {"ce_windows": 3840},
            "initial_trainer_seconds": 10.0,
        }
        totals = trainer.stage_totals(1932, {"ce": 2.0}, {"ce_windows": 3864}, 11.0, origin)
        self.assertEqual(totals["stage_updates"], 12)
        self.assertEqual(totals["stage_trainer_seconds"], 1)
        self.assertEqual(totals["stage_exposure"]["ce_windows"], 24)

    def test_forward_and_inspect_are_ast_identical_to_immutable_original(self):
        own = (PROJECT / "scripts/w1a8_generalization/modern_train_selected_sign.py").read_text()
        for name in (
            "modern_train.py",
            "modern_train_protected.py",
            "modern_train_native_replay.py",
        ):
            original = (PROJECT / "scripts/w1a8_generalization" / name).read_text()
            self.assertEqual(trainer.forward_semantics(own), trainer.forward_semantics(original))
        tree = ast.parse(own)
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "prepare_sign_gradient"
                for node in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()
