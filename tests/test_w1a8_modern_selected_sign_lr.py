"""Small CPU mature-state LR migration/resume tests; no actual checkpoint/model."""

from __future__ import annotations

# ruff: noqa: E402
import ast
import copy
import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "tests"),
    str(ROOT / "scripts/w1a8_generalization"),
    "/home/philip/binary-eagle-decoding/third_party/llama.cpp/gguf-py",
]
import modern_selected_sign as signs
import modern_train_selected_sign as original
import modern_train_selected_sign_lr as trainer
import test_w1a8_modern_selected_sign_training as fixtures
from test_w1a8_modern_selected_sign import small_actor


class MatureLearningRateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SelectedSignTrainingTests(
            "test_stage_detail_does_not_replace_inherited_physical_totals"
        )
        self.fixture.setUp()
        self.actor, self.optimizer = self.fixture.actor, self.fixture.optimizer
        parent, contract = self.fixture.resume_fixture()
        self.actor.drafter.zero_grad(set_to_none=True)
        fixtures.toy_loss(self.actor).backward()
        original.prepare_sign_gradient(self.actor, True)
        params = tuple(p for group in self.optimizer.param_groups for p in group["params"])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.optimizer.step()
        signs.clamp_selected_latents_(self.actor)
        for module in self.actor.linears.values():
            module.project_scales_()
        # Synthetic mature counters; these fixtures make no actual3840 claim.
        for path, module in self.actor.linears.items():
            self.optimizer.state[module.scale_offset]["step"].fill_(3840)
        self.optimizer.state[self.actor.linears[signs.SELECTED_PATH].latent_sign]["step"].fill_(
            1920
        )
        contract.update(
            trainer=trainer.original_source_pin(), horizon=19200, native_fit_bank_tokens=356
        )
        contract["selected_optimizer"].update(
            learn_signs=True,
            sign_lr=1e-3,
            scale_lr=1e-5,
            param_group_options=fixtures.optimizer_options(self.optimizer),
        )
        parent.update(
            cursor=3840,
            contract=contract,
            selected_state=signs.collect_selected_state(self.actor),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            trainer_seconds=166.0,
            costs={
                "ce_forward_seconds": 51.0,
                "replay_forward_seconds": 52.0,
                "shared_backward_optimizer_seconds": 63.0,
            },
        )
        parent["selected_stage"].update(updates=1920, sign_optimizer_updates=1920)
        parent["exposure"].update(
            ce_windows=7680,
            replay_windows=3840,
            ce_supported_labels=10000,
            replay_protected_tokens=6000,
        )
        self.parent = parent
        self.parent_pin = {
            "path": str(Path(self.fixture.temporary.name) / "synthetic-parent.pt"),
            "sha256": trainer.PARENT_CHECKPOINT_SHA256,
            "bytes": trainer.PARENT_CHECKPOINT_BYTES,
        }

    def tearDown(self):
        self.fixture.tearDown()

    def destination(self, name):
        path = Path(self.fixture.temporary.name) / name
        path.mkdir()
        actor = small_actor(path)
        signs.enable_selected_sign(actor)
        optimizer = torch.optim.Adam(
            signs.selected_parameter_groups(actor, sign_lr=1e-3, scale_lr=1e-5),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return actor, optimizer

    def test_migration_changes_only_declared_lr_preserving_all_moments_model_and_rng(self):
        for index, lr in enumerate((1e-3, 3e-3)):
            actor, optimizer = self.destination("target-" + str(index))
            before = copy.deepcopy(self.parent)
            rng, python_rng = torch.get_rng_state().clone(), random.getstate()
            report = trainer.migrate_mature_state(
                actor, optimizer, self.parent, self.parent_pin, lr, self.parent["contract"]
            )
            expected = copy.deepcopy(self.parent["optimizer"])
            expected["param_groups"][0]["lr"] = lr
            self.assertTrue(original.checkpoint_equal(optimizer.state_dict(), expected))
            self.assertTrue(
                original.checkpoint_equal(
                    signs.collect_selected_state(actor), self.parent["selected_state"]
                )
            )
            self.assertTrue(original.checkpoint_equal(self.parent, before))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(report["sign_moments_preserved"])

    def test_migration_parent_source_mode_identity_and_state_corruption_fail_before_mutation(self):
        for index, kind in enumerate(("source", "mode", "step", "state", "pin")):
            actor, optimizer = self.destination("bad-" + str(index))
            parent, pin = copy.deepcopy(self.parent), copy.deepcopy(self.parent_pin)
            if kind == "source":
                parent["contract"]["trainer"]["sha256"] = "0" * 64
            elif kind == "mode":
                parent["contract"]["selected_optimizer"]["learn_signs"] = False
            elif kind == "step":
                identifier = parent["optimizer"]["param_groups"][0]["params"][0]
                parent["optimizer"]["state"][identifier]["step"].fill_(1)
            elif kind == "state":
                parent["selected_state"]["scale_offsets"]["output"][0] = float("nan")
            else:
                pin["sha256"] = "0" * 64
            before = signs.collect_selected_state(actor)
            with self.assertRaises(ValueError):
                trainer.migrate_mature_state(
                    actor, optimizer, parent, pin, 3e-3, self.parent["contract"]
                )
            self.assertTrue(original.checkpoint_equal(signs.collect_selected_state(actor), before))
            self.assertFalse(optimizer.state)

    def test_control_next_updates_are_exact_original_ordinary_resume_trajectory(self):
        actor, optimizer = self.destination("control")
        trainer.migrate_mature_state(
            actor, optimizer, self.parent, self.parent_pin, 1e-3, self.parent["contract"]
        )
        for _ in range(3):
            self.actor.drafter.zero_grad(set_to_none=True)
            actor.drafter.zero_grad(set_to_none=True)
            first, second = fixtures.toy_loss(self.actor), fixtures.toy_loss(actor)
            self.assertTrue(torch.equal(first, second))
            first.backward()
            second.backward()
            for a, o in ((self.actor, self.optimizer), (actor, optimizer)):
                original.prepare_sign_gradient(a, True)
                params = tuple(p for g in o.param_groups for p in g["params"])
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                o.step()
                signs.clamp_selected_latents_(a)
                for module in a.linears.values():
                    module.project_scales_()
            self.assertTrue(
                original.checkpoint_equal(
                    signs.collect_selected_state(self.actor), signs.collect_selected_state(actor)
                )
            )
            self.assertTrue(
                original.checkpoint_equal(self.optimizer.state_dict(), optimizer.state_dict())
            )

    def own_packet(self, sign_lr=3e-3):
        origin = trainer.make_lr_origin(self.parent, self.parent_pin)
        contract = trainer.make_lr_contract(self.parent["contract"], origin, sign_lr)
        saved = copy.deepcopy(self.parent)
        saved.update(schema=trainer.CHECKPOINT_SCHEMA, contract=contract)
        saved["optimizer"]["param_groups"][0]["lr"] = sign_lr
        saved["lr_stage"] = trainer.lr_stage_totals(
            saved["cursor"],
            saved["costs"],
            saved["exposure"],
            saved["trainer_seconds"],
            saved["selected_stage"],
            contract["lr_migration"],
        )
        return saved, contract

    def test_own_resume_accepts_current_lr_and_mature_bias_counters(self):
        for lr in (1e-3, 3e-3):
            saved, contract = self.own_packet(lr)
            trainer.validate_lr_resume(self.actor, saved, contract, True)
            self.assertEqual(saved["selected_stage"]["updates"], 1920)
            self.assertEqual(saved["lr_stage"]["stage_updates"], 0)
        self.assertEqual(signs.selected_sign_metrics(self.actor)["hard_sign_flips"], 0)

    def test_own_resume_rejects_wrong_lr_source_counter_reset_and_accounting(self):
        saved, contract = self.own_packet()
        for kind in ("lr", "source", "sign_step", "global_stage", "lr_stage", "model"):
            bad = copy.deepcopy(saved)
            if kind == "lr":
                bad["optimizer"]["param_groups"][0]["lr"] = 1e-3
            elif kind == "source":
                bad["contract"]["trainer"]["sha256"] = "0" * 64
            elif kind == "sign_step":
                identifier = bad["optimizer"]["param_groups"][0]["params"][0]
                bad["optimizer"]["state"][identifier]["step"].fill_(0)
            elif kind == "global_stage":
                bad["selected_stage"]["updates"] = 0
            elif kind == "lr_stage":
                bad["lr_stage"]["stage_updates"] = 480
            else:
                bad["selected_state"]["latent_signs"][signs.SELECTED_BASES[0]][0, 0] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_lr_resume(self.actor, bad, contract, True)

    def test_lr_stage_deltas_keep1920_origin_and_do_not_double_count_mature_work(self):
        origin = trainer.make_lr_origin(self.parent, self.parent_pin)
        stage = copy.deepcopy(self.parent["selected_stage"])
        stage.update(updates=2400, sign_optimizer_updates=2400, optimizer_step_seconds=0.5)
        costs = {key: value + 1 for key, value in self.parent["costs"].items()}
        exposure = dict(self.parent["exposure"])
        exposure.update(ce_windows=8640, replay_windows=4320)
        report = trainer.lr_stage_totals(4320, costs, exposure, 169.0, stage, origin)
        self.assertEqual(report["stage_updates"], 480)
        self.assertEqual(report["stage_trainer_seconds"], 3.0)
        self.assertEqual(report["stage_exposure"]["ce_windows"], 960)
        self.assertEqual(report["selected_stage_delta"]["updates"], 480)
        self.assertEqual(stage["updates"], 2400)

    def test_forward_inspect_and_actual_training_transaction_are_unchanged(self):
        own = Path(trainer.__file__).read_text()
        old = Path(original.__file__).read_text()
        self.assertEqual(trainer.forward_semantics(own), original.forward_semantics(old))

        def transaction(source):
            nodes = [
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.With)
                and len(node.items) == 1
                and isinstance(node.items[0].context_expr, ast.Call)
                and isinstance(node.items[0].context_expr.func, ast.Name)
                and node.items[0].context_expr.func.id == "shared_round_hard_signs"
            ]
            self.assertEqual(len(nodes), 1)
            return ast.dump(nodes[0], include_attributes=False)

        self.assertEqual(transaction(own), transaction(old))


if __name__ == "__main__":
    unittest.main()
