"""Mature-head LR source/CPU fixtures; no real PT/model/array/CUDA access."""

from __future__ import annotations

# ruff: noqa: E402
import ast
import copy
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/w1a8_generalization")]
import modern_selected_head as head
import modern_train_selected_head_lr as trainer
import test_w1a8_modern_selected_head as fixtures
from modern_resident_rows import ResidentRows
from test_w1a8_modern_train_expanded_ce import continuation_fixture


class MatureHeadLRTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SelectedHeadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.directory
        self.paths = list(head.CANDIDATE_D_BASE_TO_PATH.values())
        self.actor_template = copy.deepcopy(self.fixture.actor)

    def selected(self, actor):
        return head.SelectedHead(
            actor,
            self.fixture.selection,
            base_state_pin=trainer.file_pin(self.fixture.base),
            learn_signs=True,
        )

    def mature_pair(self):
        original_actor = copy.deepcopy(self.actor_template)
        other_actor = copy.deepcopy(original_actor)
        original, other = self.selected(original_actor), self.selected(other_actor)
        optimizer, _, parameters = trainer.head_trainer.make_optimizer(original)
        for index, parameter in enumerate(parameters):
            parameter.grad = torch.full_like(parameter, (index + 1) / 10)
        optimizer.step()
        with torch.no_grad():
            original.latent[0, 0] = 0.12
        for parameter in optimizer.param_groups[0]["params"]:
            optimizer.state[parameter]["step"].fill_(5760)
        optimizer.state[original.latent]["step"].fill_(1920)
        model, adam = original.state(), copy.deepcopy(optimizer.state_dict())
        other.load_state(model)
        second, _, _ = trainer.make_optimizer(other)
        second.load_state_dict(copy.deepcopy(adam))
        return original, optimizer, other, second, model, adam

    def metadata_fixture(self, *, proof_change=None, stage_change=None):
        trainer_pin = trainer.file_pin(trainer.head_trainer.__file__)
        proof = {
            "status": "FULL5760_CHECKPOINT_AND_EXISTING_GPU_EXPORT_EXACT_REEXPORT_VERIFIED",
            "trainer": trainer_pin,
            "checkpoint": trainer.expected_mature_pin(),
            "existing_model": trainer.expected_mature_model(),
            "reexport_in_RAM": {**trainer.expected_mature_model(), "path": "/removed/zero.gguf"},
            "temporary_removed": True,
            "embedding_calls": 0,
            "forward_calls": 0,
            "training_updates_performed": 0,
            "full_saved_training_updates": 5760,
            "new_head_training_updates": 1920,
            "selected_head_metrics": {"selected_rows": 512},
        }
        proof.update(proof_change or {})
        proof_path = self.root / "proof.json"
        proof_path.write_text(json.dumps(proof))
        stage = {
            "arm": "head",
            "exit_code": 0,
            "source": trainer_pin,
            "checkpoint": trainer.expected_mature_pin(),
            "model": trainer.expected_mature_model(),
            "finalization": trainer.file_pin(proof_path),
        }
        stage.update(stage_change or {})
        matched = {
            "status": "MATCHED1920_CAPACITY_TRAINING_COMPLETE_NOT_NATIVE_ADMITTED",
            "stages": [stage],
        }
        matched_path = self.root / "matched.json"
        matched_path.write_text(json.dumps(matched))
        return matched_path, proof_path

    def parent_fixture(self):
        cache = ResidentRows()
        runtime = cache.report()
        base = {
            "path": str(trainer.PARENT_CHECKPOINT),
            "sha256": trainer.PARENT_SHA256,
            "bytes": trainer.PARENT_BYTES,
        }
        expected = {
            "schema": trainer.head_trainer.SCHEMA,
            "trainer": trainer.file_pin(trainer.head_trainer.__file__),
            "selected_head_source": trainer.file_pin(head.__file__),
            "learn_signs": True,
            "head_selection": {"count": 512},
            "head_optimizer": {"sign_lr": 0.001, "scale_lr": 1e-5},
            "head_start": {"checkpoint": base},
            "selected_head_contract": {"base_state": base},
            "horizon": 19200,
            "seed": 37,
            "resident": {
                "source": runtime["source_ancestry"]["resident_reader"],
                "proof_report": {"original_reader": runtime["source_ancestry"]["original_reader"]},
                "config": {
                    key: runtime[key]
                    for key in (
                        "max_cache_bytes",
                        "max_file_bytes",
                        "available_reserve_bytes",
                        "capacity_mode",
                    )
                },
            },
        }
        scales = [torch.nn.Parameter(torch.zeros(1)) for _ in range(9)]
        latent = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = torch.optim.Adam(
            [{"params": scales, "lr": 1e-5}, {"params": [latent], "lr": 0.001}]
        )
        for parameter in scales + [latent]:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for parameter in scales:
            optimizer.state[parameter]["step"].fill_(5760)
        optimizer.state[latent]["step"].fill_(1920)
        parent = {
            "schema": trainer.head_trainer.SCHEMA,
            "contract": copy.deepcopy(expected),
            "cursor": 5760,
            "status": "RECOVERY",
            "optimizer": optimizer.state_dict(),
            "selected_state": {"contract": expected["selected_head_contract"]},
            "resident_runtime": runtime,
            **{
                key: "state"
                for key in ("head_stage", "rng", "cuda_rng", "python_rng", "ce_stage_exposure")
            },
        }
        metadata = {"checkpoint": trainer.expected_mature_pin(), "trainer": expected["trainer"]}
        return parent, expected, metadata

    def test_metadata_admission_is_bounded_json_source_only_and_pins_fixed_mature_point(self):
        matched, proof = self.metadata_fixture()
        real_pin = trainer.file_pin

        def source_or_json(path):
            self.assertIn(Path(path).suffix, (".json", ".py"))
            return real_pin(path)

        with patch.object(trainer, "file_pin", side_effect=source_or_json):
            result = trainer.mature_metadata(matched, proof)
        self.assertEqual(result["checkpoint"], trainer.expected_mature_pin())
        self.assertEqual(result["model"], trainer.expected_mature_model())
        self.assertEqual(result["finalization"], real_pin(proof))
        for change in (
            {"forward_calls": 1},
            {"temporary_removed": False},
            {"checkpoint": {"bad": "pin"}},
            {"new_head_training_updates": 1919},
            {"trainer": {"bad": "source"}},
            {"selected_head_metrics": {"selected_rows": 511}},
        ):
            matched, proof = self.metadata_fixture(proof_change=change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                trainer.mature_metadata(matched, proof)

    def test_recovery_mature_parent_full_old_contract_is_valid_before_any_lr_change(self):
        parent, expected, metadata = self.parent_fixture()
        before = copy.deepcopy(parent)
        trainer.validate_mature_parent(parent, expected, trainer.expected_mature_pin(), metadata)
        self.assertTrue(trainer.equal(parent, before))
        for kind in ("schema", "cursor", "source", "base", "lr", "counter"):
            bad = copy.deepcopy(parent)
            if kind == "schema":
                bad["schema"] = trainer.SCHEMA
            elif kind == "cursor":
                bad["cursor"] = 5759
            elif kind == "source":
                bad["contract"]["trainer"]["sha256"] = "bad"
            elif kind == "base":
                bad["selected_state"]["contract"]["base_state"] = trainer.expected_mature_pin()
            elif kind == "lr":
                bad["optimizer"]["param_groups"][1]["lr"] = 0.003
            else:
                bad["optimizer"]["state"][9]["step"].fill_(0)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_mature_parent(
                    bad, expected, trainer.expected_mature_pin(), metadata
                )

    def test_lr_mutation_changes_exactly_head_group_lr_without_moments_order_or_rng_reset(self):
        for sign_lr in trainer.SIGN_LRS:
            _, _, _, optimizer, _, saved = self.mature_pair()
            before = copy.deepcopy(optimizer.state_dict())
            versions = {
                id(p): p._version for group in optimizer.param_groups for p in group["params"]
            }
            rng, python_rng = torch.get_rng_state().clone(), random.getstate()
            trainer.apply_sign_lr(optimizer, sign_lr)
            expected = copy.deepcopy(before)
            expected["param_groups"][1]["lr"] = sign_lr
            self.assertTrue(trainer.equal(optimizer.state_dict(), expected))
            self.assertEqual(
                versions,
                {id(p): p._version for group in optimizer.param_groups for p in group["params"]},
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(random.getstate(), python_rng)
            self.assertTrue(trainer.equal(saved, before))
        with self.assertRaises(ValueError):
            trainer.apply_sign_lr(optimizer, 0.03)

    def test_unchanged_lr_matches_original_head_next_updates_and_keeps_f4_constructor_contract(
        self,
    ):
        original, first, selected, second, model, mature_adam = self.mature_pair()
        original_contract = copy.deepcopy(selected.state()["contract"])
        trainer.apply_sign_lr(second, 0.001)
        self.assertTrue(trainer.equal(second.state_dict(), mature_adam))
        self.assertEqual(selected.state()["contract"], original_contract)
        self.assertEqual(original_contract["base_state"], trainer.file_pin(self.fixture.base))
        self.assertFalse(bool(selected.latent.detach().abs().eq(0.5).all()))
        x = torch.arange(105, dtype=torch.float32).reshape(3, 35) / 100
        for _ in range(2):
            for owner, optimizer in ((original, first), (selected, second)):
                optimizer.zero_grad(set_to_none=True)
                with owner.round():
                    sum(
                        module(x).square().sum() for module in owner.actor.linears.values()
                    ).backward()
                params = tuple(p for group in optimizer.param_groups for p in group["params"])
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                owner.clamp_()
                for path in self.paths:
                    owner.actor.linears[path].project_scales_()
            self.assertTrue(trainer.equal(original.state(), selected.state()))
            self.assertTrue(trainer.equal(first.state_dict(), second.state_dict()))
        trainer.validate_head_optimizer(second.state_dict(), 5762, True, 0.001)
        self.assertEqual(float(second.state_dict()["state"][9]["step"]), 1922)
        self.assertEqual(model["contract"], selected.state()["contract"])

    def test_strict_own_resume_and_three_stage_accounting_preserve_original_moment_age(self):
        args = continuation_fixture(6240)
        base_args, mature_args = continuation_fixture(3840), continuation_fixture(5760)
        state, base, mature = args[0], base_args[0], mature_args[0]
        base.update(contract={"origin": "resident"}, ce_stage_exposure=base_args[2])
        head_start = trainer.head_origin(
            base,
            {
                "path": str(trainer.PARENT_CHECKPOINT),
                "sha256": trainer.PARENT_SHA256,
                "bytes": trainer.PARENT_BYTES,
            },
        )
        mature["costs"]["ce"] = 4.0
        mature["trainer_seconds"] = 4.0
        parent, expected, metadata = self.parent_fixture()
        expected["head_start"] = head_start
        mature.update(contract=expected, ce_stage_exposure=mature_args[2])
        start = trainer.lr_origin(mature, trainer.expected_mature_pin())
        contract = trainer.declared_lr_contract(expected, start, 0.003, metadata)
        state["costs"]["ce"] = 4.5
        state["trainer_seconds"] = 4.5
        optimizer = copy.deepcopy(parent["optimizer"])
        optimizer["param_groups"][1]["lr"] = 0.003
        for key, row in optimizer["state"].items():
            row["step"].fill_(2400 if key == 9 else 6240)
        state.update(
            schema=trainer.SCHEMA,
            contract=contract,
            optimizer=optimizer,
            selected_state={
                "scale_offsets": {base: torch.zeros(1) for base in head.CANDIDATE_D_BASE_TO_PATH},
                "selected_latent": torch.zeros(2, 3),
            },
            resident_runtime=parent["resident_runtime"],
            rng=torch.get_rng_state(),
            cuda_rng=torch.zeros(8, dtype=torch.uint8),
            python_rng=random.getstate(),
            ce_stage_exposure=args[2],
        )
        state["stage"] = trainer.stage_totals(
            6240, state["costs"], state["exposure"], state["trainer_seconds"], args[1]
        )
        state["head_stage"] = trainer.head_totals(
            6240, state["costs"], state["exposure"], state["trainer_seconds"], args[2], head_start
        )
        state["lr_stage"] = trainer.head_totals(
            6240, state["costs"], state["exposure"], state["trainer_seconds"], args[2], start
        )
        trainer.validate_resume(state, contract)
        trainer.validate_continuation(*args)
        self.assertEqual(state["stage"]["stage_updates"], 4320)
        self.assertEqual(state["head_stage"]["stage_updates"], 2400)
        self.assertEqual(state["lr_stage"]["stage_updates"], 480)
        self.assertEqual(state["lr_stage"]["stage_costs"], {"ce": 0.5})
        for kind in ("schema", "lr", "step-reset", "lr-stage", "source"):
            bad = copy.deepcopy(state)
            if kind == "schema":
                bad["schema"] = trainer.head_trainer.SCHEMA
            elif kind == "lr":
                bad["optimizer"]["param_groups"][1]["lr"] = 0.01
            elif kind == "step-reset":
                bad["optimizer"]["state"][9]["step"].fill_(480)
            elif kind == "lr-stage":
                bad["lr_stage"]["stage_updates"] += 1
            else:
                bad["contract"]["lr_start"]["parent_contract"]["trainer"]["sha256"] = "bad"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_resume(bad, contract)
                trainer.validate_continuation(bad, *args[1:])

    def test_numeric_base_guard_unchanged_and_progress_maps_removed(
        self,
    ):
        current = ast.parse(Path(trainer.__file__).read_text())
        frozen = ast.parse(Path(trainer.head_trainer.__file__).read_text())

        def function(tree, name):
            return next(
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
            )

        for name in (
            "forward",
            "inspect",
            "head_totals",
            "native_schedule",
            "ce_at",
            "ce_order",
            "make_optimizer",
        ):
            self.assertEqual(
                ast.dump(function(current, name), include_attributes=False),
                ast.dump(function(frozen, name), include_attributes=False),
            )
        current_guard = next(
            n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "CadencedGuard"
        )
        old_guard = next(
            n for n in frozen.body if isinstance(n, ast.ClassDef) and n.name == "CadencedGuard"
        )
        self.assertEqual(
            ast.dump(current_guard, include_attributes=False),
            ast.dump(old_guard, include_attributes=False),
        )
        update = function(current, "update")
        old_update = function(frozen, "update")
        start = next(i for i, n in enumerate(update.body) if isinstance(n, ast.With))
        stop = next(
            i
            for i, n in enumerate(update.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "elapsed" for t in n.targets)
        )
        old_start = next(i for i, n in enumerate(old_update.body) if isinstance(n, ast.With))
        old_stop = next(
            i
            for i, n in enumerate(old_update.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "elapsed" for t in n.targets)
        )
        self.assertEqual(
            ast.dump(
                ast.Module(body=update.body[start:stop], type_ignores=[]), include_attributes=False
            ),
            ast.dump(
                ast.Module(body=old_update.body[old_start:old_stop], type_ignores=[]),
                include_attributes=False,
            ),
        )
        self.assertFalse(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "head_totals"
                for n in ast.walk(update)
            )
        )
        keys = [
            n.targets[0].slice.value
            for n in update.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Subscript)
            and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id == "row"
            and isinstance(n.targets[0].slice, ast.Constant)
        ]
        self.assertIn("head_stage_updates", keys)
        self.assertIn("lr_stage_updates", keys)
        self.assertNotIn("head_stage", keys)
        main = function(current, "main")
        calls = {
            name: min(
                n.lineno
                for n in ast.walk(main)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
            )
            for name in ("validate_mature_parent", "apply_sign_lr")
        }
        self.assertLess(calls["validate_mature_parent"], calls["apply_sign_lr"])
        constructors = [
            n
            for n in ast.walk(main)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "SelectedHead"
        ]
        self.assertEqual(len(constructors), 1)
        base_arg = next(k.value for k in constructors[0].keywords if k.arg == "base_state_pin")
        self.assertEqual(base_arg.id, "parent_pin")
        row_index = next(
            i
            for i, n in enumerate(update.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "row" for t in n.targets)
        )
        progress = ast.fix_missing_locations(
            ast.Module(
                body=[
                    copy.deepcopy(n)
                    for n in update.body[row_index:]
                    if not isinstance(n, ast.Return)
                ],
                type_ignores=[],
            )
        )
        exposure = {
            "ce_windows": 12480,
            "ce_supported_labels": 100,
            "replay_windows": 6240,
            "replay_protected_tokens": 6240,
        }
        values = {
            "cursor": 6240,
            "elapsed": 0.01,
            "ce_value": 0.3,
            "replay_loss": torch.tensor(0.2),
            "replay_group": "g",
            "replay_window": {"id": "w"},
            "replay_tokens": 1,
            "ce_seconds": 0.003,
            "replay_seconds": 0.002,
            "loss": torch.tensor(0.5),
            "supported": 4,
            "norm": torch.tensor(0.1),
            "train_seconds": 5.0,
            "costs": {"ce": 1.0},
            "exposure": exposure,
            "origin": {
                "global_start_cursor": 1920,
                "initial_costs": {"ce": 0.0},
                "initial_exposure": {key: 0 for key in exposure},
                "initial_trainer_seconds": 0.0,
            },
            "stage_totals": trainer.stage_totals,
            "gradients": None,
            "HEAD_START": 3840,
            "LR_START": 5760,
            "args": SimpleNamespace(sign_lr=0.003),
            "ce_stage_exposure": {"large_unused_map": {str(i): i for i in range(10000)}},
        }
        exec(compile(progress, "<actual-progress-record>", "exec"), values)
        self.assertLess(len(json.dumps(values["row"])), 2000)


if __name__ == "__main__":
    unittest.main()
