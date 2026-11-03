"""Synthetic CPU state/optimizer/capacity fixtures; no real models, captures, or CUDA."""

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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/w1a8_generalization")]
import modern_actor
import modern_selected_head as head
import modern_train_selected_head as trainer
import test_w1a8_modern_selected_head as fixtures
from modern_actual_data import ActualDataset, _Rows
from modern_actual_union import UnionActualDataset
from modern_resident_rows import ResidentRows
from test_w1a8_modern_train_expanded_ce import continuation_fixture

from w1a1_eagle.recurrent_qat import shared_round_hard_signs


class SelectedHeadTrainerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SelectedHeadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.actor = self.fixture.actor
        self.paths = list(head.CANDIDATE_D_BASE_TO_PATH.values())

    def parent_optimizer(self, paths=None):
        paths = self.paths if paths is None else paths
        optimizer = torch.optim.Adam([self.actor.linears[p].scale_offset for p in paths], lr=1e-5)
        for index, path in enumerate(self.paths):
            self.actor.linears[path].scale_offset.grad = torch.full_like(
                self.actor.linears[path].scale_offset, (index + 1) / 100
            )
        optimizer.step()
        for state in optimizer.state.values():
            state["step"].fill_(3840)
        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def selected(self, learn_signs):
        return head.SelectedHead(
            self.actor,
            self.fixture.selection,
            base_state_pin=modern_actor.file_pin(self.fixture.base),
            learn_signs=learn_signs,
        )

    def test_reachable_fit_selector_excludes_unreachable_labels_and_rejects_dev(self):
        windows = [
            {
                "role": "fit",
                "group_id": "A",
                "labels": [1, 99],
                "native_supervision_reachable_mask": [True, False],
            },
            {
                "role": "fit",
                "group_id": "B",
                "labels": [3, 5],
                "native_supervision_reachable_mask": [True, True],
            },
        ]
        records = trainer.reachable_fit_records(windows)
        self.assertEqual([r["labels"] for r in records], [[1], [3, 5]])
        result = head.select_head_rows(
            records,
            head.absolute_map(self.actor),
            source_pins=[modern_actor.file_pin(self.fixture.manifest)],
            count=2,
        )
        self.assertNotIn(99, result["target_ids"])
        for kind in ("role", "mask", "depth"):
            bad = copy.deepcopy(windows)
            if kind == "role":
                bad[0]["role"] = "development"
            elif kind == "mask":
                bad[0]["native_supervision_reachable_mask"] = [1, 0]
            else:
                bad[0]["native_supervision_reachable_mask"] = [True]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.reachable_fit_records(bad)

    def test_strict_scale_state_load_on_non_scale_only_actor_never_relaxes_legacy_loader(self):
        offsets = {
            base: torch.full_like(self.actor.linears[path].scale_offset, 0.03)
            for base, path in head.CANDIDATE_D_BASE_TO_PATH.items()
        }
        state = {
            "schema": modern_actor.SCALE_STATE_SCHEMA,
            "ancestry": modern_actor._ancestry(self.actor),
            "scale_offsets": offsets,
        }
        trainer.load_parent_scales(self.actor, state)
        self.assertFalse(self.actor.scale_only)
        for path in self.paths:
            self.assertTrue(
                torch.equal(
                    self.actor.linears[path].scale_offset,
                    torch.full_like(self.actor.linears[path].scale_offset, 0.03),
                )
            )
        with self.assertRaisesRegex(ValueError, "scale-only"):
            modern_actor.load_scale_state(self.actor, state)
        for kind in ("schema", "ancestry", "dtype", "finite"):
            bad = copy.deepcopy(state)
            if kind == "schema":
                bad["schema"] = "learned-O"
            elif kind == "ancestry":
                bad["ancestry"]["initializer"]["sha256"] = "bad"
            elif kind == "dtype":
                bad["scale_offsets"]["output"] = offsets["output"].double()
            else:
                bad["scale_offsets"]["output"][0] = float("nan")
            before = {p: m.scale_offset.detach().clone() for p, m in self.actor.linears.items()}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.load_parent_scales(self.actor, bad)
            self.assertTrue(
                all(
                    torch.equal(self.actor.linears[p].scale_offset, value)
                    for p, value in before.items()
                )
            )

    def test_false_migration_keeps_original_group_moments_rng_and_next_updates_exact(self):
        parent = self.parent_optimizer()
        parent_state = copy.deepcopy(parent.state_dict())
        reference = copy.deepcopy(self.actor)
        modern_actor.enable_scale_only(reference)
        original = torch.optim.Adam(reference.scale_parameters(), lr=1e-5)
        original.load_state_dict(copy.deepcopy(parent_state))
        selected = self.selected(False)
        optimizer, paths, clipped = trainer.make_optimizer(selected)
        rng, python_rng = torch.get_rng_state().clone(), random.getstate()
        trainer.migrate_parent_optimizer(parent_state, optimizer, self.paths, paths, False)
        self.assertTrue(trainer.equal(parent_state, optimizer.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertNotIn("name", optimizer.param_groups[0])
        self.assertEqual(len(clipped), 9)
        self.assertNotIn(id(selected.latent), {id(p) for p in clipped})
        x = torch.arange(105, dtype=torch.float32).reshape(3, 35) / 100
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            original.zero_grad(set_to_none=True)
            with selected.round():
                sum(m(x).square().sum() for m in self.actor.linears.values()).backward()
            torch.nn.utils.clip_grad_norm_(clipped, 1.0)
            optimizer.step()
            selected.clamp_()
            for path in paths:
                self.actor.linears[path].project_scales_()
            with shared_round_hard_signs(reference.linears):
                sum(m(x).square().sum() for m in reference.linears.values()).backward()
                torch.nn.utils.clip_grad_norm_(reference.scale_parameters(), 1.0)
                original.step()
                for path in paths:
                    reference.linears[path].project_scales_()
            self.assertTrue(
                all(
                    torch.equal(
                        self.actor.linears[p].scale_offset, reference.linears[p].scale_offset
                    )
                    for p in paths
                )
            )
            self.assertTrue(trainer.equal(optimizer.state_dict(), original.state_dict()))
            self.assertIsNone(selected.latent.grad)
        trainer.validate_head_optimizer(optimizer.state_dict(), 3842, False)

    def test_migration_joins_distinct_scale_moments_by_path_and_appends_empty_head_group(self):
        old_paths = list(reversed(self.paths))
        parent = self.parent_optimizer(old_paths)
        before = copy.deepcopy(parent.state_dict())
        selected = self.selected(True)
        optimizer, paths, clipped = trainer.make_optimizer(selected)
        trainer.migrate_parent_optimizer(before, optimizer, old_paths, paths, True)
        for path in self.paths:
            self.assertTrue(
                trainer.equal(
                    optimizer.state[self.actor.linears[path].scale_offset],
                    parent.state[self.actor.linears[path].scale_offset],
                )
            )
        self.assertNotIn(selected.latent, optimizer.state)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-5, 1e-3])
        self.assertEqual(len(clipped), 10)
        self.assertIs(clipped[-1], selected.latent)
        trainer.validate_head_optimizer(optimizer.state_dict(), 3840, True)
        self.assertTrue(trainer.equal(parent.state_dict(), before))
        optimizer.zero_grad(set_to_none=True)
        with selected.round():
            sum(m(torch.ones(2, 35)).square().sum() for m in self.actor.linears.values()).backward()
        evidence = trainer.gradient_metrics(selected)
        self.assertTrue(evidence["finite"])
        self.assertGreater(evidence["nonzero_elements"], 0)
        torch.nn.utils.clip_grad_norm_(clipped, 1.0)
        optimizer.step()
        selected.clamp_()
        trainer.validate_head_optimizer(optimizer.state_dict(), 3841, True)
        self.assertGreater(trainer.sign_metrics(selected)["changed_latents"], 0)
        self.assertEqual(trainer.sign_metrics(selected)["hard_sign_flips"], 0)

    def test_adam_rejects_wrong_group_scope_lr_step_defaults_and_extra_moments(self):
        self.parent_optimizer()
        selected = self.selected(True)
        optimizer, _, _ = trainer.make_optimizer(selected)
        for parameter in optimizer.param_groups[0]["params"]:
            parameter.grad = torch.ones_like(parameter)
        selected.latent.grad = torch.ones_like(selected.latent)
        optimizer.step()
        for parameter in optimizer.param_groups[0]["params"]:
            optimizer.state[parameter]["step"].fill_(3841)
        valid = optimizer.state_dict()
        trainer.validate_head_optimizer(valid, 3841, True)
        for kind in ("lr", "step", "betas", "id", "extra", "control"):
            state = copy.deepcopy(valid)
            learn = True
            if kind == "lr":
                state["param_groups"][1]["lr"] = 0.1
            elif kind == "step":
                state["state"][9]["step"].fill_(3841)
            elif kind == "betas":
                state["param_groups"][0]["betas"] = (0.8, 0.9)
            elif kind == "id":
                state["param_groups"][0]["params"] = list(reversed(range(9)))
            elif kind == "extra":
                state["state"][10] = state["state"][9]
            else:
                learn = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_head_optimizer(state, 3841, learn)

    def test_head_accounting_preserves_1920_counters_and_subtracts_parent3840_once(self):
        parent_args, args = continuation_fixture(3840), continuation_fixture(3842)
        parent, state = parent_args[0], args[0]
        origin = trainer.head_origin(
            {**parent, "contract": {"ancestry": "original"}, "ce_stage_exposure": parent_args[2]},
            {"sha256": "parent"},
        )
        state["costs"]["ce"] += 0.5
        state["trainer_seconds"] += 0.5
        state["contract"] = {"head_start": origin, "learn_signs": False}
        parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(9)]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for row in optimizer.state.values():
            row["step"].fill_(3842)
        state["optimizer"] = optimizer.state_dict()
        state["selected_state"] = {
            "scale_offsets": {base: torch.zeros(1) for base in head.CANDIDATE_D_BASE_TO_PATH},
            "selected_latent": torch.zeros(2, 3),
        }
        state["stage"] = trainer.stage_totals(
            state["cursor"], state["costs"], state["exposure"], state["trainer_seconds"], args[1]
        )
        state["head_stage"] = trainer.head_totals(
            state["cursor"],
            state["costs"],
            state["exposure"],
            state["trainer_seconds"],
            args[2],
            origin,
        )
        trainer.validate_continuation(*args)
        self.assertEqual(state["stage"]["stage_updates"], 1922)
        self.assertEqual(state["head_stage"]["stage_updates"], 2)
        self.assertEqual(state["head_stage"]["stage_costs"], {"ce": 0.5})
        self.assertEqual(
            sum(row["windows"] for row in state["head_stage"]["ce_stage_exposure"].values()), 4
        )
        state["head_stage"]["stage_updates"] += 1
        with self.assertRaisesRegex(ValueError, "head-stage"):
            trainer.validate_continuation(*args)

    def test_parent_and_own_resume_are_schema_and_contract_strict(self):
        cache = ResidentRows()
        runtime = cache.report()
        expected = {
            "trainer": trainer.file_pin(trainer.resident_trainer.__file__),
            "schema": trainer.resident_trainer.SCHEMA,
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
            "expanded_ce": True,
            "optimizer_scope": "all",
            "horizon": 19200,
            "seed": 37,
            "native_fit_bank_tokens": 356,
            "batch_windows": 2,
            "retention": 10.0,
            "optimized_scale_paths": self.paths,
        }
        parent = {
            "schema": trainer.resident_trainer.SCHEMA,
            "contract": expected,
            "cursor": 3840,
            "optimizer": {"param_groups": [{"lr": 1e-5}]},
            "resident_runtime": runtime,
            **{
                key: "state"
                for key in ("scales", "rng", "cuda_rng", "python_rng", "ce_stage_exposure")
            },
        }
        pin = {
            "path": str(trainer.PARENT_CHECKPOINT),
            "sha256": trainer.PARENT_SHA256,
            "bytes": trainer.PARENT_BYTES,
        }
        trainer.validate_parent(parent, expected, pin)
        wrong = copy.deepcopy(parent)
        wrong["cursor"] = 3839
        with self.assertRaises(ValueError):
            trainer.validate_parent(wrong, expected, pin)
        contract = {**expected, "schema": trainer.SCHEMA, "learn_signs": False}
        own = {
            "schema": trainer.SCHEMA,
            "contract": contract,
            "resident_runtime": runtime,
            **{
                key: "state"
                for key in (
                    "selected_state",
                    "head_stage",
                    "optimizer",
                    "rng",
                    "cuda_rng",
                    "python_rng",
                    "ce_stage_exposure",
                )
            },
        }
        trainer.validate_resume(own, contract)
        for kind in ("schema", "contract", "selected_state", "head_stage"):
            bad = copy.deepcopy(own)
            if kind == "schema":
                bad["schema"] = trainer.resident_trainer.SCHEMA
            elif kind == "contract":
                bad["contract"]["learn_signs"] = True
            else:
                del bad[kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_resume(bad, contract)

    def test_resident_binding_is_instance_local(self):
        union = object.__new__(UnionActualDataset)
        union._rows = _Rows()
        old = object.__new__(ActualDataset)
        old._rows = _Rows()
        cache = trainer.bind_resident(
            union, {"max_cache_bytes": 1024, "available_reserve_bytes": 0}
        )
        self.assertIs(union._rows, cache)
        self.assertIs(type(old._rows), _Rows)

    def test_numeric_forward_ce_replay_and_round_lifetime_match_declared_recipe(self):
        current = ast.parse(Path(trainer.__file__).read_text())
        frozen = ast.parse(Path(trainer.resident_trainer.__file__).read_text())

        def function(tree, name):
            found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
            self.assertEqual(len(found), 1)
            return found[0]

        for name in (
            "forward",
            "inspect",
            "native_fit_references",
            "native_schedule",
            "ce_order",
            "ce_at",
        ):
            self.assertEqual(
                ast.dump(function(current, name), include_attributes=False),
                ast.dump(function(frozen, name), include_attributes=False),
            )
        update = function(current, "update")
        selected_round = next(n for n in update.body if isinstance(n, ast.With))
        old_round = next(
            n
            for n in ast.walk(frozen)
            if isinstance(n, ast.With)
            and len(n.items) == 1
            and isinstance(n.items[0].context_expr, ast.Call)
            and isinstance(n.items[0].context_expr.func, ast.Name)
            and n.items[0].context_expr.func.id == "shared_round_hard_signs"
        )
        prefix = old_round.body[: len(selected_round.body)]
        self.assertEqual(
            ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False),
            ast.dump(
                ast.Module(body=selected_round.body, type_ignores=[]), include_attributes=False
            ),
        )
        calls = {
            n.func.attr: n
            for n in ast.walk(update)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("step", "clamp_", "zero_grad")
        }
        self.assertEqual(calls["zero_grad"].func.value.id, "optimizer")
        self.assertGreater(calls["step"].lineno, selected_round.end_lineno)
        self.assertGreater(calls["clamp_"].lineno, calls["step"].lineno)
        self.assertEqual(
            trainer.file_pin(trainer.resident_trainer.__file__)["sha256"],
            trainer.RESIDENT_TRAINER_SHA256,
        )
        self.assertEqual(trainer.file_pin(head.__file__)["sha256"], trainer.HEAD_HELPER_SHA256)


if __name__ == "__main__":
    unittest.main()
