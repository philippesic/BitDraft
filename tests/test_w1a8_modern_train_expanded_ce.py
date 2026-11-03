"""Controlled CE expansion contracts; synthetic CPU state only, no model or CUDA."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/w1a8_generalization"))
SPEC = importlib.util.spec_from_file_location(
    "modern_train_expanded_ce", ROOT / "scripts/w1a8_generalization/modern_train_expanded_ce.py"
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


def window(identifier, role, group=None):
    group = group or identifier
    return {
        "id": identifier,
        "role": role,
        "group": group,
        "group_id": "component:" + group,
        "domain": "code",
        "prefix_raw_rows": [0, 1],
        "seed_raw_row": 2,
        "labels": [11, 13],
        "native_supervision_reachable_mask": [True, False],
        "native_prefix": 1,
        "native_protections": [{"proposal_depth": 0, "token": 11, "response_index": 3}],
        "max_depth": 2,
        "response_label_start": 3,
    }


def union_fixture():
    old = [window("old-fit", "fit"), window("old-dev", "development")]
    groups = [{k: w[k] for k in ("id", "role", "group_id", "domain")} for w in old]
    original = {
        "groups": groups,
        "windows": old,
        "capture": {"path": "/fixture/old/capture.json", "sha256": "old"},
        "required_embedding_ids": [11, 13],
    }
    extra, new_groups = [], []
    for domain in ("code", "reasoning", "prose"):
        for number in range(8):
            w = window(f"new-{domain}-{number}", "fit")
            w["domain"] = domain
            extra.append(w)
            new_groups.append({k: w[k] for k in ("id", "role", "group_id", "domain")})
    # Exercise semantic preservation despite different raw storage indices.
    remapped = copy.deepcopy(old)
    for w in remapped:
        w["prefix_raw_rows"], w["seed_raw_row"] = [5, 3], 7
    pin = {"path": "/fixture/old/manifest.json", "sha256": "oldmanifest", "bytes": 10}
    union = {
        "groups": copy.deepcopy(groups + new_groups),
        "windows": remapped + extra,
        "members": [
            {
                "manifest": pin,
                "window_ids": [w["id"] for w in old],
                "index_maps": {"raw_rows": [5, 3, 7]},
            },
            {
                "metadata": {"capture": {"path": "/fixture/new/capture.json", "sha256": "new"}},
                "window_ids": [w["id"] for w in extra],
            },
        ],
        "group_provenance": {g["id"]: [{"member": 1, "group": g}] for g in new_groups},
        "missing_embedding_ids": [],
        "required_embedding_ids": [11, 13, 17],
    }
    return original, union, pin


def optimizer_fixture(paths):
    parameters = {
        path: torch.nn.Parameter(torch.tensor([float(i + 1)])) for i, path in enumerate(paths)
    }
    optimizer = torch.optim.Adam(list(parameters.values()), lr=1e-5)
    for i, parameter in enumerate(parameters.values()):
        parameter.grad = torch.tensor([float(i + 1) / 10])
    optimizer.step()
    for row in optimizer.state.values():
        row["step"].fill_(1920)
    return parameters, optimizer


def continuation_fixture(cursor=1923):
    fit = [window("old-fit", "fit"), window("new-fit", "fit")]
    order = trainer.ce_order(fit)
    refs = {w["id"]: {"protected": [True, False]} for w in fit}

    class Bank:
        def at(self, cursor):
            return "component:old-fit", fit[0]

    initial = {
        "ce_windows": 3840,
        "ce_supported_labels": 1000,
        "replay_windows": 1920,
        "replay_protected_tokens": 1920,
    }
    origin = {
        "global_start_cursor": 1920,
        "initial_exposure": initial,
        "initial_costs": {"ce": 1.0},
        "initial_trainer_seconds": 1.0,
        "original_contract": {"ce_schedule": {"order": ["old-fit"]}},
    }
    stage = trainer.empty_ce_exposure()
    for step in range(1920, cursor):
        for w in trainer.ce_at(fit, order, step):
            trainer.count_ce(stage, w, 1, {"old-fit"})
    state = {
        "cursor": cursor,
        "status": "RECOVERY",
        "costs": {"ce": 2.0},
        "trainer_seconds": 2.0,
        "optimizer": {
            "state": {
                i: {
                    "step": torch.tensor(cursor),
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                }
                for i in range(9)
            }
        },
        "exposure": {
            "ce_windows": 2 * cursor,
            "ce_supported_labels": 1000 + 2 * (cursor - 1920),
            "replay_windows": cursor,
            "replay_protected_tokens": cursor,
        },
    }
    return state, origin, stage, fit, order, {11}, Bank(), refs


class ExpandedCETrainerTests(unittest.TestCase):
    def test_new_request_protocol_requires_pinned80_token_target_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return trainer.file_pin(path)

            request = {
                "n_predict": 80,
                "temperature": 0,
                "seed": 1234,
                "cache_prompt": False,
                "return_tokens": True,
                "stream": False,
                "prompt": "frozen whole source group",
            }
            source_request = write(root / "target-only/request.json", request)
            source_response = write(root / "target-only/response.json", {"tokens": [11]})
            freeze = write(root / "freeze.json", {"input_pins": [source_request, source_response]})
            groups = {}
            for i in range(24):
                identifier = f"new-{i}"
                write(root / identifier / "request.json", request)
                write(
                    root / identifier / "target-reference.json",
                    {"request": source_request, "response": source_response},
                )
                groups[identifier] = [{"member": 1, "group": {"id": identifier}}]
            union = {
                "members": [
                    {},
                    {
                        "metadata": {
                            "capture": {"path": str(root / "summary.json")},
                            "source_pins": [freeze],
                        }
                    },
                ],
                "group_provenance": groups,
            }
            pins = trainer.capture_protocol(union, trainer.Audit())
            self.assertEqual(len(pins), 1 + 4 * 24)
            for kind in ("protocol", "prompt", "unpinned"):
                if kind == "protocol":
                    bad = {**request, "n_predict": 81}
                else:
                    bad = {**request, "prompt": "other group"}
                write(root / "new-0/request.json", bad)
                if kind == "unpinned":
                    replacement = write(root / "other-request.json", bad)
                    write(
                        root / "new-0/target-reference.json",
                        {"request": replacement, "response": source_response},
                    )
                with self.assertRaises(ValueError):
                    trainer.capture_protocol(union, trainer.Audit())

    def test_original_fit_and_dev_preserved_with_semantic_index_remap(self):
        original, union, pin = union_fixture()
        snapshot = copy.deepcopy((original, union))
        fit, dev, extra = trainer.controlled_views(original, union, pin)
        self.assertEqual([w["id"] for w in fit], ["old-fit"])
        self.assertEqual([w["id"] for w in dev], ["old-dev"])
        self.assertEqual(dev[0]["prefix_raw_rows"], [5, 3])
        self.assertEqual(len(extra), 24)
        self.assertTrue(all(w["role"] == "fit" for w in extra))
        self.assertEqual((original, union), snapshot)

    def test_rejects_changed_old_windows_new_dev_reused_capture_and_unbalanced_groups(self):
        for kind in (
            "old-fit",
            "old-dev",
            "new-dev",
            "capture",
            "domain",
            "component",
            "pin",
            "embedding",
            "reused-window",
        ):
            with self.subTest(kind=kind):
                original, union, pin = union_fixture()
                if kind.startswith("old-"):
                    union["windows"][0 if kind == "old-fit" else 1]["labels"] = [99, 99]
                elif kind == "new-dev":
                    union["windows"][2]["role"] = "development"
                elif kind == "capture":
                    union["members"][1]["metadata"]["capture"] = original["capture"]
                elif kind == "domain":
                    union["groups"][2]["domain"] = "prose"
                elif kind == "component":
                    union["groups"][2]["group_id"] = "component:old-dev"
                elif kind == "pin":
                    pin = {"wrong": True}
                elif kind == "embedding":
                    union["missing_embedding_ids"] = [17]
                else:
                    union["members"][1]["window_ids"].append("old-fit")
                with self.assertRaises(ValueError):
                    trainer.controlled_views(original, union, pin)

    def test_false_arm_exact_old1414_order_and_global_cursor_wraparound(self):
        fit = [window(f"old-{i}", "fit") for i in range(1414)]
        expected = list(range(1414))
        random.Random(37).shuffle(expected)
        self.assertEqual(trainer.ce_order(fit), expected)
        for cursor in (0, 707, 1919, 1920, 2399, 3839, 19199):
            self.assertEqual(
                trainer.ce_at(fit, expected, cursor),
                [fit[expected[(2 * cursor + offset) % 1414]] for offset in range(2)],
            )
        before = random.getstate()
        trainer.ce_order(fit)
        self.assertEqual(random.getstate(), before)

    def test_true_arm_samples_new_fit_without_touching_original_replay_bank(self):
        original, union, pin = union_fixture()
        fit, dev, extra = trainer.controlled_views(original, union, pin)
        merged = fit + extra
        selected = [
            w for c in range(1920, 1945) for w in trainer.ce_at(merged, trainer.ce_order(merged), c)
        ]
        self.assertEqual({w["id"] for w in selected}, {w["id"] for w in merged})
        self.assertNotIn(dev[0], selected)
        refs = {
            "old-fit": {
                "id": "old-fit",
                "role": "fit",
                "labels": [11, 13],
                "native_prefix": 1,
                "supported": [True, True],
                "protected": [False, False],
                "margins": [-0.5, -0.5],
            }
        }
        native = trainer.native_fit_references(fit, refs, {11, 13})
        bank = trainer.native_schedule(fit, native)
        self.assertTrue(all(bank.at(c)[1]["id"] == "old-fit" for c in range(1920, 1945)))
        self.assertEqual(native["old-fit"]["margins"][0], 0.1)

    def test_named_adam_migration_preserves_identity_order_and_reordered_paths(self):
        paths = [f"scale{i}" for i in range(9)]
        parameters, optimizer = optimizer_fixture(paths)
        saved = optimizer.state_dict()
        before = copy.deepcopy(saved)
        migrated = trainer.named_optimizer_state(saved, paths, paths)
        self.assertTrue(trainer.equal(saved, migrated))
        reordered = list(reversed(paths))
        copies = {name: torch.nn.Parameter(parameters[name].detach().clone()) for name in reordered}
        restored = torch.optim.Adam(list(copies.values()), lr=1e-5)
        restored.load_state_dict(trainer.named_optimizer_state(saved, paths, reordered))
        self.assertTrue(trainer.equal(saved, before))
        for index, name in enumerate(paths):
            gradient = torch.tensor([(index + 2) / 7])
            parameters[name].grad = gradient.clone()
            copies[name].grad = gradient.clone()
        optimizer.step()
        restored.step()
        for name in paths:
            self.assertTrue(torch.equal(parameters[name], copies[name]))
            self.assertTrue(
                trainer.equal(optimizer.state[parameters[name]], restored.state[copies[name]])
            )

    def test_named_adam_migration_rejects_missing_paths_or_moments(self):
        names = [f"scale{i}" for i in range(9)]
        _, optimizer = optimizer_fixture(names)
        for kind in ("path", "missing", "extra"):
            state = copy.deepcopy(optimizer.state_dict())
            changed = list(names)
            if kind == "path":
                changed[0] = "unknown"
            elif kind == "missing":
                state["state"].pop(0)
            else:
                state["state"][99] = {}
            with self.assertRaises(ValueError):
                trainer.named_optimizer_state(state, names, changed)

    def test_resume_schema_contract_and_required_state_are_strict(self):
        contract = {"schema": trainer.SCHEMA, "expanded_ce": False, "trainer": "full source pin"}
        saved = {
            "schema": trainer.SCHEMA,
            "contract": contract,
            **{
                key: "state"
                for key in (
                    "scales",
                    "optimizer",
                    "rng",
                    "cuda_rng",
                    "python_rng",
                    "ce_stage_exposure",
                )
            },
        }
        trainer.validate_resume(saved, contract)
        for kind in ("schema", "contract", "rng"):
            bad = copy.deepcopy(saved)
            if kind == "schema":
                bad["schema"] = "original trainer"
            elif kind == "contract":
                bad["contract"]["expanded_ce"] = True
            else:
                del bad["rng"]
            with self.assertRaises(ValueError):
                trainer.validate_resume(bad, contract)

    def test_cursor_checks_recompute_old_new_windows_labels_groups_and_replay(self):
        trainer.validate_continuation(*continuation_fixture())
        own = continuation_fixture()
        own[0]["schema"] = trainer.SCHEMA
        own[0]["stage"] = trainer.stage_totals(
            own[0]["cursor"], own[0]["costs"], own[0]["exposure"], own[0]["trainer_seconds"], own[1]
        )
        trainer.validate_continuation(*own)
        own[0]["stage"]["stage_updates"] += 1
        with self.assertRaisesRegex(ValueError, "stage totals"):
            trainer.validate_continuation(*own)
        for kind in (
            "cursor",
            "ce",
            "replay",
            "labels",
            "stage-groups",
            "stage-window",
            "adam",
            "nan",
            "rewind",
        ):
            with self.subTest(kind=kind):
                args = list(continuation_fixture())
                saved, _, stage = args[:3]
                if kind == "cursor":
                    saved["cursor"] = 1924
                elif kind == "ce":
                    saved["exposure"]["ce_windows"] += 1
                elif kind == "replay":
                    saved["exposure"]["replay_protected_tokens"] += 1
                elif kind == "labels":
                    saved["exposure"]["ce_supported_labels"] += 1
                elif kind == "stage-groups":
                    stage["new"]["groups"]["component:new-fit"] += 1
                elif kind == "stage-window":
                    stage["old"]["window_ids"]["old-fit"] += 1
                elif kind == "adam":
                    saved["optimizer"]["state"][0]["step"] -= 1
                elif kind == "nan":
                    saved["costs"]["ce"] = float("nan")
                else:
                    saved["trainer_seconds"] = 0
                with self.assertRaises(ValueError):
                    trainer.validate_continuation(*args)

    def test_cpu_python_rng_and_optimizer_are_not_touched_by_accounting_checks(self):
        args = continuation_fixture()
        before = copy.deepcopy(args[0])
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        trainer.validate_continuation(*args)
        self.assertTrue(trainer.equal(args[0], before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(random.getstate(), python_rng)

    def test_numeric_forward_readout_match_all_original_trainers(self):
        source = (ROOT / "scripts/w1a8_generalization/modern_train_expanded_ce.py").read_text()
        for name in (
            "modern_train.py",
            "modern_train_protected.py",
            "modern_train_native_replay.py",
        ):
            frozen = (ROOT / "scripts/w1a8_generalization" / name).read_text()
            self.assertEqual(trainer.forward_semantics(source), trainer.forward_semantics(frozen))

    def test_native_bank_migration_and_stage_arithmetic_match_frozen_control(self):
        native = (ROOT / "scripts/w1a8_generalization/modern_train_native_replay.py").read_text()
        expanded = (ROOT / "scripts/w1a8_generalization/modern_train_expanded_ce.py").read_text()
        for name in (
            "native_fit_references",
            "native_schedule",
            "validate_v1_warm_start",
            "stage_totals",
        ):
            bodies = []
            for source in (native, expanded):
                nodes = [
                    node
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.FunctionDef) and node.name == name
                ]
                self.assertEqual(len(nodes), 1)
                bodies.append(ast.dump(nodes[0], include_attributes=False))
            self.assertEqual(*bodies)


if __name__ == "__main__":
    unittest.main()
