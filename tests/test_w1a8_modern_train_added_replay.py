"""Synthetic native JSON and CPU numerical fixtures; no real replies/models/arrays/GPU."""

from __future__ import annotations

# ruff: noqa: E402
import ast
import copy
import json
import math
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/w1a8_generalization")]
import modern_actor
import modern_train_added_replay as trainer
import test_w1a8_modern_selected_head as actor_fixtures
from modern_resident_rows import ResidentRows
from test_w1a8_modern_train_expanded_ce import continuation_fixture

from w1a1_eagle.recurrent_qat import shared_round_hard_signs


def opaque(path, digest="a", size=1):
    return {"path": str(path), "sha256": digest * 64, "bytes": size}


class NativeFixture:
    """All883 windows/956 tokens with407 new positions; files are JSON only."""

    def __init__(self, root):
        self.root = root

        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
            return trainer.file_pin(path)

        self.write = write
        parent = opaque("/fixture/parent.gguf", trainer.INITIALIZER_SHA256[:1])
        parent["sha256"] = trainer.INITIALIZER_SHA256
        target = opaque("/fixture/target.gguf")
        target["sha256"] = trainer.TARGET_SHA256
        self.helper = opaque("/fixture/native-replay", "b")
        self.libs = [
            opaque("/fixture/libggml-cpu.so", "c"),
            opaque("/fixture/libggml-cuda.so", "d"),
            opaque("/fixture/libllama.so", "e"),
        ]
        build_ref = write("original-build.json", {"source": "original native runtime"})
        source_ref = write("original-sources.json", [])
        map_ref = opaque("/fixture/draft-map.ids", "f")
        trace_ref = write(
            "trace-build.json",
            {
                "candidate_build": build_ref,
                "candidate_source_pins": source_ref,
                "candidate_binaries": self.libs,
            },
        )
        native_ref = write(
            "native-build.json",
            {
                "original_runtime_only": True,
                "executable": self.helper,
                "runtime_libraries": self.libs,
                "cpu_library": self.libs[0],
                "cuda_library": self.libs[1],
                "candidate_build": build_ref,
                "candidate_source_pins": source_ref,
            },
        )
        windows = []
        for kind, count, double, groups in (
            ("old", 340, 16, 24),
            ("dev", 171, 22, 12),
            ("new", 372, 35, 24),
        ):
            for i in range(count):
                prefix = 2 if i < double else 1
                group = f"{kind}-group-{i % groups:02d}"
                index = len(windows)
                windows.append(
                    {
                        "id": f"{group}:round-{i:04d}",
                        "role": "development" if kind == "dev" else "fit",
                        "domain": ("code", "reasoning", "prose")[(i % groups) // (groups // 3)],
                        "group": group,
                        "group_id": "component:" + group,
                        "round": i,
                        "anchor": 1,
                        "seed_token": 7,
                        "labels": [11, 13],
                        "max_depth": 2,
                        "native_prefix": prefix,
                        "native_verified_prefix": prefix,
                        "native_proposals": [11, 13],
                        "native_success_mask": [d < prefix for d in range(2)],
                        "native_supervision_reachable_mask": [True, True],
                        "prefix_tokens": [7],
                        "prefix_raw_rows": [index],
                        "seed_raw_row": index,
                        "response_label_start": 3,
                        "native_protections": [
                            {"proposal_depth": d, "token": [11, 13][d], "response_index": 3 + d}
                            for d in range(prefix)
                        ],
                    }
                )
        self.new_fit = [w for w in windows if w["group"].startswith("new-")]
        self.old_fit = [w for w in windows if w["group"].startswith("old-")]
        raw_spec = {
            **opaque("/fixture/raw.f32"),
            "shape": [len(windows), 7680],
            "bytes": len(windows) * 7680 * 4,
        }
        members = []
        for name in ("old", "new"):
            freeze_ref = write(name + "/freeze.json", {"input_pins": [map_ref]})
            members.append(
                {
                    "metadata": {
                        "capture": opaque(root / name / "summary.json"),
                        "source_pins": [freeze_ref],
                    },
                    "window_ids": [
                        w["id"]
                        for w in windows
                        if (w["group"].startswith("new-")) == (name == "new")
                    ],
                }
            )
        self.manifest = {
            "windows": windows,
            "members": members,
            "target": target,
            "parent": parent,
            "raw_files": [raw_spec],
            "raw_rows": [
                {"file": 0, "row": i, "raw_sha256": "a" * 64, "target_position": 0}
                for i in range(len(windows))
            ],
            "runtime_ancestry": {
                "trace_build": trace_ref,
                "command": {"environment": {"W1A8_TRACE_MAP": map_ref["path"]}},
            },
        }
        self.manifest_ref = write("union.json", self.manifest)
        self.batches, packets = [], []
        for batch_index, start in enumerate(range(0, len(windows), 96)):
            subset = windows[start : start + 96]
            cases = [
                {
                    **{key: w[key] for key in trainer.CASE_FIELDS},
                    "classification": "all_original_and_new_emitted_native_successes",
                }
                for w in subset
            ]
            packet = {
                "schema": "w1a8_actual_native_cases_v1",
                "diagnostic_only": True,
                "manifest": self.manifest_ref,
                "target": target,
                "models": {"parent": parent},
                "build": native_ref,
                "helper": self.helper,
                "runtime_libraries": self.libs,
                "cpu_library": self.libs[0],
                "cuda_library": self.libs[1],
                "draft_map": map_ref,
                "files": [raw_spec],
                "raw_row_columns": ["source_id", "file", "row", "sha256", "target_position"],
                "raw_rows": [[start + j, 0, start + j, "a" * 64, 0] for j in range(len(subset))],
                "cases": cases,
            }
            packet_ref = write(f"cases-{batch_index:02d}.json", packet)
            rows = {}
            for case in cases:
                prefix = case["native_prefix"]
                rows[case["id"]] = {
                    "case": copy.deepcopy(case),
                    "correct_prefix": 2,
                    "parent_replay_prefix": 2,
                    "native_reference_prefix": prefix,
                    "calibrated_reference_prefix": prefix,
                    "native_successes_outside_parent_calibration": 0,
                    "lost_calibrated_reference_tokens": 0,
                    "steps": [
                        {
                            "depth": d,
                            "decoder_position": 1 + d,
                            "cache_position_max": 1 + d,
                            "input_token": 7 if d == 0 else 11,
                            "label": [11, 13][d],
                            "proposed": [11, 13][d],
                            "correct": True,
                            "gold_supported": True,
                            "gold_margin": 0.02 if d == 0 else 5.0,
                            "native_success": d < prefix,
                            "native_supervision_reachable": True,
                        }
                        for d in range(2)
                    ],
                }
            tokens = sum(w["native_prefix"] for w in subset)
            reply = {
                "status": "ACTUAL_NATIVE_REPLAY_COMPLETE",
                "model_role": "parent",
                "model": parent,
                "target": target,
                "cases_pin": packet_ref,
                "build": native_ref,
                "calibration": None,
                "windows": rows,
                "native_reference_tokens": tokens,
                "calibrated_reference_tokens": tokens,
                "native_successes_outside_parent_calibration": 0,
                "lost_calibrated_reference_tokens": 0,
                "context": 4096,
                "batch": 512,
                "ubatch": 512,
                "prefix_chunk": 64,
                "kv": "F16",
                "correctness_mode": 32,
                "activation_bits": 8,
                "array_bytes_written": 0,
                "draft_placement": "CUDA",
                "host_logits": "full target vocabulary; no backend sampler attached",
                "target_placement": "CPU native embedding provider only; no target forward",
                "hardware": "NVIDIA GeForce RTX 5080",
                "loaded_images": self.libs + [self.helper],
                "initial_loaded_images": self.libs + [self.helper],
            }
            write(f"expanded/batch-{batch_index:02d}/parent/replay.json", reply)
            self.batches.append((packet, reply))
            packets.append(
                {"packet": packet_ref, "windows": len(cases), "native_successes": tokens}
            )
        record = write(
            "packet-manifest.json",
            {
                "schema": "all956_native_success_cases_v1",
                "manifest": self.manifest_ref,
                "packets": packets,
                "native_tokens": 956,
                "windows": 883,
            },
        )
        self.plan_ref = write(
            "plan.json",
            {
                "arms": [
                    {"arm": "expanded", "packet_manifest": record, "output": str(root / "expanded")}
                ]
            },
        )


class AddedReplayTests(unittest.TestCase):
    def setUp(self):
        # Generated JSON evidence lives in RAM, away from the project's guarded disk floor.
        self.temp = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = NativeFixture(self.root)

    def admitted(self):
        return trainer.load_added_references(
            self.fixture.plan_ref["path"],
            self.fixture.manifest_ref["path"],
            self.fixture.manifest,
            self.fixture.new_fit,
            {11, 13},
            lambda: None,
            expected_output=self.root / "expanded",
        )

    def test_all407_new_positions372windows24groups_admitted_from_ten_parent_batches(self):
        schedule, refs, report = self.admitted()
        self.assertEqual((report["tokens"], report["groups"], report["windows"]), (407, 24, 372))
        self.assertEqual(sum(sum(row["protected"]) for row in refs.values()), 407)
        self.assertEqual(refs[self.fixture.new_fit[0]["id"]]["margins"], [0.1, 2.0])
        self.assertTrue(
            all(row["role"] == "fit" and key.startswith("new-") for key, row in refs.items())
        )
        self.assertEqual(
            len([ref for ref in report["sources"] if ref["path"].endswith("parent/replay.json")]),
            10,
        )
        exposed = trainer.empty_added_exposure()
        for cursor in range(1920):
            group, window = schedule.at(cursor)
            trainer.count_added(exposed, group, window, refs[window["id"]])
        self.assertEqual(len(exposed["positions"]), 407)
        self.assertEqual(len(exposed["window_ids"]), 372)
        self.assertEqual(set(exposed["groups"].values()), {80})
        contract = {"added_replay": True, "added_reference": report}
        self.assertEqual(trainer.expected_added_exposure(5760, contract), exposed)
        self.assertEqual(
            trainer.expected_added_exposure(5760, {**contract, "added_replay": False}),
            trainer.empty_added_exposure(),
        )

    def test_parent_case_target_build_runtime_and_loaded_image_pins_are_strict(self):
        packet, reply = self.fixture.batches[0]
        for kind in ("role", "model", "target", "case_pin", "build", "precision", "images"):
            bad = copy.deepcopy(reply)
            if kind == "role":
                bad["model_role"] = "candidate"
            elif kind == "model":
                bad["model"]["sha256"] = "0" * 64
            elif kind == "target":
                bad["target"]["sha256"] = "0" * 64
            elif kind == "case_pin":
                bad["cases_pin"]["sha256"] = "0" * 64
            elif kind == "build":
                bad["build"]["sha256"] = "0" * 64
            elif kind == "precision":
                bad["kv"] = "F32"
            else:
                bad["loaded_images"] = bad["loaded_images"][:-1]
            metadata = trainer.NativeMetadata()
            runtime = trainer.source_runtime(self.fixture.manifest, metadata)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_native_batch(
                    packet,
                    bad,
                    reply["cases_pin"],
                    self.fixture.manifest,
                    self.fixture.manifest_ref,
                    runtime,
                    metadata,
                )

    def test_missing_duplicate_dev_oldfit_and_case_lineage_are_rejected(self):
        for kind in ("missing", "duplicate", "dev", "oldfit", "case", "raw"):
            batches = copy.deepcopy(self.fixture.batches)
            chosen = self.fixture.new_fit
            if kind == "missing":
                batches[-1][1]["windows"].pop(next(iter(batches[-1][1]["windows"])))
            elif kind == "duplicate":
                batches.append(copy.deepcopy(batches[-1]))
            elif kind == "dev":
                chosen = [
                    next(w for w in self.fixture.manifest["windows"] if w["role"] == "development")
                ]
            elif kind == "oldfit":
                chosen = self.fixture.old_fit
            elif kind == "case":
                identifier = self.fixture.new_fit[0]["id"]
                for packet, reply in batches:
                    if identifier in reply["windows"]:
                        reply["windows"][identifier]["case"]["group_id"] = "other"
            else:
                batches[-1][0]["raw_rows"][0][3] = "bad"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.admit_added_references(self.fixture.manifest, chosen, batches, {11, 13})

    def test_native_margin_label_depth_support_and_response_joins_are_strict(self):
        identifier = self.fixture.new_fit[0]["id"]
        for kind in ("margin", "nan", "label", "depth", "support", "correct"):
            batches = copy.deepcopy(self.fixture.batches)
            row = next(
                reply["windows"][identifier]
                for _, reply in batches
                if identifier in reply["windows"]
            )
            step = row["steps"][0]
            if kind == "margin":
                step["gold_margin"] = None
            elif kind == "nan":
                step["gold_margin"] = math.nan
            elif kind == "label":
                step["proposed"] = 999
            elif kind == "depth":
                step["depth"] = 1
            elif kind == "support":
                step["gold_supported"] = False
            else:
                step["correct"] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.admit_added_references(
                    self.fixture.manifest, self.fixture.new_fit, batches, {11, 13}
                )
        with self.assertRaises(ValueError):
            trainer.admit_added_references(
                self.fixture.manifest, self.fixture.new_fit, self.fixture.batches, {11}
            )

    def test_json_only_reader_rejects_duplicate_keys_nonfinite_and_payload_suffixes(self):
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            trainer.NativeMetadata().read(duplicate)
        duplicate.write_text('{"x":NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            trainer.NativeMetadata().read(duplicate)
        for suffix in (".gguf", ".pt", ".f32"):
            with self.assertRaisesRegex(ValueError, "JSON metadata only"):
                trainer.NativeMetadata().read(self.root / ("not-opened" + suffix))

    def test_false_numerical_block_matches_original_adam_rng_scales_and_old_replay(self):
        fixture = actor_fixtures.SelectedHeadTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        modern_actor.enable_scale_only(fixture.actor)
        source = ast.parse(Path(trainer.__file__).read_text())
        frozen = ast.parse(Path(trainer.resident_trainer.__file__).read_text())

        def block(tree):
            return next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.With)
                and len(n.items) == 1
                and isinstance(n.items[0].context_expr, ast.Call)
                and isinstance(n.items[0].context_expr.func, ast.Name)
                and n.items[0].context_expr.func.id == "shared_round_hard_signs"
            )

        new_block, old_block = block(source), block(frozen)
        stripped = copy.deepcopy(new_block)
        stripped.body = [
            n
            for n in stripped.body
            if not (
                isinstance(n, ast.If)
                and isinstance(n.test, ast.Name)
                and n.test.id == "added_enabled"
            )
        ]
        self.assertEqual(
            ast.dump(stripped, include_attributes=False),
            ast.dump(old_block, include_attributes=False),
        )
        w = {"id": "old", "labels": [1, 3], "native_supervision_reachable_mask": [True, True]}
        refs = {"old": {"protected": [True, False], "margins": [0.2, None]}}

        def run(numeric_block, added=False):
            actor = copy.deepcopy(fixture.actor)
            parameters = actor.scale_parameters()
            optimizer = torch.optim.Adam(parameters, lr=1e-5)
            for index, parameter in enumerate(parameters):
                parameter.grad = torch.full_like(parameter, (index + 1) / 100)
            optimizer.step()
            for state in optimizer.state.values():
                state["step"].fill_(3840)
            optimizer.zero_grad(set_to_none=True)
            calls = []

            def forward(window):
                calls.append(window["id"])
                return sum(module(torch.ones(2, 35) * 0.1) for module in actor.linears.values())

            schedule = SimpleNamespace(at=Mock(return_value=("old-group", w)))
            new_window = {**w, "id": "new"}
            extra = SimpleNamespace(
                at=Mock(return_value=("new-group", new_window))
                if added
                else Mock(side_effect=AssertionError("extra stream touched"))
            )
            environment = {
                "torch": torch,
                "time": time,
                "actor": actor,
                "shared_round_hard_signs": shared_round_hard_signs,
                "loss": 0.0,
                "ce_value": 0.0,
                "supported": 0,
                "ce_counts": [],
                "cursor": 3840,
                "ce_fit": [w],
                "order": [0],
                "ce_at": trainer.ce_at,
                "forward": forward,
                "reverse": torch.tensor([-1, 0, -1, 1]),
                "device": torch.device("cpu"),
                "t": time.perf_counter(),
                "schedule": schedule,
                "native_refs": refs,
                "normalized_replay_loss": trainer.normalized_replay_loss,
                "args": SimpleNamespace(retention=10.0),
                "selected_parameters": parameters,
                "optimizer": optimizer,
                "paths": list(actor.linears),
                "added_enabled": added,
                "START": 3840,
                "added_refs": {"new": refs["old"]},
                "added_schedule": extra,
            }
            module = ast.fix_missing_locations(
                ast.Module(body=[copy.deepcopy(numeric_block)], type_ignores=[])
            )
            with patch.object(torch.cuda, "synchronize", return_value=None):
                exec(compile(module, "<actual-numerical-block>", "exec"), environment)
            return (
                actor,
                optimizer.state_dict(),
                calls,
                schedule.at.call_args_list,
                float(environment["loss"].detach()),
                extra.at.call_args_list,
                environment.get("added_value", 0.0),
            )

        rng = torch.get_rng_state().clone()
        first = run(old_block)
        torch.set_rng_state(rng)
        second = run(new_block)
        self.assertTrue(
            all(
                torch.equal(first[0].linears[p].scale_offset, second[0].linears[p].scale_offset)
                for p in first[0].linears
            )
        )
        self.assertTrue(trainer.equal(first[1], second[1]))
        self.assertEqual(first[2:], second[2:])
        self.assertEqual(second[2], ["old", "old", "old"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        third = run(new_block, added=True)
        self.assertEqual(third[2], ["old", "old", "old", "new"])
        self.assertEqual(first[3], third[3])
        self.assertEqual(third[5][0].args, (0,))
        self.assertAlmostEqual(third[4] - second[4], 10 * third[6], places=4)

    def test_own_schema_parent_source_and_legacy_helpers_are_frozen(self):
        self.assertNotEqual(trainer.SCHEMA, trainer.resident_trainer.SCHEMA)
        self.assertEqual(
            trainer.file_pin(trainer.resident_trainer.__file__)["sha256"],
            trainer.RESIDENT_TRAINER_SHA256,
        )
        current = Path(trainer.__file__).read_text()
        old = Path(trainer.resident_trainer.__file__).read_text()
        self.assertEqual(trainer.forward_semantics(current), trainer.forward_semantics(old))
        for name in (
            "native_schedule",
            "native_fit_references",
            "ce_order",
            "ce_at",
            "validate_v1_warm_start",
        ):

            def get(source):
                return next(
                    n
                    for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )

            self.assertEqual(
                ast.dump(get(current), include_attributes=False),
                ast.dump(get(old), include_attributes=False),
            )

    def test_strict_resume_counters_and_3840_stage_preserve_original1920_accounting(self):
        _, _, report = self.admitted()
        arguments = continuation_fixture(4320)
        parent_arguments = continuation_fixture(3840)
        state, parent = arguments[0], parent_arguments[0]
        parent.update(
            contract={"original": "resident parent"}, ce_stage_exposure=parent_arguments[2]
        )
        start = trainer.stage_origin(parent, {"sha256": "fixed parent fixture"})
        cache = ResidentRows()
        runtime = cache.report()
        contract = {
            "schema": trainer.SCHEMA,
            "added_replay": True,
            "stage_start": start,
            "added_reference": report,
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
        parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(9)]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for row in optimizer.state.values():
            row["step"].fill_(4320)
        state.update(
            schema=trainer.SCHEMA,
            contract=contract,
            optimizer=optimizer.state_dict(),
            ce_stage_exposure=arguments[2],
            resident_runtime=runtime,
            scales="synthetic schema payload",
            rng=torch.get_rng_state(),
            cuda_rng=torch.zeros(8, dtype=torch.uint8),
            python_rng=random.getstate(),
            added_exposure=trainer.expected_added_exposure(4320, contract),
            added_costs={"forward_seconds": 0.2},
        )
        state["costs"]["ce"] += 0.5
        state["trainer_seconds"] += 2
        state["stage"] = trainer.stage_totals(
            4320, state["costs"], state["exposure"], state["trainer_seconds"], arguments[1]
        )
        state["objective_stage"] = trainer.objective_totals(
            4320,
            state["costs"],
            state["exposure"],
            state["trainer_seconds"],
            arguments[2],
            start,
            state["added_exposure"],
            state["added_costs"],
        )
        trainer.validate_resume(state, contract)
        trainer.validate_continuation(*arguments)
        self.assertEqual(state["objective_stage"]["stage_updates"], 480)
        self.assertEqual(state["objective_stage"]["stage_costs"], {"ce": 0.5})
        self.assertEqual(state["objective_stage"]["added_costs"], {"forward_seconds": 0.2})
        self.assertEqual(state["stage"]["stage_updates"], 2400)
        for kind in ("schema", "contract", "added-counter", "old-counter", "cost", "adam"):
            bad = copy.deepcopy(state)
            if kind == "schema":
                bad["schema"] = trainer.resident_trainer.SCHEMA
            elif kind == "contract":
                bad["contract"]["added_replay"] = False
            elif kind == "added-counter":
                bad["added_exposure"]["windows"] -= 1
            elif kind == "old-counter":
                bad["exposure"]["replay_windows"] -= 1
            elif kind == "cost":
                bad["added_costs"]["forward_seconds"] = math.nan
            else:
                bad["optimizer"]["param_groups"][0]["lr"] = 0.1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_resume(bad, contract)
                trainer.validate_continuation(bad, *arguments[1:])


if __name__ == "__main__":
    unittest.main()
