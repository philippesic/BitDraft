"""Synthetic CPU admission, guard, and continuation tests; no model/CUDA access."""

from __future__ import annotations

# Explicit project imports and frozen synthetic fixtures need no installation.
# ruff: noqa: E402
import ast
import copy
import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/w1a8_generalization"))
SPEC = importlib.util.spec_from_file_location(
    "modern_train_expanded_ce_resident",
    ROOT / "scripts/w1a8_generalization/modern_train_expanded_ce_resident.py",
)
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)
from modern_actual_data import ActualDataset, _Rows
from modern_actual_union import UnionActualDataset
from modern_resident_rows import ResidentRows
from test_w1a8_modern_native_replay import warm_fixture
from test_w1a8_modern_train_expanded_ce import continuation_fixture

CONFIG = {
    "max_cache_bytes": 1024**3,
    "max_file_bytes": 1024**3,
    "available_reserve_bytes": 2 * 1024**3,
    "capacity_mode": "fallback",
}


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


class ResidentTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = Clock()
        self.guard = trainer.CadencedGuard(
            self.root / "wsl",
            self.root / "c/out",
            control_path=self.root / "control.json",
            c_filesystem=self.root / "c",
            clock=self.clock,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def stat(self, path):
        return SimpleNamespace(f_bavail=trainer.STORAGE_FLOOR + 1024**3, f_frsize=1)

    def controls(self):
        return patch.object(Path, "read_text", return_value='{"rtx5080":{}}')

    def proof_fixture(self):
        raw = {"path": str(self.root / "raw.f32"), "bytes": 16, "shape": [2, 2], "sha256": "a" * 64}
        embedding = {
            "path": str(self.root / "emb.f32"),
            "bytes": 8,
            "shape": [1, 2],
            "sha256": "b" * 64,
        }
        manifest = {
            "raw_files": [raw, copy.deepcopy(raw)],
            "embedding_files": [embedding],
            "groups": [{"id": "first"}, {"id": "second"}],
            "required_embedding_ids": [11, 13, 17],
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        proof = {
            "schema": trainer.PROOF_SCHEMA,
            "status": trainer.PROOF_STATUS,
            "manifest": trainer.file_pin(path),
            "helper": trainer.file_pin(trainer.resident_helpers.__file__),
            "original_reader": trainer.file_pin(
                ROOT / "scripts/w1a8_generalization/modern_actual_data.py"
            ),
            "all_matrices_byte_equal": True,
            "all_sampled_materializations_byte_equal": True,
            "all_embeddings_byte_equal": True,
            "sampled_windows": 2,
            "embedding_rows": 3,
            "unique_descriptors": 2,
            "matrix_bytes_compared": 24,
        }
        proof_path = self.root / "proof.json"
        proof_path.write_text(json.dumps(proof))
        return manifest, path, proof, proof_path

    def test_read_only_audit_guard_throttles_polls_and_stats_each_filesystem_once(self):
        with (
            self.controls() as control,
            patch.object(trainer.os, "statvfs", side_effect=self.stat) as disk,
        ):
            self.guard()
            self.assertEqual(control.call_count, 1)
            self.assertEqual(disk.call_count, 2)
            for _ in range(1000):
                self.guard()
            self.clock.value = 0.249
            self.guard()
            self.assertEqual(control.call_count, 1)
            self.assertEqual(disk.call_count, 2)
            self.clock.value = 0.25
            self.guard()
            self.assertEqual(control.call_count, 2)
            self.clock.value = 0.999
            self.guard()
            self.assertEqual(control.call_count, 3)
            self.assertEqual(disk.call_count, 2)
            self.clock.value = 1.0
            self.guard()
            self.assertEqual(disk.call_count, 4)
            self.assertEqual(
                [args.args[0] for args in disk.call_args_list],
                [self.root / "wsl", self.root / "c"] * 2,
            )

    def test_signal_flag_is_immediate_inside_poll_quiet_period(self):
        with self.controls(), patch.object(trainer.os, "statvfs", side_effect=self.stat):
            self.guard()
        self.clock.value = 0.01
        self.guard.request_stop(signal.SIGTERM, None)
        with (
            patch.object(Path, "read_text", side_effect=AssertionError("control reread")),
            patch.object(trainer.os, "statvfs", side_effect=AssertionError("stat")),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop requested"):
                self.guard()
            with self.assertRaisesRegex(RuntimeError, "stop requested"):
                self.guard.before_write(1)

    def test_pause_and_stop_file_are_checked_at_declared_poll_deadline(self):
        with (
            self.controls(),
            patch.object(trainer.os, "statvfs", side_effect=self.stat),
            patch.object(Path, "exists", return_value=False),
        ):
            self.guard()
        self.clock.value = 0.25
        with (
            self.controls(),
            patch.object(Path, "exists", return_value=True),
            self.assertRaisesRegex(RuntimeError, "stop requested"),
        ):
            self.guard()
        other = trainer.CadencedGuard(self.root, self.root, clock=self.clock)
        with (
            patch.object(Path, "read_text", return_value='{"rtx5080":{"pause_requested":true}}'),
            self.assertRaisesRegex(RuntimeError, "stop requested"),
        ):
            other()

    def test_force_write_freshly_checks_both_disks_and_exact_projected_bytes(self):
        with self.controls(), patch.object(trainer.os, "statvfs", side_effect=self.stat) as disk:
            self.guard()
            self.guard.before_write(123)
            self.assertEqual(disk.call_count, 4)

        def boundary(path):
            available = trainer.STORAGE_FLOOR + (10 if path == self.root / "c" else 0)
            return SimpleNamespace(f_bavail=available, f_frsize=1)

        with self.controls(), patch.object(trainer.os, "statvfs", side_effect=boundary):
            self.guard.before_write(10)
            with self.assertRaisesRegex(RuntimeError, "write reserve"):
                self.guard.before_write(11)
        with (
            self.controls(),
            patch.object(
                trainer.os,
                "statvfs",
                return_value=SimpleNamespace(f_bavail=trainer.STORAGE_FLOOR - 1, f_frsize=1),
            ),
            self.assertRaisesRegex(RuntimeError, "storage floor"),
        ):
            self.guard.before_write(0)

    def test_recovery_checkpoint_write_can_finish_after_stop_but_keeps_fresh_floor(self):
        self.guard.request_stop(signal.SIGTERM)
        with self.controls(), patch.object(trainer.os, "statvfs", side_effect=self.stat) as disk:
            self.guard.before_write(100, recovery=True)
            self.assertEqual(disk.call_count, 2)
        with (
            patch.object(Path, "read_text", side_effect=OSError("control unavailable")),
            patch.object(trainer.os, "statvfs", side_effect=self.stat),
        ):
            self.guard.before_write(100, recovery=True)
        with (
            self.controls(),
            patch.object(
                trainer.os,
                "statvfs",
                return_value=SimpleNamespace(f_bavail=trainer.STORAGE_FLOOR, f_frsize=1),
            ),
            self.assertRaisesRegex(RuntimeError, "write reserve"),
        ):
            self.guard.before_write(1, recovery=True)
        with self.assertRaisesRegex(RuntimeError, "stop requested"):
            self.guard()

    def test_guard_rejects_unbounded_cadence_and_invalid_write_reserve(self):
        for options in (
            {"stop_poll_seconds": 0.251},
            {"disk_poll_seconds": 1.01},
            {"stop_poll_seconds": 0},
            {"disk_poll_seconds": float("nan")},
        ):
            with self.assertRaises(ValueError):
                trainer.CadencedGuard(self.root, self.root, **options)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.guard.before_write(value)

    def test_json_and_checkpoint_writers_reserve_exact_temporary_size(self):
        guard = Mock()
        json_path = self.root / "readout.json"
        trainer.guarded_write(json_path, {"unicode": "é", "rows": [1, 2]}, guard)
        self.assertEqual(guard.before_write.call_args.args, (json_path.stat().st_size,))
        state = {
            "scales": torch.tensor([0.0, -0.0]),
            "cursor": 2400,
            "rng": torch.get_rng_state().clone(),
        }
        rng = torch.get_rng_state().clone()
        checkpoint = self.root / "checkpoint.pt"
        trainer.guarded_checkpoint(checkpoint, state, guard, recovery=True)
        self.assertEqual(guard.before_write.call_args.args, (checkpoint.stat().st_size,))
        self.assertEqual(guard.before_write.call_args.kwargs, {"recovery": True})
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertTrue(trainer.equal(state, loaded))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(checkpoint.with_suffix(".tmp").exists())
        blocked = Mock()
        blocked.before_write.side_effect = RuntimeError("storage floor")
        destination = self.root / "blocked.pt"
        with self.assertRaises(RuntimeError):
            trainer.guarded_checkpoint(destination, state, blocked)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".tmp").exists())

    def test_real_union_proof_pins_all_denominators_and_true_byte_parity(self):
        manifest, manifest_path, proof, proof_path = self.proof_fixture()
        pin, admitted = trainer.validate_resident_proof(proof_path, manifest_path, manifest)
        self.assertEqual(pin, trainer.file_pin(proof_path))
        self.assertEqual(admitted, proof)
        for key in (
            "schema",
            "status",
            "manifest",
            "helper",
            "original_reader",
            "all_matrices_byte_equal",
            "all_sampled_materializations_byte_equal",
            "all_embeddings_byte_equal",
            "sampled_windows",
            "embedding_rows",
            "unique_descriptors",
            "matrix_bytes_compared",
        ):
            bad = copy.deepcopy(proof)
            bad[key] = False if key.startswith("all_") else "wrong"
            proof_path.write_text(json.dumps(bad))
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.validate_resident_proof(proof_path, manifest_path, manifest)
        for key, value in (
            ("sampled_windows", 1),
            ("embedding_rows", 2),
            ("unique_descriptors", 3),
            ("matrix_bytes_compared", 23),
            ("all_embeddings_byte_equal", 1),
        ):
            proof_path.write_text(json.dumps({**proof, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.validate_resident_proof(proof_path, manifest_path, manifest)

    def test_cache_is_bound_to_union_instance_before_ordered_embedding_reads(self):
        data = object.__new__(UnionActualDataset)
        data._rows = _Rows()
        untouched = object.__new__(ActualDataset)
        untouched._rows = _Rows()
        data.manifest = {"required_embedding_ids": [13, 11, 17]}
        observed = []

        def lookup(token, device):
            self.assertIsInstance(data._rows, ResidentRows)
            self.assertEqual(device, "cpu")
            observed.append(token)
            return token * 2

        data.embedding_lookup = lookup
        cache = trainer.bind_resident(data, CONFIG)
        self.assertIs(data._rows, cache)
        guard = Mock()
        embeddings = trainer.load_embeddings(data, "cpu", guard)
        self.assertEqual(observed, [13, 11, 17])
        self.assertEqual(list(embeddings), observed)
        self.assertEqual(guard.call_count, 3)
        self.assertIs(type(untouched._rows), _Rows)
        with self.assertRaisesRegex(ValueError, "already"):
            trainer.bind_resident(data, CONFIG)
        with self.assertRaisesRegex(ValueError, "union reader"):
            trainer.bind_resident(untouched, CONFIG)

    def test_own_resume_is_strict_while_process_cache_can_start_cold(self):
        _, _, proof, proof_path = self.proof_fixture()
        resident = trainer.resident_contract(CONFIG, trainer.file_pin(proof_path), proof)
        contract = {"schema": trainer.SCHEMA, "resident": resident, "expanded_ce": False}
        cache = ResidentRows(**CONFIG)
        runtime = cache.report()
        runtime.update(hits=100, misses=60, cached_bytes=512, cached_files=10, backing_mmap_files=4)
        saved = {
            "schema": trainer.SCHEMA,
            "contract": copy.deepcopy(contract),
            "resident_runtime": runtime,
            **{
                key: "untouched state"
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
        before = copy.deepcopy(saved)
        trainer.validate_resume(saved, contract)
        self.assertEqual(cache.report()["cached_bytes"], 0)
        self.assertEqual(saved, before)
        for kind in (
            "legacy-schema",
            "proof",
            "config",
            "source",
            "runtime-bytes",
            "runtime-counter",
            "missing-state",
        ):
            bad = copy.deepcopy(saved)
            if kind == "legacy-schema":
                bad["schema"] = trainer.frozen_ce_helpers.SCHEMA
            elif kind == "proof":
                bad["contract"]["resident"]["proof"]["sha256"] = "f" * 64
            elif kind == "config":
                bad["contract"]["resident"]["config"]["max_cache_bytes"] += 1
            elif kind == "source":
                bad["resident_runtime"]["source_ancestry"]["resident_reader"]["sha256"] = "f" * 64
            elif kind == "runtime-bytes":
                bad["resident_runtime"]["cached_bytes"] = CONFIG["max_cache_bytes"] + 1
            elif kind == "runtime-counter":
                bad["resident_runtime"]["hits"] = -1
            else:
                del bad["rng"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                trainer.validate_resume(bad, contract)

    def test_old_sampler_warm_state_and_incremental_counters_match_frozen_control(self):
        windows = [{"id": str(i)} for i in range(1414)]
        self.assertEqual(trainer.ce_order(windows), trainer.frozen_ce_helpers.ce_order(windows))
        for cursor in (1920, 2399, 3839, 19199):
            order = trainer.ce_order(windows)
            self.assertEqual(
                trainer.ce_at(windows, order, cursor),
                trainer.frozen_ce_helpers.ce_at(windows, order, cursor),
            )
        packet, expected, schedule, helper = warm_fixture()
        before = copy.deepcopy(packet)
        trainer.validate_v1_warm_start(packet, expected, schedule, helper)
        self.assertEqual(packet, before)
        packet["cursor"] += 1
        with self.assertRaises(ValueError):
            trainer.validate_v1_warm_start(packet, expected, schedule, helper)
        arguments = continuation_fixture()
        arguments[0]["schema"] = trainer.SCHEMA
        state = arguments[0]
        state["stage"] = trainer.stage_totals(
            state["cursor"],
            state["costs"],
            state["exposure"],
            state["trainer_seconds"],
            arguments[1],
        )
        trainer.validate_continuation(*arguments)
        state["exposure"]["ce_windows"] += 1
        with self.assertRaises(ValueError):
            trainer.validate_continuation(*arguments)

    def test_frozen_numeric_helpers_and_training_arithmetic_are_identical(self):
        current = ast.parse(Path(trainer.__file__).read_text())
        frozen = ast.parse(Path(trainer.frozen_ce_helpers.__file__).read_text())
        for name in (
            "forward",
            "inspect",
            "native_fit_references",
            "native_schedule",
            "validate_v1_warm_start",
            "stage_totals",
            "ce_order",
            "ce_at",
            "count_ce",
            "named_optimizer_state",
            "validate_continuation",
        ):

            def body(tree):
                matches = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name
                ]
                self.assertEqual(len(matches), 1)
                return ast.dump(matches[0], include_attributes=False)

            self.assertEqual(body(current), body(frozen), name)

        def training_block(tree):
            matches = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.With)
                and len(node.items) == 1
                and isinstance(node.items[0].context_expr, ast.Call)
                and isinstance(node.items[0].context_expr.func, ast.Name)
                and node.items[0].context_expr.func.id == "shared_round_hard_signs"
            ]
            self.assertEqual(len(matches), 1)
            return ast.dump(matches[0], include_attributes=False)

        self.assertEqual(training_block(current), training_block(frozen))
        main = next(
            node
            for node in current.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = {
            name: min(
                node.lineno
                for node in ast.walk(main)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name
            )
            for name in (
                "validate_resident_proof",
                "bind_resident",
                "load_embeddings",
                "validate_original_reference",
                "validate_v1_warm_start",
                "resident_contract",
            )
        }
        self.assertLess(calls["validate_resident_proof"], calls["bind_resident"])
        self.assertLess(calls["bind_resident"], calls["load_embeddings"])
        self.assertLess(calls["validate_original_reference"], calls["resident_contract"])
        self.assertLess(calls["validate_v1_warm_start"], calls["resident_contract"])
        self.assertEqual(
            trainer.file_pin(trainer.frozen_ce_helpers.__file__)["sha256"], trainer.FROZEN_CE_SHA256
        )
        self.assertEqual(
            trainer.file_pin(trainer.resident_helpers.__file__)["sha256"],
            trainer.RESIDENT_HELPER_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
