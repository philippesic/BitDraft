"""Synthetic CPU tests for immutable native-array RAM snapshots; no model or CUDA."""

from __future__ import annotations

# Project module path is explicit; no package installation is required.
# ruff: noqa: E402
import copy
import gc
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/w1a8_generalization"))
import modern_resident_rows as resident
from modern_actual_data import _Rows


class ResidentRowsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, name="array", *, bits=None, shape=(2, 3)):
        if bits is None:
            bits = np.arange(np.prod(shape), dtype="<u4")
        payload = np.asarray(bits, dtype="<u4").tobytes()
        path = self.root / (name + ".f32")
        path.write_bytes(payload)
        spec = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "shape": list(shape),
        }
        return spec, payload

    def cache(self, **kwargs):
        return resident.ResidentRows(available_reserve_bytes=0, **kwargs)

    def test_defaults_and_configuration_reject_invalid_limits(self):
        cache = resident.ResidentRows()
        self.assertEqual(cache.max_cache_bytes, 1024**3)
        self.assertEqual(cache.available_reserve_bytes, 2 * 1024**3)
        for arguments in (
            {"max_cache_bytes": -1},
            {"max_cache_bytes": True},
            {"max_cache_bytes": 10, "max_file_bytes": 11},
            {"available_reserve_bytes": -1},
            {"capacity_mode": "evict"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resident.ResidentRows(**arguments)

    def test_exact_bits_signed_zero_nan_payload_and_immutable_bytes_backing(self):
        bits = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC01234, 0xFF800000]
        spec, payload = self.fixture(bits=bits)
        cache = self.cache()
        result = cache.matrix(spec)
        self.assertEqual(result.tobytes(), payload)
        self.assertEqual(result.view("<u4").ravel().tolist(), bits)
        self.assertEqual(result.dtype, np.dtype("<f4"))
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(result.flags.writeable)
        with self.assertRaises(ValueError):
            result[0, 0] = 9
        with self.assertRaises(ValueError):
            result.setflags(write=True)
        base = result
        while isinstance(base, np.ndarray):
            self.assertFalse(base.flags.writeable)
            with self.assertRaises(ValueError):
                base.setflags(write=True)
            base = base.base
        self.assertIsInstance(base, bytes)
        self.assertFalse(np.shares_memory(result, next(iter(cache._backing.maps.values()))))

    def test_known_path_hit_performs_no_filesystem_or_available_memory_calls(self):
        spec, payload = self.fixture()
        cache = self.cache()
        first = cache.matrix(spec)
        with (
            patch.object(Path, "resolve", side_effect=AssertionError("resolve")),
            patch.object(Path, "open", side_effect=AssertionError("open")),
            patch("os.stat", side_effect=AssertionError("stat")),
            patch.object(np, "memmap", side_effect=AssertionError("mmap")),
            patch.object(resident, "linux_available_bytes", side_effect=AssertionError("memory")),
        ):
            second = cache.matrix(copy.deepcopy(spec))
            self.assertIs(first, second)
            self.assertEqual(second.tobytes(), payload)
        self.assertEqual(cache.report()["hits"], 1)

    def test_cached_snapshot_survives_source_edit_without_aliasing_mutable_mmap(self):
        spec, payload = self.fixture()
        cache = self.cache()
        snapshot = cache.matrix(spec)
        Path(spec["path"]).write_bytes(bytes([255]) * spec["bytes"])
        self.assertIs(cache.matrix(spec), snapshot)
        self.assertEqual(snapshot.tobytes(), payload)
        # A new cache must authenticate current bytes and reject the stale pin.
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.cache().matrix(spec)

    def test_descriptor_conflicts_on_hits_reject_before_any_payload_read(self):
        spec, _ = self.fixture()
        cache = self.cache()
        cache.matrix(spec)
        for kind in ("sha", "shape", "size", "dtype", "suffix"):
            changed = copy.deepcopy(spec)
            if kind == "sha":
                changed["sha256"] = "f" * 64
            elif kind == "shape":
                changed["shape"] = [3, 2]
            elif kind == "size":
                changed["bytes"] += 4
            elif kind == "dtype":
                changed["shape"] = [True, 6]
            else:
                changed["path"] = str(self.root / "weights.gguf")
            with (
                self.subTest(kind=kind),
                patch.object(cache._backing, "matrix", side_effect=AssertionError("opened")),
                self.assertRaises(ValueError),
            ):
                cache.matrix(changed)

    def test_resolved_aliases_share_snapshot_and_cannot_change_identity(self):
        spec, _ = self.fixture()
        cache = self.cache()
        first = cache.matrix(spec)
        alias = self.root / "alias.f32"
        alias.symlink_to(spec["path"])
        alternate = {**spec, "path": str(alias)}
        self.assertIs(cache.matrix(alternate), first)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            cache.matrix({**alternate, "sha256": "f" * 64})
        self.assertEqual(cache.report()["cached_files"], 1)
        bad_alias = self.root / "weights.f32"
        bad_target = self.root / "weights.pt"
        bad_target.write_bytes(b"not a native capture")
        bad_alias.symlink_to(bad_target)
        with self.assertRaisesRegex(ValueError, "resolved source"):
            cache.matrix({**spec, "path": str(bad_alias)})

    def test_source_changed_before_first_read_and_fallback_read_both_fail_authentication(self):
        spec, _ = self.fixture()
        Path(spec["path"]).write_bytes(b"x" * spec["bytes"])
        for size in (0, 1024):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "source changed"):
                self.cache(max_cache_bytes=size).matrix(spec)

    def test_byte_budget_and_file_limit_fallback_are_deterministic_without_eviction(self):
        first, first_bytes = self.fixture("first")
        second, second_bytes = self.fixture("second", bits=np.arange(6, 12, dtype="<u4"))
        cache = self.cache(max_cache_bytes=first["bytes"])
        original = cache.matrix(first)
        uncached = cache.matrix(second)
        self.assertIsInstance(uncached, np.memmap)
        self.assertEqual(uncached.tobytes(), second_bytes)
        self.assertIs(cache.matrix(first), original)
        self.assertEqual(original.tobytes(), first_bytes)
        report = cache.report()
        self.assertEqual(report["cached_bytes"], first["bytes"])
        self.assertEqual(report["fallback_reasons"], {"cache_limit": 1})
        oversized = self.cache(max_cache_bytes=1024, max_file_bytes=first["bytes"] - 1)
        self.assertEqual(oversized.matrix(first).tobytes(), first_bytes)
        self.assertEqual(oversized.report()["fallback_reasons"], {"file_limit": 1})

    def test_available_ram_guard_checks_again_after_hashing_before_copy(self):
        spec, payload = self.fixture()
        cache = resident.ResidentRows(available_reserve_bytes=100)
        with patch.object(resident, "linux_available_bytes", side_effect=[124, 123]) as available:
            result = cache.matrix(spec)
        self.assertEqual(available.call_count, 2)
        self.assertIsInstance(result, np.memmap)
        self.assertEqual(result.tobytes(), payload)
        self.assertEqual(cache.report()["cached_bytes"], 0)
        self.assertEqual(cache.report()["fallback_reasons"], {"available_limit": 1})
        enough = resident.ResidentRows(available_reserve_bytes=100)
        with patch.object(resident, "linux_available_bytes", return_value=124):
            result = enough.matrix(spec)
        self.assertNotIsInstance(result, np.memmap)
        self.assertEqual(enough.report()["cached_bytes"], 24)

    def test_explicit_error_mode_never_silently_falls_back(self):
        spec, _ = self.fixture()
        for kwargs, available in (({"max_cache_bytes": 0}, 10**9), ({}, 0)):
            cache = self.cache(capacity_mode="error", **kwargs)
            with (
                patch.object(resident, "linux_available_bytes", return_value=available),
                self.assertRaises(MemoryError),
            ):
                cache.matrix(spec)
            self.assertEqual(cache.report()["fallbacks"], 0)
            self.assertEqual(cache.report()["cached_bytes"], 0)
        cache = self.cache(capacity_mode="error")
        with (
            patch.object(resident, "linux_available_bytes", side_effect=[1000, 0]),
            self.assertRaises(MemoryError),
        ):
            cache.matrix(spec)

    def test_missing_memavailable_is_explicit_and_safe_in_each_policy(self):
        spec, payload = self.fixture()
        cache = self.cache()
        with patch.object(resident, "linux_available_bytes", side_effect=OSError("unreadable")):
            self.assertEqual(cache.matrix(spec).tobytes(), payload)
        self.assertEqual(cache.report()["fallback_reasons"], {"available_unreadable": 1})
        strict = self.cache(capacity_mode="error")
        with (
            patch.object(resident, "linux_available_bytes", side_effect=OSError("unreadable")),
            self.assertRaises(MemoryError),
        ):
            strict.matrix(spec)

    def test_private_four_mmap_backing_and_no_descriptor_growth_from_ram_arrays(self):
        fixtures = [self.fixture(f"file-{i}") for i in range(20)]
        baseline = len(os.listdir("/proc/self/fd"))
        cache = self.cache(max_cache_bytes=1024**2)
        resident_arrays = [cache.matrix(spec) for spec, _ in fixtures]
        self.assertEqual(len(resident_arrays), 20)
        self.assertIsInstance(cache._backing, _Rows)
        self.assertEqual(cache._backing.limit, 4)
        self.assertLessEqual(len(cache._backing.maps), 4)
        gc.collect()
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), baseline + 4)
        self.assertTrue(all(not isinstance(x, np.memmap) for x in resident_arrays))
        # Keeping RAM arrays alive must not retain mmap descriptors after cache release.
        del cache
        gc.collect()
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), baseline)
        fallback = self.cache(max_cache_bytes=0)
        for spec, _ in fixtures:
            fallback.matrix(spec)
        self.assertLessEqual(len(fallback._backing.maps), 4)
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), baseline + 4)

    def test_accounting_and_source_ancestry_are_serializable_defensive_copies(self):
        spec, _ = self.fixture()
        cache = self.cache()
        cache.matrix(spec)
        cache.matrix(spec)
        value = cache.report(include_descriptors=True)
        self.assertEqual(value["hits"], 1)
        self.assertEqual(value["misses"], 1)
        self.assertEqual(value["fallbacks"], 0)
        self.assertEqual(value["cached_bytes"], spec["bytes"])
        self.assertEqual(value["descriptors"][0], {**spec, "resident": True})
        for pin in value["source_ancestry"].values():
            self.assertEqual(
                hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest(), pin["sha256"]
            )
        json.dumps(value, allow_nan=False)
        value["source_ancestry"]["resident_reader"]["sha256"] = "changed"
        self.assertNotEqual(
            cache.report()["source_ancestry"]["resident_reader"]["sha256"], "changed"
        )
        self.assertEqual(cache.report()["authenticated_descriptors"], 1)
        self.assertNotIn("descriptors", cache.report())

    def test_linux_memavailable_parser_requires_kernel_units(self):
        from io import StringIO

        with patch.object(
            Path, "open", return_value=StringIO("MemTotal: 10 kB\nMemAvailable: 7 kB\n")
        ):
            self.assertEqual(resident.linux_available_bytes(), 7 * 1024)
        for text in ("MemFree: 9 kB\n", "MemAvailable: 7 bytes\n", "MemAvailable: -1 kB\n"):
            with (
                patch.object(Path, "open", return_value=StringIO(text)),
                self.assertRaises((RuntimeError, ValueError)),
            ):
                resident.linux_available_bytes()


if __name__ == "__main__":
    unittest.main()
