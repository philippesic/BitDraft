"""Bounded synthetic data-contract tests; the real36 join is separate evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts/w1a8_generalization"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "modern_actual_data", SCRIPTS / "modern_actual_data.py"
)
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


class ActualDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = np.arange(4 * data.WIDTH, dtype=np.float32).reshape(4, data.WIDTH)
        self.raw_path = self.root / "raw.f32"
        self.raw.tofile(self.raw_path)
        self.emb = np.stack(
            [
                np.full(data.HIDDEN, 0.25, dtype=np.float32),
                np.full(data.HIDDEN, -0.5, dtype=np.float32),
            ]
        )
        self.emb_path = self.root / "layer0.f32"
        self.emb.tofile(self.emb_path)
        self.window = {
            "id": "example:round-0000",
            "anchor": 2,
            "prefix_tokens": [19, 23],
            "prefix_raw_rows": [2, 0],
            "seed_raw_row": 3,
            "seed_token": 27,
            "labels": [31, 37],
            "max_depth": 2,
            "native_prefix": 1,
            "native_protections": [{"proposal_depth": 0, "token": 31}],
        }
        self.manifest = {
            "schema": data.SCHEMA,
            "target": {"sha256": data.TARGET_SHA256},
            "parent": {"sha256": data.PARENT_SHA256},
            "windows": [self.window],
            "raw_files": [{**data.pin(self.raw_path), "shape": list(self.raw.shape)}],
            "raw_rows": [
                {
                    "file": 0,
                    "row": i,
                    "raw_sha256": data.row_hash(self.raw[i]),
                    "target_position": 17 + i,
                }
                for i in range(4)
            ],
            "embedding_files": [{**data.pin(self.emb_path), "shape": list(self.emb.shape)}],
            "embedding_rows": {"19": {"file": 0, "row": 1, "sha256": data.row_hash(self.emb[1])}},
        }
        self.path = self.root / "manifest.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_metadata_only_load_and_bounded_source_indexed_materialization(self):
        with patch.object(data._Rows, "matrix", side_effect=AssertionError("payload loaded early")):
            dataset = data.ActualDataset(self.path)
        got = dataset.materialize(dataset.windows[0])
        self.assertEqual(got.prefix_tokens, [19, 23])
        np.testing.assert_array_equal(got.prefix_rawfeatures.numpy(), self.raw[[2, 0]])
        np.testing.assert_array_equal(got.seed_rawfeature.numpy(), self.raw[3])
        self.assertEqual(str(got.prefix_rawfeatures.dtype), "torch.float32")
        # Crossed target positions17..20 are preserved; cache positions0..1 are not substituted.
        self.assertEqual(dataset.manifest["raw_rows"][2]["target_position"], 19)
        self.assertEqual(
            set(p.name for p in self.root.iterdir()), {"raw.f32", "layer0.f32", "manifest.json"}
        )

    def test_wrong_target_parent_schema_fail_before_payload(self):
        for name in ("target", "parent", "schema"):
            with self.subTest(name=name):
                modified = copy.deepcopy(self.manifest)
                if name == "schema":
                    modified[name] = "unknown"
                else:
                    modified[name]["sha256"] = "0" * 64
                self.path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(ValueError, "ancestry/schema"):
                    data.ActualDataset(self.path)

    def test_changed_file_and_changed_row_fail_closed(self):
        changed = self.raw.copy()
        changed[0, 0] = -1
        changed.tofile(self.raw_path)
        dataset = data.ActualDataset(self.path)
        with self.assertRaisesRegex(ValueError, "source changed"):
            dataset.materialize(dataset.windows[0])
        self.manifest["raw_files"][0].update(data.pin(self.raw_path))
        self.save()
        dataset = data.ActualDataset(self.path)
        with self.assertRaisesRegex(ValueError, "row shape/hash/finite"):
            dataset.materialize(dataset.windows[0])

    def test_nonfinite_raw_rows_are_rejected_even_with_updated_hashes(self):
        self.raw[2, 0] = np.nan
        self.raw.tofile(self.raw_path)
        self.manifest["raw_files"][0].update(data.pin(self.raw_path))
        self.manifest["raw_rows"][2]["raw_sha256"] = data.row_hash(self.raw[2])
        self.save()
        dataset = data.ActualDataset(self.path)
        with self.assertRaisesRegex(ValueError, "finite"):
            dataset.materialize(dataset.windows[0])

    def test_prefix_geometry_unknown_row_and_foreign_window(self):
        dataset = data.ActualDataset(self.path)
        with self.assertRaisesRegex(ValueError, "not from this manifest"):
            dataset.materialize({**self.window, "id": "other"})
        for change in ({"anchor": 3}, {"prefix_raw_rows": [999, 0]}):
            with self.subTest(change=change):
                self.manifest["windows"] = [{**self.window, **change}]
                self.save()
                dataset = data.ActualDataset(self.path)
                with self.assertRaises(ValueError):
                    dataset.materialize(dataset.windows[0])

    def test_exact_native_embedding_lookup_and_missing_id(self):
        dataset = data.ActualDataset(self.path)
        got = dataset.embedding_lookup(19)
        np.testing.assert_array_equal(got.numpy(), self.emb[1])
        self.assertFalse(got.requires_grad)
        with self.assertRaisesRegex(KeyError, "unavailable"):
            dataset.embedding_lookup(20)

    def test_source_aliases_ambiguity_and_missing_source(self):
        aliases = [
            {"kind": "fc", "semantic": "same-raw", "call": 3},
            {"kind": "fc", "semantic": "same-raw", "call": 6},
        ]
        self.assertEqual(data.select_source({"state": aliases}, "state")["call"], 6)
        for choices in (
            [],
            [aliases[0], {"kind": "fc", "semantic": "other-raw"}],
            [aliases[0], {"kind": "prenorm", "semantic": (5, 0)}],
        ):
            with self.assertRaises(ValueError):
                data.select_source({"state": choices}, "state")

    def test_cache_overwrite_truncation_and_hole_fail_closed(self):
        cache = ["pos0", "pos1", "pos2"]
        self.assertEqual(data.truncate_cache(cache, 1), ["pos0"])
        self.assertEqual(cache, ["pos0", "pos1", "pos2"])
        for position in (-1, 4, 1.0):
            with self.assertRaisesRegex(ValueError, "hole"):
                data.truncate_cache(cache, position)


if __name__ == "__main__":
    unittest.main()
