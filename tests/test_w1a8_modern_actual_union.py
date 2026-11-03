"""Synthetic source-index union audits; no real captures/models/GPU are opened."""

from __future__ import annotations

# ruff: noqa: E402
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/w1a8_generalization")]
import modern_actual_union as union
from modern_actual_data import ActualDataset


def write_json(path, value):
    path.write_bytes(union.encoded(value) + b"\n")
    return union.pin(path)


def array(path, values):
    values = np.asarray(values, dtype="<f4")
    path.write_bytes(values.tobytes())
    return {**union.pin(path), "shape": list(values.shape)}


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


class ActualUnionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.target = {
            "path": str(self.root / "target.gguf"),
            "bytes": 100,
            "sha256": union.TARGET_SHA256,
        }
        self.parent = {
            "path": str(self.root / "parent.gguf"),
            "bytes": 80,
            "sha256": union.PARENT_SHA256,
        }
        self.binaries = []
        for name in ("llama-server", "trace.so", "core.so"):
            path = self.root / name
            path.write_bytes(name.encode())
            self.binaries.append(union.pin(path))
        self.trace_build = write_json(self.root / "trace-build.json", {"fixture": "same runtime"})

    def tearDown(self):
        self.temporary.cleanup()

    def member(self, name, *, reverse=False, embedding_delta=0):
        root = self.root / name
        root.mkdir()
        raw = np.stack([np.full(union.WIDTH, i, np.float32) for i in (1, 2, 3, 4)])
        fc = np.stack([np.full(union.HIDDEN, i / 10, np.float32) for i in (1, 2, 3, 4)])
        order = [3, 2, 1, 0] if reverse else [0, 1, 2, 3]
        raw_spec = array(root / "raw.f32", raw[order])
        fc_spec = array(root / "fc.f32", fc[order])
        tokens = [10, 11, 12, 99]
        embedding = np.stack([np.full(union.HIDDEN, t / 100, np.float32) for t in tokens])
        embedding[0] += embedding_delta
        emb_spec = array(root / "emb.f32", embedding[order])
        groups, windows, raws, proofs, captures = [], [], [], [], []
        row_index = {}
        for source_index, logical in enumerate(order):
            group = "dev-old" if logical < 2 else "fit-" + name
            row_index[logical] = source_index
            raws.append(
                {
                    "file": 0,
                    "row": source_index,
                    "group": group,
                    "encoder_call": 3,
                    "target_call": 2,
                    "target_position": logical % 2,
                    "raw_sha256": digest(raw[logical]),
                    "native_fc_sha256": digest(fc[logical]),
                    "native_fc_file": fc_spec,
                    "native_fc_row": source_index,
                }
            )
        for group_id, role, component, start in (
            ("dev-old", "development", "component:old", 0),
            ("fit-" + name, "fit", "component:" + name, 2),
        ):
            directory = root / group_id
            directory.mkdir()
            evidence = {
                key: write_json(directory / (key + ".json"), {"fixture": key, "group": group_id})
                for key in ("protection", "response", "trace_manifest")
            }
            proofs.extend(evidence.values())
            groups.append(
                {
                    "id": group_id,
                    "role": role,
                    "domain": "code",
                    "group_id": component,
                    "windows": 1,
                    "native_protections": 1,
                    "native_draft_denominator": 2,
                    "selected_native_rows": 2,
                    "unreachable_or_extra_selected_rows": 0,
                    **evidence,
                }
            )
            captures.append(
                {
                    "id": group_id,
                    "role": role,
                    "domain": "code",
                    "group_id": component,
                    "status": "ACTUAL_PARENT_TRACE_AND_PROTECTION_AUDIT_PASS",
                    "token_text_stop_parity": True,
                    "source_identity": {"source_split": "TRAIN"},
                }
            )
            windows.append(
                {
                    "id": group_id + ":round-0000",
                    "role": role,
                    "domain": "code",
                    "group_id": component,
                    "group": group_id,
                    "round": 0,
                    "anchor": 1,
                    "seed_token": 10,
                    "labels": [11, 12],
                    "max_depth": 2,
                    "response_label_start": 0,
                    "native_prefix": 1,
                    "native_verified_prefix": 1,
                    "native_proposals": [11, 99],
                    "native_protections": [{"proposal_depth": 0, "token": 11, "response_index": 0}],
                    "native_success_mask": [True, False],
                    "native_supervision_reachable_mask": [True, True],
                    "prefix_tokens": [10],
                    "prefix_raw_rows": [row_index[start]],
                    "seed_raw_row": row_index[start + 1],
                    "native_selected": [],
                }
            )
            values = {
                "-c": "4096",
                "-np": "1",
                "-b": "512",
                "-ub": "512",
                "-t": "4",
                "-fa": "on",
                "--cache-type-k": "f16",
                "--cache-type-v": "f16",
                "--cache-type-k-draft": "f16",
                "--cache-type-v-draft": "f16",
                "--spec-type": "draft-eagle3",
                "--spec-draft-n-max": "2",
                "--spec-draft-n-min": "0",
                "--spec-draft-p-min": "0",
                "--spec-draft-sampling": "greedy",
                "-m": self.target["path"],
                "-md": self.parent["path"],
                "--port": str(9000 + start),
            }
            command = {
                "argv": [self.binaries[0]["path"]] + [x for item in values.items() for x in item],
                "environment": {
                    "LD_PRELOAD": self.binaries[1]["path"],
                    "W1A8_TRACE_LLAMA": self.binaries[2]["path"],
                    "W1A8_TRACE_DIR": str(directory / "trace"),
                    "GGML_CUDA_DECODE_CONSISTENT": "32",
                    "GGML_W1AX_ACT_BITS": "8",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "OMP_NUM_THREADS": "4",
                },
            }
            write_json(directory / "command.json", command)
        freeze = write_json(
            root / "freeze.json",
            {
                "target": self.target,
                "parent": self.parent,
                "depth": 2,
                "trace_build": self.trace_build,
                "input_pins": [self.target, self.parent] + self.binaries,
            },
        )
        capture = write_json(
            root / "summary.json",
            {
                "status": "ACTUAL_PARENT_COLLECTION_COMPLETE_WITH_ACCOUNTED_FAILURES",
                "training_updates": 0,
                "groups": captures,
            },
        )
        document = {
            "schema": union.MEMBER_SCHEMA,
            "target": self.target,
            "parent": self.parent,
            "capture": capture,
            "source_pins": [freeze, capture] + proofs,
            "raw_files": [raw_spec],
            "raw_rows": raws,
            "windows": windows,
            "groups": groups,
            "embedding_files": [emb_spec],
            "embedding_rows": {
                str(token): {
                    "file": 0,
                    "row": order.index(index),
                    "sha256": digest(embedding[index]),
                }
                for index, token in enumerate(tokens)
            },
            "required_embedding_ids": tokens,
            "missing_embedding_ids": [],
            "native_protections": {"development": 1, "fit": 1},
            "crossed_fc_source_rows": [],
            "counts": {"arbitrary_original_counter": 17},
            "rollout_contract": "own prefix unchanged",
            "source_position_contract": "observed native provenance",
            "scope": "fixture",
            "unknown_extension": {"preserve": name},
        }
        path = root / "manifest.json"
        write_json(path, document)
        return path

    def test_merge_remaps_rows_preserves_old_ids_roles_masks_and_all_provenance(self):
        first, second = self.member("A"), self.member("B", reverse=True)
        before = [p.read_bytes() for p in (first, second)]
        result = union.build_union([first, second])
        self.assertEqual(result["schema"], union.SCHEMA)
        self.assertNotIn("capture", result)
        self.assertEqual(len(result["groups"]), 3)
        self.assertEqual(len(result["windows"]), 3)
        self.assertEqual(len(result["raw_rows"]), 6)
        self.assertEqual(result["native_protections"], {"development": 1, "fit": 2})
        old = json.loads(first.read_bytes())
        merged = {w["id"]: w for w in result["windows"]}
        for window in old["windows"]:
            self.assertEqual(merged[window["id"]]["role"], window["role"])
            self.assertEqual(merged[window["id"]]["labels"], window["labels"])
            self.assertEqual(
                merged[window["id"]]["native_success_mask"], window["native_success_mask"]
            )
        self.assertEqual(len(result["group_provenance"]["dev-old"]), 2)
        self.assertEqual(len(result["embedding_provenance"]["10"]), 2)
        self.assertEqual(result["members"][1]["metadata"]["unknown_extension"], {"preserve": "B"})
        self.assertEqual([p.read_bytes() for p in (first, second)], before)
        self.assertFalse(result["storage_audit"]["dense_arrays_copied"])

    def test_explicit_new_reader_materializes_and_old_loader_rejects_union(self):
        first, second = self.member("A"), self.member("B", reverse=True)
        result = union.build_union([first, second])
        path = self.root / "union.json"
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, "schema"):
            ActualDataset(path)
        data = union.UnionActualDataset(path)
        for window in data.windows:
            materialized = data.materialize(window)
            self.assertEqual(tuple(materialized.prefix_rawfeatures.shape), (1, union.WIDTH))
            expected = 2 if window["role"] == "development" else 4
            self.assertTrue(bool(materialized.seed_rawfeature.eq(expected).all()))
        self.assertEqual(tuple(data.embedding_lookup(10).shape), (union.HIDDEN,))

    def test_same_token_with_different_admitted_row_bytes_is_rejected(self):
        first, second = self.member("A"), self.member("B", embedding_delta=0.2)
        with self.assertRaisesRegex(ValueError, "different native F32 embeddings"):
            union.build_union([first, second])

    def test_duplicate_dev_labels_and_unknown_semantic_fields_cannot_change(self):
        first, second = self.member("A"), self.member("B")
        original = json.loads(second.read_bytes())
        for change in ("labels", "unknown"):
            bad = copy.deepcopy(original)
            if change == "labels":
                bad["windows"][0]["labels"][1] = 13
                bad["required_embedding_ids"] = sorted(set(bad["required_embedding_ids"]) | {13})
                bad["missing_embedding_ids"] = [13]
            else:
                bad["windows"][0]["new_semantics"] = True
            write_json(second, bad)
            with self.assertRaisesRegex(ValueError, "duplicate window differs"):
                union.build_union([first, second])

    def test_duplicate_group_cannot_change_window_set_or_development_role(self):
        first, second = self.member("A"), self.member("B")
        original = json.loads(second.read_bytes())
        bad = copy.deepcopy(original)
        bad["windows"][0]["id"] = "dev-old:round-0001"
        write_json(second, bad)
        with self.assertRaisesRegex(ValueError, "duplicate group changes"):
            union.build_union([first, second])
        bad = copy.deepcopy(original)
        bad["groups"][0]["role"] = bad["windows"][0]["role"] = "fit"
        bad["native_protections"] = {"fit": 2}
        capture = json.loads(Path(bad["capture"]["path"]).read_bytes())
        capture["groups"][0]["role"] = "fit"
        ref = write_json(Path(bad["capture"]["path"]), capture)
        bad["capture"] = ref
        bad["source_pins"] = [ref if p["path"] == ref["path"] else p for p in bad["source_pins"]]
        write_json(second, bad)
        with self.assertRaisesRegex(ValueError, "duplicate group changes"):
            union.build_union([first, second])

    def test_trained_parent_other_target_and_native_mask_loss_are_rejected(self):
        first, second = self.member("A"), self.member("B")
        original = json.loads(second.read_bytes())
        for kind in ("parent", "target", "mask"):
            bad = copy.deepcopy(original)
            if kind == "mask":
                bad["windows"][0]["native_success_mask"] = [False, False]
            else:
                bad[kind]["sha256"] = "0" * 64
            write_json(second, bad)
            with self.assertRaises(ValueError):
                union.build_union([first, second])

    def test_runtime_drift_and_fresh_source_roles_reject_before_union(self):
        first, second = self.member("A"), self.member("B")
        command_path = second.parent / "dev-old/command.json"
        command = json.loads(command_path.read_bytes())
        command["environment"]["GGML_CUDA_DECODE_CONSISTENT"] = "16"
        write_json(command_path, command)
        with self.assertRaisesRegex(ValueError, "environment differs"):
            union.build_union([first, second])

    def test_file_corruption_and_row_hash_corruption_are_detected(self):
        first, second = self.member("A"), self.member("B")
        document = json.loads(second.read_bytes())
        document["raw_rows"][0]["raw_sha256"] = "0" * 64
        write_json(second, document)
        with self.assertRaisesRegex(ValueError, "row bytes/hash"):
            union.build_union([first, second])
        document["raw_rows"][0]["raw_sha256"] = digest(np.full(union.WIDTH, 1, np.float32))
        write_json(second, document)
        with Path(document["raw_files"][0]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "size changed"):
            union.build_union([first, second])

    def test_repeated_manifest_and_non_C_publication_rejected(self):
        first = self.member("A")
        with self.assertRaisesRegex(ValueError, "path repeated"):
            union.build_union([first, first])
        with self.assertRaisesRegex(ValueError, "unique C"):
            union.prepare_union([first, first], self.root / "output")
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
