"""Synthetic local corpus fixtures only; never select actual confirmation inputs."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import unittest
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization"
sys.path.insert(0, str(SCRIPTS))
import modern_confirm as confirm
import modern_fresh as fresh
import modern_natural_confirmation as natural
import test_w1a8_modern_confirm as confirm_fixture


class NaturalConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fixture-natural-confirmation-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = []
        original = []
        for index in range(512):
            domain = ("code", "reasoning", "prose")[index % 3]
            text = f"Previously trained fixture prompt {index}."
            messages = [{"role": "user", "content": text}]
            row = {
                "group_id": f"historical-{index:03d}",
                "prompt_id": f"old:row-{index:03d}",
                "domain": domain,
                "messages": messages,
                "source_ancestry": {"source_split": "TRAIN"},
            }
            self.history.append(row)
            if index < 36:
                original.append(
                    {
                        **row,
                        "id": f"fit-fixture-{index:03d}",
                        "role": "fit" if index < 24 else "development",
                        "rendered_prompt_sha256": fresh.digest(fresh.render(text).encode()),
                    }
                )
        self.original = original
        self.original_ref = self.put(
            "original.json",
            {
                "schema": "w1a8_modern_phase1_generalization_roster_v2",
                "rows": original,
                "roster": original,
            },
        )
        union_groups = [
            {**row, "role": "development" if 24 <= i < 36 else "fit"}
            for i, row in enumerate(self.history[:60])
        ]
        self.union_ref = self.put(
            "union.json", {"schema": "w1a8_actual_source_indexed_union_v1", "groups": union_groups}
        )
        binding1 = self.put("bindings1.json", {"rows": self.history[:256]})
        binding2 = self.put("bindings2.json", {"rows": self.history[256:]})
        indexes = []
        counts = {}
        self.role_texts = {
            "actual_train": [r["messages"][0]["content"] for r in self.history],
            "development": [
                r["messages"][0]["content"] for r in original if r["role"] == "development"
            ],
            "pilot": ["Previously used pilot fixture."],
            "old_matrix": ["Previously used matrix fixture."],
        }
        for role, texts in self.role_texts.items():
            file = self.put(
                role + "-hashes.json",
                {
                    "schema": "modern_fresh_exclusion_index_v1",
                    "records": [fresh.identities(text) for text in texts],
                },
            )
            indexes.append(
                {
                    "role": role,
                    "format": "hash_index_v1",
                    "file": file,
                    "previously_used": True,
                    "sealed": False,
                    "complete": True,
                }
            )
            counts[role] = {"unique_identity_records": len(texts)}
        candidate_selection = self.put("old-candidate.json", {"historical": "fixture"})
        self.inventory = self.put(
            "inventory.json",
            {
                "schema": "modern_confirmation_explicit_prior_input_inventory_v1",
                "candidate_selection": candidate_selection,
                "counts": counts,
                "sources": [{"role": "actual_train", "file": ref} for ref in (binding1, binding2)],
            },
        )
        self.prior_selection = self.put(
            "old-selected.json",
            {
                "schema": fresh.SCHEMA,
                "exclusion_inventory": self.inventory,
                "candidate_selection": candidate_selection,
                "exclusions": indexes,
            },
        )
        self.consumed_texts = [f"Consumed earlier fresh case {i}." for i in range(48)]
        consumed_manifest = self.put(
            "old-inputs/manifest.json",
            {
                "schema": "modern_fresh_confirmation_inputs_v1",
                "admitted": True,
                "groups": [
                    {"group_id": f"consumed-{i}", "identities": fresh.identities(text)}
                    for i, text in enumerate(self.consumed_texts)
                ],
            },
        )
        old_files = {"requests.json": "a" * 64, "quality-cases.json": "b" * 64}
        consumed_receipt = self.put(
            "old-inputs/receipt.json",
            {
                "schema": "modern_fresh_confirmation_receipt_v1",
                "admitted": True,
                "manifest_sha256": consumed_manifest["sha256"],
                "performance_prompts": 24,
                "quality_tasks": 24,
                "files": old_files,
            },
        )
        marker = self.put(
            "consumed.json",
            {
                "schema": "modern_fresh_cohort_consumption_v1",
                "evaluated": True,
                "future_exclusion_sources": [
                    {"path": str(self.root / "old-inputs" / name), "bytes": 1, "sha256": sha}
                    for name, sha in old_files.items()
                ],
            },
        )
        self.catalog_rows = {}
        source_entries = []
        for domain in natural.DOMAIN_COUNTS:
            sid = "fixture-" + domain
            rows = []
            for i in range(30):
                text = f"Natural {domain} fixture task number {i}. Preserve the complete task."
                if i == 0:
                    text = self.history[0]["messages"][0]["content"]
                if i == 1:
                    text = self.role_texts["pilot"][0]
                if i == 2:
                    text = self.role_texts["old_matrix"][0]
                if i == 3:
                    text = self.consumed_texts[0]
                if i == 4:
                    text = "\t" + self.consumed_texts[0].replace(" ", "  ") + "\n"
                row = {
                    "id": f"row-{i:02d}",
                    "group_id": f"row-{i:02d}",
                    "messages": [{"role": "user", "content": text}],
                }
                if i in (10, 11):
                    row["topic_id"] = "one-whole-shared-topic"
                if i == 29:
                    row["group_id"] = "row-28"
                rows.append(row)
            self.catalog_rows[domain] = rows
            file = self.jsonl(f"{sid}.jsonl", rows)
            source_entries.append(
                {
                    "id": sid,
                    "domain": domain,
                    "path": Path(file["path"]).name,
                    "sha256": file["sha256"],
                    "revision": "fixture-revision",
                    "license": "fixture",
                    "uri": "https://example.invalid/fixture-only",
                }
            )
        partitions = {}
        for role, row_index in [("train_large", 5), ("dev", 6), ("final", 7)]:
            rows = []
            for domain in natural.DOMAIN_COUNTS:
                source_row = self.catalog_rows[domain][row_index]
                sid = "fixture-" + domain
                rows.append(
                    {
                        "id": sid + ":" + source_row["id"],
                        "group": "group:" + sid + ":" + source_row["group_id"],
                        "content_sha256": fresh.digest(
                            json.dumps(
                                source_row["messages"],
                                sort_keys=True,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode()
                        ),
                        "topic": None,
                    }
                )
            file = self.jsonl(role + ".index.jsonl", rows)
            partitions[role] = {
                "index": Path(file["path"]).name,
                "index_sha256": file["sha256"],
                "prompts_count": len(rows),
                "prompts": "NEVER-OPEN-" + role + ".jsonl",
            }
        self.partition_ref = self.put(
            "partitions.json", {"schema": "w1a_data_manifest_v1", "files": partitions}
        )
        self.catalog = self.put(
            "catalog.json",
            {
                "schema": natural.CATALOG_SCHEMA,
                "sources": source_entries,
                "partition_manifests": [self.partition_ref],
            },
        )
        opaque = {
            "path": str(self.root / "NEVER-OPEN-model.gguf"),
            "bytes": 100,
            "sha256": "c" * 64,
        }
        server = {**opaque, "path": str(self.root / "NEVER-OPEN-server")}
        runtime = {
            "server": server,
            "libraries": [server],
            "environment": {"fixture": "not executed"},
            "source_pins": [fresh.reference(SCRIPTS / "modern_confirm.py")],
        }
        self.models = {
            "target": opaque,
            "baseline": {**opaque, "path": str(self.root / "baseline.json")},
            "arms": [
                {
                    "id": kind,
                    "kind": kind,
                    "model": None if kind == "target" else opaque,
                    "runtime": copy.deepcopy(runtime),
                }
                for kind in ("target", "dflash", "w1a8", "eagle_q4")
            ],
        }
        models_ref = self.put("models.json", self.models)
        self.freeze = {
            "schema": natural.FREEZE_SCHEMA,
            "selection_frozen": True,
            "evaluation_inputs_generated": False,
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "candidate": opaque,
            "runtime": server,
            "selected_arm_id": "w1a8",
            "source_freeze": models_ref,
            "generator": fresh.reference(SCRIPTS / "modern_natural_confirmation.py"),
            "quality_generator": fresh.reference(SCRIPTS / "modern_fresh.py"),
            "confirmation_runner": fresh.reference(SCRIPTS / "modern_confirm.py"),
            "judge": fresh.reference(fresh.TRUSTED_JUDGE),
            "settings": {"output_limit": 256},
            "natural_sources": {
                "catalog": self.catalog,
                "partition_manifests": [self.partition_ref],
                "prior_selection": self.prior_selection,
                "prior_inventory": self.inventory,
                "current_union": self.union_ref,
                "original_roster": self.original_ref,
                "consumed_manifest": consumed_manifest,
                "consumed_receipt": consumed_receipt,
                "consumed_marker": marker,
            },
        }
        self.freeze_ref = self.put("selected.json", self.freeze)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fresh.encoded(value))
        return natural.reference(path)

    def jsonl(self, name, rows):
        path = self.root / name
        path.write_bytes(
            b"".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n" for row in rows
            )
        )
        return natural.reference(path)

    def write_freeze(self):
        self.freeze_ref = self.put("selected.json", self.freeze)

    def prep(self, name="packet"):
        output = self.root / name
        receipt = natural.prepare(self.freeze_ref["path"], output)
        return output, receipt

    def test_freeze_precedes_all_catalog_selection_and_generation(self):
        self.freeze["selection_frozen"] = False
        self.write_freeze()
        with (
            patch.object(natural, "build_packet") as build,
            patch.object(natural.fresh, "_instance") as instance,
        ):
            with self.assertRaisesRegex(ValueError, "must freeze"):
                self.prep()
        build.assert_not_called()
        instance.assert_not_called()
        self.assertFalse((self.root / "packet").exists())

    def test_packet_exact_exclusions_components_domains_and_oracle_controls(self):
        output, receipt = self.prep()
        self.assertTrue(receipt["admitted"])
        requests = json.loads((output / "requests.json").read_text())
        cases = json.loads((output / "quality-cases.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(Counter(r["domain"] for r in requests), natural.DOMAIN_COUNTS)
        self.assertEqual(len({r["family"] for r in requests}), 24)
        self.assertEqual(len({r["group_id"] for r in requests}), 24)
        forbidden = {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 28, 29}
        self.assertFalse(
            {int(r["source_ancestry"]["source_row_id"].split("-")[1]) for r in requests} & forbidden
        )
        for row in requests:
            source_row = self.catalog_rows[row["domain"]][
                int(row["source_ancestry"]["source_row_id"].split("-")[1])
            ]
            self.assertEqual(row["messages"], source_row["messages"])
            self.assertEqual(
                row["request"]["prompt"], fresh.render(source_row["messages"][0]["content"])
            )
            confirm.validate_payload(row["request"], 256)
        self.assertEqual(len(cases), 24)
        self.assertEqual(Counter(c["kind"] for c in cases), {"python": 6, "sql": 2, "json": 16})
        self.assertEqual(receipt["quality_correct_controls_passed"], 24)
        self.assertEqual(receipt["quality_known_wrong_controls_rejected"], 24)
        self.assertEqual(manifest["historical_TRAIN_inputs_excluded"], 512)
        self.assertFalse(manifest["selection"]["synthetic_performance_family_clusters"])
        self.assertFalse(manifest["tokenization_performed"])
        self.assertEqual(receipt["model_requests_executed"], 0)
        self.assertEqual(
            receipt["manifest_sha256"], natural.reference(output / "manifest.json")["sha256"]
        )
        second, _ = self.prep("same-frozen-selection")
        for name in (
            "manifest.json",
            "requests.json",
            "quality-cases.json",
            "quality-controls.json",
        ):
            self.assertEqual((output / name).read_bytes(), (second / name).read_bytes())
        with self.assertRaisesRegex(ValueError, "replace"):
            self.prep()

    def test_sources_pins_complete_runtime_and_no_payload_access(self):
        original = Path.open
        forbidden = []

        def checked_open(path, *args, **kwargs):
            if "NEVER-OPEN" in str(path) or path.suffix in (".gguf", ".f32", ".pt"):
                forbidden.append(str(path))
                raise AssertionError("forbidden payload access")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", checked_open):
            self.prep()
        self.assertFalse(forbidden)
        self.models["arms"][1]["runtime"]["libraries"] = []
        self.freeze["source_freeze"] = self.put("models.json", self.models)
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "complete runtime"):
            self.prep("bad-runtime")

    def test_missing_partition_index_and_omitted_catalog_partition_fail_closed(self):
        (self.root / "final.index.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prep()
        self.assertFalse((self.root / "packet").exists())
        # The known master catalog cannot omit the second continuous freeze inventory.
        with (
            patch.object(natural, "KNOWN_CATALOG", Path(self.catalog["path"])),
            patch.object(
                natural,
                "REQUIRED_PARTITIONS",
                (Path(self.partition_ref["path"]), self.root / "missing-continuous.json"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "known master catalog"):
                self.prep("missing-complete-partitions")

    def test_changed_catalog_source_and_fewer_than512_exclusions_rejected(self):
        path = self.root / "fixture-code.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "catalog source pin"):
            self.prep()
        # Restore source then deliberately omit a historical TRAIN hash.
        self.jsonl("fixture-code.jsonl", self.catalog_rows["code"])
        prior = json.loads(Path(self.prior_selection["path"]).read_text())
        entry = next(e for e in prior["exclusions"] if e["role"] == "actual_train")
        document = json.loads(Path(entry["file"]["path"]).read_text())
        document["records"].pop()
        entry["file"] = self.put("actual_train-hashes.json", document)
        self.freeze["natural_sources"]["prior_selection"] = self.put("old-selected.json", prior)
        self.write_freeze()
        with self.assertRaisesRegex(ValueError, "all512"):
            self.prep("incomplete-history")

    def runner(self, output):
        self.output = self.root / "runner-output"
        self.output.mkdir()
        args = argparse.Namespace(
            output=self.output,
            models=self.root / "models.json",
            inputs=self.root / "inputs.json",
            repetitions=5,
            allow_cuda=False,
        )
        runner = natural.NaturalConfirmation(args)
        inputs = {
            key: natural.reference(output / name) for key, name in confirm.PACKET_FILES.items()
        }
        inputs["receipt"] = natural.reference(output / "receipt.json")
        inputs["selection"] = self.freeze_ref
        warmups = []
        for row in self.original[:2]:
            request = confirm_fixture.ConfirmationTests.payload(
                fresh.render(row["messages"][0]["content"]), 80
            )
            warmups.append(
                {
                    "id": row["id"],
                    "role": "fit",
                    "request": self.put(row["id"] + "-request.json", request),
                    "reference": self.put(
                        row["id"] + "-response.json", confirm_fixture.ConfirmationTests.response()
                    ),
                }
            )
        inputs["warmups"] = warmups
        inputs["warmup_source"] = self.put(
            "warmup-inputs.json", {"roster": self.original_ref, "warmups": warmups}
        )
        runner.inputs, runner.models, runner.arms = inputs, self.models, self.models["arms"]
        runner.records, runner.stopping = [], False
        runner.guard = Mock()
        return runner

    def test_adapter_authenticates_new_schema_and_reuses_native_geometry_gate(self):
        output, _ = self.prep()
        runner = self.runner(output)
        runner.authenticate_inputs()
        self.assertEqual(runner.request_schema, "modern_natural_confirmation_requests_v1")
        self.assertIs(natural.NaturalConfirmation.prepare_cell, confirm.Confirmation.prepare_cell)
        self.assertIs(natural.NaturalConfirmation.completion, confirm.Confirmation.completion)
        self.assertIs(natural.NaturalConfirmation.run, confirm.Confirmation.run)
        cell = self.root / "tokenize-cell"
        cell.mkdir()
        runner.request = Mock(return_value={"tokens": [3] * 3840})
        runner.prepare_cell(123, cell, self.models["arms"][0], 0)
        self.assertEqual(len(runner.geometry), 48)
        self.assertTrue(all(r["fits"] for r in runner.geometry))
        self.assertEqual(runner.request.call_count, 48)
        geometry = json.loads((self.output / "native-geometry.json").read_text())
        self.assertEqual(geometry["context"], 4096)
        self.assertEqual(geometry["output_reserve"], 256)

    def test_native_geometry_rejects_whole_packet_without_completion_or_replacement(self):
        output, _ = self.prep()
        runner = self.runner(output)
        runner.authenticate_inputs()
        cell = self.root / "bad-tokenize-cell"
        cell.mkdir()
        runner.request = Mock(return_value={"tokens": [3] * 3841})
        with self.assertRaisesRegex(ValueError, "no truncation/drop"):
            runner.prepare_cell(123, cell, self.models["arms"][0], 0)
        self.assertIsNone(runner.geometry)
        self.assertEqual(len(runner.rows), 24)
        self.assertTrue(all(call.args[1] == "tokenize" for call in runner.request.call_args_list))
        with self.assertRaisesRegex(ValueError, "geometry must precede"):
            runner.completion(123, runner.rows[0], cell, 0, False, self.models["arms"][0], 0)

    def test_existing_five_alternating_repetitions_and_natural_cluster_annotations(self):
        output, _ = self.prep()
        runner = self.runner(output)
        runner.authenticate_inputs()
        seen = []
        runner.authenticate = Mock()
        runner.cell = lambda repetition, arm: seen.append((repetition, arm["id"]))
        runner.summarize = Mock(return_value=([], []))
        self.put("runner-output/native-geometry.json", {"fixture": "no native execution"})
        with patch.object(confirm.signal, "signal"):
            runner.run()
        expected = [
            (rep, arm["id"])
            for rep in range(5)
            for arm in (self.models["arms"] if rep % 2 == 0 else self.models["arms"][::-1])
        ]
        expected += [(5, arm["id"]) for arm in self.models["arms"]]
        self.assertEqual(seen, expected)
        del runner.summarize
        with patch.object(
            confirm.Confirmation, "summarize", return_value=([{"arm": "fixture"}], [])
        ):
            performance, _ = runner.summarize()
        self.assertEqual(performance[0]["performance_clusters"], 24)
        self.assertEqual(performance[0]["domain_strata"], natural.DOMAIN_COUNTS)
        self.assertFalse(performance[0]["synthetic_family_cluster_count_applicable"])

    def test_oracle_failure_retains_complete_inconclusive_packet(self):
        rejected = Mock()
        rejected.judge.return_value = {"passed": False, "reason": "synthetic failure fixture"}
        with patch.object(natural.fresh, "load_judge", return_value=rejected):
            output, receipt = self.prep()
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["status"], "INCONCLUSIVE_INPUTS_OR_ORACLE")
        self.assertEqual(len(json.loads((output / "requests.json").read_text())), 24)
        self.assertEqual(len(json.loads((output / "quality-cases.json").read_text())), 24)

    def test_legacy_partition_index_reference_is_mandatory(self):
        manifest = json.loads(Path(self.partition_ref["path"]).read_text())
        manifest["opaque_exclusions"] = [
            {"file": "prior-sealed.index.jsonl", "sha256": "f" * 64, "rows": 1}
        ]
        self.partition_ref = self.put("partitions.json", manifest)
        self.freeze["natural_sources"]["partition_manifests"] = [self.partition_ref]
        catalog = json.loads(Path(self.catalog["path"]).read_text())
        catalog["partition_manifests"] = [self.partition_ref]
        self.freeze["natural_sources"]["catalog"] = self.put("catalog.json", catalog)
        self.write_freeze()
        with self.assertRaises(FileNotFoundError):
            self.prep()
        self.assertFalse((self.root / "packet").exists())

    def test_indexed_master_rows_are_skipped_before_message_inspection(self):
        self.catalog_rows["code"][6]["messages"] = "UNREADABLE-HELD-OUT-BODY"
        source = self.jsonl("fixture-code.jsonl", self.catalog_rows["code"])
        catalog = json.loads(Path(self.catalog["path"]).read_text())
        entry = next(row for row in catalog["sources"] if row["domain"] == "code")
        entry["sha256"] = source["sha256"]
        self.freeze["natural_sources"]["catalog"] = self.put("catalog.json", catalog)
        self.write_freeze()
        output, receipt = self.prep()
        self.assertTrue(receipt["admitted"])
        requests = json.loads((output / "requests.json").read_text())
        self.assertNotIn("fixture-code:row-06", {r["original_prompt_id"] for r in requests})

    def test_packet_tampering_and_old_schema_not_admitted(self):
        output, _ = self.prep()
        runner = self.runner(output)
        requests = json.loads((output / "requests.json").read_text())
        requests[0]["request"]["prompt"] += "\nmutation"
        runner.inputs["requests"] = self.put("tampered-requests.json", requests)
        with self.assertRaisesRegex(ValueError, "deterministic frozen"):
            runner.authenticate_inputs()


if __name__ == "__main__":
    unittest.main()
