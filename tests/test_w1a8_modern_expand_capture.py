"""CPU-only source/exclusion/storage/protocol checks; no model or native launch."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/w1a8_generalization"
sys.path.insert(0, str(SCRIPTS))
import modern_expand_capture as capture


def digest(value):
    return capture.hashlib.sha256(value.encode()).hexdigest()


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rendering = {
            "prefix": "<|im_start|>system\nTest<|im_end|>\n",
            "assistant_suffix": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        }
        self.original = []
        self.rows, self.cases = [], []
        for domain in ("code", "reasoning", "prose"):
            for index in range(28):
                text = f"{domain} question {index}"
                gid = f"group-{domain}-{index:02d}"
                messages = [{"role": "user", "content": text}]
                ancestry = {
                    "domain": domain,
                    "prompt_id": gid,
                    "source_split": "TRAIN",
                    "prompt_sha256": digest(text),
                }
                old = self.write(
                    gid + "-original.json",
                    {
                        "prompt": {"messages": messages},
                        "prompt_source_sha256": digest(text),
                        "chain_ancestry": ancestry,
                    },
                )
                old_prompt = (
                    "<|im_start|>user\n"
                    + text
                    + "<|im_end|>\n"
                    + self.rendering["assistant_suffix"]
                )
                split = self.write(
                    gid + "-split.json",
                    {
                        "prompt": {"text": old_prompt},
                        "chain_ancestry": ancestry,
                        "tokens": [12, 34, 56],
                    },
                )
                self.rows.append(
                    {
                        "role": "expanded_fit",
                        "domain": domain,
                        "group_id": gid,
                        "prompt_id": gid,
                        "messages": messages,
                        "source_ancestry": ancestry,
                        "original_receipt": capture.lifecycle.pin(old),
                    }
                )
                role = "development" if 8 <= index < 12 else "fit"
                self.cases.append(
                    {
                        "role": role,
                        "domain": domain,
                        "group_id": gid,
                        "source_split": "TRAIN",
                        "ancestry": capture.lifecycle.pin(split),
                        "prompt_length": 2,
                        "prompt_tokens": [12, 34],
                    }
                )
                if index < 12:
                    self.original.append(
                        {
                            "id": "original-" + gid,
                            "group_id": gid,
                            "domain": domain,
                            "role": role,
                            "messages": messages,
                        }
                    )
        self.current = self.write(
            "current.json",
            {
                "schema": "w1a8_modern_phase1_generalization_roster_v2",
                "rows": self.original,
                "roster": self.original,
                "rendering": self.rendering,
            },
        )
        self.split = self.write("split.json", {"cases": self.cases})
        self.binding = self.write("bindings.json", {"rows": self.rows})
        self.fresh = self.root / "fresh"
        self.fresh.mkdir()
        self.groups = []
        requests = []
        for i in range(48):
            text = "fresh question " + str(i)
            self.groups.append(
                {
                    "group_id": "fresh-" + str(i),
                    "identities": {
                        "raw_sha256": digest(text),
                        "normalized_raw_sha256": digest(capture.normalized(text)),
                    },
                }
            )
            if i < 24:
                requests.append(
                    {"prompt": capture.render([{"role": "user", "content": text}], self.rendering)}
                )
        self.requests = self.write("fresh/requests.json", requests)
        self.update_fresh()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def update_fresh(self):
        manifest = self.write("fresh/manifest.json", {"admitted": True, "groups": self.groups})
        self.write(
            "fresh/receipt.json",
            {
                "manifest_sha256": capture.lifecycle.pin(manifest)["sha256"],
                "files": {"requests.json": capture.lifecycle.pin(self.requests)["sha256"]},
            },
        )

    def select(self):
        return capture.select_rows(self.current, self.split, [self.binding], self.fresh)

    def test_deterministic_whole_balanced_fit_and_original_exclusion(self):
        rows, pins = self.select()
        self.assertEqual(rows, self.select()[0])
        self.assertEqual(
            Counter(r["domain"] for r in rows), {"code": 8, "reasoning": 8, "prose": 8}
        )
        self.assertEqual({r["role"] for r in rows}, {"fit"})
        self.assertFalse({r["group_id"] for r in rows} & {r["group_id"] for r in self.original})
        self.assertEqual(len({r["group_id"] for r in rows}), 24)
        self.assertEqual(len({r["normalized_message_identity_sha256"] for r in rows}), 24)
        # Original+split receipts authenticated for every selected whole group.
        paths = {p["path"] for p in pins}
        self.assertTrue(
            all(
                r["original_receipt"]["path"] in paths
                and r["historical_split_receipt"]["path"] in paths
                for r in rows
            )
        )
        self.assertFalse(any("quality-cases" in p for p in paths))

    def test_fresh_quality_manifest_exact_and_normalized_exclusion(self):
        before = self.select()[0]
        candidates = [r for r in before if r["domain"] == "code"][:2]
        self.groups[24]["identities"]["raw_sha256"] = digest(
            candidates[0]["messages"][0]["content"]
        )
        # Different exact bytes, same Unicode/whitespace-normalized identity.
        text = candidates[1]["messages"][0]["content"]
        self.groups[25]["identities"]["raw_sha256"] = digest("  " + text.replace(" ", "\t") + "\n")
        self.groups[25]["identities"]["normalized_raw_sha256"] = digest(capture.normalized(text))
        self.update_fresh()
        after = self.select()[0]
        self.assertEqual(len(after), 24)
        self.assertFalse({r["group_id"] for r in candidates} & {r["group_id"] for r in after})

    def test_original_identity_exclusion_even_with_different_group(self):
        selected = self.select()[0][0]
        self.original[0]["messages"] = selected["messages"]
        value = json.loads(self.current.read_text())
        value["rows"] = value["roster"] = self.original
        self.current.write_text(json.dumps(value))
        self.assertNotIn(selected["group_id"], {r["group_id"] for r in self.select()[0]})

    def test_receipt_mutation_or_false_ancestry_fails_closed(self):
        selected = self.select()[0][0]
        path = Path(selected["original_receipt"]["path"])
        value = json.loads(path.read_text())
        value["prompt"]["messages"] = [{"role": "user", "content": "substitution"}]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "pin changed"):
            self.select()
        # Even updating a binding pin cannot authenticate different source messages.
        original_row = next(r for r in self.rows if r["group_id"] == selected["group_id"])
        original_row["original_receipt"] = capture.lifecycle.pin(path)
        self.binding.write_text(json.dumps({"rows": self.rows}))
        with self.assertRaisesRegex(ValueError, "source messages"):
            self.select()

    def test_incomplete_consumed_packet_fails(self):
        self.groups.pop()
        self.update_fresh()
        with self.assertRaisesRegex(ValueError, "fresh48"):
            self.select()

    def test_insufficient_independent_fit_pool_fails_without_role_reassignment(self):
        for case in self.cases:
            if case["domain"] == "code" and case["group_id"] not in {
                r["group_id"] for r in self.original
            }:
                case["role"] = "development"
        self.split.write_text(json.dumps({"cases": self.cases}))
        with self.assertRaisesRegex(
            ValueError, "not enough independent unused TRAIN FIT groups: code"
        ):
            self.select()

    def test_storage_both_floors_capture_and_future_reserves(self):
        floor = capture.lifecycle.FLOOR + capture.lifecycle.RESERVE
        with patch.object(
            capture.shutil,
            "disk_usage",
            return_value=SimpleNamespace(
                free=floor + capture.PLANNED_BYTES + capture.FUTURE_STATE_BYTES
            ),
        ) as usage:
            capture.storage_guard(self.root, capture.PLANNED_BYTES, capture.FUTURE_STATE_BYTES)
            self.assertEqual(usage.call_count, 2)
        for values in (
            (floor + capture.METADATA_BYTES - 1, 10**10),
            (10**10, floor + capture.PLANNED_BYTES + capture.FUTURE_STATE_BYTES - 1),
        ):
            with patch.object(
                capture.shutil, "disk_usage", side_effect=[SimpleNamespace(free=v) for v in values]
            ):
                with self.assertRaisesRegex(ValueError, "insufficient storage"):
                    capture.storage_guard(
                        self.root, capture.PLANNED_BYTES, capture.FUTURE_STATE_BYTES
                    )
        with self.assertRaisesRegex(ValueError, "future state"):
            capture.storage_guard(self.root, capture.PLANNED_BYTES, 0)

    def test_planned_limit_reserves_full_native_process_cap(self):
        args = SimpleNamespace(
            output=self.root,
            planned_bytes=capture.PLANNED_BYTES,
            future_state_bytes=capture.FUTURE_STATE_BYTES,
        )
        worker = capture.ExpansionCollector(args)
        with (
            patch.object(
                capture, "directory_bytes", return_value=capture.PLANNED_BYTES - capture.TRACE_CAP
            ),
            patch.object(capture, "storage_guard") as guard,
        ):
            with self.assertRaisesRegex(ValueError, "next64MiB"):
                worker.budget(1)
            guard.assert_not_called()

    def test_destination_direct_c_unique_and_protocol_signature(self):
        with patch.object(capture, "C_ROOT", self.root):
            capture.check_destination(self.root / "new-capture")
            with self.assertRaisesRegex(ValueError, "already exists"):
                capture.check_destination(self.fresh)
            with self.assertRaisesRegex(ValueError, "directly"):
                capture.check_destination(self.root.parent / "outside")
        self.assertEqual(
            capture.protocol(),
            {
                "n_predict": 80,
                "temperature": 0,
                "seed": 1234,
                "cache_prompt": False,
                "return_tokens": True,
                "stream": False,
            },
        )
        value = {
            "tokens": [3],
            "content": "x",
            "stop": True,
            "stop_type": "limit",
            "stopping_word": "",
            "truncated": False,
            "tokens_predicted": 1,
            "has_new_line": False,
        }
        self.assertEqual(capture.response_signature(value), value)
        value["tokens_predicted"] = 2
        with self.assertRaisesRegex(ValueError, "invalid raw"):
            capture.response_signature(value)

    def test_target_reference_protocol_and_oversize_rejection_without_launch(self):
        prompt = self.root / "prompt.txt"
        prompt.write_text("a prompt")
        row = {"id": "fit-expand-v1-code-00", "rendered_text_path": str(prompt)}
        fake_process = Mock(pid=987654321)
        fake_process.poll.return_value = None
        response = {
            "tokens": [3],
            "content": "x",
            "stop": True,
            "stop_type": "limit",
            "stopping_word": "",
            "truncated": False,
            "tokens_predicted": 1,
            "has_new_line": False,
            "tokens_evaluated": 2,
        }
        worker = SimpleNamespace(
            output=self.root,
            permission=Mock(),
            budget=Mock(),
            checked=Mock(),
            request=Mock(side_effect=[{}, {"tokens": [10, 11]}, response]),
            reap=Mock(),
            active=None,
            active_identity=None,
        )
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                return "1 2 3 /fake/libggml-cuda.so\n"
            return original_read(path, *args, **kwargs)

        patches = (
            patch.object(capture.subprocess, "Popen", return_value=fake_process),
            patch.object(
                capture.lifecycle, "process_identity", return_value={"pid": fake_process.pid}
            ),
            patch.object(Path, "read_text", read_text),
        )
        with patches[0], patches[1], patches[2]:
            result = capture.target_references(
                worker,
                [row],
                {"path": "/fake/server"},
                {"target": {"path": "/fake/model"}},
                {"binaries": [{"path": "/fake/libggml-cuda.so"}]},
                {},
            )
        self.assertEqual(result[row["id"]][0], {"prompt": "a prompt", **capture.protocol()})
        self.assertEqual(worker.request.call_args.args[2]["n_predict"], 80)
        worker.reap.assert_called_once()
        # A selected oversize group aborts; it is never replaced with another group.
        worker.output = self.root / "second"
        worker.output.mkdir()
        worker.request = Mock(side_effect=[{}, {"tokens": [1] * 305}])
        with (
            patch.object(capture.subprocess, "Popen", return_value=fake_process),
            patch.object(capture.lifecycle, "process_identity", return_value={}),
            patch.object(Path, "read_text", read_text),
        ):
            with self.assertRaisesRegex(ValueError, "no replacement"):
                capture.target_references(
                    worker,
                    [row],
                    {"path": "/fake/server"},
                    {"target": {"path": "/fake/model"}},
                    {"binaries": [{"path": "/fake/libggml-cuda.so"}]},
                    {},
                )
        self.assertEqual(worker.request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
