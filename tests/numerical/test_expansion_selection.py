import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import expansion
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.records import canonical, identity


class FakeTokenizer:
    template_sha256 = "template"

    def __init__(self, binary: Path, target: Path, log: Path) -> None:
        self.binary = binary

    def render(self, messages: list[Record], tools: list[Record], thinking: bool = False) -> str:
        return ("<think>" if thinking else "") + canonical(messages).decode()

    def tokenize(self, rendered: str) -> tuple[int, ...]:
        return tuple(rendered.encode())

    def query(self, payload: Record) -> Record:
        return {"tokens": [7], "n_tokens": 1}

    @staticmethod
    def identifiers(reply: Record) -> tuple[int, ...]:
        return tuple(integer(value) for value in sequence(reply["tokens"]))

    def close(self) -> None:
        pass


def reference(path: Path) -> Record:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path: Path, value: object) -> Record:
    path.write_bytes(canonical(value))
    return reference(path)


def pool_row(source: str, subtype: str, category: str, index: int) -> Record:
    messages: list[Record] = [{"role": "user", "content": source + " request " + str(index)}]
    rendered = canonical(messages).decode()
    tokens = list(rendered.encode())
    return {
        "id": source + "-" + str(index),
        "source": source,
        "source_key": source + "-key-" + str(index),
        "group": source + ":group-" + str(index),
        "role": "train",
        "category": category,
        "subtype": subtype,
        "prior_use": "unknown",
        "messages": messages,
        "tools": [],
        "input_tokens": len(tokens),
        "length_band": 0,
        "token_sha256": identity({"tokens": tokens}),
        "rendered_sha256": identity({"text": rendered}),
    }


def expansion_spec(root: Path) -> Record:
    rows = [pool_row("commitpack", "editing", "code", index) for index in range(8)] + [
        pool_row("ultrachat", "non_arithmetic", "reasoning", index) for index in range(8)
    ]
    pool = root / "pool.jsonl"
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    target = root / "target.gguf"
    target.write_bytes(b"target")
    base = root / "base.jsonl"
    base.write_text(json.dumps(rows[0]) + "\n")
    binary = root / "tokenizer"
    binary.write_bytes(b"tokenizer")
    return {
        "schema": 1,
        "pool": reference(pool),
        "pool_result": write_json(
            root / "pool-result.json",
            {"target": reference(target), "template_sha256": "template"},
        ),
        "base_banks": [reference(base)],
        "sessions": [],
        "sources": [],
        "tokenizer": reference(binary),
        "exclusions": write_json(root / "exclusions.json", {"identities": []}),
        "heldout_exclusions": write_json(root / "heldout.json", {"identities": []}),
        "quotas": {"commitpack|editing|*": 2},
        "output_cap": 1024,
        "maximum_feature_bytes": 10**12,
        "seed": 5,
    }


def run_expansion(root: Path, raw: Record, name: str) -> tuple[Record, list[Record]]:
    specification = root / (name + ".json")
    specification.write_bytes(canonical(raw))
    result = expansion.expand(specification, root / name)
    rows = [
        record(json.loads(line))
        for line in (root / name / "prompts.jsonl").read_text().splitlines()
    ]
    return result, rows


class TestExpansionSelection(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, value in (
            ("Tokenizer", FakeTokenizer),
            ("freeze_source", lambda *values: None),
            ("source_identity", lambda root: "code"),
        ):
            patcher = mock.patch.object(expansion, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_thinking_quotas_render_thinking_and_keep_default_selection(self) -> None:
        raw = expansion_spec(self.root)
        plain, plain_rows = run_expansion(self.root, raw, "plain")
        self.assertNotIn("thinking", plain)
        self.assertNotIn("exclusion_inputs", plain)
        self.assertTrue(all("thinking" not in row for row in plain_rows))
        self.assertTrue(all(row["output_cap"] == 1024 for row in plain_rows))
        self.assertEqual(
            identity(plain_rows),
            "96e950e21f141e196303746899a056d3765bb3b66b1c85ddbf86f24d88e7a524",
        )
        raw["thinking_quotas"] = {"commitpack|editing|*": 2, "ultrachat|non_arithmetic|*": 3}
        raw["thinking_output_cap"] = 2048
        result, rows = run_expansion(self.root, raw, "thinking")
        thinking = [row for row in rows if row.get("thinking") is True]
        self.assertEqual(record(result["thinking"])["prompts"], 5)
        self.assertEqual(len(thinking), 5)
        for row in thinking:
            rendered = "<think>" + canonical(row["messages"]).decode()
            self.assertEqual(row["output_cap"], 2048)
            self.assertEqual(row["token_sha256"], identity({"tokens": list(rendered.encode())}))
            self.assertEqual(row["input_tokens"], len(rendered.encode()))
        self.assertEqual(len({row["group"] for row in rows}), 7)
        self.assertEqual([row for row in rows if "thinking" not in row], plain_rows)

    def test_exclusions_refuse_dev_workloads_mixtures_and_identities(self) -> None:
        raw = expansion_spec(self.root)
        raw["quotas"] = {"commitpack|editing|*": 4}
        raw["thinking_quotas"] = {"ultrachat|non_arithmetic|*": 5}
        raw["thinking_output_cap"] = 2048
        corpus = write_json(
            self.root / "corpus.json",
            {"records": [{"id": "commitpack-1", "group": "commitpack:group-3", "aliases": []}]},
        )
        raw["excluded_workloads"] = [
            write_json(
                self.root / "workload.json",
                {"tasks": [{"id": "ultrachat-1", "group": "ultrachat:group-2"}]},
            )
        ]
        raw["excluded_mixtures"] = [
            write_json(
                self.root / "mixture.json",
                {"components": [{"kind": "corpus-v1", "source": corpus}]},
            )
        ]
        raw["additional_exclusions"] = [
            write_json(self.root / "extra.json", {"identities": ["ultrachat-key-4"]})
        ]
        result, rows = run_expansion(self.root, raw, "excluded")
        chosen = {string(row["id"]) for row in rows}
        self.assertFalse(chosen & {"commitpack-0", "commitpack-1", "commitpack-3"})
        self.assertFalse(chosen & {"ultrachat-1", "ultrachat-2", "ultrachat-4"})
        self.assertEqual(len(chosen), 9)
        self.assertIn("exclusion_inputs", result)
        raw["thinking_quotas"] = {"ultrachat|non_arithmetic|*": 6}
        with self.assertRaisesRegex(ValueError, "cannot fill quotas"):
            run_expansion(self.root, raw, "short")
        raw["thinking_quotas"] = {"ultrachat|non_arithmetic|*": 1}
        del raw["thinking_output_cap"]
        with self.assertRaisesRegex(ValueError, "quotas required"):
            run_expansion(self.root, raw, "unpaired")
        raw["thinking_output_cap"] = 2048
        with mock.patch.object(
            FakeTokenizer, "render", lambda self, messages, tools, thinking=False: "same"
        ):
            with self.assertRaisesRegex(ValueError, "ignores thinking"):
                run_expansion(self.root, raw, "ignored")
        raw["session_input_limit"] = 40000
        with self.assertRaisesRegex(ValueError, "session input limit"):
            run_expansion(self.root, raw, "limit")

    def test_long_session_records_use_a_band_beyond_the_bank_limit(self) -> None:
        self.assertEqual(expansion.session_band(16384), 3)
        self.assertEqual(expansion.session_band(16385), 4)
        self.assertEqual(expansion.session_band(30000), 4)
