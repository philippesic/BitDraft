import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from benchmarks.sources import commitpack, qasper
from w1a1_eagle.contracts import Record, record, sequence


class TestPrompts(unittest.TestCase):
    def test_qasper_nullable_headings_preserve_training_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.tgz"
            body = json.dumps(
                {
                    "paper": {
                        "title": "Title",
                        "full_text": [
                            {
                                "section_name": None,
                                "paragraphs": ["", "   ", "A retained observation."],
                            }
                        ],
                        "qas": [{"question": "What happened?"}],
                    }
                }
            ).encode()
            with tarfile.open(path, "w:gz") as archive:
                for name, payload in (
                    ("qasper-train-v0.3.json", body),
                    ("qasper-dev-v0.3.json", b"unread invalid JSON"),
                ):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            rows = list(qasper(path))
            self.assertEqual(len(rows), 1)
            self.assertIn("A retained observation.", str(rows[0]["messages"]))
            self.assertEqual(rows[0]["source_key"], "paper")

    def test_commit_edits_exclude_new_contents_and_empty_originals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commits.jsonl"
            source = {
                "old_contents": "old function",
                "new_contents": "answer must stay hidden",
                "commit": "one",
                "old_file": "module.py",
                "repos": "fixture/repo",
                "message": "Repair the function",
                "license": "mit",
            }
            path.write_text(
                json.dumps(source | {"old_contents": ""}) + "\n" + json.dumps(source) + "\n"
            )
            rows = list(commitpack(path))
            self.assertEqual(len(rows), 1)
            self.assertIn("old function", str(rows[0]["messages"]))
            self.assertNotIn("answer must stay hidden", str(rows[0]["messages"]))

    def test_tool_argument_rendering_survives_canonical_json(self) -> None:
        from w1a1_eagle.numerical.prompting import Tokenizer
        from w1a1_eagle.records import canonical

        class FakeTokenizer(Tokenizer):
            def query(self, payload: Record) -> Record:
                messages = payload["messages"]
                prompt = json.dumps(messages)
                return {"prompt": prompt, "tokens": [1], "n_tokens": 1}

        tokenizer = FakeTokenizer.__new__(FakeTokenizer)
        messages: list[Record] = [{"arguments": {"z": 1, "a": 2}}]
        restored = json.loads(canonical(messages))
        self.assertEqual(tokenizer.render(messages, []), tokenizer.render(restored, []))
        self.assertEqual(tokenizer.tokenize(str(tokenizer.last_prompt)), (1,))

    def test_tool_stage_selection_excludes_future_calls_and_labels_last_role(self) -> None:
        from benchmarks.sources import glaive

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            tool = {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {"type": "object", "properties": {}},
            }
            chat = 'USER: Look it up\n\nASSISTANT: <functioncall> {"name":"lookup","arguments":{}}\n\nFUNCTION RESPONSE: {"value":1}\n\nASSISTANT: The value is one\n\nUSER: Thank you\n\nASSISTANT: You are welcome'
            path.write_text(
                json.dumps([{"system": json.dumps(tool), "chat": chat} for _ in range(100)])
            )
            rows = list(glaive(path))
            self.assertEqual(
                {row["subtype"] for row in rows}, {"tool_request", "tool_result", "tool_abstention"}
            )
            for row in rows:
                last = record(sequence(row["messages"])[-1])
                self.assertEqual(
                    last["role"], "tool" if row["subtype"] == "tool_result" else "user"
                )
                if row["subtype"] == "tool_request":
                    self.assertEqual(last["content"], "Look it up")
                    self.assertNotIn("tool_calls", str(row["messages"]))

    def test_request_labels_do_not_scan_words_inside_source_context(self) -> None:
        from typing import Iterator
        from unittest.mock import patch

        from benchmarks.sources import ultrachat

        values: list[Record] = [
            {
                "prompt_id": "context",
                "messages": [
                    {
                        "role": "user",
                        "content": "Read this passage: the translation puzzle required planning.",
                    }
                ],
            },
            {
                "prompt_id": "plan",
                "messages": [
                    {"role": "user", "content": "Create a weekly study plan with deadlines."}
                ],
            },
            {
                "prompt_id": "translate",
                "messages": [
                    {"role": "user", "content": "Translate into Spanish: the train is late."}
                ],
            },
            {"prompt_id": "blank", "messages": [{"role": "user", "content": ""}]},
        ]

        def rows(path: Path) -> Iterator[Record]:
            yield from values

        with patch("benchmarks.sources.parquet_rows", new=rows):
            selected = list(ultrachat(Path("unused")))
        self.assertEqual({row["source_key"] for row in selected}, {"plan", "translate"})
        self.assertEqual({row["subtype"] for row in selected}, {"non_arithmetic", "translation"})
