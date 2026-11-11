import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from benchmarks.sources import commitpack, qasper


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
                "repos": ["fixture/repo"],
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
