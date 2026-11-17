import hashlib
from pathlib import Path

import pytest

from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact
from w1a1_eagle.drafting import THINK_CLOSE, THINK_OPEN
from w1a1_eagle.numerical.labels import label_report
from w1a1_eagle.records import canonical, reference


def authenticated(path: Path) -> Artifact:
    return Artifact(path, hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)


def write(path: Path, value: object) -> Artifact:
    path.write_bytes(canonical(value))
    return authenticated(path)


def request(prompt: list[int], answer: list[int]) -> Record:
    tokens = prompt + answer
    return {
        "input_tokens": tokens,
        "prompt_tokens": len(prompt),
        "logit_positions": list(range(len(prompt) - 1, len(tokens) - 1)),
        "greedy_ids": answer,
        "eog_ids": [9],
        "window_policy": "up-to-15",
    }


def mixture(root: Path, rows: list[tuple[str, str, str, Record]], role: str = "train") -> Artifact:
    records = []
    for identity, group, topic, metadata in rows:
        records.append(
            {
                "id": identity,
                "group": group,
                "topic": topic,
                "role": role,
                "metadata": reference(write(root / f"{identity}.json", metadata)),
                "files": [],
            }
        )
    corpus = write(root / "corpus.json", {"target_sha256": "t", "records": records})
    return write(
        root / "mixture.json",
        {
            "schema": "corpus-mixture-v1",
            "target_sha256": "t",
            "components": [{"kind": "corpus-v1", "source": reference(corpus)}],
        },
    )


def test_label_report_counts_conversations_labels_and_sources(tmp_path: Path) -> None:
    long_prompt = [1] * 2999 + [THINK_OPEN, 2, THINK_CLOSE, 3]
    session = {
        "input_tokens": [1] * 20 + [4, 4, 4, 9] + [1] * 6 + [5, 5, 9],
        "logit_positions": list(range(19, 23)) + list(range(29, 32)),
        "greedy_ids": [4, 4, 4, 9, 5, 5, 9],
        "eog_ids": [9],
        "supervised_spans": [[19, 22], [29, 31]],
        "window_policy": "up-to-15",
    }
    reference_mixture = mixture(
        tmp_path,
        [
            ("short", "g1", "code", request([1, 2, THINK_OPEN, 4, THINK_CLOSE, 3], [6, 7, 9])),
            ("think", "g1", "reasoning", request([1, THINK_OPEN, 3], [6, 9])),
            ("paper", "g2", "documents", request(long_prompt, [6, 7, 8, 9])),
            ("tools", "g3", "code", session),
        ],
    )
    report = label_report(reference_mixture)
    scenarios = record(report["by_scenario"])
    assert scenarios["ordinary-code"] == {"conversations": 1, "labels": 3, "sources": 1}
    assert scenarios["thinking-on"] == {"conversations": 1, "labels": 2, "sources": 1}
    assert scenarios["long-documents"] == {"conversations": 1, "labels": 4, "sources": 1}
    session_cell = record(scenarios["agentic-sessions"])
    session_labels = session_cell["labels"]
    assert isinstance(session_labels, int) and session_labels > 0
    assert session_cell["conversations"] == 1 and session_cell["sources"] == 1
    assert record(report["total"]) == {
        "conversations": 4,
        "labels": 9 + session_labels,
        "sources": 3,
    }
    assert record(report["by_thinking"])["on"] == {"conversations": 1, "labels": 2, "sources": 1}
    assert record(report["by_thinking"])["absent"] == session_cell
    assert record(record(report["by_context_band"])["2K-8K"])["labels"] == 4
    components = report["components"]
    assert isinstance(components, list)
    assert record(components[0])["labels"] == 9 + session_labels
    (tmp_path / "dev").mkdir()
    with pytest.raises(ValueError, match="TRAIN"):
        label_report(mixture(tmp_path / "dev", [("x", "g", "code", request([1], [9]))], "dev"))
