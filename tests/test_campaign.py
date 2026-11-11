from collections import Counter
from pathlib import Path

import pytest

from benchmarks.campaign import select
from benchmarks.comparison import Arm, Comparison
from benchmarks.runner import execution_blocks
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact, position_count


def test_selection_keeps_earlier_ties_and_refuses_partial_or_invalid() -> None:
    leader: Record = {"state": "earlier", "score": 200.0}
    result: Record = {
        "status": "complete",
        "summary": {
            "candidate": {"correct": True, "tokens_per_second": 200.0},
            "prior-leader": {"correct": True, "tokens_per_second": 200.0},
        },
    }
    assert select(result, leader, "new-state", "new-export", {}) is leader
    result["summary"] = {
        "candidate": {"correct": True, "tokens_per_second": 201.0},
        "prior-leader": {"correct": True, "tokens_per_second": 200.0},
    }
    selected = select(result, leader, "new-state", "new-export", {})
    assert record(selected)["state"] == "new-state"
    with pytest.raises(ValueError, match="partial"):
        select({"status": "incomplete"}, None, "state", "export", {})
    with pytest.raises(ValueError, match="invalid"):
        select(
            {"status": "complete", "summary": {"candidate": {"correct": False}}},
            None,
            "state",
            "export",
            {},
        )


def test_mixed_partial_windows_count_actual_positions() -> None:
    assert position_count((10, 11, 12), (3, 2, 1)) == 3
    assert position_count((10, 11, 12)) == 17
    with pytest.raises(ValueError, match="lengths"):
        position_count((10, 11), (3,))
    with pytest.raises(ValueError, match="lengths"):
        position_count((10,), (16,))


def test_grouped_blocks_balance_arm_positions_and_preserve_every_request(tmp_path: Path) -> None:
    ref = Artifact(tmp_path / "input", "a" * 64, 1)
    arms = (Arm("target", None), Arm("baseline", ref), Arm("candidate", ref))
    tasks: tuple[Record, ...] = tuple({"id": "task-" + str(i)} for i in range(90))
    spec = Comparison(
        ref, ref, (ref,), ref, arms, ref, tasks, 2, 0, 16, 2, "dev", "harness", frozenset()
    )
    positions: Counter[tuple[str, int]] = Counter()
    requests: Counter[tuple[int, str, str]] = Counter()
    for rep, _, selected, ordered, seed in execution_blocks(spec):
        assert seed == 8101
        for position, arm in enumerate(ordered):
            positions[arm.name, position] += 1
            for task in selected:
                requests[rep, str(task["id"]), arm.name] += 1
    assert set(positions.values()) == {2}
    assert len(requests) == 90 * 3 * 2 and set(requests.values()) == {1}
