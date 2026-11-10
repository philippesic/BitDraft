import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pytest

from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record
from w1a1_eagle.lifecycle import Journal, Ownership, control
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, read
from w1a1_eagle.training import Budget, Progress, train


@dataclass
class FakeClock:
    value: float = 0.0

    def now(self) -> float:
        return self.value


@dataclass
class FakeWorker:
    clock: FakeClock
    fail: bool = False

    def step(self, cursor: int) -> float:
        self.clock.value += 2
        if self.fail:
            raise RuntimeError("injected update failure")
        return float(cursor)


def test_cpu_control_budget_and_cooperative_stop() -> None:
    assert Monotonic().now() >= 0
    clock = FakeClock()
    progress = Progress()
    assert (
        train(FakeWorker(clock), clock, Budget(10, 3, 10), progress, lambda: False) == "wall-budget"
    )
    assert (progress.updates, progress.cursor, progress.wall_seconds) == (2, 2, 4)
    assert train(FakeWorker(clock), clock, Budget(10, 9, 10), progress, lambda: True) == "stopped"
    assert progress.updates == 2
    progress.validate()


def test_failed_attempt_cost_and_retry_limit() -> None:
    clock = FakeClock()
    progress = Progress()
    with pytest.raises(RuntimeError, match="failure"):
        train(FakeWorker(clock, fail=True), clock, Budget(10, 9, 1), progress, lambda: False)
    assert (progress.updates, progress.failed, progress.attempts, progress.wall_seconds) == (
        0,
        1,
        1,
        2,
    )
    progress.validate()
    assert (
        train(FakeWorker(clock), clock, Budget(10, 9, 1), progress, lambda: False)
        == "attempt-budget"
    )


def test_journal_integrity_and_truthful_health(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append({"kind": "started", "attempt": "one"})
    journal.append({"kind": "terminal", "attempt": "one", "outcome": "failed", "wall_seconds": 3.0})
    report = Journal(tmp_path).report()
    assert report["operational"] == "terminal"
    assert report["cumulative_attempt_seconds"] == 3
    assert str(report["scientific"]).startswith("unknown")
    assert str(report["objective"]).startswith("unknown")
    path = tmp_path / "events.jsonl"
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"wall_seconds":3.0', b'"wall_seconds":0.0'))
    with pytest.raises(ValueError, match="chain"):
        Journal(tmp_path).events()
    path.write_bytes(original + b'{"partial":')
    with pytest.raises(ValueError, match="partial"):
        Journal(tmp_path).events()
    missing = tmp_path / "missing"
    assert Journal(missing).report()["operational"] == "incomplete"
    assert not missing.exists()


def test_exact_process_control_and_busy_refusal(tmp_path: Path) -> None:
    owner = ProcessIdentity.read(os.getpid())
    assert owner.alive()
    assert not ProcessIdentity(owner.pid, owner.boot, owner.birth + "0").alive()
    message: Record = {
        "schema": 1,
        "attempt": "one",
        "owner": cast(Record, asdict(owner)),
        "sequence": 1,
        "action": "stop",
    }
    path = tmp_path / "control.json"
    path.write_bytes(canonical(message))
    with pytest.raises(ValueError, match="foreign"):
        control(tmp_path, "two", owner, 1)
    assert control(tmp_path, "one", owner, 1)
    with pytest.raises(ValueError, match="replayed"):
        control(tmp_path, "one", owner, 1)
    ownership = Ownership(tmp_path)
    try:
        with pytest.raises(BlockingIOError):
            Ownership(tmp_path)
    finally:
        ownership.close()


def test_records_reject_duplicate_keys_and_optimized_guards(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="duplicate"):
        read(path)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-B",
            "-c",
            "from w1a1_eagle.training import Budget; Budget(True, 1, 1)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "positive update budget" in result.stderr
