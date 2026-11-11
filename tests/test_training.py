import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pytest

from w1a1_eagle.__main__ import human
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record
from w1a1_eagle.identity import freeze_source, source_identity
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
    rendered = human(report)
    assert "controller outcome failed" in rendered
    assert "worker stop unknown" in rendered
    assert "no completed worker evidence" in rendered
    path = tmp_path / "events.jsonl"
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"wall_seconds":3.0', b'"wall_seconds":0.0'))
    with pytest.raises(ValueError, match="chain"):
        Journal(tmp_path).events()
    path.write_bytes(original + b'{"partial":')
    with pytest.raises(ValueError, match="partial"):
        Journal(tmp_path).events()
    Journal(tmp_path).recover_tail()
    recovered = Journal(tmp_path).events()
    assert recovered[:2] == Journal.decode(original)[0]
    assert recovered[-1]["kind"] == "tail-recovered"
    receipt = read(next((tmp_path / "incidents").glob("*.json")))
    assert bytes.fromhex(str(receipt["tail_hex"])) == b'{"partial":'
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
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
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


def test_independent_wall_timer_and_invalid_clock() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "w1a1_eagle.bootstrap",
            "0.001",
            "--preflight",
            "/missing-disposable-fixture",
        ],
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == -signal.SIGALRM
    clock = FakeClock(float("nan"))
    progress = Progress()
    with pytest.raises(ValueError, match="before update"):
        train(FakeWorker(clock), clock, Budget(1, 1, 1), progress, lambda: False)
    assert progress.attempts == progress.updates == 0


def test_runtime_snapshot_survives_checkout_changes(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "w1a1_eagle").mkdir(parents=True)
    (checkout / "policy").mkdir()
    for name, data in {
        "w1a1_eagle/__init__.py": b"",
        "w1a1_eagle/value.py": b"VALUE = 17\n",
        "uv.lock": b"lock",
        "pyproject.toml": b"project",
        "policy/numerical.toml": b"schema = 1",
    }.items():
        (checkout / name).write_bytes(data)
    expected = source_identity(checkout)
    frozen = freeze_source(checkout, tmp_path / "sources", expected, 0)
    (checkout / "w1a1_eagle/value.py").write_text("VALUE = 99\n")
    assert source_identity(frozen) == expected != source_identity(checkout)
    result = subprocess.run(
        [sys.executable, "-B", "-c", "from w1a1_eagle.value import VALUE; print(VALUE)"],
        cwd=frozen,
        env={**os.environ, "PYTHONPATH": str(frozen), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "17"
    with pytest.raises(ValueError, match="changed before"):
        freeze_source(checkout, tmp_path / "sources", expected, 0)
    (frozen / "w1a1_eagle/value.py").chmod(0o644)
    (frozen / "w1a1_eagle/value.py").write_text("VALUE = 0\n")
    (checkout / "w1a1_eagle/value.py").write_text("VALUE = 17\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        freeze_source(checkout, tmp_path / "sources", expected, 0)


def test_report_refuses_ambiguous_or_negative_attempt_cost(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append({"kind": "started", "attempt": "one"})
    journal.append(
        {"kind": "terminal", "attempt": "one", "outcome": "complete", "wall_seconds": -1}
    )
    with pytest.raises(ValueError, match="terminal outcome or cost"):
        journal.report()
    other = Journal(tmp_path / "duplicate")
    other.append({"kind": "started", "attempt": "one"})
    for _ in range(2):
        other.append(
            {"kind": "terminal", "attempt": "one", "outcome": "complete", "wall_seconds": 1}
        )
    with pytest.raises(ValueError, match="ambiguous"):
        other.report()


def test_report_json_flag_positions_and_failure_exit(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    journal.append({"kind": "started", "attempt": "fixture"})
    journal.append(
        {"kind": "terminal", "attempt": "fixture", "outcome": "failed", "wall_seconds": 6.25}
    )
    outputs = []
    for arguments in (["--json", "report", str(tmp_path)], ["report", str(tmp_path), "--json"]):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "w1a1_eagle", *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert '"cumulative_attempt_seconds": 6.25' in outputs[0]


def test_task_sampling_balances_tasks_and_retains_locality() -> None:
    from w1a1_eagle.sampling import TaskSampler

    sampler = TaskSampler(
        (1, 100000, 2), ("first", "first", "second"), {"first": 1, "second": 3}, 8101, 8
    )
    batches = [sampler.locate(cursor * 8)[0] for cursor in range(2000)]
    first = [value for value in batches if value in (0, 1)]
    assert 0.17 < len(first) / len(batches) < 0.33
    assert 0.4 < first.count(0) / len(first) < 0.6
    for cursor in range(0, 80, 8):
        assert len({sampler.locate(cursor + offset)[0] for offset in range(8)}) == 1
        assert sampler.locate(cursor) == sampler.locate(cursor)


def test_streamed_assets_and_retention_preserve_recoverable_states(tmp_path: Path) -> None:
    from typing import BinaryIO

    from w1a1_eagle.checkpoints.store import Store

    store = Store(tmp_path)

    def interrupted(output: BinaryIO) -> None:
        output.write(b"partial")
        raise ValueError("interrupted asset")

    with pytest.raises(ValueError, match="interrupted"):
        store.stream("layer.f16", interrupted, 32)
    assert not (tmp_path / "layer.f16").exists()
    assert list(tmp_path.glob(".pending-*")) == []
    states = [store.commit(bytes([index]), "recipe", index) for index in range(4)]
    retired = store.prune(2, (states[0].artifact.sha256,))
    assert set(retired) == {states[1].artifact.sha256, states[2].artifact.sha256}
    assert states[0].artifact.path.exists()
    assert store.latest("recipe") == states[3]
