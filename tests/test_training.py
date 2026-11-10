from dataclasses import dataclass

import pytest

from w1a1_eagle.clock import Monotonic
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
