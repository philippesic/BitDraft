import math
from dataclasses import dataclass
from typing import Callable, Protocol

from w1a1_eagle.clock import Clock


@dataclass(frozen=True)
class Budget:
    updates: int
    wall_seconds: float
    attempts: int

    def __post_init__(self) -> None:
        if type(self.updates) is not int or self.updates < 1:
            raise ValueError("positive update budget required")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("positive wall budget required")
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("positive attempt budget required")


@dataclass
class Progress:
    updates: int = 0
    cursor: int = 0
    attempts: int = 0
    failed: int = 0
    wall_seconds: float = 0.0
    last_loss: float | None = None

    def validate(self) -> None:
        if any(
            type(v) is not int or v < 0
            for v in (self.updates, self.cursor, self.attempts, self.failed)
        ):
            raise ValueError("invalid progress counters")
        if self.attempts != self.updates + self.failed or self.cursor != self.updates:
            raise ValueError("inconsistent progress accounting")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("invalid cumulative wall time")
        if self.last_loss is not None and not math.isfinite(self.last_loss):
            raise ValueError("nonfinite loss")


class Worker(Protocol):
    def step(self, cursor: int) -> float: ...


def train(
    worker: Worker,
    clock: Clock,
    budget: Budget,
    progress: Progress,
    stop: Callable[[], bool],
) -> str:
    progress.validate()
    while progress.updates < budget.updates:
        if progress.attempts >= budget.attempts:
            return "attempt-budget"
        if progress.wall_seconds >= budget.wall_seconds:
            return "wall-budget"
        if stop():
            return "stopped"
        started = clock.now()
        progress.attempts += 1
        try:
            loss = worker.step(progress.cursor)
            if not math.isfinite(loss):
                raise ValueError("nonfinite training result")
        except BaseException:
            progress.failed += 1
            raise
        else:
            progress.last_loss = loss
            progress.updates += 1
            progress.cursor += 1
        finally:
            elapsed = clock.now() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("operational clock moved backwards or became nonfinite")
            progress.wall_seconds += elapsed
    return "update-budget"
