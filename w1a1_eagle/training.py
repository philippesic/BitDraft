import math
from dataclasses import dataclass
from typing import Callable, Protocol

from w1a1_eagle.clock import Clock

ROW_COPIES = 15.0
ROW_TOKEN_COPIES = 1.25
CONTEXT_COPIES = 1.0


class Shape(Protocol):
    @property
    def hidden(self) -> int: ...
    @property
    def intermediate(self) -> int: ...
    @property
    def heads(self) -> int: ...
    @property
    def layers(self) -> int: ...
    @property
    def slots(self) -> int: ...
    @property
    def taps(self) -> int: ...


@dataclass(frozen=True)
class Grouping:
    max_windows: int = 32
    activation_bytes: int = 3 * 1024**3
    device_bytes: int = 14 * 1024**3

    def __post_init__(self) -> None:
        if type(self.max_windows) is not int or not 1 <= self.max_windows <= 64:
            raise ValueError("windows per update must be between 1 and 64")
        if type(self.activation_bytes) is not int or self.activation_bytes < 1:
            raise ValueError("positive activation byte budget required")
        if type(self.device_bytes) is not int or self.device_bytes < self.activation_bytes:
            raise ValueError("device memory limit must cover the activation budget")

    def windows(self, shape: Shape, context: int) -> int:
        if type(context) is not int or context < 0:
            raise ValueError("nonnegative context length required")
        row = ROW_COPIES * (shape.hidden + shape.intermediate) * shape.layers
        row += ROW_TOKEN_COPIES * shape.heads * shape.layers * context
        shared = CONTEXT_COPIES * (shape.taps + shape.layers) * shape.hidden * context
        available = int((self.activation_bytes - 4 * shared) // (4 * shape.slots * row))
        return max(1, min(self.max_windows, available))


@dataclass(frozen=True)
class Budget:
    updates: int
    wall_seconds: float
    attempts: int

    def __post_init__(self) -> None:
        if type(self.updates) is not int or self.updates < 1:
            raise ValueError("positive update budget required")
        if (
            type(self.wall_seconds) not in (int, float)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
        ):
            raise ValueError("positive wall budget required")
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("positive attempt budget required")


@dataclass
class Progress:
    updates: int = 0
    cursor: int = 0
    attempts: int = 0
    failed: int = 0
    discarded: int = 0
    needs_recovery: bool = False
    wall_seconds: float = 0.0
    last_loss: float | None = None

    def validate(self) -> None:
        if any(
            type(v) is not int or v < 0
            for v in (self.updates, self.cursor, self.attempts, self.failed, self.discarded)
        ):
            raise ValueError("invalid progress counters")
        if (
            self.attempts != self.updates + self.failed + self.discarded
            or type(self.needs_recovery) is not bool
        ):
            raise ValueError("inconsistent progress accounting")
        if (
            type(self.wall_seconds) not in (int, float)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise ValueError("invalid cumulative wall time")
        if self.last_loss is not None and (
            type(self.last_loss) not in (int, float) or not math.isfinite(self.last_loss)
        ):
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
        if progress.needs_recovery:
            return "recovery-required"
        if progress.wall_seconds >= budget.wall_seconds:
            return "wall-budget"
        if stop():
            return "stopped"
        started = clock.now()
        if not math.isfinite(started):
            raise ValueError("operational clock is nonfinite before update")
        progress.attempts += 1
        try:
            loss = worker.step(progress.cursor)
            if not math.isfinite(loss):
                raise ValueError("nonfinite training result")
        except BaseException:
            progress.failed += 1
            progress.needs_recovery = True
            raise
        else:
            progress.last_loss = loss
            progress.updates += 1
            progress.cursor += 1
        finally:
            elapsed = clock.now() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                progress.needs_recovery = True
                raise ValueError("operational clock moved backwards or became nonfinite")
            progress.wall_seconds += elapsed
    return "update-budget"
