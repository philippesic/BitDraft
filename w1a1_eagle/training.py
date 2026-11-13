import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

from w1a1_eagle.clock import Clock

PARAMETER_GROUPS = ("sign", "scale", "midpoint", "gain", "clip", "norm")
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


def default_step_sizes() -> dict[str, float]:
    return {"sign": 1e-3, "scale": 1e-5, "midpoint": 1e-5, "gain": 1e-5, "clip": 1e-5, "norm": 1e-5}


@dataclass(frozen=True)
class Schedule:
    decay: Literal["constant", "cosine"] = "constant"
    warmup_updates: int = 0
    start_update: int = 0
    final_update: int | None = None

    def __post_init__(self) -> None:
        if self.decay not in ("constant", "cosine"):
            raise ValueError("schedule decay must be constant or cosine")
        if any(
            type(value) is not int or value < 0
            for value in (self.warmup_updates, self.start_update)
        ):
            raise ValueError("schedule start and warm-up must be nonnegative update counts")
        if self.decay == "constant" and self.final_update is not None:
            raise ValueError("a constant schedule has no final update")
        if self.decay == "cosine" and (
            type(self.final_update) is not int
            or self.final_update <= self.start_update + self.warmup_updates
        ):
            raise ValueError("cosine decay requires a final update after its warm-up")

    def multiplier(self, updates: int) -> float:
        if type(updates) is not int or updates < self.start_update:
            raise ValueError("update precedes the schedule start")
        elapsed = updates - self.start_update
        if elapsed < self.warmup_updates:
            return (elapsed + 1) / self.warmup_updates
        if self.final_update is None:
            return 1.0
        span = self.final_update - self.start_update - self.warmup_updates
        progress = (elapsed - self.warmup_updates) / span
        if progress >= 1:
            raise ValueError("update is at or beyond the schedule's final update")
        return 0.5 * (1 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class Optimization:
    step_sizes: dict[str, float] = field(default_factory=default_step_sizes)
    schedule: Schedule = Schedule()
    latent_multiple: float | None = None

    def __post_init__(self) -> None:
        if tuple(sorted(self.step_sizes)) != tuple(sorted(PARAMETER_GROUPS)) or any(
            type(value) is not float or not math.isfinite(value) or value <= 0
            for value in self.step_sizes.values()
        ):
            raise ValueError("positive finite step sizes required for every parameter group")
        if self.latent_multiple is not None and (
            type(self.latent_multiple) is not float
            or not math.isfinite(self.latent_multiple)
            or self.latent_multiple <= 0
        ):
            raise ValueError("bounded latent scale requires a positive finite multiple")


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
