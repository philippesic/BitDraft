import math
import random
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, Sequence

from w1a1_eagle.clock import Clock

PARAMETER_GROUPS = ("sign", "scale", "midpoint", "gain", "clip", "norm")
TABLE_GROUP = "table"
SOFT_FAMILY = "soft"
ARITHMETIC = ("reference", "fast-exact", "fast-exact-tf32ff")
Arithmetic = Literal["reference", "fast-exact", "fast-exact-tf32ff"]
MAXIMUM_SLOTS = 16
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

    def windows(self, shape: Shape, context: int, slots: int | None = None) -> int:
        if type(context) is not int or context < 0:
            raise ValueError("nonnegative context length required")
        block = shape.slots if slots is None else slots
        if type(block) is not int or not 1 <= block <= shape.slots:
            raise ValueError("block size must be between one slot and the model's slots")
        row = ROW_COPIES * (shape.hidden + shape.intermediate) * shape.layers
        row += ROW_TOKEN_COPIES * shape.heads * shape.layers * context
        shared = CONTEXT_COPIES * (shape.taps + shape.layers) * shape.hidden * context
        available = int((self.activation_bytes - 4 * shared) // (4 * block * row))
        return max(1, min(self.max_windows * shape.slots // block, available))

    def admit_device(self, frozen_bytes: int, trained_bytes: int, retained_bytes: int) -> int:
        if any(
            type(value) is not int or value < 0
            for value in (frozen_bytes, trained_bytes, retained_bytes)
        ):
            raise ValueError("nonnegative resident byte counts required")
        demand = frozen_bytes + 4 * trained_bytes + retained_bytes + self.activation_bytes
        if demand > self.device_bytes:
            raise ValueError(
                f"resident state ({frozen_bytes} frozen, 4 x {trained_bytes} trained with"
                f" gradients and Adam moments, {retained_bytes} retained fusion input) plus"
                f" activation_bytes {self.activation_bytes} needs {demand} bytes, more than"
                f" device_bytes {self.device_bytes}; lower activation_bytes or raise device_bytes"
            )
        return demand


@dataclass(frozen=True)
class Blocks:
    slots: dict[int, float] = field(default_factory=lambda: {MAXIMUM_SLOTS: 1.0})
    position_weights: Literal["decay", "reach"] = "decay"

    def __post_init__(self) -> None:
        if not self.slots or any(
            type(size) is not int
            or not 2 <= size <= MAXIMUM_SLOTS
            or type(weight) is not float
            or not math.isfinite(weight)
            or weight <= 0
            for size, weight in self.slots.items()
        ):
            raise ValueError("block sizes must be 2-16 slots with positive finite weights")
        if self.position_weights not in ("decay", "reach"):
            raise ValueError("position weights must be decay or reach")

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self.slots))

    @property
    def fixed(self) -> int | None:
        return self.sizes[0] if len(self.slots) == 1 else None

    def draw(self, generator: random.Random) -> int:
        sizes = self.sizes
        return generator.choices(sizes, [self.slots[size] for size in sizes])[0]

    def encode(self) -> dict[str, object]:
        return {
            "slots": {str(size): self.slots[size] for size in self.sizes},
            "position_weights": self.position_weights,
        }


def reach_weights(windows: Sequence[int], survivors: Sequence[int]) -> tuple[float, ...]:
    if (
        len(windows) != MAXIMUM_SLOTS - 1
        or len(survivors) != len(windows)
        or any(type(v) is not int or v < 0 for v in (*windows, *survivors))
        or any(alive > count for alive, count in zip(survivors, windows, strict=True))
    ):
        raise ValueError("reach weights need survivor and window counts for 15 positions")
    weights = [1.0]
    for alive, count in zip(survivors[:-1], windows[:-1], strict=True):
        weights.append((alive + 1) / (count + 2))
    return tuple(weights)


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
        if set(self.step_sizes) not in (
            set(PARAMETER_GROUPS),
            {*PARAMETER_GROUPS, TABLE_GROUP},
        ) or any(
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

    @property
    def groups(self) -> tuple[str, ...]:
        return PARAMETER_GROUPS + ((TABLE_GROUP,) if TABLE_GROUP in self.step_sizes else ())


@dataclass(frozen=True)
class Features:
    scale: Literal["learned", "fixed"] = "learned"
    clip: Literal["learned", "fixed"] = "learned"
    gain: Literal["learned", "off"] = "learned"
    midpoint: Literal["learned", "off"] = "learned"

    def __post_init__(self) -> None:
        if (
            self.scale not in ("learned", "fixed")
            or self.clip not in ("learned", "fixed")
            or self.gain not in ("learned", "off")
            or self.midpoint not in ("learned", "off")
        ):
            raise ValueError(
                "features: scale/clip are learned or fixed; gain/midpoint learned or off"
            )

    @property
    def default(self) -> bool:
        return self == Features()

    @property
    def untrained(self) -> frozenset[str]:
        return frozenset(
            name
            for name, value in (
                ("scale", self.scale),
                ("clip", self.clip),
                ("gain", self.gain),
                ("midpoint", self.midpoint),
            )
            if value != "learned"
        )

    @property
    def off(self) -> frozenset[str]:
        return frozenset(
            name
            for name, value in (("gain", self.gain), ("midpoint", self.midpoint))
            if value == "off"
        )

    def encode(self) -> dict[str, str]:
        return {
            "scale": self.scale,
            "clip": self.clip,
            "gain": self.gain,
            "midpoint": self.midpoint,
        }


@dataclass(frozen=True)
class Sample:
    sequences: int
    windows: int
    seed: int

    def __post_init__(self) -> None:
        if (
            type(self.sequences) is not int
            or not 1 <= self.sequences <= 64
            or type(self.windows) is not int
            or not 1 <= self.windows <= 32
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("a sample needs 1-64 sequences, 1-32 windows each and a seed >= 0")

    def encode(self) -> dict[str, int]:
        return {"sequences": self.sequences, "windows": self.windows, "seed": self.seed}


@dataclass(frozen=True)
class StartCheck:
    sample: Sample
    max_relative_error: float
    mean_relative_error: float

    def __post_init__(self) -> None:
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0
            for value in (self.max_relative_error, self.mean_relative_error)
        ):
            raise ValueError("start check tolerances must be positive finite numbers")


@dataclass(frozen=True)
class GradualPrecision:
    start_temperature: float
    end_temperature: float
    shape: Literal["linear", "exponential"]
    soft_updates: int
    step_size: float
    start_check: StartCheck

    def __post_init__(self) -> None:
        if (
            type(self.start_temperature) is not float
            or type(self.end_temperature) is not float
            or not math.isfinite(self.start_temperature)
            or not math.isfinite(self.end_temperature)
            or not 0 < self.start_temperature < self.end_temperature
        ):
            raise ValueError("gradual precision needs 0 < start temperature < end temperature")
        if self.shape not in ("linear", "exponential"):
            raise ValueError("gradual precision shape is linear or exponential")
        if type(self.soft_updates) is not int or self.soft_updates < 1:
            raise ValueError("gradual precision needs at least one soft update")
        if (
            type(self.step_size) is not float
            or not math.isfinite(self.step_size)
            or self.step_size <= 0
        ):
            raise ValueError("gradual precision needs a positive finite soft step size")

    def encode(self) -> dict[str, object]:
        return {
            "start_temperature": self.start_temperature,
            "end_temperature": self.end_temperature,
            "shape": self.shape,
            "soft_updates": self.soft_updates,
            "step_size": self.step_size,
            "start_check": {
                "sample": self.start_check.sample.encode(),
                "max_relative_error": self.start_check.max_relative_error,
                "mean_relative_error": self.start_check.mean_relative_error,
            },
        }

    def temperature(self, updates: int) -> float | None:
        if type(updates) is not int or updates < 0:
            raise ValueError("nonnegative update count required")
        if updates >= self.soft_updates:
            return None
        fraction = updates / (self.soft_updates - 1) if self.soft_updates > 1 else 0.0
        if self.shape == "linear":
            return (
                self.start_temperature + (self.end_temperature - self.start_temperature) * fraction
            )
        return self.start_temperature * math.pow(
            self.end_temperature / self.start_temperature, fraction
        )


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
