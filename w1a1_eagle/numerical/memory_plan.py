import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast

import torch

from w1a1_eagle.contracts import Record
from w1a1_eagle.model import Geometry
from w1a1_eagle.numerical.weights import is_table, shapes

MIB = 1024**2
DEVICE_MARGIN = 512 * MIB
PROCESS_HOST = 3072 * MIB
FLOAT = 4


class FatalMemory(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    name: str
    device: int
    host: int

    def record(self) -> Record:
        return {
            "step": self.name,
            "device_gib": round(self.device / 1024**3, 2),
            "host_gib": round(self.host / 1024**3, 2),
        }


def he4_elements(geometry: Geometry, rows: int) -> tuple[int, int]:
    frozen = rows * geometry.hidden
    trainable = 0
    for name, shape in shapes(geometry).items():
        size = 1
        for extent in shape:
            size *= extent
        if is_table(name):
            frozen += size
        elif ".ffn_" in name and len(shape) == 2:
            trainable += size + 2 * shape[0]
        else:
            trainable += size
    trainable += geometry.layers * (geometry.hidden + geometry.intermediate + 2)
    return frozen, trainable


def baseline_elements(geometry: Geometry, rows: int) -> int:
    total = rows * geometry.hidden
    for shape in shapes(geometry).values():
        size = 1
        for extent in shape:
            size *= extent
        total += size
    return total


def working(
    geometry: Geometry, context: int, anchors: int, slots: int, rows: int, scored: int
) -> int:
    cfg = geometry
    features = context * cfg.taps * cfg.hidden * (FLOAT + 2)
    fused = 4 * context * cfg.hidden * FLOAT
    keys = 12 * context * cfg.kv_heads * cfg.head_dim * FLOAT
    scores = 3 * cfg.heads * anchors * slots * (context + slots) * FLOAT
    states = 16 * anchors * slots * max(cfg.hidden, cfg.intermediate) * FLOAT
    logits = 2 * anchors * scored * rows * FLOAT
    return (features + fused + keys + scores + states + logits) * 5 // 4


def capture_host(geometry: Geometry, context: int) -> int:
    return 3 * context * geometry.taps * geometry.hidden * FLOAT


@dataclass(frozen=True)
class Shapes:
    geometry: Geometry
    candidate_rows: int | None
    baseline_rows: int | None
    rounds_context: int
    windows_context: int
    chunk: int
    depth: int
    exports: int
    ranges: bool
    state_bytes: int = 0
    scoring: int = 0
    scoring_host: int = 0


def plan(shape: Shapes, cuda: bool) -> tuple[Step, ...]:
    cfg = shape.geometry
    steps: list[Step] = []
    on_device: Callable[[int], int] = (lambda value: value) if cuda else (lambda _: 0)
    on_host: Callable[[int], int] = (lambda _: 0) if cuda else (lambda value: value)
    if shape.candidate_rows is not None:
        frozen, trainable = he4_elements(cfg, shape.candidate_rows)
        resident = (frozen + trainable) * FLOAT
        moments = 2 * trainable * FLOAT
        largest = shape.candidate_rows * cfg.hidden * FLOAT
        loading = resident * 5 // 2
        steps.append(Step("candidate load", resident, PROCESS_HOST + loading))
        steps.append(
            Step(
                "candidate restore",
                on_device(resident + moments),
                PROCESS_HOST + largest + shape.state_bytes + on_host(resident + moments),
            )
        )
        if shape.rounds_context:
            rounds = (
                resident
                + working(
                    cfg,
                    shape.rounds_context,
                    shape.chunk,
                    shape.depth + 1,
                    shape.candidate_rows,
                    shape.depth,
                )
                + shape.scoring
            )
            steps.append(
                Step(
                    "candidate rounds",
                    on_device(rounds),
                    PROCESS_HOST
                    + capture_host(cfg, shape.rounds_context)
                    + shape.scoring_host
                    + on_host(rounds),
                )
            )
        if shape.windows_context:
            window = working(cfg, shape.windows_context, 1, 4, shape.candidate_rows, 4)
            probe = 64 * MIB if shape.ranges else 0
            steps.append(
                Step(
                    "candidate windows",
                    on_device(resident + window + probe),
                    PROCESS_HOST
                    + capture_host(cfg, shape.windows_context)
                    + on_host(resident + window + probe),
                )
            )
        if shape.exports:
            parameters = cfg.intermediate * cfg.hidden
            scratch = (
                parameters * 32 + (frozen + trainable) * 2 + shape.candidate_rows * cfg.hidden * 9
            )
            steps.append(
                Step(
                    "candidate exports",
                    on_device(resident + moments),
                    max(
                        PROCESS_HOST + scratch,
                        PROCESS_HOST + largest + shape.state_bytes,
                    )
                    + on_host(resident + moments),
                )
            )
    if shape.baseline_rows is not None:
        resident = baseline_elements(cfg, shape.baseline_rows) * FLOAT
        steps.append(
            Step(
                "baseline load",
                resident,
                PROCESS_HOST + 8 * 8192 * cfg.hidden * FLOAT + on_host(resident),
            )
        )
        if shape.rounds_context:
            extra = (
                working(
                    cfg,
                    shape.rounds_context,
                    shape.chunk,
                    shape.depth + 1,
                    shape.baseline_rows,
                    shape.depth,
                )
                + shape.scoring
            )
            steps.append(
                Step(
                    "baseline rounds",
                    on_device(resident + extra),
                    PROCESS_HOST
                    + capture_host(cfg, shape.rounds_context)
                    + shape.scoring_host
                    + on_host(resident + extra),
                )
            )
    if not cuda:
        steps = [Step(step.name, 0, step.host) for step in steps]
    return tuple(steps)


def refusal(steps: tuple[Step, ...], device_free: int | None, host_limit: int) -> str | None:
    for step in steps:
        if step.host > host_limit:
            return (
                f"{step.name} needs about {step.host / 1024**3:.1f} GiB of host memory;"
                f" the limit is {host_limit / 1024**3:.1f} GiB"
            )
        if device_free is not None and step.device > device_free - DEVICE_MARGIN:
            return (
                f"{step.name} needs about {step.device / 1024**3:.1f} GiB of GPU memory;"
                f" {device_free / 1024**3:.1f} GiB is free"
            )
    return None


def host_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("available host memory unknown")


def host_resident() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def host_peak() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise ValueError("peak host memory unknown")


def out_of_memory_count() -> int:
    initialized = cast(Callable[[], bool], getattr(torch.cuda, "is_initialized"))
    if not torch.cuda.is_available() or not initialized():
        return 0
    statistics = cast(Callable[[], dict[str, int]], getattr(torch.cuda, "memory_stats"))
    return int(statistics().get("num_ooms", 0))


@dataclass
class Guard:
    cuda: bool
    host_limit: int
    failures: Callable[[], int] = out_of_memory_count
    resident: Callable[[], int] = host_resident
    measured: list[Record] = field(default_factory=list)

    def check(self) -> None:
        if self.failures():
            raise FatalMemory("CUDA reported an out-of-memory failure; stopping")
        if self.resident() > self.host_limit:
            raise FatalMemory("host memory exceeded the declared limit; stopping")

    def begin(self) -> None:
        self.check()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def end(self, name: str) -> None:
        self.check()
        self.measured.append(
            {
                "step": name,
                "device_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                if self.cuda
                else None,
                "host_resident_gib": round(self.resident() / 1024**3, 2),
                "host_peak_gib": round(host_peak() / 1024**3, 2),
            }
        )

    def release(self, stage: str) -> None:
        if self.cuda:
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated() > 64 * MIB:
                raise FatalMemory(stage + " left model memory on the GPU")
        self.check()
