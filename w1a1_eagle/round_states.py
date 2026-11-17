import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, Supervision
from w1a1_eagle.drafting import Ranked, segments, simulate
from w1a1_eagle.records import read

MAP_SCHEMA = "round-state-map-v1"
MAP_BYTES = 512 * 1024 * 1024
SCOPES = ("full", "intent", "sample")

Span = tuple[int, int]


def needed_spans(anchors: Sequence[int], sampled: Iterable[int] | None = None) -> tuple[Span, ...]:
    whole = segments(anchors)
    if sampled is None:
        return whole
    chosen = sorted(set(sampled))
    if not set(chosen) <= set(anchors):
        raise ValueError("sampled anchors lie outside the sequence's windows")
    result: list[Span] = []
    for first, last in whole:
        inside = chosen[bisect.bisect_left(chosen, first) : bisect.bisect_right(chosen, last)]
        if inside:
            result.append((first, inside[-1]))
    return tuple(result)


def contains(spans: Sequence[Span], anchor: int) -> bool:
    index = bisect.bisect_right(spans, anchor, key=span_start) - 1
    return index >= 0 and spans[index][0] <= anchor <= spans[index][1]


def span_start(span: Span) -> int:
    return span[0]


def covered_windows(
    windows: Mapping[int, Sequence[int]], spans: Sequence[Span]
) -> dict[int, Sequence[int]]:
    return {
        anchor: windows[anchor]
        for first, last in spans
        for anchor in range(first, last + 1)
        if anchor in windows
    }


def round_starts(
    windows: Mapping[int, Sequence[int]], proposals: Mapping[int, Ranked], depth: int
) -> tuple[int, ...]:
    return tuple(item.anchor for item in simulate(windows, proposals, depth))


def window_shares(flags: Sequence[bool], weight: float) -> tuple[float, ...]:
    if not flags or not math.isfinite(weight) or weight <= 0:
        raise ValueError("window shares need windows and a positive finite weight")
    raw = [weight if flag else 1.0 for flag in flags]
    total = sum(raw)
    return tuple(value / total for value in raw)


@dataclass(frozen=True)
class RoundStateMap:
    sha256: str
    size: int
    corpus_sha256: str
    corpus_bytes: int
    target: str
    depth: int
    scope: Record
    drafter: Record
    covered: dict[str, tuple[Span, ...]]
    starts: dict[str, frozenset[int]]

    def covers(self, trajectory: str, anchor: int) -> bool:
        return contains(self.covered.get(trajectory, ()), anchor)

    def flags(self, windows: Sequence[Supervision]) -> tuple[bool, ...]:
        result: list[bool] = []
        for window in windows:
            if not self.covers(window.trajectory, window.anchor):
                raise ValueError(
                    "round-state map does not cover trained window "
                    + window.trajectory
                    + "@"
                    + str(window.anchor)
                )
            result.append(window.anchor in self.starts.get(window.trajectory, frozenset()))
        return tuple(result)

    def summary(self) -> Record:
        return {
            "sha256": self.sha256,
            "bytes": self.size,
            "depth": self.depth,
            "scope": self.scope,
            "drafter": self.drafter,
        }


@dataclass(frozen=True)
class RoundStateWeighting:
    states: RoundStateMap
    weight: float

    def __post_init__(self) -> None:
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("round-state weight must be a positive finite number")

    def shares(self, windows: Sequence[Supervision]) -> tuple[float, ...]:
        return window_shares(self.states.flags(windows), self.weight)


def parse_map(raw: Record, sha256: str, size: int) -> RoundStateMap:
    if (
        set(raw)
        != {
            "schema",
            "corpus",
            "target_sha256",
            "depth",
            "scope",
            "drafter",
            "code_sha256",
            "rows",
            "totals",
        }
        or raw["schema"] != MAP_SCHEMA
    ):
        raise ValueError("unsupported round-state map")
    corpus = record(raw["corpus"])
    if set(corpus) != {"sha256", "bytes", "format"}:
        raise ValueError("round-state map requires its corpus identity")
    scope = record(raw["scope"])
    if scope.get("kind") not in SCOPES:
        raise ValueError("unknown round-state map scope")
    depth = integer(raw["depth"])
    if depth < 1:
        raise ValueError("round-state map depth must be positive")
    covered: dict[str, tuple[Span, ...]] = {}
    starts: dict[str, frozenset[int]] = {}
    for value in sequence(raw["rows"]):
        row = record(value)
        if set(row) != {"id", "covered", "starts"}:
            raise ValueError("unsupported round-state map row")
        name = string(row["id"])
        if name in covered:
            raise ValueError("duplicate round-state map row")
        spans: list[Span] = []
        for item in sequence(row["covered"]):
            first, last = (integer(part) for part in sequence(item))
            if first < 0 or last < first or (spans and first <= spans[-1][1] + 1):
                raise ValueError("round-state map spans must be ordered and separate")
            spans.append((first, last))
        anchors = tuple(integer(item) for item in sequence(row["starts"]))
        if list(anchors) != sorted(set(anchors)) or any(
            not (contains(spans, anchor) and contains(spans, anchor - 1)) for anchor in anchors
        ):
            raise ValueError("round starts must be ordered, covered and follow a covered anchor")
        covered[name] = tuple(spans)
        starts[name] = frozenset(anchors)
    totals = record(raw["totals"])
    if (
        set(totals) != {"rows", "covered_windows", "proposals", "round_starts"}
        or integer(totals["rows"]) != len(covered)
        or integer(totals["round_starts"]) != sum(len(value) for value in starts.values())
        or integer(totals["covered_windows"]) < integer(totals["proposals"])
    ):
        raise ValueError("round-state map totals disagree with its rows")
    return RoundStateMap(
        sha256,
        size,
        string(corpus["sha256"]),
        integer(corpus["bytes"]),
        string(raw["target_sha256"]),
        depth,
        scope,
        record(raw["drafter"]),
        covered,
        starts,
    )


def load_map(reference: Artifact) -> RoundStateMap:
    reference.verify()
    return parse_map(read(reference.path, MAP_BYTES), reference.sha256, reference.size)
