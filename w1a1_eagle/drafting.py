from dataclasses import dataclass, field
from random import Random
from typing import Mapping, Sequence

from w1a1_eagle.contracts import Record

CONTEXT_BANDS = ("<512", "512-2K", "2K-8K", "8K-16K", ">16K")
THINK_OPEN = 248068
THINK_CLOSE = 248069

Ranked = Sequence[Sequence[int]]


def context_band(tokens: int) -> str:
    if type(tokens) is not int or tokens < 0:
        raise ValueError("context length must be a nonnegative integer")
    if tokens < 512:
        return CONTEXT_BANDS[0]
    if tokens < 2048:
        return CONTEXT_BANDS[1]
    if tokens < 8192:
        return CONTEXT_BANDS[2]
    if tokens <= 16384:
        return CONTEXT_BANDS[3]
    return CONTEXT_BANDS[4]


def thinking_mode(tokens: Sequence[int], end: int) -> str:
    prefix = tokens[:end]
    opened = max((index for index, token in enumerate(prefix) if token == THINK_OPEN), default=-1)
    if opened < 0:
        return "absent"
    return "off" if THINK_CLOSE in prefix[opened + 1 :] else "on"


@dataclass(frozen=True)
class Round:
    anchor: int
    proposed: int
    accepted: int


def segments(anchors: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(anchors))
    result: list[tuple[int, int]] = []
    for anchor in ordered:
        if result and result[-1][1] == anchor - 1:
            result[-1] = (result[-1][0], anchor)
        else:
            result.append((anchor, anchor))
    return tuple(result)


def leading(labels: Sequence[int], ranked: Ranked, depth: int) -> tuple[int, int]:
    proposed = min(depth, len(labels), len(ranked))
    accepted = 0
    while accepted < proposed and ranked[accepted][0] == labels[accepted]:
        accepted += 1
    return proposed, accepted


def simulate(
    windows: Mapping[int, Sequence[int]], proposals: Mapping[int, Ranked], depth: int
) -> tuple[Round, ...]:
    if type(depth) is not int or depth < 1:
        raise ValueError("positive draft depth required")
    rounds: list[Round] = []
    for first, last in segments(tuple(windows)):
        anchor = first + 1
        while anchor <= last:
            proposed, accepted = leading(windows[anchor], proposals[anchor], depth)
            if proposed == 0:
                break
            rounds.append(Round(anchor, proposed, accepted))
            anchor += accepted + 1
    return tuple(rounds)


def generated_anchors(windows: Mapping[int, Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        anchor
        for first, last in segments(tuple(windows))
        for anchor in range(first + 1, last + 1)
        if windows[anchor]
    )


@dataclass
class Tally:
    depth: int
    top: int
    positions: int = 0
    proposed: int = 0
    accepted: int = 0
    reached: list[int] = field(default_factory=list)
    first_choice: list[int] = field(default_factory=list)
    shortlist: list[int] = field(default_factory=list)
    survived: list[int] = field(default_factory=list)
    first_wrong: list[int] = field(default_factory=list)
    accepted_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth < 1 or type(self.top) is not int:
            raise ValueError("positive depth and shortlist size required")
        self.reached = self.reached or [0] * self.depth
        self.first_choice = self.first_choice or [0] * self.depth
        self.shortlist = self.shortlist or [0] * self.depth
        self.survived = self.survived or [0] * self.depth
        self.first_wrong = self.first_wrong or [0] * (self.depth + 2)
        self.accepted_counts = self.accepted_counts or [0] * (self.depth + 1)

    def add(self, labels: Sequence[int], ranked: Ranked) -> None:
        proposed, accepted = leading(labels, ranked, self.depth)
        if proposed == 0:
            return
        self.positions += 1
        self.proposed += proposed
        self.accepted += accepted
        self.accepted_counts[accepted] += 1
        alive = True
        for slot in range(proposed):
            choices = ranked[slot]
            hit = choices[0] == labels[slot]
            alive = alive and hit
            self.reached[slot] += 1
            self.first_choice[slot] += int(hit)
            self.shortlist[slot] += int(labels[slot] in choices[: self.top])
            self.survived[slot] += int(alive)
        if accepted < proposed:
            self.first_wrong[accepted] += 1
        elif proposed == self.depth:
            self.first_wrong[self.depth] += 1
        else:
            self.first_wrong[self.depth + 1] += 1

    def merge(self, other: "Tally") -> None:
        if (other.depth, other.top) != (self.depth, self.top):
            raise ValueError("tallies of different depth or shortlist")
        self.positions += other.positions
        self.proposed += other.proposed
        self.accepted += other.accepted
        for mine, theirs in (
            (self.reached, other.reached),
            (self.first_choice, other.first_choice),
            (self.shortlist, other.shortlist),
            (self.survived, other.survived),
            (self.first_wrong, other.first_wrong),
            (self.accepted_counts, other.accepted_counts),
        ):
            for index, value in enumerate(theirs):
                mine[index] += value

    def record(self) -> Record:
        def share(numerator: int, denominator: int) -> float | None:
            return numerator / denominator if denominator else None

        wrong_names = [str(slot + 1) for slot in range(self.depth)] + ["none", "ended"]
        return {
            "positions": self.positions,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance": share(self.accepted, self.proposed),
            "accepted_per_round": share(self.accepted, self.positions),
            "accepted_distribution": {
                str(count): value for count, value in enumerate(self.accepted_counts)
            },
            "first_wrong_slot": dict(zip(wrong_names, self.first_wrong, strict=True)),
            "reached": list(self.reached),
            "first_choice_recall": [
                share(hit, total) for hit, total in zip(self.first_choice, self.reached)
            ],
            "top_recall": [share(hit, total) for hit, total in zip(self.shortlist, self.reached)],
            "prefix_survival": [
                share(hit, total) for hit, total in zip(self.survived, self.reached)
            ],
        }


def tally(
    anchors: Sequence[int],
    windows: Mapping[int, Sequence[int]],
    proposals: Mapping[int, Ranked],
    depth: int,
    top: int,
) -> Tally:
    result = Tally(depth, top)
    for anchor in anchors:
        result.add(windows[anchor], proposals[anchor])
    return result


Pair = tuple[str, float, float, float, float]


def ratio_difference(rows: Sequence[Pair]) -> float | None:
    first = sum(row[2] for row in rows)
    second = sum(row[4] for row in rows)
    if not first or not second:
        return None
    return sum(row[1] for row in rows) / first - sum(row[3] for row in rows) / second


def paired_interval(rows: Sequence[Pair], resamples: int, generator: Random) -> Record:
    estimate = ratio_difference(rows)
    groups: dict[str, list[Pair]] = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    names = sorted(groups)
    if estimate is None or len(names) < 2 or resamples < 1:
        return {"difference": estimate, "groups": len(names), "interval": None}
    values: list[float] = []
    for _ in range(resamples):
        drawn = [row for _ in names for row in groups[names[generator.randrange(len(names))]]]
        value = ratio_difference(drawn)
        if value is not None:
            values.append(value)
    values.sort()
    if len(values) < resamples // 2:
        return {"difference": estimate, "groups": len(names), "interval": None}
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1) + 0.5)]
    return {
        "difference": estimate,
        "groups": len(names),
        "interval": [low, high],
        "resamples": len(values),
    }
