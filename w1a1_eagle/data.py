import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from w1a1_eagle.contracts import components

Role = Literal["train", "dev", "calibration", "heldout"]
PriorUse = Literal["unobserved", "historical-training", "unknown"]


def prior_use(value: object) -> PriorUse:
    choices: dict[str, PriorUse] = {
        "unobserved": "unobserved",
        "historical-training": "historical-training",
        "unknown": "unknown",
    }
    if not isinstance(value, str) or value not in choices:
        raise ValueError("unsupported prior-use classification")
    return choices[value]


def optional_topic(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("topic must be text or null")
    return value.strip() or None


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    size: int
    boundary: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or type(self.size) is not int
            or self.size < 0
            or not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("invalid artifact identity")

    def verify(self) -> None:
        if self.boundary is not None and not self.path.resolve().is_relative_to(
            self.boundary.resolve()
        ):
            raise ValueError("artifact escaped its declared input boundary")
        if not self.path.is_file() or self.path.stat().st_size != self.size:
            raise ValueError("artifact size or existence mismatch")
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ValueError("artifact content mismatch")


@dataclass(frozen=True)
class Prompt:
    identity: str
    source: str
    group: str
    topic: str | None
    aliases: tuple[str, ...]
    role: Role
    ancestry: str
    prior_use: PriorUse = "unknown"

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v.strip()
            for v in (self.identity, self.source, self.group, self.ancestry)
        ):
            raise ValueError("missing prompt identity")
        if self.role not in {"train", "dev", "calibration", "heldout"}:
            raise ValueError("unsupported role")
        if not isinstance(self.aliases, tuple) or any(
            not isinstance(a, str) or not a.strip() for a in self.aliases
        ):
            raise ValueError("empty alias")
        object.__setattr__(self, "topic", optional_topic(self.topic))
        prior_use(self.prior_use)


@dataclass(frozen=True)
class CaptureSpec:
    identity: str
    metadata: Artifact
    files: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity, str)
            or not self.identity.strip()
            or not isinstance(self.files, tuple)
        ):
            raise ValueError("invalid capture identity or mutable file inventory")


@dataclass(frozen=True)
class CaptureRow:
    prompt: Prompt
    capture: CaptureSpec
    anchors: tuple[int, ...]
    unique_positions: int
    label_lengths: tuple[int, ...] = ()


@dataclass(frozen=True)
class Dataset:
    source: Artifact
    target: str
    rows: tuple[CaptureRow, ...]
    protected: frozenset[str]
    metadata_bytes: int

    def __post_init__(self) -> None:
        if type(self.metadata_bytes) is not int or self.metadata_bytes < 0:
            raise ValueError("invalid retained metadata estimate")
        Ledger(tuple(row.prompt for row in self.rows), self.protected)
        if not self.rows or any(row.prompt.identity != row.capture.identity for row in self.rows):
            raise ValueError("dataset identity mismatch")
        if len(self.target) != 64 or any(char not in "0123456789abcdef" for char in self.target):
            raise ValueError("dataset target identity invalid")
        for row in self.rows:
            if row.anchors != tuple(
                sorted(set(row.anchors))
            ) or row.unique_positions != position_count(row.anchors, row.label_lengths):
                raise ValueError("dataset anchor accounting mismatch")


def position_count(anchors: tuple[int, ...], lengths: tuple[int, ...] = ()) -> int:
    if lengths and (
        len(lengths) != len(anchors) or any(type(n) is not int or not 1 <= n <= 15 for n in lengths)
    ):
        raise ValueError("invalid supervision lengths")
    end = total = 0
    for index, anchor in enumerate(anchors):
        if type(anchor) is not int or anchor < 0:
            raise ValueError("invalid supervision anchor")
        width = lengths[index] if lengths else 15
        total += max(0, anchor + width - max(end, anchor))
        end = max(end, anchor + width)
    return total


@dataclass(frozen=True)
class Ledger:
    prompts: tuple[Prompt, ...]
    protected: frozenset[str]

    def __post_init__(self) -> None:
        identifiers = [p.identity for p in self.prompts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate prompt")
        families = components(
            [
                {"id": p.identity, "source": p.source, "group": p.group, "topic": p.topic}
                for p in self.prompts
            ]
        )
        by_id = {p.identity: p for p in self.prompts}
        for family in families:
            if len({by_id[identity].role for identity in family}) != 1:
                raise ValueError("related prompts cross roles")
        roles: dict[str, Role] = {}
        for prompt in self.prompts:
            names = {prompt.identity, *prompt.aliases}
            if names & self.protected and prompt.role != "heldout":
                raise ValueError("protected identity outside heldout")
            for name in names:
                if name in roles and roles[name] != prompt.role:
                    raise ValueError("alias crosses roles")
                roles[name] = prompt.role

    def admit_training(self, expected: int) -> tuple[Prompt, ...]:
        selected = tuple(p for p in self.prompts if p.role == "train")
        if len(selected) != expected or not selected:
            raise ValueError("training count mismatch")
        if any(p.ancestry != "verified" for p in selected):
            observed = {
                status: sum(prompt.ancestry == status for prompt in selected)
                for status in {prompt.ancestry for prompt in selected}
            }
            raise ValueError(
                "unverified training provenance: "
                + str(observed)
                + "; metadata integrity does not waive training-origin policy"
            )
        identities: dict[str, str] = {}
        for prompt in selected:
            names = {prompt.identity, *prompt.aliases}
            conflict = names.intersection(identities)
            if conflict:
                previous = identities[min(conflict)]
                raise ValueError(
                    "duplicate training identity or alias: " + previous + " and " + prompt.identity
                )
            identities.update({name: prompt.identity for name in names})
        return selected


@dataclass(frozen=True)
class Window:
    trajectory: str
    anchor: int
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trajectory, str)
            or not self.trajectory.strip()
            or type(self.anchor) is not int
            or self.anchor < 0
            or not isinstance(self.labels, tuple)
            or len(self.labels) != 15
        ):
            raise ValueError("invalid training window")
        if any(type(v) is not int or v < 0 for v in self.labels):
            raise ValueError("invalid label")

    def positions(self) -> tuple[tuple[str, int], ...]:
        return tuple((self.trajectory, self.anchor + j) for j in range(15))


def windows(
    trajectory: str,
    tokens: tuple[int, ...],
    prompt_tokens: int,
    positions: tuple[int, ...],
    labels: tuple[int, ...],
    eog: int | None = None,
) -> tuple[Window, ...]:
    if not trajectory or not 0 <= prompt_tokens <= len(tokens):
        raise ValueError("invalid capture identity or prompt length")
    if len(positions) != len(labels) or positions != tuple(sorted(set(positions))):
        raise ValueError("capture positions are not unique and ordered")
    if any(type(p) is not int or not 0 <= p < len(tokens) for p in positions):
        raise ValueError("capture position outside next-token range")
    if any(type(v) is not int or v < 0 for v in (*tokens, *labels)):
        raise ValueError("invalid token identity")
    if eog is not None and eog in labels and labels.index(eog) != len(labels) - 1:
        raise ValueError("post-EOG labels")
    stop = (
        next(
            (index for index in range(prompt_tokens, len(tokens)) if tokens[index] == eog),
            len(tokens),
        )
        if eog is not None
        else len(tokens)
    )
    mapping = dict(zip(positions, labels, strict=True))
    return tuple(
        Window(trajectory, anchor, tuple(mapping[anchor + j] for j in range(15)))
        for anchor in positions
        if max(0, prompt_tokens - 1) <= anchor < len(tokens)
        and anchor + 14 < stop
        and all(anchor + j in mapping for j in range(15))
    )


@dataclass(frozen=True)
class TailWindow:
    trajectory: str
    anchor: int
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.trajectory, str)
            or not self.trajectory.strip()
            or type(self.anchor) is not int
            or self.anchor < 0
            or not isinstance(self.labels, tuple)
            or not 1 <= len(self.labels) < 15
            or any(type(value) is not int or value < 0 for value in self.labels)
        ):
            raise ValueError("invalid explicit partial supervision window")

    def positions(self) -> tuple[tuple[str, int], ...]:
        return tuple((self.trajectory, self.anchor + index) for index in range(len(self.labels)))


Supervision = Window | TailWindow


def supervision(
    trajectory: str,
    tokens: tuple[int, ...],
    prompt_tokens: int,
    positions: tuple[int, ...],
    labels: tuple[int, ...],
    eog: int | None,
    policy: str,
) -> tuple[Supervision, ...]:
    complete = windows(trajectory, tokens, prompt_tokens, positions, labels, eog)
    if policy == "full15":
        return complete
    if policy != "up-to-15":
        raise ValueError("explicit supported supervision policy required")
    existing = {window.anchor: window for window in complete}
    mapping = dict(zip(positions, labels, strict=True))
    stop = (
        next(
            (index for index in range(prompt_tokens, len(tokens)) if tokens[index] == eog),
            len(tokens),
        )
        if eog is not None
        else len(tokens)
    )
    selected: list[Supervision] = []
    for anchor in positions:
        if anchor < max(0, prompt_tokens - 1) or anchor >= stop:
            continue
        if anchor in existing:
            selected.append(existing[anchor])
            continue
        available: list[int] = []
        for offset in range(15):
            position = anchor + offset
            if position >= stop or position not in mapping:
                break
            available.append(mapping[position])
        if available:
            selected.append(TailWindow(trajectory, anchor, tuple(available)))
    return tuple(selected)


def supervision_positions(selected: tuple[Supervision, ...]) -> int:
    count, end = 0, -1
    for window in sorted(selected, key=window_anchor):
        last = window.anchor + len(window.labels)
        count += max(0, last - max(window.anchor, end))
        end = max(end, last)
    return count


def window_anchor(window: Supervision) -> int:
    return window.anchor
