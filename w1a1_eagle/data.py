import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from w1a1_eagle.contracts import components

Role = Literal["train", "dev", "calibration", "heldout"]


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    size: int

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

    def __post_init__(self) -> None:
        if not all(v.strip() for v in (self.identity, self.source, self.group, self.ancestry)):
            raise ValueError("missing prompt identity")
        if self.role not in {"train", "dev", "calibration", "heldout"}:
            raise ValueError("unsupported role")
        if any(not a.strip() for a in self.aliases):
            raise ValueError("empty alias")


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
            raise ValueError("unverified training provenance")
        identities: set[str] = set()
        for prompt in selected:
            names = {prompt.identity, *prompt.aliases}
            if identities & names:
                raise ValueError("duplicate training identity or alias")
            identities.update(names)
        return selected


@dataclass(frozen=True)
class Window:
    trajectory: str
    anchor: int
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.trajectory or self.anchor < 0 or len(self.labels) != 15:
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
    stop = tokens.index(eog) if eog is not None and eog in tokens else len(tokens)
    mapping = dict(zip(positions, labels, strict=True))
    return tuple(
        Window(trajectory, anchor, tuple(mapping[anchor + j] for j in range(15)))
        for anchor in positions
        if prompt_tokens <= anchor < len(tokens)
        and anchor + 14 < stop
        and all(anchor + j in mapping for j in range(15))
    )
