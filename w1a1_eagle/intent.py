from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from w1a1_eagle.contracts import Record, integer, number, record, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import artifact, identity, read
from w1a1_eagle.training import Budget


@dataclass(frozen=True)
class Intent:
    raw: Record
    sha256: str
    source: Artifact
    vocabulary: Artifact
    corpus: Artifact
    parent: Artifact | None
    mode: Literal["fresh", "resume", "branch"]
    budget: Budget
    memory_bytes: int
    state_bytes: int
    disk_reserve_bytes: int
    backing_available_bytes: int
    seed: int
    checkpoint_updates: int
    checkpoint_seconds: float

    @classmethod
    def load(cls, path: Path) -> "Intent":
        raw = read(path)
        required = {
            "schema",
            "objective",
            "hypothesis",
            "decision",
            "risk",
            "authority_scope",
            "source",
            "vocabulary",
            "corpus",
            "parent",
            "mode",
            "budget",
            "memory_bytes",
            "state_bytes",
            "disk_reserve_bytes",
            "backing_available_bytes",
            "seed",
            "device",
            "science_policy",
            "code_sha256",
            "geometry",
            "checkpoint_updates",
            "checkpoint_seconds",
        }
        if set(raw) != required or type(raw["schema"]) is not int or raw["schema"] != 2:
            raise ValueError(
                "supported intent schema is 2 with exact fields; preserve older intents"
            )
        if integer(raw["checkpoint_updates"]) < 1 or number(raw["checkpoint_seconds"]) <= 0:
            raise ValueError("positive explicit checkpoint cadence required")
        for key in (
            "objective",
            "hypothesis",
            "decision",
            "risk",
            "authority_scope",
            "code_sha256",
        ):
            if not string(raw[key]).strip():
                raise ValueError("intent requires " + key)
        if raw["device"] != "cpu":
            raise ValueError("only CPU execution is qualified; hardware admission is separate")
        geometry = record(raw["geometry"])
        if set(geometry) != {
            "hidden",
            "intermediate",
            "heads",
            "kv_heads",
            "head_dim",
            "layers",
        } or any(integer(value) < 1 for value in geometry.values()):
            raise ValueError("explicit positive geometry required")
        mode = raw["mode"]
        if mode not in {"fresh", "resume", "branch"}:
            raise ValueError("unsupported training mode")
        if (mode == "fresh") != (raw["parent"] is None):
            raise ValueError("fresh requires no parent; resume/branch require an explicit parent")
        budget = record(raw["budget"])
        if set(budget) != {"updates", "wall_seconds", "attempts", "retries"}:
            raise ValueError("unsupported budget")
        if integer(budget["retries"]) != 0:
            raise ValueError(
                "automatic retries are not supported; recover explicitly within budget"
            )
        sizes = tuple(
            integer(raw[key])
            for key in (
                "memory_bytes",
                "state_bytes",
                "disk_reserve_bytes",
                "backing_available_bytes",
            )
        )
        if any(v < 1 for v in sizes) or integer(raw["seed"]) < 0:
            raise ValueError("positive resource bounds and nonnegative seed required")
        policy = record(raw["science_policy"])
        if set(policy) != {"id", "min_unique_positions", "max_mean_visits"}:
            raise ValueError("unsupported supplied data policy")
        if (
            not string(policy["id"]).strip()
            or integer(policy["min_unique_positions"]) < 1
            or number(policy["max_mean_visits"]) <= 0
        ):
            raise ValueError("explicit positive scientific admission policy required")
        root = path.resolve().parent
        return cls(
            raw,
            identity(raw),
            artifact(raw["source"], root),
            artifact(raw["vocabulary"], root),
            artifact(raw["corpus"], root),
            artifact(raw["parent"], root) if raw["parent"] is not None else None,
            "fresh" if mode == "fresh" else "resume" if mode == "resume" else "branch",
            Budget(
                integer(budget["updates"]),
                number(budget["wall_seconds"]),
                integer(budget["attempts"]),
            ),
            sizes[0],
            sizes[1],
            sizes[2],
            sizes[3],
            integer(raw["seed"]),
            integer(raw["checkpoint_updates"]),
            number(raw["checkpoint_seconds"]),
        )

    @property
    def recipe(self) -> str:
        return identity(
            {
                "source": {"sha256": self.source.sha256, "bytes": self.source.size},
                "vocabulary": {"sha256": self.vocabulary.sha256, "bytes": self.vocabulary.size},
                "corpus": {"sha256": self.corpus.sha256, "bytes": self.corpus.size},
                "seed": self.seed,
                "device": self.raw["device"],
                "geometry": self.raw["geometry"],
                "code": self.raw["code_sha256"],
            }
        )

    @property
    def peak_new_bytes(self) -> int:
        import math

        intermediate = min(
            self.budget.updates - 1,
            math.ceil(self.budget.updates / self.checkpoint_updates)
            + math.ceil(self.budget.wall_seconds / self.checkpoint_seconds),
        )
        return (intermediate + 3) * self.state_bytes + 8 * 1024 * 1024
