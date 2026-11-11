from dataclasses import dataclass
from pathlib import Path

from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import artifact, read


@dataclass(frozen=True)
class Arm:
    name: str
    drafter: Artifact | None


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seeds: tuple[int, ...] = (8101,)
    qualification: Artifact | None = None


@dataclass(frozen=True)
class Comparison:
    source: Artifact
    binary: Artifact
    libraries: tuple[Artifact, ...]
    target: Artifact
    arms: tuple[Arm, ...]
    workload: Artifact
    tasks: tuple[Record, ...]
    repeats: int
    warmups: int
    context: int
    depth: int
    role: str
    tier: str
    protected: frozenset[str]
    diagnostic: bool = False
    sampling: Sampling = Sampling()

    @classmethod
    def load(cls, path: Path) -> "Comparison":
        raw = read(path)
        if (
            set(raw) - {"exclusions", "diagnostic", "sampling"}
            != {
                "schema",
                "source",
                "binary",
                "libraries",
                "target",
                "arms",
                "workload",
                "repeats",
                "warmups",
                "context",
                "depth",
                "role",
                "tier",
            }
            or raw["schema"] != "native-comparison-v1"
        ):
            raise ValueError("unsupported comparison specification")
        root = path.resolve().parent
        if type(raw.get("diagnostic", False)) is not bool:
            raise ValueError("diagnostic must be an explicit boolean")
        sampling = Sampling()
        if "sampling" in raw:
            settings = record(raw["sampling"])
            if set(settings) != {
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "seeds",
                "qualification",
            }:
                raise ValueError("explicit complete sampling settings required")
            sampling = Sampling(
                number(settings["temperature"]),
                number(settings["top_p"]),
                integer(settings["top_k"]),
                number(settings["min_p"]),
                tuple(integer(v) for v in sequence(settings["seeds"])),
                artifact(settings["qualification"], root)
                if settings["qualification"] is not None
                else None,
            )
            if (
                not 0 < sampling.temperature <= 2
                or not 0 < sampling.top_p <= 1
                or not 0 <= sampling.min_p <= 1
                or not 0 <= sampling.top_k <= 1000
                or not 1 <= len(sampling.seeds) <= 5
                or len(set(sampling.seeds)) != len(sampling.seeds)
                or any(not 0 <= seed < 2**32 - 1 for seed in sampling.seeds)
            ):
                raise ValueError("sampling parameters outside supported bounds")
        protected = {"heldout24"}
        if "exclusions" in raw:
            exclusion = artifact(raw["exclusions"], root)
            exclusion.verify()
            protected.update(
                string(value) for value in sequence(read(exclusion.path)["identities"])
            )
        arms = []
        for value in sequence(raw["arms"]):
            row = record(value)
            if set(row) != {"name", "drafter"}:
                raise ValueError("unsupported comparison arm")
            name = string(row["name"])
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("unsafe arm identifier")
            arms.append(
                Arm(name, artifact(row["drafter"], root) if row["drafter"] is not None else None)
            )
        if len({a.name for a in arms}) != len(arms) or len(arms) < 2:
            raise ValueError("unique target and speculative arms required")
        if sum(a.drafter is None for a in arms) != 1 or arms[0] != Arm("target", None):
            raise ValueError("first arm must be target-only, named target")
        workload = artifact(raw["workload"], root)
        workload.verify()
        panel = read(workload.path)
        if set(panel) != {"schema", "tasks"} or panel["schema"] != 1:
            raise ValueError("unsupported workload schema")
        tasks = tuple(record(v) for v in sequence(panel["tasks"]))
        role, tier = string(raw["role"]), string(raw["tier"])
        if role not in {"dev", "confirmation"} or tier not in {
            "pilot",
            "short",
            "full",
            "harness",
            "live",
            "sampled",
        }:
            raise ValueError("explicit workload role/tier required")
        if sampling.temperature > 0 and tier not in {"pilot", "sampled"}:
            raise ValueError("sampled requests require an explicit sampled tier or pilot")
        if tier == "sampled" and (
            sampling.temperature <= 0 or len(sampling.seeds) != 3 or sampling.qualification is None
        ):
            raise ValueError(
                "sampled robustness requires three fixed seeds and native verifier qualification"
            )
        expected = {
            "short": 120,
            "full": 600,
            "harness": 90,
            "live": 12 if role == "dev" else 24,
            "sampled": 120,
        }
        if not tasks or len(tasks) > 600 or (tier in expected and len(tasks) != expected[tier]):
            raise ValueError("workload count differs from declared tier")
        seen: set[str] = set()
        for task in tasks:
            if set(task) - {"tools", "harness", "ancestry", "fixture", "followup"} != {
                "id",
                "group",
                "role",
                "category",
                "messages",
                "output_cap",
            }:
                raise ValueError("unsupported workload task fields")
            identity = string(task["id"])
            if identity in seen or not identity.replace("-", "").replace("_", "").isalnum():
                raise ValueError("duplicate or unsafe task identity")
            seen.add(identity)
            if task["role"] != role or integer(task["output_cap"]) < 1:
                raise ValueError("workload role or output cap mismatch")
            ancestry = record(task.get("ancestry", {}))
            names = {
                identity,
                string(task["group"]),
                string(ancestry.get("source_key", identity)),
                *(string(v) for v in sequence(ancestry.get("aliases", []))),
            }
            if names & protected:
                raise ValueError("protected heldout identity in evaluation workload")
            string(task["group"])
            string(task["category"])
            messages = sequence(task["messages"])
            if not messages:
                raise ValueError("empty conversation")
            for value in messages:
                message = record(value)
                string(message["role"])
                if not isinstance(message.get("content"), str):
                    raise ValueError("message content must be text")
        repeats, warmups = integer(raw["repeats"]), integer(raw["warmups"])
        context, depth = integer(raw["context"]), integer(raw["depth"])
        if (
            not 1 <= repeats <= 20
            or not 0 <= warmups <= 20
            or not 16 <= context <= 65536
            or not 1 <= depth <= 15
        ):
            raise ValueError("comparison execution bounds invalid")
        return cls(
            artifact(raw["source"], root),
            artifact(raw["binary"], root),
            tuple(artifact(v, root) for v in sequence(raw["libraries"])),
            artifact(raw["target"], root),
            tuple(arms),
            workload,
            tasks,
            repeats,
            warmups,
            context,
            depth,
            role,
            tier,
            frozenset(protected),
            raw.get("diagnostic", False) is True,
            sampling,
        )

    def verify(self) -> None:
        for ref in (
            self.source,
            self.binary,
            self.target,
            self.workload,
            *self.libraries,
            *(arm.drafter for arm in self.arms if arm.drafter is not None),
        ):
            ref.verify()
        if not self.libraries:
            raise ValueError("explicit native library identities required")
        if read(self.source.path).get("verification_trace", False) != self.diagnostic:
            raise ValueError(
                "instrumented runtime must be explicitly diagnostic; primary timings require the clean runtime"
            )
        if self.sampling.qualification is not None:
            self.sampling.qualification.verify()
            qualification = read(self.sampling.qualification.path)
            if (
                qualification.get("status") != "passed"
                or qualification.get("eos_stopped") is not True
                or qualification.get("target_sha256") != self.target.sha256
                or qualification.get("runtime_source_sha256") != self.source.sha256
                or {string(v) for v in sequence(qualification.get("functions_tested", []))}
                != {"sample-and-match", "rejection"}
            ):
                raise ValueError("sampled verifier qualification does not bind this target/runtime")
