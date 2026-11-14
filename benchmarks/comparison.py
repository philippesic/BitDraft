from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from benchmarks.coverage import FULL, SHORT, length_band
from benchmarks.runtime_policy import LIVE_SECONDS, LIVE_TURNS, arguments, payload
from benchmarks.sandbox import TOOLS
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import artifact, read, reference


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
    references: tuple[tuple[str, Artifact], ...] = ()

    @classmethod
    def load(cls, path: Path) -> "Comparison":
        raw = read(path)
        if (
            set(raw) - {"exclusions", "diagnostic", "sampling", "references"}
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
        references = tuple(
            (name, artifact(value, root))
            for name, value in record(raw.get("references", {})).items()
        )
        arm_names = [a.name for a in arms] + [name for name, _ in references]
        if len(set(arm_names)) != len(arm_names) or not arms or len(arm_names) < 2:
            raise ValueError("unique target and speculative arms required")
        if references and (
            "target" not in dict(references) or any(a.drafter is None for a in arms)
        ):
            raise ValueError("cached execution requires a saved target and speculative-only arms")
        if not references and (
            sum(a.drafter is None for a in arms) != 1 or arms[0] != Arm("target", None)
        ):
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
            "screen",
            "comparison",
            "prefix",
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
            "screen": 60,
            "comparison": 120,
            "prefix": 18,
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
        require_cohort(tier, tasks)
        context, depth = integer(raw["context"]), integer(raw["depth"])
        if (
            not 1 <= repeats <= 20
            or not 0 <= warmups <= 20
            or not 16 <= context <= 65536
            or not 1 <= depth <= 15
        ):
            raise ValueError("comparison execution bounds invalid")
        if (references or tier == "screen") and (repeats != 1 or sampling.temperature):
            raise ValueError("lean/cached panels require one greedy pass")
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
            references,
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
        built = read(self.source.path)
        if built.get("verification_trace", False) != self.diagnostic:
            raise ValueError(
                "instrumented runtime must be explicitly diagnostic; primary timings require the clean runtime"
            )
        produced = [
            artifact(value, self.source.path.parent) for value in sequence(built["artifacts"])
        ]
        if not any(
            ref.sha256 == self.binary.sha256
            and ref.size == self.binary.size
            and ref.path.name == "llama-server"
            for ref in produced
        ) or {ref.sha256 for ref in produced if ".so" in ref.path.name} != {
            ref.sha256 for ref in self.libraries
        }:
            raise ValueError("comparison server/libraries do not match the authenticated build")

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
        if self.references:
            self.cached()

    def reference_protocol(self) -> Record:
        return {
            "policy": "native-greedy-request-v3" if self.tier != "live" else "native-live-v3",
            "target_arguments": arguments(
                Path("server"), Path("target"), None, self.context, self.depth, 0
            ),
            "draft_arguments": arguments(
                Path("server"), Path("target"), Path("drafter"), self.context, self.depth, 0
            ),
            "request": payload(
                [],
                0,
                self.sampling.temperature,
                self.sampling.top_p,
                self.sampling.top_k,
                self.sampling.min_p,
                8101,
            ),
            "serving": list(self.serving_identity()),
            "tasks": list(self.tasks),
            "sampling": [
                self.sampling.temperature,
                self.sampling.top_p,
                self.sampling.top_k,
                self.sampling.min_p,
                *self.sampling.seeds,
            ],
            "live": {
                "turns": LIVE_TURNS,
                "seconds": LIVE_SECONDS,
                "tools": TOOLS,
                "cache_followups": True,
            }
            if self.tier == "live"
            else None,
        }

    def cached(self) -> dict[str, Record]:
        found: dict[str, Record] = {}
        for name, ref in self.references:
            ref.verify()
            value = read(ref.path, 16 * 1024 * 1024)
            if (
                value.get("schema") != "benchmark-reference-v1"
                or value.get("protocol") != self.reference_protocol()
                or value.get("status") != "complete"
            ):
                raise ValueError(
                    "reference protocol/target/workload differs or is incomplete: " + name
                )
            for key in ("result", "specification"):
                source = artifact(value[key], ref.path.parent)
                source.verify()
            result_source = artifact(value["result"], ref.path.parent)
            result = read(result_source.path, 64 * 1024 * 1024)
            source_spec = Comparison.load(artifact(value["specification"], ref.path.parent).path)
            source_protocol = source_spec.reference_protocol()
            desired_protocol = self.reference_protocol()
            source_protocol.pop("tasks")
            desired_protocol.pop("tasks")
            original_tasks = {string(task["id"]): task for task in source_spec.tasks}
            source_arm = next(
                (arm for arm in source_spec.arms if arm.name == value["source_arm"]), None
            )
            recorded_model = (
                artifact(value["drafter"], ref.path.parent)
                if value["drafter"] is not None
                else None
            )
            if (
                source_protocol != desired_protocol
                or source_arm is None
                or (source_arm.drafter.sha256 if source_arm.drafter else None)
                != (recorded_model.sha256 if recorded_model else None)
                or any(original_tasks.get(string(task["id"])) != task for task in self.tasks)
                or source_spec.diagnostic
                or source_spec.sampling.temperature
            ):
                raise ValueError("reference model/workload identity differs from its source")
            wanted = {task["id"] for task in self.tasks}
            recorded = [
                row
                for row in sequence(result["rows"])
                if record(row)["arm"] == value["source_arm"]
                and record(row)["rep"] == value["source_repeat"]
                and record(row)["task"] in wanted
            ]
            if (
                result.get("status") != "complete"
                or recorded != value["rows"]
                or (result_source.path.parent / "gpu.txt").read_text() != value["gpu"]
            ):
                raise ValueError("reference facts differ from their source execution")
            for receipt in sequence(value["raw"]):
                artifact(receipt, ref.path.parent).verify()
            rows = [record(row) for row in sequence(value["rows"])]
            if (
                len(rows) != len(self.tasks)
                or {row["task"] for row in rows} != {task["id"] for task in self.tasks}
                or any(row.get("correct") is not True for row in rows)
                or set(record(value["signatures"])) != {string(task["id"]) for task in self.tasks}
            ):
                raise ValueError(
                    "reference must contain one complete verified observation per task"
                )
            drafter = value["drafter"]
            if (name == "target") != (drafter is None):
                raise ValueError(
                    "target reference must be target-only; competitors require drafters"
                )
            if drafter is not None:
                artifact(drafter, ref.path.parent)
            for row in rows:
                counts = record(row["native"])
                if (
                    not 0 <= integer(counts["accepted"]) <= integer(counts["proposed"])
                    or number(row["wall_seconds"]) <= 0
                ):
                    raise ValueError("invalid reference counters or timing")
            found[name] = {
                **value,
                "reference": reference(ref),
                "rows": [{**row, "arm": name, "rep": 0} for row in rows],
            }
        if found and ("target" not in found or "baseline" not in found):
            raise ValueError(
                "cached evaluation requires established target and baseline references"
            )
        if found and any(
            value["signatures"] != found["target"]["signatures"] for value in found.values()
        ):
            raise ValueError("cached competitors disagree with target output signatures")
        if len({string(value["gpu"]) for value in found.values()}) > 1:
            raise ValueError("cached references come from different hardware/driver conditions")
        return found

    def serving_identity(self) -> tuple[str, ...]:
        return (
            self.source.sha256,
            self.binary.sha256,
            self.target.sha256,
            *sorted(ref.sha256 for ref in self.libraries),
            str(self.context),
            str(self.depth),
            self.role,
        )

    def block_count(self) -> int:
        return (
            min(len(self.arms), len(self.tasks))
            if self.tier in {"full", "harness", "comparison", "prefix"}
            else 1
        )


def require_cohort(tier: str, tasks: tuple[Record, ...]) -> None:
    if tier in {"short", "full", "sampled", "screen", "comparison"}:
        categories = ("code", "reasoning", "documents", "conversation", "structured", "writing")
        counts = (15, 10, 12, 10, 8, 5) if tier == "screen" else FULL if tier == "full" else SHORT
        bands = (
            (30, 18, 9, 3)
            if tier == "screen"
            else (300, 180, 90, 30)
            if tier == "full"
            else (60, 36, 18, 6)
        )
        observed = Counter(string(task["category"]) for task in tasks)
        lengths = Counter(
            length_band(integer(record(task["ancestry"])["input_tokens"])) for task in tasks
        )
        if (
            observed != dict(zip(categories, counts, strict=True))
            or lengths != dict(enumerate(bands))
            or len({string(task["group"]) for task in tasks}) != len(tasks)
        ):
            raise ValueError(
                "declared request cohort violates category, length or independent-group coverage"
            )
        if tier == "screen" and any(task.get("output_cap") != 128 for task in tasks):
            raise ValueError("training screen requires 128-token output caps")
    if tier in {"harness", "prefix"}:
        groups: dict[str, list[str]] = {}
        for task in tasks:
            groups.setdefault(string(task["group"]), []).append(string(task["harness"]))
        if len(groups) != (6 if tier == "prefix" else 30) or any(
            sorted(values) != ["coding", "minimal", "tools"] for values in groups.values()
        ):
            raise ValueError("harness requires declared groups with all three configurations")
