from dataclasses import dataclass
from pathlib import Path

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.records import read


@dataclass(frozen=True)
class CaptureContract:
    target: str
    tokenizer: str
    template: str
    taps: tuple[int, ...]
    semantics: str
    hidden: int
    objective: str

    @classmethod
    def parse(cls, value: object) -> "CaptureContract":
        raw = record(value)
        if set(raw) != {
            "target",
            "tokenizer",
            "template",
            "taps",
            "semantics",
            "hidden",
            "objective",
        }:
            raise ValueError("capture contract fields differ; use schema 1")
        result = cls(
            string(raw["target"]),
            string(raw["tokenizer"]),
            string(raw["template"]),
            tuple(integer(v) for v in sequence(raw["taps"])),
            string(raw["semantics"]),
            integer(raw["hidden"]),
            string(raw["objective"]),
        )
        for digest in (result.target, result.tokenizer, result.template):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("contract requires exact SHA256 identities")
        if result.hidden < 1 or not result.taps or len(set(result.taps)) != len(result.taps):
            raise ValueError("positive width and unique feature taps required")
        if min(result.taps) < 0:
            raise ValueError("negative feature tap")
        return result

    def require(self, actual: "CaptureContract") -> None:
        pairs = (
            ("target", self.target, actual.target),
            ("tokenizer", self.tokenizer, actual.tokenizer),
            ("template", self.template, actual.template),
            ("taps", self.taps, actual.taps),
            ("semantics", self.semantics, actual.semantics),
            ("hidden", self.hidden, actual.hidden),
            ("objective", self.objective, actual.objective),
        )
        for field, expected, supplied in pairs:
            if expected != supplied:
                raise ValueError(
                    "candidate/capture mismatch: " + field + "; recapture compatible inputs"
                )


@dataclass(frozen=True)
class Task:
    identity: str
    group: str
    role: str
    category: str
    input_tokens: int
    output_cap: int

    @classmethod
    def parse(cls, value: object) -> "Task":
        raw = record(value)
        if set(raw) != {"id", "group", "role", "category", "input_tokens", "output_cap"}:
            raise ValueError("unexpected task fields")
        result = cls(
            string(raw["id"]),
            string(raw["group"]),
            string(raw["role"]),
            string(raw["category"]),
            integer(raw["input_tokens"]),
            integer(raw["output_cap"]),
        )
        if result.role not in {"train", "dev", "confirmation"}:
            raise ValueError("unknown task role")
        if result.category not in {
            "code",
            "reasoning",
            "documents",
            "conversation",
            "structured",
            "writing",
        }:
            raise ValueError("unknown category")
        if result.input_tokens < 1 or result.output_cap < 1:
            raise ValueError("positive exact input count and output cap required")
        return result


@dataclass(frozen=True)
class Specification:
    contract: CaptureContract
    tasks: tuple[Task, ...]
    features: tuple[Artifact, ...]
    limit: int
    storage_bytes: int

    @classmethod
    def load(cls, path: Path) -> "Specification":
        raw = read(path, 16 * 1024 * 1024)
        if (
            set(raw)
            != {
                "schema",
                "candidate",
                "capture",
                "tasks",
                "features",
                "storage_bytes",
                "capture_limit_bytes",
            }
            or type(raw["schema"]) is not int
            or raw["schema"] != 1
        ):
            raise ValueError("supported benchmark specification schema is 1")
        candidate = CaptureContract.parse(raw["candidate"])
        candidate.require(CaptureContract.parse(raw["capture"]))
        tasks = tuple(Task.parse(v) for v in sequence(raw["tasks"]))
        if not tasks or len({v.identity for v in tasks}) != len(tasks):
            raise ValueError("nonempty unique task identities required")
        groups: dict[str, str] = {}
        for task in tasks:
            if task.group in groups and groups[task.group] != task.role:
                raise ValueError("related task group crosses data roles: " + task.group)
            groups[task.group] = task.role
        features: list[Artifact] = []
        for value in sequence(raw["features"]):
            item = record(value)
            if set(item) != {"path", "sha256", "bytes"}:
                raise ValueError("unexpected feature artifact fields")
            locator = Path(string(item["path"]))
            if not locator.is_absolute():
                locator = path.parent / locator
            features.append(Artifact(locator, string(item["sha256"]), integer(item["bytes"])))
        storage = integer(raw["storage_bytes"])
        limit = integer(raw["capture_limit_bytes"])
        if storage not in (2, 4) or limit < 1:
            raise ValueError("capture dtype must use 2/4 bytes and limit must be positive")
        return cls(candidate, tasks, tuple(features), limit, storage)

    def report(self) -> Record:
        train = [v for v in self.tasks if v.role == "train"]
        forecast = (
            sum(v.input_tokens + v.output_cap for v in train)
            * len(self.contract.taps)
            * self.contract.hidden
            * self.storage_bytes
        )
        blockers = []
        if forecast > self.limit:
            blockers.append(
                "maximum capture payload exceeds ceiling; revise selection or continuation policy"
            )
        if not self.features:
            blockers.append("no authenticated feature artifacts; acquisition is not implemented")
        return {
            "schema": 1,
            "stage": "plan",
            "tasks": len(self.tasks),
            "train_tasks": len(train),
            "maximum_feature_bytes": forecast,
            "capture_limit_bytes": self.limit,
            "blockers": blockers,
            "runtime_qualified": False,
            "scope": "metadata compatibility and worst-case feature payload only; no execution admission",
        }
