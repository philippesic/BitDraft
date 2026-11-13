import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.index import IndexSpec, Mount
from w1a1_eagle.records import artifact, canonical, identity, read
from w1a1_eagle.storage import relocated
from w1a1_eagle.training import Budget, Grouping


def grouping(value: object) -> Grouping:
    raw = record(value)
    if set(raw) != {"max_windows", "activation_bytes", "device_bytes"}:
        raise ValueError("grouping requires max_windows, activation_bytes and device_bytes")
    return Grouping(
        integer(raw["max_windows"]), integer(raw["activation_bytes"]), integer(raw["device_bytes"])
    )


@dataclass(frozen=True)
class Intent:
    encoded: bytes
    sha256: str
    source: Artifact
    vocabulary: Artifact
    corpus: Artifact
    parent: Artifact | None
    mode: Literal["fresh", "resume", "branch", "continue"]
    budget: Budget
    memory_bytes: int
    state_bytes: int
    disk_reserve_bytes: int
    backing_available_bytes: int
    seed: int
    checkpoint_updates: int
    checkpoint_seconds: float
    base: Path
    index_spec: IndexSpec | None = None

    @property
    def raw(self) -> Record:
        return record(cast(object, json.loads(self.encoded)))

    @classmethod
    def load(cls, path: Path) -> "Intent":
        return cls.parse(read(path, 1024 * 1024), path.resolve().parent)

    @classmethod
    def load_sealed(cls, path: Path) -> "Intent":
        raw = read(path)
        if (
            set(raw) != {"schema", "payload", "base", "sha256"}
            or type(raw["schema"]) is not int
            or raw["schema"] != 1
        ):
            raise ValueError("unsupported sealed intent receipt")
        payload = record(raw["payload"])
        base = Path(string(raw["base"]))
        if (
            not base.is_absolute()
            or identity(payload) != raw["sha256"]
            or path.name != str(raw["sha256"]) + ".json"
        ):
            raise ValueError("sealed intent identity mismatch")
        return cls.parse(payload, base)

    def seal(self, directory: Path) -> Path:
        store = Store(directory, self.disk_reserve_bytes)
        path = store.root / (self.sha256 + ".json")
        data = canonical(
            {"schema": 1, "payload": self.raw, "base": str(self.base), "sha256": self.sha256}
        )
        if path.exists():
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError("existing intent receipt disagrees with admitted input")
        else:
            store.atomic(path, data)
            path.chmod(0o444)
        return path

    @classmethod
    def parse(cls, raw: Record, root: Path) -> "Intent":
        if len(canonical(raw)) > 1024 * 1024:
            raise ValueError("intent exceeds its explicit 1 MiB bound")
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
        if (
            set(raw)
            - {
                "input_adapter",
                "target_sha256",
                "sampling",
                "retention",
                "corpus_format",
                "development",
                "model_update_seconds",
                "development_cadence_updates",
                "initialization",
                "cpu_threads",
                "grouping",
            }
            != required
            or type(raw["schema"]) is not int
            or raw["schema"] != 2
        ):
            raise ValueError(
                "supported intent schema is 2 with exact fields; preserve older intents"
            )
        if integer(raw["checkpoint_updates"]) < 1 or number(raw["checkpoint_seconds"]) <= 0:
            raise ValueError("positive explicit checkpoint cadence required")
        if not 1 <= integer(raw.get("cpu_threads", 2)) <= 16:
            raise ValueError("cpu_threads must be between1 and16")
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
        if raw["device"] not in {"cpu", "cuda"}:
            raise ValueError(
                "supported execution devices are cpu/cuda; GPU requires exclusive admission"
            )
        if raw["device"] == "cuda" or "target_sha256" in raw:
            target = string(raw.get("target_sha256"))
            if len(target) != 64 or any(c not in "0123456789abcdef" for c in target):
                raise ValueError(
                    "CUDA execution requires explicit target SHA256, distinct from head source"
                )
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
        if mode not in {"fresh", "resume", "branch", "continue"}:
            raise ValueError("unsupported training mode")
        if (mode == "fresh") != (raw["parent"] is None):
            raise ValueError("fresh requires no parent; resume/branch require an explicit parent")
        if "retention" in raw:
            retention = record(raw["retention"])
            if (
                set(retention) != {"states", "exports"}
                or not 2 <= integer(retention["states"]) <= 64
                or not 1 <= integer(retention["exports"]) <= 64
            ):
                raise ValueError("explicit bounded state/export retention required")
        if "development_cadence_updates" in raw and (
            "development" not in raw or integer(raw["development_cadence_updates"]) < 1
        ):
            raise ValueError("positive diagnostic cadence requires an explicit DEV specification")
        if "grouping" in raw:
            grouping(raw["grouping"])
        if "model_update_seconds" in raw and number(raw["model_update_seconds"]) <= 0:
            raise ValueError("positive cumulative model-update time limit required")
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
        adapter: IndexSpec | None = None
        if "corpus_format" in raw and (
            raw["corpus_format"] != "corpus-mixture-v1" or raw.get("input_adapter") is not None
        ):
            raise ValueError("explicit mixture format cannot use a direct input adapter")
        if raw.get("input_adapter") is not None:
            values = record(raw["input_adapter"])
            fields = {
                "kind",
                "logical_root",
                "metadata_root",
                "manifest_sha256",
                "target_sha256",
                "sequences",
                "windows",
                "unique_positions",
                "exclusions",
            }
            if set(values) - {"data_root"} != fields or values["kind"] != "he4fresh-index-v1":
                raise ValueError("unsupported input adapter")
            logical = Path(string(values["logical_root"]))
            if not logical.is_absolute():
                raise ValueError("index logical root must be absolute")
            mount = Mount(logical, relocated(root / string(values["metadata_root"])).resolve())
            data_mount = Mount(
                logical,
                relocated(
                    root / string(values.get("data_root", values["metadata_root"]))
                ).resolve(),
            )
            adapter = IndexSpec(
                string(values["manifest_sha256"]),
                string(values["target_sha256"]),
                integer(values["sequences"]),
                integer(values["windows"]),
                integer(values["unique_positions"]),
                integer(geometry["hidden"]),
                mount,
                mount.map(values["exclusions"]),
                data_mount,
            )
        return cls(
            canonical(raw),
            identity(raw),
            artifact(raw["source"], root),
            artifact(raw["vocabulary"], root),
            artifact(raw["corpus"], root),
            artifact(raw["parent"], root) if raw["parent"] is not None else None,
            "fresh"
            if mode == "fresh"
            else "resume"
            if mode == "resume"
            else "continue"
            if mode == "continue"
            else "branch",
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
            root,
            adapter,
        )

    @property
    def initialization(self) -> Artifact | None:
        return (
            artifact(self.raw["initialization"], self.base)
            if "initialization" in self.raw
            else None
        )

    @property
    def grouping(self) -> Grouping:
        return grouping(self.raw["grouping"]) if "grouping" in self.raw else Grouping()

    @property
    def cpu_threads(self) -> int:
        return integer(self.raw.get("cpu_threads", 2))

    @property
    def sampling(self) -> Artifact | None:
        return artifact(self.raw["sampling"], self.base) if "sampling" in self.raw else None

    @property
    def recipe(self) -> str:
        return identity(
            {
                "source": {"sha256": self.source.sha256, "bytes": self.source.size},
                "vocabulary": {"sha256": self.vocabulary.sha256, "bytes": self.vocabulary.size},
                "corpus": {"sha256": self.corpus.sha256, "bytes": self.corpus.size},
                **(
                    {"corpus_format": self.raw["corpus_format"]}
                    if "corpus_format" in self.raw
                    else {}
                ),
                **(
                    {"development": artifact(self.raw["development"], self.base).sha256}
                    if "development" in self.raw
                    else {}
                ),
                **(
                    {"development_cadence_updates": self.raw["development_cadence_updates"]}
                    if "development_cadence_updates" in self.raw
                    else {}
                ),
                "seed": self.seed,
                "device": self.raw["device"],
                "initialization": None
                if self.initialization is None
                else self.initialization.sha256,
                "cpu_threads": self.cpu_threads,
                "grouping": cast(Record, asdict(self.grouping)),
                "sampling": None
                if self.sampling is None
                else {"sha256": self.sampling.sha256, "bytes": self.sampling.size},
                "geometry": self.raw["geometry"],
                "code": self.raw["code_sha256"],
                "input_adapter": None
                if self.index_spec is None
                else {
                    "kind": "he4fresh-index-v1",
                    "manifest_sha256": self.index_spec.manifest_sha256,
                    "target_sha256": self.index_spec.target_sha256,
                    "sequences": self.index_spec.sequences,
                    "windows": self.index_spec.windows,
                    "unique_positions": self.index_spec.unique_positions,
                    "exclusions_sha256": self.index_spec.exclusions.sha256,
                },
            }
        )

    @property
    def peak_new_bytes(self) -> int:
        import math

        if "retention" in self.raw:
            retention = record(self.raw["retention"])
            return (
                integer(retention["states"]) + integer(retention["exports"]) + 1
            ) * self.state_bytes + 80 * 1024 * 1024
        intermediate = min(
            self.budget.updates - 1,
            math.ceil(self.budget.updates / self.checkpoint_updates)
            + math.ceil(self.budget.wall_seconds / self.checkpoint_seconds),
        )
        return (intermediate + 3) * self.state_bytes + 80 * 1024 * 1024
