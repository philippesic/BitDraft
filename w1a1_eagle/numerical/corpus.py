import bisect
import math
import random

from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import (
    Artifact,
    CaptureRow,
    Dataset,
    Ledger,
    Prompt,
    Role,
    Supervision,
    optional_topic,
    prior_use,
    supervision,
    supervision_positions,
)
from w1a1_eagle.numerical.captures import TAPS, Capture, CaptureCache, CaptureSpec
from w1a1_eagle.records import artifact, read
from w1a1_eagle.sampling import TaskSampler


class Corpus:
    def __init__(
        self,
        manifest: Artifact | Dataset,
        seed: int,
        byte_budget: int,
        hidden: int = 2560,
        sampling: Artifact | None = None,
    ) -> None:
        if type(byte_budget) is not int or byte_budget < 1 or type(seed) is not int or seed < 0:
            raise ValueError("positive memory budget and nonnegative seed required")
        self.sampling = sampling
        self.task_sampler: TaskSampler | None = None
        if isinstance(manifest, Dataset):
            Ledger(tuple(row.prompt for row in manifest.rows), manifest.protected).admit_training(
                len(manifest.rows)
            )
            self.target = manifest.target
            self.rows = manifest.rows
            self._index(seed, byte_budget, hidden, manifest.metadata_bytes)
            return
        manifest.verify()
        raw = read(manifest.path, min(byte_budget // 8, 64 * 1024 * 1024))
        if set(raw) != {"schema", "target_sha256", "protected", "expected_train", "records"}:
            raise ValueError("unsupported corpus fields")
        if type(raw["schema"]) is not int or raw["schema"] != 1:
            raise ValueError("supported corpus schema is 1; preserve originals on migration")
        self.target = string(raw["target_sha256"])
        if len(self.target) != 64 or any(c not in "0123456789abcdef" for c in self.target):
            raise ValueError("target identity required")
        rows: list[CaptureRow] = []
        prompts: list[Prompt] = []
        inputs: list[tuple[Prompt, Record]] = []
        metadata_bytes = manifest.size * 16
        retained_bytes = 0
        for value in sequence(raw["records"]):
            item = record(value)
            if set(item) - {"prior_use"} != {
                "id",
                "source",
                "group",
                "topic",
                "aliases",
                "role",
                "ancestry",
                "metadata",
                "files",
            }:
                raise ValueError("unsupported corpus row")
            role_text = string(item["role"])
            roles: dict[str, Role] = {
                "train": "train",
                "dev": "dev",
                "calibration": "calibration",
                "heldout": "heldout",
            }
            if role_text not in roles:
                raise ValueError("unknown data role")
            prompt = Prompt(
                string(item["id"]),
                string(item["source"]),
                string(item["group"]),
                optional_topic(item["topic"]),
                tuple(string(v) for v in sequence(item["aliases"])),
                roles[role_text],
                string(item["ancestry"]),
                prior_use(item.get("prior_use", "unknown")),
            )
            prompts.append(prompt)
            inputs.append((prompt, item))
        ledger = Ledger(
            tuple(prompts),
            frozenset({"heldout24", *(string(v) for v in sequence(raw["protected"]))}),
        )
        admitted = ledger.admit_training(integer(raw["expected_train"]))
        for prompt, item in inputs:
            if prompt.role != "train":
                continue
            root = manifest.path.parent
            spec = CaptureSpec(
                prompt.identity,
                artifact(item["metadata"], root),
                tuple(artifact(v, root) for v in sequence(item["files"])),
            )
            metadata_bytes = max(metadata_bytes, spec.metadata.size * 32)
            if metadata_bytes >= byte_budget // 2:
                raise ValueError("corpus metadata exceeds memory budget")
            spec.metadata.verify()
            metadata = read(spec.metadata.path, min(byte_budget // 32, 16 * 1024 * 1024))
            if (
                metadata.get("taps") != list(TAPS)
                or metadata.get("feature_semantics") != "zero_based_pre_block_input_all_tokens"
            ):
                raise ValueError("capture semantics mismatch")
            if metadata.get("target_sha256", self.target) != self.target:
                raise ValueError("individual capture and corpus target identities differ")
            eog = {integer(v) for v in sequence(metadata["eog_ids"])}
            labels = tuple(integer(v) for v in sequence(metadata["greedy_ids"]))
            if any(v in eog for v in labels[:-1]):
                raise ValueError("post-EOG training labels")
            selected = supervision(
                prompt.identity,
                tuple(integer(v) for v in sequence(metadata["input_tokens"])),
                integer(metadata["prompt_tokens"]),
                tuple(integer(v) for v in sequence(metadata["logit_positions"])),
                labels,
                labels[-1] if labels and labels[-1] in eog else None,
                string(metadata.get("window_policy", "full15")),
            )
            anchors = tuple(window.anchor for window in selected)
            retained_bytes += (
                len(anchors) * 40
                + 1024
                + sum(len(str(ref.path).encode()) * 4 + 512 for ref in spec.files)
            )
            metadata_bytes = max(metadata_bytes, retained_bytes)
            if metadata_bytes >= byte_budget // 2:
                raise ValueError("corpus windows exceed memory budget")
            rows.append(
                CaptureRow(
                    prompt,
                    spec,
                    anchors,
                    supervision_positions(selected),
                    tuple(len(window.labels) for window in selected),
                )
            )
        if {p.identity for p in admitted} != {r.prompt.identity for r in rows}:
            raise ValueError("ledger and reader disagree")
        self.rows = tuple(rows)
        self._index(seed, byte_budget, hidden, retained_bytes)

    def _index(self, seed: int, byte_budget: int, hidden: int, metadata_bytes: int) -> None:
        ends: list[int] = []
        for row in self.rows:
            ends.append((ends[-1] if ends else 0) + len(row.anchors))
        if not ends or not ends[-1]:
            raise ValueError("training corpus has no supervised windows")
        self.ends = tuple(ends)
        self.count = self.ends[-1]
        self.unique_positions = sum(row.unique_positions for row in self.rows)
        metadata_bytes += self.count * 40
        if metadata_bytes >= byte_budget // 2:
            raise ValueError("sampler metadata exceeds memory budget")
        self.cache = CaptureCache(byte_budget - metadata_bytes, hidden)
        self.seed = seed
        self.order: list[int] = []
        if self.sampling is not None:
            self.sampling.verify()
            configuration = read(self.sampling.path)
            if (
                set(configuration) != {"schema", "policy", "burst", "weights", "strata"}
                or configuration["schema"] != 1
                or configuration["policy"] != "task-position-v1"
            ):
                raise ValueError("unsupported sampling specification")
            strata = record(configuration["strata"])
            if set(strata) != {row.prompt.identity for row in self.rows}:
                raise ValueError("sampling strata must cover the exact admitted corpus")
            weights = {
                name: number(value) for name, value in record(configuration["weights"]).items()
            }
            self.task_sampler = TaskSampler(
                tuple(len(row.anchors) for row in self.rows),
                tuple(string(strata[row.prompt.identity]) for row in self.rows),
                weights,
                seed,
                integer(configuration["burst"]),
            )

    def locate(self, cursor: int) -> tuple[CaptureRow, int]:
        if type(cursor) is not int or cursor < 0:
            raise ValueError("nonnegative sampler cursor required")
        if self.task_sampler is not None:
            row_index, position = self.task_sampler.locate(cursor)
            row = self.rows[row_index]
            return row, row.anchors[position]
        offset = cursor % self.count
        if not self.order:
            self.order = list(range(self.count))
            random.Random(self.seed).shuffle(self.order)
        index = self.order[offset]
        row_index = bisect.bisect_right(self.ends, index)
        row = self.rows[row_index]
        return row, row.anchors[index - (self.ends[row_index - 1] if row_index else 0)]

    def sample(self, cursor: int) -> tuple[Capture, Supervision]:
        row, anchor = self.locate(cursor)
        capture = self.cache.get(row.capture)
        index = bisect.bisect_left(row.anchors, anchor)
        if len(capture.windows) != len(row.anchors) or capture.windows[index].anchor != anchor:
            raise ValueError("admitted and materialized window disagreement")
        return capture, capture.windows[index]

    def exposure(self, updates: int) -> dict[str, int | float]:
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid planned updates")
        return {
            "sequences": len(self.rows),
            "eligible_sequences": sum(bool(row.anchors) for row in self.rows),
            "zero_window_sequences": sum(not row.anchors for row in self.rows),
            "historical_training_sequences": sum(
                row.prompt.prior_use == "historical-training" for row in self.rows
            ),
            "unknown_prior_use_sequences": sum(
                row.prompt.prior_use == "unknown" for row in self.rows
            ),
            "unobserved_prior_use_sequences": sum(
                row.prompt.prior_use == "unobserved" for row in self.rows
            ),
            "windows": self.count,
            "distinct_positions": self.unique_positions,
            "label_exposures_upper_bound": updates * 15,
            "mean_visits_upper_bound": updates * 15 / self.unique_positions,
            "window_passes": updates / self.count,
            "maximum_position_visits_bound": updates
            if self.task_sampler is not None
            else math.ceil(updates / self.count) * 15,
            "sampling_with_replacement": int(self.task_sampler is not None),
        }
