import bisect
import math
import random
from dataclasses import dataclass

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, Ledger, Prompt, Role, Window, windows
from w1a1_eagle.numerical.captures import TAPS, Capture, CaptureCache, CaptureSpec
from w1a1_eagle.records import artifact, read


@dataclass(frozen=True)
class Row:
    prompt: Prompt
    capture: CaptureSpec
    windows: tuple[Window, ...]


class Corpus:
    def __init__(self, manifest: Artifact, seed: int, byte_budget: int, hidden: int = 2560) -> None:
        if type(byte_budget) is not int or byte_budget < 1 or type(seed) is not int or seed < 0:
            raise ValueError("positive memory budget and nonnegative seed required")
        manifest.verify()
        raw = read(manifest.path, min(byte_budget // 8, 64 * 1024 * 1024))
        if set(raw) != {"schema", "target_sha256", "protected", "expected_train", "records"}:
            raise ValueError("unsupported corpus fields")
        if type(raw["schema"]) is not int or raw["schema"] != 1:
            raise ValueError("supported corpus schema is 1; preserve originals on migration")
        self.target = string(raw["target_sha256"])
        if len(self.target) != 64 or any(c not in "0123456789abcdef" for c in self.target):
            raise ValueError("target identity required")
        self.rows: list[Row] = []
        prompts: list[Prompt] = []
        inputs: list[tuple[Prompt, Record]] = []
        metadata_bytes = manifest.size * 16
        for value in sequence(raw["records"]):
            item = record(value)
            if set(item) != {
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
                string(item["topic"]) if item["topic"] is not None else None,
                tuple(string(v) for v in sequence(item["aliases"])),
                roles[role_text],
                string(item["ancestry"]),
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
            metadata_bytes += spec.metadata.size * 32
            if metadata_bytes >= byte_budget // 2:
                raise ValueError("corpus metadata exceeds memory budget")
            spec.metadata.verify()
            metadata = read(spec.metadata.path, min(byte_budget // 32, 16 * 1024 * 1024))
            if (
                metadata.get("taps") != list(TAPS)
                or metadata.get("feature_semantics") != "zero_based_pre_block_input_all_tokens"
            ):
                raise ValueError("capture semantics mismatch")
            eog = {integer(v) for v in sequence(metadata["eog_ids"])}
            labels = tuple(integer(v) for v in sequence(metadata["greedy_ids"]))
            if any(v in eog for v in labels[:-1]):
                raise ValueError("post-EOG training labels")
            selected = windows(
                prompt.identity,
                tuple(integer(v) for v in sequence(metadata["input_tokens"])),
                integer(metadata["prompt_tokens"]),
                tuple(integer(v) for v in sequence(metadata["logit_positions"])),
                labels,
                labels[-1] if labels and labels[-1] in eog else None,
            )
            if not selected:
                raise ValueError("admitted training capture has no windows")
            metadata_bytes += len(selected) * (15 * 32 + 64)
            if metadata_bytes >= byte_budget // 2:
                raise ValueError("corpus windows exceed memory budget")
            self.rows.append(Row(prompt, spec, selected))
        if {p.identity for p in admitted} != {r.prompt.identity for r in self.rows}:
            raise ValueError("ledger and reader disagree")
        self.rows.sort(key=lambda row: row.prompt.identity)
        self.ends: list[int] = []
        for row in self.rows:
            self.ends.append((self.ends[-1] if self.ends else 0) + len(row.windows))
        self.count = self.ends[-1]
        self.unique_positions = sum(
            len({p for window in row.windows for p in window.positions()}) for row in self.rows
        )
        self.cache = CaptureCache(byte_budget - metadata_bytes, hidden)
        self.seed = seed
        self.epoch = -1
        self.order: list[int] = []

    def locate(self, cursor: int) -> tuple[Row, Window]:
        if type(cursor) is not int or cursor < 0:
            raise ValueError("nonnegative sampler cursor required")
        epoch, offset = divmod(cursor, self.count)
        if self.epoch != epoch:
            self.order = list(range(self.count))
            random.Random(self.seed + epoch).shuffle(self.order)
            self.epoch = epoch
        index = self.order[offset]
        row_index = bisect.bisect_right(self.ends, index)
        row = self.rows[row_index]
        return row, row.windows[index - (self.ends[row_index - 1] if row_index else 0)]

    def sample(self, cursor: int) -> tuple[Capture, Window]:
        row, expected = self.locate(cursor)
        capture = self.cache.get(row.capture)
        if expected not in capture.windows:
            raise ValueError("admitted and materialized window disagreement")
        return capture, expected

    def exposure(self, updates: int) -> dict[str, int | float]:
        if type(updates) is not int or updates < 0:
            raise ValueError("invalid planned updates")
        return {
            "sequences": len(self.rows),
            "windows": self.count,
            "distinct_positions": self.unique_positions,
            "label_exposures": updates * 15,
            "mean_visits": updates * 15 / self.unique_positions,
            "window_passes": updates / self.count,
            "maximum_position_visits_bound": math.ceil(updates / self.count) * 15,
        }
