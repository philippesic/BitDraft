import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, capture_supervision
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.numerical.captures import TAP_SEMANTICS, TAPS
from w1a1_eagle.records import artifact, canonical, read, reference

ROW_FIELDS = ("id", "source", "group", "topic", "aliases", "role", "ancestry")
DEFAULT_BANDS = (512, 2048, 8192)


@dataclass
class Group:
    records: int = 0
    identities_exact: int = 0
    labels: int = 0
    labels_equal: int = 0
    labels_valid: int = 0
    bit_identical_taps: int = 0
    taps: int = 0
    maximum_relative_l2: float = 0.0
    maximum_absolute: float = 0.0
    reference_windows: list[int] = field(default_factory=lambda: [0] * 15)
    replay_windows: list[int] = field(default_factory=lambda: [0] * 15)

    def add(self, row: Record) -> None:
        self.records += 1
        self.identities_exact += row["identities_exact"] is True
        labels = record(row["labels"])
        self.labels += integer(labels["count"])
        self.labels_equal += integer(labels["equal"])
        self.labels_valid += labels["valid"] is True
        for tap in sequence(row["features"]):
            values = record(tap)
            self.taps += 1
            self.bit_identical_taps += values["bit_identical"] is True
            self.maximum_relative_l2 = max(
                self.maximum_relative_l2, float(cast(float, values["relative_l2"]))
            )
            self.maximum_absolute = max(
                self.maximum_absolute, float(cast(float, values["max_absolute"]))
            )
        windows = record(row["windows"])
        for index, value in enumerate(sequence(windows["reference"])):
            self.reference_windows[index] += integer(value)
        for index, value in enumerate(sequence(windows["replay"])):
            self.replay_windows[index] += integer(value)

    def encode(self) -> Record:
        return {
            "records": self.records,
            "identities_exact": self.identities_exact,
            "labels": self.labels,
            "labels_equal": self.labels_equal,
            "labels_valid_records": self.labels_valid,
            "taps": self.taps,
            "bit_identical_taps": self.bit_identical_taps,
            "maximum_relative_l2": self.maximum_relative_l2,
            "maximum_absolute": self.maximum_absolute,
            "windows_by_labels": {
                "reference": list(self.reference_windows),
                "replay": list(self.replay_windows),
            },
        }


def feature_files(row: Record, root: Path, half: bool) -> list[Artifact]:
    suffix = "f16" if half else "f32"
    by_name = {artifact(v, root).path.name: artifact(v, root) for v in sequence(row["files"])}
    files = []
    for tap in TAPS:
        ref = by_name.get(f"layer-{tap}.{suffix}")
        if ref is None:
            raise ValueError("agreement requires eight-tap feature files")
        files.append(ref)
    return files


def compare_tap(
    left: Artifact,
    right: Artifact,
    left_half: bool,
    right_half: bool,
    count: int,
    hidden: int,
    rows: int,
) -> Record:
    left_width = 2 if left_half else 4
    right_width = 2 if right_half else 4
    if left.size != count * hidden * left_width or right.size != count * hidden * right_width:
        raise ValueError("feature geometry differs from the token stream")
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    identical = left_half == right_half
    error_squared = norm_squared = absolute_sum = 0.0
    maximum = 0.0
    with left.path.open("rb") as left_file, right.path.open("rb") as right_file:
        for start in range(0, count, rows):
            take = min(rows, count - start)
            left_bytes = left_file.read(take * hidden * left_width)
            right_bytes = right_file.read(take * hidden * right_width)
            left_digest.update(left_bytes)
            right_digest.update(right_bytes)
            identical = identical and left_bytes == right_bytes
            reference_values = torch.frombuffer(
                bytearray(left_bytes), dtype=torch.float16 if left_half else torch.float32
            ).double()
            replay_values = torch.frombuffer(
                bytearray(right_bytes), dtype=torch.float16 if right_half else torch.float32
            ).double()
            delta = replay_values - reference_values
            error_squared += float(delta.square().sum())
            norm_squared += float(reference_values.square().sum())
            absolute_sum += float(delta.abs().sum())
            maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.0)
    if left_digest.hexdigest() != left.sha256 or right_digest.hexdigest() != right.sha256:
        raise ValueError("feature content mismatch during agreement")
    return {
        "bit_identical": identical,
        "max_absolute": maximum,
        "mean_absolute": absolute_sum / (count * hidden),
        "relative_l2": math.sqrt(error_squared / norm_squared) if norm_squared else 0.0,
    }


def window_counts(name: str, metadata: Record) -> tuple[list[int], bool]:
    counts = [0] * 15
    try:
        windows = capture_supervision(name, metadata)
    except ValueError:
        return counts, False
    for window in windows:
        counts[len(window.labels) - 1] += 1
    return counts, True


def band(length: int, bands: tuple[int, ...]) -> str:
    lower = 0
    for upper in bands:
        if length < upper:
            return f"{lower}-{upper}"
        lower = upper
    return f"{lower}+"


def source_row(row: Record, root: Path) -> Record | None:
    for value in sequence(row["files"]):
        ref = artifact(value, root)
        if ref.path.name == "source.json":
            ref.verify()
            return read(ref.path)
    return None


def compare_record(
    name: str,
    reference_row: Record,
    reference_root: Path,
    replay_row: Record,
    replay_root: Path,
    hidden: int,
    rows: int,
) -> Record:
    reference_ref = artifact(reference_row["metadata"], reference_root)
    replay_ref = artifact(replay_row["metadata"], replay_root)
    reference_ref.verify()
    replay_ref.verify()
    reference_metadata = read(reference_ref.path)
    replay_metadata = read(replay_ref.path)
    for metadata in (reference_metadata, replay_metadata):
        if metadata.get("feature_semantics") != TAP_SEMANTICS or metadata.get("taps") != list(TAPS):
            raise ValueError("agreement compares eight-tap captures")
    replayed = record(replay_metadata.get("token_replay", {}))
    stream: Record = {}
    for value in sequence(replay_row["files"]):
        ref = artifact(value, replay_root)
        if ref.path.name == "token-stream.json":
            ref.verify()
            stream = read(ref.path)
    reference_source = source_row(reference_row, reference_root)
    replay_source = source_row(replay_row, replay_root)
    tokens = tuple(integer(v) for v in sequence(reference_metadata["input_tokens"]))
    replay_tokens = tuple(integer(v) for v in sequence(replay_metadata["input_tokens"]))
    reference_positions = tuple(integer(v) for v in sequence(reference_metadata["logit_positions"]))
    replay_positions = tuple(integer(v) for v in sequence(replay_metadata["logit_positions"]))
    reference_labels = tuple(integer(v) for v in sequence(reference_metadata["greedy_ids"]))
    replay_labels = tuple(integer(v) for v in sequence(replay_metadata["greedy_ids"]))
    checks = {
        "row": all(reference_row.get(key) == replay_row.get(key) for key in ROW_FIELDS),
        "request": record(replayed.get("source_metadata", {})).get("sha256")
        == record(
            record(reference_metadata.get("token_replay", {})).get(
                "source_metadata", reference(reference_ref)
            )
        ).get("sha256")
        and reference_source == replay_source,
        "tokens": tokens == replay_tokens
        and stream.get("tokens_sha256") == replayed.get("tokens_sha256"),
        "positions": reference_positions == replay_positions,
        "target": reference_metadata.get("target_sha256") == replay_metadata.get("target_sha256"),
    }
    paired = list(zip(reference_labels, replay_labels, strict=False))
    equal = sum(left == right for left, right in paired)
    first = next(
        (reference_positions[index] for index, (left, right) in enumerate(paired) if left != right),
        None,
    )
    reference_counts, _ = window_counts(name, reference_metadata)
    replay_counts, valid = window_counts(name, replay_metadata)
    reference_half = reference_metadata.get("feature_dtype", "float32") == "float16"
    replay_half = replay_metadata.get("feature_dtype", "float32") == "float16"
    features = (
        [
            {"tap": tap}
            | compare_tap(left, right, reference_half, replay_half, len(tokens), hidden, rows)
            for tap, left, right in zip(
                TAPS,
                feature_files(reference_row, reference_root, reference_half),
                feature_files(replay_row, replay_root, replay_half),
                strict=True,
            )
        ]
        if checks["tokens"]
        else []
    )
    return {
        "id": name,
        "checks": cast(Record, checks),
        "identities_exact": all(checks.values()),
        "tokens": len(tokens),
        "session": "supervised_spans" in reference_metadata,
        "labels": {
            "count": len(reference_labels),
            "equal": equal if len(reference_labels) == len(replay_labels) else 0,
            "first_difference": first,
            "valid": valid,
        },
        "features": features,
        "windows": {"reference": reference_counts, "replay": replay_counts},
        "scenario": "session"
        if "supervised_spans" in reference_metadata
        else string(
            (reference_source or {}).get("category", reference_row.get("topic") or "unknown")
        ),
        "thinking": (reference_source or {}).get("thinking", False) is True,
    }


def agreement(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) - {"ids", "bands", "rows"} != {"schema", "reference", "replay"} or (
        raw["schema"] != "replay-agreement-v1"
    ):
        raise ValueError("unsupported replay agreement specification")
    root = specification.resolve().parent
    bands = tuple(integer(v) for v in sequence(raw.get("bands", list(DEFAULT_BANDS))))
    rows = integer(raw.get("rows", 512))
    if not bands or list(bands) != sorted(set(bands)) or bands[0] < 1 or not 1 <= rows <= 8192:
        raise ValueError("increasing positive length bands and a bounded row chunk required")
    corpora = {}
    for key in ("reference", "replay"):
        ref = artifact(raw[key], root)
        ref.verify()
        manifest = read(ref.path, 64 * 1024 * 1024)
        if manifest.get("schema") != 1:
            raise ValueError("agreement compares corpus-v1 manifests")
        corpora[key] = (
            ref,
            {string(record(v)["id"]): record(v) for v in sequence(manifest["records"])},
        )
    replay_rows = corpora["replay"][1]
    reference_rows = corpora["reference"][1]
    selected = (
        tuple(string(v) for v in sequence(raw["ids"])) if "ids" in raw else tuple(replay_rows)
    )
    if (
        not selected
        or not set(selected) <= set(replay_rows)
        or not set(selected) <= set(reference_rows)
    ):
        raise ValueError("every compared record needs a reference and a replay")
    store = Store(output)
    code_root = Path(__file__).resolve().parents[2]
    code = source_identity(code_root)
    freeze_source(code_root, store.root / "sources", code, 0)
    store.atomic(store.root / "specification.json", canonical(raw))
    results = []
    groups: dict[str, Group] = {}
    total = Group()
    for name in selected:
        row = compare_record(
            name,
            reference_rows[name],
            corpora["reference"][0].path.parent,
            replay_rows[name],
            corpora["replay"][0].path.parent,
            2560,
            rows,
        )
        row["band"] = band(integer(row["tokens"]), bands)
        results.append(row)
        total.add(row)
        key = (
            string(row["scenario"])
            + ("/thinking" if row["thinking"] is True else "")
            + "|"
            + string(row["band"])
        )
        groups.setdefault(key, Group()).add(row)
    summary = total.encode()
    identities = total.identities_exact == total.records
    labels_identical = total.labels_equal == total.labels
    labels_valid = total.labels_valid == total.records
    bit_identical = (
        labels_identical and total.bit_identical_taps == total.taps == total.records * len(TAPS)
    )
    result: Record = {
        "status": "compared",
        "reference": reference(corpora["reference"][0]),
        "replay": reference(corpora["replay"][0]),
        "records": len(results),
        "identities_exact": identities,
        "labels_identical": labels_identical,
        "labels_valid": labels_valid,
        "bit_identical": bit_identical,
        "decision": "refused: identities or labels"
        if not identities or not labels_valid
        else "bit-identical"
        if bit_identical
        else "values differ: matched training and serving check required",
        "summary": summary,
        "groups": {key: group.encode() for key, group in sorted(groups.items())},
        "code_sha256": code,
    }
    store.atomic(store.root / "records.json", canonical({"records": results}))
    store.atomic(store.root / "result.json", canonical(result))
    return result
