import hashlib
from dataclasses import dataclass
from pathlib import Path

from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import (
    Artifact,
    CaptureRow,
    CaptureSpec,
    Dataset,
    Prompt,
    optional_topic,
    position_count,
    windows,
)
from w1a1_eagle.json_stream import ArrayRows
from w1a1_eagle.records import artifact, canonical, read

TAPS = (2, 6, 10, 14, 18, 22, 26, 30)


@dataclass(frozen=True)
class Mount:
    logical: Path
    physical: Path

    def __post_init__(self) -> None:
        if not self.logical.is_absolute() or ".." in self.logical.parts:
            raise ValueError("logical input root must be absolute")
        object.__setattr__(self, "physical", self.physical.absolute())

    def map(self, value: object) -> Artifact:
        raw = artifact(value, self.logical, resolve=False)
        if ".." in raw.path.parts:
            raise ValueError("input reference escaped its declared logical mount")
        try:
            relative = raw.path.relative_to(self.logical)
        except ValueError as error:
            raise ValueError("input reference escaped its declared logical mount") from error
        return Artifact(self.physical / relative, raw.sha256, raw.size, self.physical)


@dataclass(frozen=True)
class IndexSpec:
    manifest_sha256: str
    target_sha256: str
    sequences: int
    windows: int
    unique_positions: int
    hidden: int
    mount: Mount
    exclusions: Artifact
    data_mount: Mount | None = None

    def __post_init__(self) -> None:
        for digest in (self.manifest_sha256, self.target_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid declared index identity")
        if any(
            type(value) is not int or value < 1
            for value in (self.sequences, self.windows, self.unique_positions, self.hidden)
        ):
            raise ValueError("positive declared index counts required")


def source_prompt(identity: str, source: Record, protected: frozenset[str]) -> Prompt:
    if source.get("id") != identity:
        raise ValueError("capture/source identity mismatch: " + identity)
    ancestry_status = "unknown"
    previous_use = (
        "historical-training" if source.get("pool") == "historical_optimizer" else "unknown"
    )
    if source.get("role") in {"fit", "train"}:
        ancestry = record(source["source_ancestry"])
        if ancestry and ancestry.get("certainty") != "unknown":
            ancestry_status = "verified"
        if source.get("pool") == "unused_modern_reserved":
            previous_use = "unobserved"
        namespace = string(source["source"])
        aliases = record(source["identities"])
    elif source.get("role") is None and source.get("split") == "train":
        if integer(source["shard"]) < 0 or integer(source["row"]) < 0:
            raise ValueError("invalid retained catalog locator: " + identity)
        namespace = "retained-modern-catalog"
        aliases = {"content": string(source["content_sha256"])}
        ancestry_status = "verified"
    elif (
        source.get("role") is None
        and source.get("split") is None
        and source.get("pool") == "authored_broader_domains"
    ):
        if set(source) != {"id", "group_id", "domain", "pool", "rendered_text"}:
            raise ValueError("unsupported authored source schema: " + identity)
        namespace = "authored-broader-domains"
        aliases = {"rendered": hashlib.sha256(string(source["rendered_text"]).encode()).hexdigest()}
        ancestry_status = "verified"
    else:
        raise ValueError("unsupported or non-training source role: " + identity)
    names = {identity}
    for key in (
        "prompt_id",
        "original_prompt_id",
        "message_identity_sha256",
        "normalized_message_identity_sha256",
        "rendered_sha256",
        "rendered_prompt_sha256",
    ):
        if source.get(key) is not None:
            names.add(string(source[key]))
    names.update(string(value) for value in aliases.values())
    groups = {string(source["group_id"])}
    if source.get("original_group") is not None:
        groups.add(string(source["original_group"]))
    if (names | groups) & protected:
        raise ValueError("protected source identity or group in training index: " + identity)
    topic = source.get("source_topic")
    return Prompt(
        identity,
        namespace,
        string(source["group_id"]),
        optional_topic(topic),
        tuple(sorted(names - {identity})),
        "train",
        ancestry_status,
        "historical-training"
        if previous_use == "historical-training"
        else "unobserved"
        if previous_use == "unobserved"
        else "unknown",
    )


def admit_row(value: Record, spec: IndexSpec, protected: frozenset[str]) -> CaptureRow:
    if set(value) != {"id", "receipt_pin", "receipt", "metadata", "anchors"}:
        raise ValueError("unsupported modern index row fields")
    identity = string(value["id"])
    receipt_ref = spec.mount.map(value["receipt_pin"])
    receipt_ref.verify()
    receipt = read(receipt_ref.path)
    if canonical(receipt) != canonical(value["receipt"]):
        raise ValueError("embedded and retained capture receipt disagree: " + identity)
    if (
        receipt.get("status") != "VERIFIED_NATIVE_HARD_LABEL_CAPTURE"
        or receipt.get("id") != identity
        or receipt.get("greedy_vs_full_logits_argmax") is not True
        or receipt.get("no_post_eog_labels") is not True
        or receipt.get("taps") != list(TAPS)
        or receipt.get("feature_semantics") != "zero_based_pre_block_input_all_tokens"
    ):
        raise ValueError("capture proof contract mismatch: " + identity)
    metadata_ref = spec.mount.map(receipt["metadata"])
    metadata_ref.verify()
    metadata = read(metadata_ref.path)
    if canonical(metadata) != canonical(value["metadata"]):
        raise ValueError("embedded and retained metadata disagree: " + identity)
    files = tuple(
        (
            spec.mount
            if Path(string(record(item)["path"])).name == "source.json"
            else spec.data_mount or spec.mount
        ).map(item)
        for item in sequence(receipt["files"])
    )
    by_name = {item.path.name: item for item in files}
    required = {
        *(f"layer-{tap}.f32" for tap in TAPS),
        "source.json",
        "inputs.npy",
        "label_positions.npy",
        "labels.npy",
    }
    if len(by_name) != len(files) or not required.issubset(by_name):
        raise ValueError("missing or duplicate capture files: " + identity)
    source_ref = by_name["source.json"]
    source_ref.verify()
    prompt = source_prompt(identity, read(source_ref.path), protected)
    if receipt.get("group_id") != prompt.group:
        raise ValueError("capture/source group disagreement: " + identity)
    tokens = tuple(integer(item) for item in sequence(metadata["input_tokens"]))
    positions = tuple(integer(item) for item in sequence(metadata["logit_positions"]))
    labels = tuple(integer(item) for item in sequence(metadata["greedy_ids"]))
    eog = {integer(item) for item in sequence(metadata["eog_ids"])}
    if (
        metadata.get("taps") != list(TAPS)
        or metadata.get("feature_semantics") != receipt["feature_semantics"]
        or any(item in eog for item in labels[:-1])
        or receipt.get("feature_shape") != [len(tokens), spec.hidden]
        or any(by_name[f"layer-{tap}.f32"].size != len(tokens) * spec.hidden * 4 for tap in TAPS)
    ):
        raise ValueError("capture shape or EOG contract mismatch: " + identity)
    selected = windows(
        identity,
        tokens,
        integer(metadata["prompt_tokens"]),
        positions,
        labels,
        labels[-1] if labels and labels[-1] in eog else None,
    )
    anchors = tuple(window.anchor for window in selected)
    unique_positions = position_count(anchors)
    if (
        tuple(integer(item) for item in sequence(value["anchors"])) != anchors
        or integer(receipt["windows"]) != len(anchors)
        or integer(receipt["unique_label_positions_in_windows"]) != unique_positions
        or integer(receipt["unique_label_positions"]) != len(positions)
        or integer(receipt["verified_rows"]) != len(positions)
        or integer(receipt["sequences"]) != 1
    ):
        raise ValueError("native label/window accounting disagreement: " + identity)
    return CaptureRow(prompt, CaptureSpec(identity, metadata_ref, files), anchors, unique_positions)


class IndexReader:
    def __init__(self, source: Artifact, spec: IndexSpec, byte_budget: int) -> None:
        if type(byte_budget) is not int or byte_budget < 32 * 1024 * 1024:
            raise ValueError("explicit index metadata memory budget required")
        self.source = source
        self.spec = spec
        self.byte_budget = byte_budget
        self.observed_rows = 0
        self.observed_windows = 0
        self.observed_unique_positions = 0
        self.peak_buffer_bytes = 0

    def load(self) -> Dataset:
        self.source.verify()
        self.spec.exclusions.verify()
        exclusions = read(self.spec.exclusions.path)
        protected = frozenset(
            {"heldout24", *(string(item) for item in sequence(exclusions["identities"]))}
        )
        stream = ArrayRows(self.source.path, "rows")
        rows: list[CaptureRow] = []
        self.observed_rows = self.observed_windows = self.observed_unique_positions = 0
        retained_bytes = self.spec.exclusions.size * 16
        for value in stream:
            if self.observed_rows >= self.spec.sequences:
                raise ValueError("index exceeds declared sequence count")
            row = admit_row(value, self.spec, protected)
            retained_bytes += (
                len(row.anchors) * 40
                + 2048
                + sum(len(alias.encode()) * 4 + 128 for alias in row.prompt.aliases)
                + sum(len(str(item.path).encode()) * 4 + 512 for item in row.capture.files)
            )
            if retained_bytes >= self.byte_budget // 2:
                raise ValueError("admitted index metadata exceeds memory budget")
            rows.append(row)
            self.observed_rows += 1
            self.observed_windows += len(row.anchors)
            self.observed_unique_positions += row.unique_positions
        header = stream.header
        required = {
            "status",
            "manifest_sha256",
            "target_sha256",
            "exclusions",
            "sequences",
            "windows",
            "unique_labels_in_windows",
            "note",
        }
        if (
            not stream.complete
            or set(header) != required
            or header["status"] != "ADMITTED_METADATA"
        ):
            raise ValueError("unsupported modern index envelope")
        if (
            header["manifest_sha256"] != self.spec.manifest_sha256
            or header["target_sha256"] != self.spec.target_sha256
            or self.spec.mount.map(header["exclusions"]) != self.spec.exclusions
        ):
            raise ValueError("index lineage differs from declared manifest/target/exclusions")
        observed = (self.observed_rows, self.observed_windows, self.observed_unique_positions)
        declared = (self.spec.sequences, self.spec.windows, self.spec.unique_positions)
        embedded = tuple(
            integer(header[key]) for key in ("sequences", "windows", "unique_labels_in_windows")
        )
        if observed != declared or observed != embedded:
            raise ValueError("index coverage differs from declared counts: " + str(observed))
        self.source.verify()
        self.peak_buffer_bytes = stream.peak_buffer_bytes
        return Dataset(self.source, self.spec.target_sha256, tuple(rows), protected, retained_bytes)
