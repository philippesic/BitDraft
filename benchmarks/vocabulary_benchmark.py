import math
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from benchmarks.kernel_benchmark import Weights, samples
from benchmarks.measurement import now
from benchmarks.native import file_reference
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.records import artifact, canonical, read

ROW_BYTES = {12: 144, 14: 210}
FORMATS = {12: "Q4_K", 14: "Q6_K"}
EXPORTED = (
    "hidden.f32",
    "signs.u32",
    "scales.f32",
    "low-rank-left.f16",
    "low-rank-right.f16",
    "frequent.i32",
)
COPIED = ("signs.u32", "scales.f32", "low-rank-left.f16", "low-rank-right.f16")
REQUIRED = {
    "schema",
    "build",
    "tables",
    "variants",
    "sets",
    "warmup",
    "repeats",
    "trials",
    "bandwidth_gb_s",
}


def integers(data: bytes) -> tuple[int, ...]:
    if len(data) % 4:
        raise ValueError("int32 data has a partial value")
    return cast(tuple[int, ...], struct.unpack("<" + str(len(data) // 4) + "i", data))


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    keep: int = 0
    shortlist: int = 0
    rank: int = 0


def variant(name: str) -> Variant:
    parts = name.split("-")
    try:
        if name in {"topk", "argmax"}:
            return Variant(name, name)
        if parts[0] == "shortlist" and len(parts) == 2:
            return Variant(name, "shortlist", shortlist=int(parts[1]))
        if parts[:2] == ["one", "bit"] and len(parts) == 3:
            return Variant(name, "one-bit", keep=int(parts[2]))
        if parts[:2] == ["low", "rank"] and len(parts) == 4:
            return Variant(name, "low-rank", keep=int(parts[3]), rank=int(parts[2]))
    except ValueError as error:
        raise ValueError("invalid vocabulary variant: " + name) from error
    raise ValueError("unknown vocabulary variant: " + name)


@dataclass(frozen=True)
class Table:
    name: str
    manifest: Record
    files: dict[str, Artifact]
    model: Artifact
    tensor: str
    kind: int
    rows: int
    width: int
    depth: int
    rank: int
    frequent: tuple[int, ...]

    @property
    def row_bytes(self) -> int:
        return self.width // 256 * ROW_BYTES[self.kind]


def load_table(value: Record, root: Path) -> Table:
    if set(value) != {"name", "scoring"}:
        raise ValueError("each table needs a name and its scoring export")
    name = string(value["name"])
    if not name or Path(name).name != name:
        raise ValueError("table names are single path components")
    exported = artifact(value["scoring"], root)
    exported.verify()
    manifest = read(exported.path)
    if manifest.get("schema") != "vocabulary-scoring-export-v1":
        raise ValueError("vocabulary scoring export required")
    listed = record(manifest["files"])
    if set(listed) != set(EXPORTED):
        raise ValueError("vocabulary scoring export is incomplete")
    files = {key: artifact(listed[key], exported.path.parent) for key in EXPORTED}
    for item in files.values():
        item.verify()
    table = record(manifest["table"])
    model = artifact(table["model"], exported.path.parent)
    model.verify()
    tensor = string(table["tensor"])
    weights = Weights(model.path)
    stored = weights.tensors[tensor]
    rows, width = integer(manifest["rows"]), integer(manifest["hidden"])
    if stored.kind not in ROW_BYTES or stored.shape != (width, rows) or width % 256:
        raise ValueError("vocabulary table must be a Q4_K or Q6_K rows x hidden table")
    ranked = files["frequent.i32"].path.read_bytes()
    frequent = integers(ranked)
    return Table(
        name,
        manifest,
        files,
        model,
        tensor,
        stored.kind,
        rows,
        width,
        integer(manifest["depth"]),
        integer(manifest["rank"]),
        frequent,
    )


def weight_bytes(table: Table, chosen: Variant) -> int:
    rescoring = chosen.keep * table.depth * table.row_bytes
    if chosen.kind in {"topk", "argmax"}:
        return table.rows * table.row_bytes
    if chosen.kind == "shortlist":
        return chosen.shortlist * table.row_bytes
    if chosen.kind == "one-bit":
        return table.rows * (table.width // 8 + 4) + rescoring
    return table.rows * table.rank * 2 + table.rank * table.width * 2 + rescoring


def multiply_adds(table: Table, chosen: Variant) -> int:
    rescoring = chosen.keep * table.width
    if chosen.kind in {"topk", "argmax"}:
        return table.rows * table.width
    if chosen.kind == "shortlist":
        return chosen.shortlist * table.width
    if chosen.kind == "one-bit":
        return table.rows * table.width + rescoring
    return table.width * table.rank + table.rank * table.rows + rescoring


def prepare_table(table: Table, variants: list[Variant], sets: int, output: Path) -> Path:
    directory = output / table.name
    owner = Store(directory)
    weights = Weights(table.model.path)
    raw = weights.tensor(table.tensor, set(ROW_BYTES))
    owner.atomic(directory / "table.bin", raw)
    hidden = table.files["hidden.f32"].path.read_bytes()
    owner.atomic(directory / "hidden.f32", hidden[: sets * table.depth * table.width * 4])
    for name in COPIED:
        owner.atomic(directory / name, table.files[name].path.read_bytes())
    for size in sorted({item.shortlist for item in variants if item.kind == "shortlist"}):
        identifiers = table.frequent[:size]
        owner.atomic(
            directory / f"shortlist-{size}.bin",
            b"".join(
                raw[token * table.row_bytes : (token + 1) * table.row_bytes]
                for token in identifiers
            ),
        )
        owner.atomic(
            directory / f"shortlist-{size}.i32",
            struct.pack("<" + str(len(identifiers)) + "i", *identifiers),
        )
    return directory


def command(
    executable: Path,
    directory: Path,
    table: Table,
    chosen: Variant,
    prefix: Path,
    sets: int,
    timing: tuple[int, int, int],
) -> list[str]:
    return [
        str(executable),
        "vocabulary",
        str(directory),
        chosen.kind,
        str(prefix),
        str(table.kind),
        str(table.rows),
        str(table.width),
        str(table.depth),
        str(sets),
        str(chosen.keep),
        str(chosen.shortlist),
        str(chosen.rank),
        *(str(value) for value in timing),
        "v1",
    ]


Choice = tuple[int, int]


def choices(text: str, sets: int, slots: int) -> tuple[str, dict[tuple[int, int], Choice]]:
    mode = ""
    result: dict[tuple[int, int], Choice] = {}
    for line in text.splitlines():
        fields = line.split(",")
        if fields[0] == "mode" and len(fields) == 2 and fields[1] in {"graph", "compute"}:
            mode = fields[1]
        elif fields[0] == "choice":
            if len(fields) != 5:
                raise ValueError("invalid native choice record")
            key = (int(fields[1]), int(fields[2]))
            if key in result or not 0 <= key[0] < sets or not 0 <= key[1] < slots:
                raise ValueError("duplicate or out-of-range native choice")
            result[key] = (int(fields[3]), int(fields[4], 16))
    if not mode or len(result) != sets * slots:
        raise ValueError("native vocabulary run is missing its mode or choices")
    return mode, result


def share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def checks(
    chosen: Variant,
    measured: dict[tuple[int, int], Choice],
    reference: dict[tuple[int, int], Choice],
    candidates: dict[tuple[int, int], set[int]] | None,
    frequent: tuple[int, ...],
) -> Record:
    keys = sorted(reference)
    equal = sum(measured[key][0] == reference[key][0] for key in keys)
    values = sum(measured[key][1] == reference[key][1] for key in keys)
    result: Record = {
        "rounds_checked": len(keys),
        "token_agreement": share(equal, len(keys)),
        "value_bits_agreement": share(values, len(keys)),
    }
    if chosen.kind in {"topk", "argmax"}:
        result["exact"] = values == len(keys)
        result["claim"] = "same winning score as the merged argmax; equal tokens unless tied"
        return result
    if chosen.kind == "shortlist":
        allowed = set(frequent[: chosen.shortlist])
        reachable = [key for key in keys if reference[key][0] in allowed]
    else:
        if candidates is None:
            raise ValueError("coarse variants need their candidate sets")
        reachable = [key for key in keys if reference[key][0] in candidates[key]]
    hits = sum(measured[key][0] == reference[key][0] for key in reachable)
    result["reference_reachable"] = share(len(reachable), len(keys))
    result["agreement_when_reachable"] = share(hits, len(reachable))
    result["claim"] = "lossy; exact rescoring is checked only where the full choice is reachable"
    return result


def candidate_sets(path: Path, sets: int, slots: int, keep: int) -> dict[tuple[int, int], set[int]]:
    data = path.read_bytes()
    if len(data) != sets * slots * keep * 4:
        raise ValueError("candidate output size differs")
    values = integers(data)
    return {
        (index, slot): set(
            values[(index * slots + slot) * keep : (index * slots + slot + 1) * keep]
        )
        for index in range(sets)
        for slot in range(slots)
    }


def timing(times: list[float], weight: int, madds: int, bandwidth: float) -> Record:
    median = statistics.median(times)
    seconds = median * 1e-6
    return {
        "microseconds": median,
        "minimum_microseconds": min(times),
        "maximum_microseconds": max(times),
        "spread_percent": (max(times) - min(times)) / median * 100,
        "weight_bytes": weight,
        "weight_bytes_per_second": weight / seconds,
        "multiply_adds_per_second": madds / seconds,
        "memory_floor_microseconds": weight / (bandwidth * 1e9) * 1e6,
    }


def validate(raw: Record) -> tuple[list[Variant], int, tuple[int, int, int], float]:
    if set(raw) != REQUIRED or raw["schema"] != "vocabulary-benchmark-v1":
        raise ValueError("unsupported vocabulary benchmark specification")
    names = [string(value) for value in sequence(raw["variants"])]
    if len(set(names)) != len(names) or "argmax" not in names:
        raise ValueError("distinct variants including the merged argmax reference required")
    variants = [variant(name) for name in names]
    if any(
        item.keep < 0
        or item.shortlist < 0
        or (item.kind in {"one-bit", "low-rank"} and not item.keep)
        for item in variants
    ):
        raise ValueError("invalid vocabulary variant size")
    variants.sort(key=lambda item: item.name != "argmax")
    sets = integer(raw["sets"])
    warmup, repeats, trials = (integer(raw[name]) for name in ("warmup", "repeats", "trials"))
    bandwidth = number(raw["bandwidth_gb_s"])
    if (
        not 1 <= sets <= 4096
        or not 1 <= warmup <= 10000
        or not 1 <= repeats <= 100000
        or not 3 <= trials <= 101
        or not math.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise ValueError("invalid timing bounds")
    return variants, sets, (warmup, repeats, trials), bandwidth


def benchmark_vocabulary(specification: Path, output: Path) -> Record:
    raw = read(specification)
    variants, requested_sets, timing_settings, bandwidth = validate(raw)
    root = specification.parent
    build = artifact(raw["build"], root)
    build.verify()
    built = read(build.path)
    executable = artifact(built["executable"], build.path.parent)
    executable.verify()
    libraries = [artifact(value, build.path.parent) for value in sequence(built["libraries"])]
    for library in libraries:
        library.verify()
    if "kernel-vocabulary.cuh" not in {
        Path(string(record(value)["path"])).name for value in sequence(built["sources"])
    }:
        raise ValueError("kernel build predates the vocabulary harness")
    tables = [load_table(record(value), root) for value in sequence(raw["tables"])]
    if not tables or len({table.name for table in tables}) != len(tables):
        raise ValueError("distinct named tables required")
    for table in tables:
        for item in variants:
            if item.kind == "low-rank" and item.rank != table.rank:
                raise ValueError("low-rank variant rank differs from the exported factor")
            if item.keep > table.rows or item.shortlist > len(table.frequent):
                raise ValueError("variant size exceeds the table or the exported ranking")
    if output.exists():
        raise ValueError("vocabulary measurement output must be new")
    owner = Store(output)
    prepared: list[tuple[Table, Path, int]] = []
    for table in tables:
        sets = min(requested_sets, len(sequence(table.manifest["sets"])))
        prepared.append((table, prepare_table(table, variants, sets, output), sets))
    owner.atomic(output / "specification.json", canonical(raw))
    rows: list[Record] = []
    with device_lease("cuda"):
        identity = cuda_identity()
        began = now()
        for table, directory, sets in prepared:
            reference: dict[tuple[int, int], Choice] = {}
            for item in variants:
                prefix = output / f"{table.name}-{item.name}"
                arguments = command(
                    executable.path, directory, table, item, prefix, sets, timing_settings
                )
                measured = subprocess.run(
                    arguments,
                    env=native_environment(libraries[0].path.parent),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                owner.atomic(
                    Path(str(prefix) + ".log"), (measured.stdout + measured.stderr).encode()
                )
                if measured.returncode:
                    raise ValueError(
                        "native vocabulary benchmark failed; see " + str(prefix) + ".log"
                    )
                times, stable = samples(measured.stdout)
                if len(times) != timing_settings[2]:
                    raise ValueError("native trial count differs")
                mode, picked = choices(measured.stdout, sets, table.depth)
                if item.name == "argmax":
                    reference = picked
                coarse = (
                    candidate_sets(Path(str(prefix) + ".candidates"), sets, table.depth, item.keep)
                    if item.kind in {"one-bit", "low-rank"}
                    else None
                )
                exported = [
                    tuple(integer(token) for token in sequence(record(value)["first_choice"]))
                    for value in sequence(table.manifest["sets"])[:sets]
                ]
                row: Record = {
                    "table": table.name,
                    "table_format": FORMATS[table.kind],
                    "variant": item.name,
                    "slots": table.depth,
                    "sets": sets,
                    "mode": mode,
                    "stable_outputs": stable,
                    "samples_microseconds": times,
                    **timing(
                        times, weight_bytes(table, item), multiply_adds(table, item), bandwidth
                    ),
                    "checks": checks(item, picked, reference, coarse, table.frequent),
                    "exported_first_choice_agreement": share(
                        sum(
                            picked[(index, slot)][0] == exported[index][slot]
                            for index in range(sets)
                            for slot in range(table.depth)
                        ),
                        sets * table.depth,
                    ),
                }
                rows.append(row)
                owner.atomic(output / "partial.json", canonical({"rows": rows}))
        seconds = now() - began
    for table in tables:
        own = [row for row in rows if row["table"] == table.name]
        current = next((row for row in own if row["variant"] == "topk"), None)
        for row in own:
            row["saving_vs_topk_microseconds"] = (
                None
                if current is None
                else number(current["microseconds"]) - number(row["microseconds"])
            )
    result: Record = {
        "schema": "vocabulary-measurement-v1",
        "specification": raw,
        "build": file_reference(build.path),
        "tables": {
            table.name: {
                "model": file_reference(table.model.path),
                "tensor": table.tensor,
                "format": FORMATS[table.kind],
                "rows": table.rows,
                "hidden": table.width,
                "rank": table.rank,
            }
            for table in tables
        },
        "gpu": identity,
        "gpu_seconds": seconds,
        "rows": rows,
        "all_stable": all(row["stable_outputs"] is True for row in rows),
        "merged_selection_exact": all(
            record(row["checks"]).get("exact", True) is True for row in rows
        ),
        "artifact_disposition": "Retain the specification, logs, raw samples, candidate sets and result; per-table input copies are reproducible from the retained scoring exports and models.",
    }
    owner.atomic(output / "result.json", canonical(result))
    return result
