import hashlib
import struct
from pathlib import Path

import pytest

from benchmarks.vocabulary_benchmark import (
    EXPORTED,
    Variant,
    candidate_sets,
    checks,
    choices,
    command,
    integers,
    load_table,
    multiply_adds,
    prepare_table,
    validate,
    variant,
    weight_bytes,
)
from tests.test_kernel_benchmark import write_kernel_weights
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.records import canonical

ROWS, WIDTH, DEPTH, RANK = 6, 256, 3, 2


def reference(path: Path) -> Record:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def export(root: Path, kind: int = 12) -> Path:
    row_bytes = {12: 144, 14: 210}[kind]
    table = bytes((row * 7 + offset) % 256 for row in range(ROWS) for offset in range(row_bytes))
    model = root / "model.gguf"
    write_kernel_weights(model, {}, [("token_embd.weight", (WIDTH, ROWS), kind, table)])
    sizes = {
        "hidden.f32": 2 * DEPTH * WIDTH * 4,
        "signs.u32": ROWS * WIDTH // 8,
        "scales.f32": ROWS * 4,
        "low-rank-left.f16": ROWS * RANK * 2,
        "low-rank-right.f16": RANK * WIDTH * 2,
    }
    files: Record = {}
    for name, size in sizes.items():
        (root / name).write_bytes(bytes(index % 251 for index in range(size)))
        files[name] = reference(root / name)
    (root / "frequent.i32").write_bytes(struct.pack("<4i", 4, 1, 5, 0))
    files["frequent.i32"] = reference(root / "frequent.i32")
    manifest = root / "manifest.json"
    manifest.write_bytes(
        canonical(
            {
                "schema": "vocabulary-scoring-export-v1",
                "drafter": "candidate",
                "table": {"model": reference(model), "tensor": "token_embd.weight"},
                "rows": ROWS,
                "hidden": WIDTH,
                "depth": DEPTH,
                "rank": RANK,
                "frequent_mixture": "m",
                "sets": [
                    {"id": "a", "anchor": 3, "first_choice": [1, 2, 3]},
                    {"id": "b", "anchor": 9, "first_choice": [4, 5, 0]},
                ],
                "files": files,
                "definitions": {},
            }
        )
    )
    return manifest


def test_variant_names_and_costs(tmp_path: Path) -> None:
    assert variant("shortlist-8192") == Variant("shortlist-8192", "shortlist", shortlist=8192)
    assert variant("one-bit-64") == Variant("one-bit-64", "one-bit", keep=64)
    assert variant("low-rank-256-1024") == Variant(
        "low-rank-256-1024", "low-rank", keep=1024, rank=256
    )
    for bad in ("top", "shortlist-x", "low-rank-256", "one-bit"):
        with pytest.raises(ValueError):
            variant(bad)
    table = load_table({"name": "candidate", "scoring": reference(export(tmp_path))}, tmp_path)
    assert (table.rows, table.width, table.depth, table.rank, table.row_bytes) == (
        6,
        256,
        3,
        2,
        144,
    )
    assert table.frequent == (4, 1, 5, 0)
    assert weight_bytes(table, variant("argmax")) == 6 * 144
    assert weight_bytes(table, variant("shortlist-2")) == 2 * 144
    assert weight_bytes(table, variant("one-bit-2")) == 6 * (32 + 4) + 2 * 3 * 144
    assert weight_bytes(table, variant("low-rank-2-2")) == 6 * 2 * 2 + 2 * 256 * 2 + 2 * 3 * 144
    assert multiply_adds(table, variant("low-rank-2-2")) == 256 * 2 + 2 * 6 + 2 * 256


@pytest.mark.parametrize("kind", [12, 14])
def test_prepared_inputs_gather_rows_and_keep_exports(tmp_path: Path, kind: int) -> None:
    table = load_table({"name": "baseline", "scoring": reference(export(tmp_path, kind))}, tmp_path)
    variants = [variant("argmax"), variant("shortlist-2"), variant("one-bit-2")]
    directory = prepare_table(table, variants, 1, tmp_path / "out")
    raw = (directory / "table.bin").read_bytes()
    row = table.row_bytes
    assert len(raw) == ROWS * row
    assert (directory / "shortlist-2.bin").read_bytes() == raw[4 * row : 5 * row] + raw[
        row : 2 * row
    ]
    assert integers((directory / "shortlist-2.i32").read_bytes()) == (4, 1)
    assert (directory / "hidden.f32").stat().st_size == DEPTH * WIDTH * 4
    for name in ("signs.u32", "scales.f32", "low-rank-left.f16", "low-rank-right.f16"):
        assert (directory / name).read_bytes() == (tmp_path / name).read_bytes()
    arguments = command(
        Path("/x"), directory, table, variant("one-bit-2"), tmp_path / "p", 1, (5, 6, 7)
    )
    assert arguments[1:5] == ["vocabulary", str(directory), "one-bit", str(tmp_path / "p")]
    assert arguments[5:] == [str(kind), "6", "256", "3", "1", "2", "0", "0", "5", "6", "7", "v1"]


def test_export_must_be_complete_and_authenticated(tmp_path: Path) -> None:
    manifest = export(tmp_path)
    (tmp_path / "scales.f32").write_bytes(b"\0" * 24)
    with pytest.raises(ValueError):
        load_table({"name": "t", "scoring": reference(manifest)}, tmp_path)
    assert len(EXPORTED) == 6


def test_choices_and_checks() -> None:
    text = "mode,graph\nchoice,0,0,4,3f800000\nchoice,0,1,2,40000000\nsample,0,1.5,1\n"
    mode, picked = choices(text, 1, 2)
    assert mode == "graph" and picked == {(0, 0): (4, 0x3F800000), (0, 1): (2, 0x40000000)}
    with pytest.raises(ValueError):
        choices("mode,graph\nchoice,0,0,4,3f800000\n", 1, 2)
    with pytest.raises(ValueError):
        choices(text + "choice,0,1,2,40000000\n", 1, 2)
    reference_ = {(0, 0): (4, 1), (0, 1): (2, 2)}
    exact = checks(variant("topk"), {(0, 0): (4, 1), (0, 1): (3, 2)}, reference_, None, ())
    assert exact["exact"] is True and exact["token_agreement"] == 0.5
    wrong = checks(variant("topk"), {(0, 0): (4, 1), (0, 1): (3, 3)}, reference_, None, ())
    assert wrong["exact"] is False
    shortlist = checks(
        variant("shortlist-1"), {(0, 0): (4, 1), (0, 1): (4, 0)}, reference_, None, (4, 2)
    )
    assert shortlist["reference_reachable"] == 0.5
    assert shortlist["agreement_when_reachable"] == 1.0
    sets = {(0, 0): {4, 1}, (0, 1): {0, 1}}
    coarse = checks(variant("one-bit-2"), {(0, 0): (1, 0), (0, 1): (1, 0)}, reference_, sets, ())
    assert coarse["reference_reachable"] == 0.5
    assert coarse["agreement_when_reachable"] == 0.0


def test_candidate_sets_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "c.candidates"
    path.write_bytes(struct.pack("<8i", 1, 2, 3, 4, 5, 6, 7, 8))
    assert candidate_sets(path, 2, 2, 2)[(1, 0)] == {5, 6}
    with pytest.raises(ValueError):
        candidate_sets(path, 2, 2, 3)
    raw: Record = {
        "schema": "vocabulary-benchmark-v1",
        "build": {},
        "tables": [],
        "variants": ["topk", "argmax", "one-bit-64"],
        "sets": 8,
        "warmup": 10,
        "repeats": 20,
        "trials": 5,
        "bandwidth_gb_s": 960,
    }
    variants, sets, timing, bandwidth = validate(raw)
    assert [item.name for item in variants] == ["argmax", "topk", "one-bit-64"]
    assert (sets, timing, bandwidth) == (8, (10, 20, 5), 960)
    for change in (
        {"variants": ["topk"]},
        {"variants": ["argmax", "argmax"]},
        {"variants": ["argmax", "one-bit-0"]},
        {"trials": 2},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            validate(record({**raw, **change}))
