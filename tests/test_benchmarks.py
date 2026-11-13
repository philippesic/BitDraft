import json
from pathlib import Path

import pytest

from benchmarks.acceptance import reduce_rounds
from benchmarks.specification import CaptureContract, Specification
from w1a1_eagle.contracts import Record, record


def contract() -> Record:
    return {
        "target": "a" * 64,
        "tokenizer": "b" * 64,
        "template": "c" * 64,
        "taps": [2, 6],
        "hidden": 16,
        "semantics": "pre-block",
        "objective": "hard-label",
    }


def spec() -> Record:
    return {
        "schema": 1,
        "candidate": contract(),
        "capture": contract(),
        "features": [],
        "storage_bytes": 2,
        "capture_limit_bytes": 100000,
        "tasks": [
            {
                "id": "one",
                "group": "one",
                "role": "train",
                "category": "code",
                "input_tokens": 128,
                "output_cap": 32,
            }
        ],
    }


def test_contract_refuses_cross_target_and_semantics() -> None:
    expected = CaptureContract.parse(contract())
    for field, value in (
        ("target", "d" * 64),
        ("semantics", "post-block"),
        ("hidden", 32),
        ("objective", "soft-label"),
        ("taps", [3, 7]),
    ):
        actual = contract()
        actual[field] = value
        with pytest.raises(ValueError, match=field):
            expected.require(CaptureContract.parse(actual))


def test_forecast_and_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    raw = spec()
    path.write_text(json.dumps(raw))
    assert Specification.load(path).report()["maximum_feature_bytes"] == 10240
    raw["tasks"] = [
        {
            "id": "a",
            "group": "same",
            "role": "train",
            "category": "code",
            "input_tokens": 1,
            "output_cap": 1,
        },
        {
            "id": "b",
            "group": "same",
            "role": "dev",
            "category": "code",
            "input_tokens": 1,
            "output_cap": 1,
        },
    ]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="crosses data roles"):
        Specification.load(path)


def test_acceptance_excludes_target_correction_and_bonus() -> None:
    result = reduce_rounds(
        [
            {"proposed": [10, 11, 12], "accepted": 1, "target_tokens": [20], "emitted": [10, 20]},
            {"proposed": [30, 31], "accepted": 2, "target_tokens": [40], "emitted": [30, 31, 40]},
        ]
    )
    assert result["accepted"] == 3
    assert result["proposed"] == 5
    assert result["acceptance"] == 0.6
    assert result["emitted"] == 5
    with pytest.raises(ValueError, match="emitted tokens"):
        reduce_rounds([{"proposed": [10], "accepted": 1, "target_tokens": [], "emitted": [99]}])


def test_summary_reports_rounds_and_speed_relative_to_controls() -> None:
    from benchmarks.metrics import relative_speed, server_seconds, totals

    def row(tokens: int, proposed: int, accepted: int) -> Record:
        return {
            "tokens": tokens,
            "wall_seconds": 1.0,
            "correct": True,
            "stop_type": "eos",
            "native": {"proposed": proposed, "accepted": accepted},
            **server_seconds([{"timings": {"prompt_ms": 250.0, "predicted_ms": 500.0}}]),
        }

    drafted = totals([row(11, 8, 6), row(9, 8, 2)])
    assert drafted["derived_rounds"] == 10
    assert drafted["proposed_per_round"] == 1.6
    assert drafted["accepted_per_round"] == 0.8
    assert drafted["emitted_per_round"] == 1.8
    assert drafted["server_prompt_seconds"] == 0.5
    assert drafted["server_prediction_seconds"] == 1.0
    mixed = [
        row(11, 8, 6),
        {key: value for key, value in row(9, 8, 2).items() if "server" not in key},
    ]
    assert totals(mixed)["server_prompt_seconds"] is None
    assert server_seconds([{"timings": {"predicted_ms": 1.0}}]) == {}
    plain = totals([row(5, 0, 0)])
    assert plain["derived_rounds"] is None and plain["accepted_per_round"] is None
    summaries = relative_speed(
        {"target": plain, "baseline": totals([row(40, 8, 8)]), "candidate": drafted}
    )
    assert record(summaries["candidate"])["speed_vs_target"] == 2.0
    assert record(summaries["candidate"])["speed_vs_baseline"] == 0.25
    assert record(summaries["target"])["speed_vs_target"] == 1.0
    assert "speed_vs_baseline" not in record(relative_speed({"candidate": drafted})["candidate"])


def test_fallback_has_no_acceptance_denominator() -> None:
    result = reduce_rounds([{"proposed": [], "accepted": 0, "target_tokens": [2], "emitted": [2]}])
    assert result["acceptance"] is None
    assert result["activation"] == "fallback_only"


def test_coverage_requires_real_strata_and_length_supply() -> None:
    from benchmarks.coverage import select

    rows: list[Record] = [
        {"id": str(i), "category": category, "subtype": subtype, "length_band": band}
        for i, (category, subtype, band) in enumerate(
            (
                ("code", "editing", 0),
                ("code", "generation", 2),
                ("conversation", "followup", 0),
                ("conversation", "followup", 2),
            )
        )
    ]
    chosen = select(rows, (2, 0, 0, 2, 0, 0), (2, 0, 2, 0))
    assert {row["id"] for row in chosen} == {"0", "1", "2", "3"}
    with pytest.raises(ValueError, match="cannot satisfy"):
        select(rows, (2, 0, 0, 2, 0, 0), (2, 2, 0, 0))


def test_paired_statistics_preserve_groups_and_missing_pairs() -> None:
    from benchmarks.statistics import paired

    rows: list[Record] = [
        {
            "arm": arm,
            "group": group,
            "task": group,
            "rep": 0,
            "tokens": 10,
            "wall_seconds": seconds,
            "native": {"proposed": 4, "accepted": accepted},
        }
        for group in ("one", "two")
        for arm, seconds, accepted in (("baseline", 1.0, 2), ("candidate", 0.5, 3))
    ]
    result = paired(rows, "candidate", "baseline", 100)
    assert result["speed_ratio"] == 2
    assert result["acceptance_difference"] == 0.25
    assert result["speed_ratio_95_interval"] == {"lower": 2.0, "upper": 2.0}
    with pytest.raises(ValueError, match="both arms"):
        paired(rows[:-1], "candidate", "baseline", 100)


def test_capture_replies_preserve_messages_and_refuse_eof() -> None:
    import os

    from benchmarks.capture import Replies

    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as source, os.fdopen(write_fd, "wb") as target:
        target.write(b'{"ready":true}\n{"status":"complete"}\n')
        target.flush()
        replies = Replies(source)
        assert replies.receive(1) == {"ready": True}
        assert replies.receive(1) == {"status": "complete"}
        target.close()
        with pytest.raises(ValueError, match="exited"):
            replies.receive(1)


def test_native_bundle_refuses_paths_and_links(tmp_path: Path) -> None:
    import io
    import tarfile

    from benchmarks.native import unpack_bundle

    for index, name in enumerate(("../escape", "/escape", "source/file")):
        path = tmp_path / (str(index) + ".tgz")
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(name)
            if name == "source/file":
                member.type = tarfile.SYMTYPE
                member.linkname = "../../escape"
                archive.addfile(member)
            else:
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
        if name == "source/file":
            with tarfile.open(path, "r:gz") as archive:
                assert archive.getmember(name).linkname == "../../escape"
        with pytest.raises(ValueError, match="regular source"):
            unpack_bundle(path, tmp_path / "output", 1024)


def test_runtime_source_edits_apply_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import PurePosixPath

    import benchmarks.native as native
    from benchmarks.runtime_edits import SOURCE_EDITS

    for edits in SOURCE_EDITS.values():
        for relative, old, new in edits:
            parts = PurePosixPath(relative).parts
            assert relative == PurePosixPath(relative).as_posix() and ".." not in parts
            assert not relative.startswith("/") and old and old != new
    monkeypatch.setattr(native, "SOURCE_EDITS", {"fix": (("src/a.cpp", "int a;\n", "int b;\n"),)})
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/a.cpp").write_text("x\nint a;\ny\n")
    native.apply_source_edits(source, "fix")
    assert (source / "src/a.cpp").read_text() == "x\nint b;\ny\n"
    with pytest.raises(ValueError, match="does not match"):
        native.apply_source_edits(source, "fix")
    (source / "src/a.cpp").write_text("int a;\nint a;\n")
    with pytest.raises(ValueError, match="does not match"):
        native.apply_source_edits(source, "fix")
    with pytest.raises(ValueError, match="unknown"):
        native.apply_source_edits(source, "other")


def test_fixed_cohort_rejects_narrow_categories_lengths_and_related_groups() -> None:
    from benchmarks.comparison import require_cohort
    from benchmarks.coverage import SHORT

    categories = ("code", "reasoning", "documents", "conversation", "structured", "writing")
    labels = [name for name, count in zip(categories, SHORT, strict=True) for _ in range(count)]
    lengths = [100] * 60 + [1000] * 36 + [4000] * 18 + [10000] * 6
    tasks: tuple[Record, ...] = tuple(
        {"id": str(i), "group": str(i), "category": category, "ancestry": {"input_tokens": length}}
        for i, (category, length) in enumerate(zip(labels, lengths, strict=True))
    )
    require_cohort("short", tasks)
    for broken in (
        ({**tasks[0], "category": "documents"}, *tasks[1:]),
        ({**tasks[0], "ancestry": {"input_tokens": 10000}}, *tasks[1:]),
        ({**tasks[0], "group": tasks[1]["group"]}, *tasks[1:]),
    ):
        with pytest.raises(ValueError, match="coverage"):
            require_cohort("short", broken)


def test_storage_selection_keeps_coverage_and_required_harness() -> None:
    from benchmarks.coverage import select

    rows: list[Record] = [
        {
            "id": name,
            "group": name,
            "source": "unit",
            "role": "train",
            "category": "code",
            "subtype": kind,
            "length_band": 0,
            "input_tokens": tokens,
        }
        for name, kind, tokens in [
            ("edit-long", "editing", 500),
            ("edit-short", "editing", 100),
            ("generate-long", "generation", 400),
            ("generate-short", "generation", 80),
        ]
    ]
    selected = select(rows, (2, 0, 0, 0, 0, 0), (2, 0, 0, 0), 180)
    assert {row["id"] for row in selected} == {"edit-short", "generate-short"}
    with pytest.raises(ValueError, match="input-token budget"):
        select(rows, (2, 0, 0, 0, 0, 0), (2, 0, 0, 0), 179)
    rows[0]["harness"] = "project-training"
    assert rows[0] in select(rows, (2, 0, 0, 0, 0, 0), (2, 0, 0, 0), 580)
    rows[0]["source"] = "project-replay"
    with pytest.raises(ValueError, match="roles"):
        select(rows, (2, 0, 0, 0, 0, 0), (2, 0, 0, 0))


def test_native_comparison_refuses_server_not_produced_by_declared_build(tmp_path: Path) -> None:
    from benchmarks.comparison import Arm, Comparison
    from benchmarks.native import file_reference
    from w1a1_eagle.records import artifact, canonical

    server, library, source = (
        tmp_path / name for name in ["llama-server", "libtest.so", "build.json"]
    )
    server.write_bytes(b"server")
    library.write_bytes(b"library")
    source.write_bytes(canonical({"artifacts": [file_reference(library)]}))
    binary = artifact(file_reference(server), tmp_path)
    lib = artifact(file_reference(library), tmp_path)
    built = artifact(file_reference(source), tmp_path)
    spec = Comparison(
        built,
        binary,
        (lib,),
        binary,
        (Arm("target", None), Arm("baseline", binary)),
        binary,
        (),
        1,
        0,
        16,
        2,
        "dev",
        "pilot",
        frozenset(),
    )
    with pytest.raises(ValueError, match="authenticated build"):
        spec.verify()
    source.write_bytes(canonical({"artifacts": [file_reference(library), file_reference(server)]}))
    from dataclasses import replace

    replace(spec, source=artifact(file_reference(source), tmp_path)).verify()
