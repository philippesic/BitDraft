import json
from pathlib import Path

import pytest

from benchmarks.acceptance import reduce_rounds
from benchmarks.specification import CaptureContract, Specification
from w1a1_eagle.contracts import Record


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
