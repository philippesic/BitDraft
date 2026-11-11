import hashlib
import io
import json
import os
import tarfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest

from tools.snapshot_metadata import pack, unpack
from w1a1_eagle.contracts import Record
from w1a1_eagle.data import Artifact, Ledger, Prompt, Window, windows
from w1a1_eagle.index import IndexReader, IndexSpec, Mount
from w1a1_eagle.json_stream import ArrayRows
from w1a1_eagle.records import canonical


def test_artifact_hash_reuse_detects_same_size_edits_restored_mtime_and_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"original")
    captured = Artifact.capture(path)
    captured.verify()
    original = path.stat()
    path.write_bytes(b"changed!")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="mismatch"):
        captured.verify()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"changed!")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(ValueError, match="mismatch"):
        captured.verify()
    path.write_bytes(b"original")
    captured.verify()
    assert captured == Artifact.capture(path)


def test_null_topics_and_role_aliases() -> None:
    first = Prompt("a", "source", "one", None, (), "train", "verified")
    second = Prompt("b", "source", "two", None, (), "dev", "verified")
    assert Ledger((first, second), frozenset()).admit_training(1) == (first,)
    with pytest.raises(ValueError, match="protected"):
        Ledger((first,), frozenset({"a"}))
    alias = Prompt("b", "source", "two", None, ("a",), "dev", "verified")
    with pytest.raises(ValueError, match="alias"):
        Ledger((first, alias), frozenset())


def test_admission_count_ancestry_and_relations() -> None:
    first = Prompt("a", "source", "one", "topic", (), "train", "verified")
    second = Prompt("b", "source", "two", "topic", (), "dev", "verified")
    with pytest.raises(ValueError, match="related"):
        Ledger((first, second), frozenset())
    with pytest.raises(ValueError, match="count"):
        Ledger((first,), frozenset()).admit_training(2)
    unknown = Prompt("u", "source", "one", None, (), "train", "unknown")
    with pytest.raises(ValueError, match="provenance"):
        Ledger((unknown,), frozenset()).admit_training(1)


def test_window_boundaries_and_eog() -> None:
    tokens = tuple(range(32))
    positions = tuple(range(31))
    actual = windows("t", tokens, 4, positions, tokens[1:])
    assert [w.anchor for w in actual] == list(range(3, 17))
    assert actual[0].labels == tuple(range(4, 19))
    assert actual[0].positions() == tuple(("t", p) for p in range(3, 18))
    assert windows("t", tokens, 4, tuple(range(10)), tuple(range(1, 11)), eog=10) == ()
    with pytest.raises(ValueError, match="EOG"):
        windows("t", tokens, 4, positions, tokens[1:], eog=10)
    tail = windows(
        "t", tuple(range(24)), 8, tuple(range(8, 24)), tuple(range(100, 115)) + (999,), eog=999
    )
    assert [window.anchor for window in tail] == [8, 9]
    assert tail[-1].labels[-1] == 999
    with pytest.raises(ValueError, match="position"):
        windows("t", tokens, 4, (-1,), (0,))


def test_native_last_prompt_anchor_and_prompt_eog() -> None:
    tokens = (0, 999) + tuple(range(2, 24))
    selected = windows(
        "native", tokens, 9, tuple(range(8, 24)), tuple(range(100, 115)) + (999,), 999
    )
    assert [window.anchor for window in selected] == [8, 9]
    assert selected[-1].labels[-1] == 999
    assert len({position for window in selected for position in window.positions()}) == 16


def test_window_rejects_invalid_trajectory() -> None:
    invalid: tuple[object, ...] = ("", "  ", "\t", None, 0, 1, True, b"t", ["t"], {"t": 1})
    for trajectory in invalid:
        with pytest.raises(ValueError, match="training window"):
            Window(cast(str, trajectory), 0, tuple(range(15)))


def test_window_rejects_invalid_anchor() -> None:
    invalid: tuple[object, ...] = (-1, True, False, 0.0, 1.5, "0", None)
    for anchor in invalid:
        with pytest.raises(ValueError, match="training window"):
            Window("t", cast(int, anchor), tuple(range(15)))


def test_window_rejects_invalid_labels() -> None:
    invalid: tuple[object, ...] = (
        list(range(15)),
        None,
        "012345678901234",
        bytes(range(15)),
        tuple(range(14)),
        tuple(range(16)),
    )
    for labels in invalid:
        with pytest.raises(ValueError, match="training window"):
            Window("t", 0, cast(tuple[int, ...], labels))
    invalid_tokens: tuple[object, ...] = (-1, True, False, 1.0, "1", None, [1])
    for token in invalid_tokens:
        labels_with_invalid = tuple(range(7)) + (token,) + tuple(range(8, 15))
        with pytest.raises(ValueError, match="invalid label"):
            Window("t", 0, cast(tuple[int, ...], labels_with_invalid))


def test_window_preserves_valid_immutable_values() -> None:
    labels = tuple(range(15))
    replacement_labels = tuple(range(1, 16))
    for anchor in (0, 4):
        window = Window("t", anchor, labels)
        assert window.trajectory == "t"
        assert window.anchor == anchor
        assert window.labels == labels
        assert window.positions() == tuple(("t", anchor + j) for j in range(15))
        assert hash(window) == hash(Window("t", anchor, labels))
        with pytest.raises(FrozenInstanceError):
            setattr(window, "labels", replacement_labels)


def test_artifact_tampering(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"original")
    artifact = Artifact(path, hashlib.sha256(b"original").hexdigest(), 8)
    artifact.verify()
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="content"):
        artifact.verify()


def test_streaming_json_reference_and_refusals(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    expected = {
        "note": 'quoted "rows":[ is data',
        "rows": [{"id": str(index), "text": "é" * 25} for index in range(1000)],
        "count": 1000,
    }
    path.write_text(json.dumps(expected), encoding="utf-8")
    stream = ArrayRows(path, "rows", 8192)
    assert list(stream) == expected["rows"]
    assert stream.complete and stream.header == {"note": expected["note"], "count": 1000}
    assert stream.peak_buffer_bytes <= 8192
    for malformed in [
        '{"rows":[{"id":"x","id":"y"}]}',
        '{"rows":[{}],"rows":[]}',
        '{"rows":[{}]} trailing',
        '{"rows":[{}]',
        '{"rows":[{"n":NaN}]}',
        '{"rows":[{"text":"' + "x" * 10000 + '"}]}',
    ]:
        path.write_text(malformed)
        stream = ArrayRows(path, "rows", 8192)
        with pytest.raises(ValueError):
            list(stream)
        assert not stream.complete


def test_modern_index_metadata_counts_and_protected_aliases(tmp_path: Path) -> None:
    logical = tmp_path
    mount = Mount(logical, tmp_path)

    def ref(path: Path) -> Record:
        return {
            "path": str(logical / path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    exclusions_path = tmp_path / "exclusions.json"
    exclusions_path.write_bytes(canonical({"identities": ["heldout24"]}))
    exclusions = mount.map(ref(exclusions_path))
    source_path = tmp_path / "source.json"
    source: Record = {
        "id": "one",
        "group_id": "g",
        "source": "fixture",
        "role": "fit",
        "source_ancestry": {"fixture": "retained receipt"},
        "prompt_id": "p0",
        "identities": {"rendered_sha256": "c" * 64},
    }
    metadata_path = tmp_path / "capture.json"
    metadata: Record = {
        "taps": [2, 6, 10, 14, 18, 22, 26, 30],
        "feature_semantics": "zero_based_pre_block_input_all_tokens",
        "input_tokens": [0, 999] + list(range(2, 24)),
        "prompt_tokens": 9,
        "logit_positions": list(range(8, 24)),
        "greedy_ids": list(range(100, 115)) + [999],
        "eog_ids": [999],
    }
    metadata_path.write_bytes(canonical(metadata))
    receipt_path = tmp_path / "proof.json"
    index_path = tmp_path / "index.json"

    def write_index() -> Artifact:
        source_path.write_bytes(canonical(source))
        files: list[object] = [ref(source_path)]
        for name in [
            *(f"layer-{tap}.f32" for tap in (2, 6, 10, 14, 18, 22, 26, 30)),
            "inputs.npy",
            "label_positions.npy",
            "labels.npy",
        ]:
            files.append(
                {
                    "path": str(logical / name),
                    "sha256": "0" * 64,
                    "bytes": 384 if name.endswith("f32") else 64,
                }
            )
        receipt: Record = {
            "status": "VERIFIED_NATIVE_HARD_LABEL_CAPTURE",
            "id": "one",
            "group_id": "g",
            "greedy_vs_full_logits_argmax": True,
            "no_post_eog_labels": True,
            "taps": [2, 6, 10, 14, 18, 22, 26, 30],
            "feature_semantics": "zero_based_pre_block_input_all_tokens",
            "metadata": ref(metadata_path),
            "files": files,
            "feature_shape": [24, 4],
            "windows": 2,
            "unique_label_positions_in_windows": 16,
            "unique_label_positions": 16,
            "verified_rows": 16,
            "sequences": 1,
        }
        receipt_path.write_bytes(canonical(receipt))
        raw: Record = {
            "status": "ADMITTED_METADATA",
            "manifest_sha256": "b" * 64,
            "target_sha256": "a" * 64,
            "exclusions": ref(exclusions_path),
            "rows": [
                {
                    "id": "one",
                    "receipt_pin": ref(receipt_path),
                    "receipt": receipt,
                    "metadata": metadata,
                    "anchors": [8, 9],
                }
            ],
            "sequences": 1,
            "windows": 2,
            "unique_labels_in_windows": 16,
            "note": "synthetic metadata-only fixture; tensor files deliberately absent",
        }
        index_path.write_bytes(canonical(raw))
        return Artifact(
            index_path,
            hashlib.sha256(index_path.read_bytes()).hexdigest(),
            index_path.stat().st_size,
        )

    source_ref = write_index()
    spec = IndexSpec("b" * 64, "a" * 64, 1, 2, 16, 4, mount, exclusions)
    reader = IndexReader(source_ref, spec, 32 * 1024 * 1024)
    dataset = reader.load()
    assert reader.observed_rows == 1 and reader.observed_windows == 2
    assert dataset.rows[0].anchors == (8, 9) and dataset.rows[0].unique_positions == 16
    assert dataset.rows[0].prompt.aliases == ("c" * 64, "p0")
    assert reader.peak_buffer_bytes <= 1024 * 1024
    assert not (tmp_path / "layer-2.f32").exists()
    with pytest.raises(ValueError, match="declared counts"):
        IndexReader(
            source_ref,
            IndexSpec("b" * 64, "a" * 64, 1, 2, 17, 4, mount, exclusions),
            32 * 1024 * 1024,
        ).load()
    packed = io.BytesIO()
    summary = pack(index_path, tmp_path, packed, source_ref.sha256, 1024 * 1024)
    archive = tmp_path / "metadata.gz"
    archive.write_bytes(packed.getvalue())
    destination = tmp_path / "snapshot"
    transported = unpack(archive, destination, source_ref.sha256, 1024 * 1024)
    assert summary["files"] == transported["files"] == 5
    snapshot_mount = Mount(logical, destination)
    relocated = IndexSpec(
        "b" * 64,
        "a" * 64,
        1,
        2,
        16,
        4,
        snapshot_mount,
        snapshot_mount.map(ref(exclusions_path)),
        mount,
    )
    snapshot_ref = Artifact(destination / "index.json", source_ref.sha256, source_ref.size)
    assert IndexReader(snapshot_ref, relocated, 32 * 1024 * 1024).load().rows[0].anchors == (8, 9)
    with pytest.raises(ValueError, match="bound"):
        pack(index_path, tmp_path, io.BytesIO(), source_ref.sha256, 32)
    source["prompt_id"] = "heldout24"
    with pytest.raises(ValueError, match="protected"):
        IndexReader(write_index(), spec, 32 * 1024 * 1024).load()
    with pytest.raises(ValueError, match="logical mount"):
        mount.map({"path": "/foreign/source.json", "sha256": "0" * 64, "bytes": 1})


def test_metadata_unpack_rejects_escape(tmp_path: Path) -> None:
    archive = tmp_path / "bad.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("../foreign.json")
        member.size = 2
        output.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="unsafe"):
        unpack(archive, tmp_path / "owned", "0" * 64, 1024)
    assert not (tmp_path / "foreign.json").exists()


def test_partial_supervision_is_explicit_and_stops_at_eos() -> None:
    from w1a1_eagle.data import TailWindow, supervision, supervision_positions

    tokens = (1, 2, 3, 4)
    positions = (1, 2, 3)
    labels = (3, 4, 9)
    assert supervision("short", tokens, 2, positions, labels, 9, "full15") == ()
    selected = supervision("short", tokens, 2, positions, labels, 9, "up-to-15")
    assert all(isinstance(window, TailWindow) for window in selected)
    assert tuple(window.labels for window in selected) == ((3, 4, 9), (4, 9), (9,))
    assert supervision_positions(selected) == 3
    with pytest.raises(ValueError, match="post-EOG"):
        supervision("short", tokens, 2, positions, (3, 9, 4), 9, "up-to-15")
    with pytest.raises(ValueError):
        Window("short", 1, (3, 4, 9))
