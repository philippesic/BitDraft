import hashlib
from pathlib import Path

import pytest

from w1a1_eagle.data import Artifact, Ledger, Prompt, windows


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
    assert [w.anchor for w in actual] == list(range(4, 17))
    assert actual[0].labels == tuple(range(5, 20))
    assert actual[0].positions() == tuple(("t", p) for p in range(4, 19))
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


def test_artifact_tampering(tmp_path: Path) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(b"original")
    artifact = Artifact(path, hashlib.sha256(b"original").hexdigest(), 8)
    artifact.verify()
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="content"):
        artifact.verify()
