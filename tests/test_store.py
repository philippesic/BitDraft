from pathlib import Path
from typing import Callable

import pytest

from w1a1_eagle.checkpoints.store import Store


def test_committed_state_and_tamper(tmp_path: Path) -> None:
    store = Store(tmp_path)
    reference = store.commit(b"state one", "recipe", 1)
    assert store.latest("recipe") == reference
    with pytest.raises(ValueError, match="configuration"):
        store.latest("other")
    reference.artifact.path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="content"):
        store.latest("recipe")


def test_failed_publication_preserves_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    parent = store.commit(b"parent", "recipe", 1)
    original: Callable[[Path, bytes], None] = store.atomic

    def failing(path: Path, payload: bytes) -> None:
        if path.name == "latest.json":
            raise OSError("injected publication failure")
        original(path, payload)

    monkeypatch.setattr(store, "atomic", failing)
    with pytest.raises(OSError, match="publication"):
        store.commit(b"child", "recipe", 2)
    assert store.latest("recipe") == parent
    assert not list(tmp_path.glob(".pending-*"))
