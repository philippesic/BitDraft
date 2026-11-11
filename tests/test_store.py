import select
import signal
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO, Callable

import pytest

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.data import Artifact
from w1a1_eagle.processes import ProcessIdentity


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


def test_stream_budget_preserves_parent(tmp_path: Path) -> None:
    store = Store(tmp_path)
    parent = store.commit(b"parent", "recipe", 1)

    def oversized(output: BinaryIO) -> None:
        output.write(b"1234")
        assert output.tell() == 4
        output.write(b"5678")

    with pytest.raises(ValueError, match="budget"):
        store.commit_with(oversized, "recipe", 2, 6)
    assert store.latest("recipe") == parent
    assert not list(tmp_path.glob(".pending-*"))


def test_process_crash_and_write_limit_preserve_parent(tmp_path: Path) -> None:
    store = Store(tmp_path)
    parent = store.commit(b"parent", "recipe", 1)
    manifest = (tmp_path / "latest.json").read_bytes()
    source = "\n".join(
        [
            "import signal,sys",
            "from pathlib import Path",
            "from w1a1_eagle.checkpoints.store import Store",
            "def write(output):",
            "    output.write(b'uncommitted child')",
            "    print('ready',flush=True)",
            "    signal.pause()",
            "Store(Path(sys.argv[1])).commit_with(write,'recipe',2,1024)",
        ]
    )
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", source, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    owner = ProcessIdentity.read(process.pid)
    try:
        assert process.stdout is not None
        poller = select.poll()
        poller.register(process.stdout, select.POLLIN)
        assert poller.poll(5000)
        assert process.stdout.readline().strip() == "ready"
        owner.signal_group(signal.SIGKILL)
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            owner.signal_group(signal.SIGKILL)
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
    assert store.latest("recipe") == parent
    assert (tmp_path / "latest.json").read_bytes() == manifest
    failure = "\n".join(
        [
            "import resource,signal,sys",
            "from pathlib import Path",
            "from w1a1_eagle.checkpoints.store import Store",
            "signal.signal(signal.SIGXFSZ,signal.SIG_IGN)",
            "resource.setrlimit(resource.RLIMIT_FSIZE,(32,32))",
            "Store(Path(sys.argv[1])).commit(b'x'*1024,'recipe',2)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", failure, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert store.latest("recipe") == parent
    assert (tmp_path / "latest.json").read_bytes() == manifest
    assert Store(tmp_path).commit(b"recovered", "recipe", 2).updates == 2


def test_validation_and_root_ownership_before_publication(tmp_path: Path) -> None:
    store = Store(tmp_path)
    parent = store.commit(b"parent", "recipe", 1)

    def write(output: BinaryIO) -> None:
        output.write(b"invalid producer output")

    def reject(artifact: Artifact) -> None:
        artifact.verify()
        raise ValueError("independent consumer rejected output")

    with pytest.raises(ValueError, match="consumer"):
        store.commit_with(write, "recipe", 2, 64, reject)
    assert store.latest("recipe") == parent
    with pytest.raises(ValueError, match="escaped"):
        store.atomic(tmp_path.parent / "foreign.json", b"wrong")
    with pytest.raises(ValueError, match="configuration"):
        store.commit(b"other task", "other", 1)
    assert store.latest("recipe") == parent


def test_closed_retention_preserves_selected_and_valid_recovery(tmp_path: Path) -> None:
    old = Store(tmp_path / "old")
    unselected = old.commit(b"unselected", "recipe", 1)
    selected = old.commit(b"selected", "recipe", 2)
    recovery = Store(tmp_path / "new").commit(b"recovery", "recipe", 3)
    retired = old.retire_closed(frozenset({selected.artifact.path}), recovery.artifact)
    assert retired == (unselected.artifact.sha256,)
    assert not unselected.artifact.path.exists()
    assert old.describe(selected.artifact) == selected
    assert old.retire_closed(frozenset({selected.artifact.path}), recovery.artifact) == ()
    recovery.artifact.path.write_bytes(b"badstate")
    with pytest.raises(ValueError, match="content"):
        old.retire_closed(frozenset(), recovery.artifact)
    assert selected.artifact.path.exists()
