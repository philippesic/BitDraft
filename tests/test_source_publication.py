import errno
import shutil
from pathlib import Path

import pytest

from w1a1_eagle.identity import freeze_source, source_identity


def test_concurrent_source_publisher_accepts_only_identical_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    for name, data in {
        "w1a1_eagle/__init__.py": "",
        "policy/numerical.toml": "schema=1",
        "uv.lock": "lock",
        "pyproject.toml": "project",
    }.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    expected = source_identity(checkout)
    changed = False

    def publish_other(source: Path, target: Path) -> None:
        shutil.copytree(source, target)
        if changed:
            (target / "w1a1_eagle/injected.py").write_text("VALUE=1\n")
        raise OSError(errno.ENOTEMPTY, "simultaneous publisher")

    monkeypatch.setattr("w1a1_eagle.identity.os.rename", publish_other)
    destination = tmp_path / "matching"
    assert source_identity(freeze_source(checkout, destination, expected, 0)) == expected
    assert not list(destination.glob(".pending-source-*"))
    changed = True
    with pytest.raises(ValueError, match="concurrent source snapshot"):
        freeze_source(checkout, tmp_path / "changed", expected, 0)
