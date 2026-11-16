import os
import platform
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from w1a1_eagle import capacity


class Usage(NamedTuple):
    total: int
    used: int
    free: int


@pytest.mark.parametrize("release", ["7.0.0-generic", "6.6.0-microsoft-standard-WSL2"])
def test_backing_capacity_selects_host_volume(
    monkeypatch: pytest.MonkeyPatch, release: str
) -> None:
    observed: list[Path] = []

    def usage(path: Path) -> Usage:
        observed.append(path)
        return Usage(1000, 877, 123)

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "release", lambda: release)

    def mounted(path: Path) -> bool:
        return path == Path("/mnt/c")

    monkeypatch.setattr(os.path, "ismount", mounted)
    monkeypatch.setattr(shutil, "disk_usage", usage)
    assert capacity.backing_available() == 123
    expected = (
        Path("/mnt/c")
        if "microsoft" in release
        else Path(capacity.__file__).resolve().parent.parent
    )
    assert observed == [expected]


def test_wsl_refuses_missing_backing_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "release", lambda: "6.6.0-microsoft-standard-WSL2")

    def unmounted(path: Path) -> bool:
        return False

    monkeypatch.setattr(os.path, "ismount", unmounted)
    with pytest.raises(ValueError, match="backing-volume capacity is unknown"):
        capacity.backing_available()
