import os
import platform
import shutil
from pathlib import Path


def backing_available() -> int:
    if platform.system() == "Linux" and "microsoft" not in platform.release().lower():
        return shutil.disk_usage(Path(__file__).resolve().parent.parent).free
    path = Path("/mnt/c")
    if not os.path.ismount(path):
        raise ValueError(
            "backing-volume capacity is unknown on this host; provide a qualified host adapter"
        )
    return shutil.disk_usage(path).free


def require(root: Path, peak: int, reserve: int) -> None:
    if type(peak) is not int or type(reserve) is not int or peak < 0 or reserve < 0:
        raise ValueError("invalid disk requirement")
    if shutil.disk_usage(root).free < peak + reserve:
        raise ValueError("observed disk headroom cannot cover peak plus reserve")
