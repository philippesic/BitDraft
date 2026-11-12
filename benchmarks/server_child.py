import ctypes
import os
import signal
import sys
from typing import Callable, cast


def main() -> int:
    if len(sys.argv) < 3:
        raise ValueError("private server launcher requires parent and executable")
    parent = int(sys.argv[1])
    if os.getppid() != parent:
        return 125
    library = ctypes.CDLL(None, use_errno=True)
    prctl = cast(Callable[[int, int, int, int, int], int], library.prctl)
    if prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot bind native server to its parent lifetime")
    if os.getppid() != parent:
        return 125
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
