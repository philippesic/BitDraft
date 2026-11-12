import math
import os
import signal
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        raise ValueError("private bootstrap requires a wall limit and fixed worker arguments")
    seconds = float(sys.argv[1])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite wall limit required")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    arguments = sys.argv[2:]
    module = "w1a1_eagle.numerical.entry"
    if arguments[0] == "--entrypoint":
        module, arguments = arguments[1], arguments[2:]
        if not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("invalid frozen worker module")
        path = Path(__file__).resolve().parents[1].joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file() or path.is_symlink():
            raise ValueError("worker module is outside the frozen source closure")
    os.execv(sys.executable, [sys.executable, "-B", "-m", module, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
