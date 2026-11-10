import math
import os
import signal
import sys


def main() -> int:
    if len(sys.argv) < 3:
        raise ValueError("private bootstrap requires a wall limit and fixed worker arguments")
    seconds = float(sys.argv[1])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("positive finite wall limit required")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    os.execv(
        sys.executable, [sys.executable, "-B", "-m", "w1a1_eagle.numerical.entry", *sys.argv[2:]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
