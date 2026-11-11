import fcntl
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from w1a1_eagle.clock import Clock


def active_cuda_processes() -> tuple[int, ...]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        candidate = Path("/usr/lib/wsl/lib/nvidia-smi")
        if not candidate.is_file():
            raise ValueError("nvidia-smi unavailable; cannot prove GPU ownership")
        executable = str(candidate)
    result = subprocess.run(
        [executable, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


@contextmanager
def device_lease(device: str) -> Iterator[None]:
    if device == "cpu":
        yield
        return
    if device != "cuda":
        raise ValueError("supported devices are cpu/cuda")
    path = Path("/tmp/binary-eagle-cuda.lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise ValueError("GPU lock is owned by another user")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another task holds the GPU lease; preserve its job") from error
        busy = tuple(pid for pid in active_cuda_processes() if pid != os.getpid())
        if busy:
            raise ValueError("GPU has existing compute processes; wait for verified release")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)


class DeviceClock:
    def __init__(self, clock: Clock, synchronize: Callable[[], None] | None) -> None:
        self.clock = clock
        self.synchronize = synchronize

    def now(self) -> float:
        if self.synchronize is not None:
            self.synchronize()
        return self.clock.now()
