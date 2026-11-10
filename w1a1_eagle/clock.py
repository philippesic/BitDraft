import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class Monotonic:
    def now(self) -> float:
        return time.monotonic()
