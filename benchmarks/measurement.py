import json
import signal
import socket
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.client import HTTPResponse
from typing import Iterator, cast

from w1a1_eagle.contracts import Record, record
from w1a1_eagle.records import canonical


def now() -> float:
    return time.monotonic()


def pause(seconds: float) -> None:
    time.sleep(seconds)


@contextmanager
def deadline(seconds: float | None, nested: bool = False) -> Iterator[None]:
    if seconds is None:
        yield
        return
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    if remaining and not nested:
        raise ValueError("screen cannot replace an existing process deadline")
    previous = signal.getsignal(signal.SIGALRM)
    began = now()

    def expired(_signal: int, _frame: object) -> None:
        if remaining and remaining <= seconds:
            signal.signal(signal.SIGALRM, previous)
            signal.raise_signal(signal.SIGALRM)
        raise ValueError("training screen exceeded120 seconds; incomplete attempt retained")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining) if remaining else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if remaining:
            budget = remaining - (now() - began)
            if budget <= 0:
                signal.raise_signal(signal.SIGALRM)
            else:
                signal.setitimer(signal.ITIMER_REAL, budget)


def port_number() -> int:
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return cast(tuple[str, int], stream.getsockname())[1]


def request(port: int, route: str, payload: Record | None = None) -> Record:
    query = urllib.request.Request(
        "http://127.0.0.1:" + str(port) + "/" + route,
        data=canonical(payload) if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with cast(HTTPResponse, urllib.request.urlopen(query, timeout=900)) as response:
        body = response.read(16 * 1024 * 1024 + 1)
    if len(body) > 16 * 1024 * 1024:
        raise ValueError("native response exceeds 16 MiB; preserve attempt and inspect limits")
    return record(cast(object, json.loads(body)))


def ready(port: int) -> bool:
    try:
        return request(port, "health").get("status") == "ok"
    except (OSError, urllib.error.URLError, ValueError):
        return False
