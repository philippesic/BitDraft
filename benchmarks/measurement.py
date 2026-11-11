import json
import socket
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from typing import cast

from w1a1_eagle.contracts import Record, record
from w1a1_eagle.records import canonical


def now() -> float:
    return time.monotonic()


def pause(seconds: float) -> None:
    time.sleep(seconds)


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
