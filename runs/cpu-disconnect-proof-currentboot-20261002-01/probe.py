#!/usr/bin/env python3
"""Stdlib-only bounded heartbeat for remote tmux/SSH lifecycle proof."""

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path


def process_identity():
    pid = os.getpid()
    ppid = os.getppid()
    try:
        parent_comm = Path(f"/proc/{ppid}/comm").read_text().strip()
    except OSError:
        parent_comm = None
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "parent_comm": parent_comm,
        "torch_already_loaded": any(
            name == "torch" or name.startswith("torch.") for name in sys.modules
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_seconds > 240 or args.interval_seconds <= 0:
        raise ValueError("probe bound must be within 1..240 seconds")
    args.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.heartbeat.with_suffix(".summary.json")
    signal_seen = None
    identity = process_identity()

    def receive(signum, _frame):
        nonlocal signal_seen
        signal_seen = int(signum)

    signal.signal(signal.SIGTERM, receive)
    signal.signal(signal.SIGINT, receive)
    started_wall = time.time()
    started_mono = time.monotonic()
    count = 0

    def append(event):
        record = {
            **identity,
            "event": event,
            "counter": count,
            "unix_time": time.time(),
            "elapsed_seconds": time.monotonic() - started_mono,
            "received_signal": signal_seen,
        }
        with args.heartbeat.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def save_summary(exit_reason, exit_code):
        result = {
            "schema": "remote_tmux_disconnect_probe_v1",
            "identity": identity,
            "started_unix": started_wall,
            "ended_unix": time.time(),
            "heartbeat_count": count,
            "received_signal": signal_seen,
            "exit_reason": exit_reason,
            "exit_code": exit_code,
            "heartbeat_path": str(args.heartbeat.resolve()),
        }
        fd, name = tempfile.mkstemp(prefix=summary_path.name + ".", dir=summary_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, summary_path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    deadline = started_mono + args.max_seconds
    while True:
        append("heartbeat")
        if signal_seen is not None:
            append("signal_received")
            save_summary("signal", 0)
            return 0
        if time.monotonic() >= deadline:
            append("time_limit")
            save_summary("time_limit", 2)
            return 2
        count += 1
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
