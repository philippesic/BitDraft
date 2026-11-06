#!/usr/bin/env python3
"""Bounded CPU-only WSL interop keeper for the DSpark resume path."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

DURATION_SECONDS = 259200
stop_requested = False

def on_term(_signum, _frame):
    global stop_requested
    stop_requested = True

def proc_start_ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return int(raw[raw.rfind(")") + 2:].split()[19])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(run_dir, 0o700)
    pid = os.getpid()
    interop = os.environ.get("WSL_INTEROP", "")
    interop_stat = None
    if interop:
        try:
            st = os.stat(interop)
            interop_stat = {"device": st.st_dev, "inode": st.st_ino, "mode": oct(st.st_mode), "socket": __import__("stat").S_ISSOCK(st.st_mode)}
        except OSError as exc:
            interop_stat = {"error": f"{type(exc).__name__}: {exc}"}
    now = time.time()
    record = {
        "schema": "bounded_wsl_interop_keeper_v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": DURATION_SECONDS,
        "deadline_unix": now + DURATION_SECONDS,
        "pid": pid,
        "birth_ticks": proc_start_ticks(pid),
        "pgid": os.getpgid(pid),
        "ppid": os.getppid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "wsl_distro_name": os.environ.get("WSL_DISTRO_NAME"),
        "wsl_interop": interop,
        "wsl_interop_socket": interop_stat,
        "stop_file": str(run_dir / "STOP"),
        "stop_method": "verify boot_id + pid + birth_ticks, create STOP, SIGTERM exact PID",
        "status": "running",
    }
    metadata = run_dir / "keeper.json"
    tmp = run_dir / ".keeper.json.tmp"
    tmp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, metadata)
    os.chmod(metadata, 0o600)
    signal.signal(signal.SIGTERM, on_term)
    deadline = now + DURATION_SECONDS
    while not stop_requested and time.time() < deadline and not (run_dir / "STOP").exists():
        time.sleep(5)
    record["status"] = "stopped" if stop_requested or (run_dir / "STOP").exists() else "expired"
    record["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["stop_signal_received"] = stop_requested
    tmp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, metadata)
    os.chmod(metadata, 0o600)

if __name__ == "__main__":
    main()
