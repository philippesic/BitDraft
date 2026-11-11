import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, cast

from benchmarks.comparison import Arm, Comparison
from benchmarks.measurement import now, pause, port_number, ready
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, reference


def server_arguments(spec: Comparison, arm: Arm, port: int) -> list[str]:
    argv = [
        str(spec.binary.path),
        "-m",
        str(spec.target.path),
        "-ngl",
        "99",
        "-c",
        str(spec.context),
        "-np",
        "1",
        "-b",
        "512",
        "-ub",
        "512",
        "-t",
        "4",
        "-fa",
        "on",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--no-warmup",
        "--spec-type",
        "draft-dflash" if arm.drafter else "none",
        "-lv",
        "3",
    ]
    if arm.drafter:
        argv += [
            "-md",
            str(arm.drafter.path),
            "-ngld",
            "99",
            "--cache-type-k-draft",
            "f16",
            "--cache-type-v-draft",
            "f16",
            "--spec-draft-n-max",
            str(spec.depth),
            "--spec-draft-n-min",
            "0",
            "--spec-draft-p-min",
            "0",
            "--spec-draft-sampling",
            "greedy",
        ]
    return argv


@contextmanager
def session(spec: Comparison, arm: Arm, cell: Store) -> Iterator[int]:
    port = port_number()
    argv = server_arguments(spec, arm, port)
    cell.atomic(
        cell.root / "command.json",
        canonical(
            {"argv": argv, "binary": reference(spec.binary), "target": reference(spec.target)}
        ),
    )
    with (cell.root / "server.log").open("wb") as log:
        process = subprocess.Popen(
            argv,
            env=native_environment(spec.binary.path.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = now() + 180
            while not ready(port):
                if process.poll() is not None or now() >= deadline:
                    raise ValueError("native server startup failed; inspect server.log")
                pause(0.25)
            owner = ProcessIdentity.read(process.pid)
            cell.atomic(cell.root / "server-owner.json", canonical(cast(Record, asdict(owner))))
            cell.atomic(
                cell.root / "library-maps.txt", Path(f"/proc/{process.pid}/maps").read_bytes()
            )
            yield port
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            cell.atomic(
                cell.root / "release.json",
                canonical({"returncode": process.returncode, "reaped": True}),
            )
