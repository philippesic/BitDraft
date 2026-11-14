import os
import subprocess
import sys
from pathlib import Path

from benchmarks.measurement import now, pause, port_number, ready, request
from benchmarks.native import file_reference
from benchmarks.runner import prompt_tokens
from benchmarks.runtime_policy import arguments, payload
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import kernel_capture_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.records import artifact, canonical, read


def dev_task(workload: Record, identifier: str) -> Record:
    tasks = [record(value) for value in sequence(workload["tasks"])]
    selected = [task for task in tasks if task.get("id") == identifier]
    if len(selected) != 1 or selected[0].get("role") != "dev":
        raise ValueError("one declared DEV request required")
    return selected[0]


def capture_kernel_activations(specification: Path, output: Path) -> Record:
    raw = read(specification)
    fields = {"schema", "build", "target", "drafter", "workload", "task", "slots", "context", "cap"}
    if set(raw) != fields or raw["schema"] != "kernel-capture-v1":
        raise ValueError("unsupported kernel capture specification")
    inputs = {
        key: artifact(raw[key], specification.parent)
        for key in ("build", "target", "drafter", "workload")
    }
    for value in inputs.values():
        value.verify()
    build = read(inputs["build"].path)
    helper = artifact(build["capture_library"], inputs["build"].path.parent)
    helper.verify()
    runtime = artifact(build["runtime"], inputs["build"].path.parent)
    runtime.verify()
    native = read(runtime.path)
    files = [artifact(value, runtime.path.parent) for value in sequence(native["artifacts"])]
    for value in files:
        value.verify()
    servers = [value for value in files if value.path.name == "llama-server"]
    if len(servers) != 1:
        raise ValueError("one native server required")
    task = dev_task(read(inputs["workload"].path), string(raw["task"]))
    slots = [integer(value) for value in sequence(raw["slots"])]
    context, cap = integer(raw["context"]), integer(raw["cap"])
    if (
        not slots
        or len(set(slots)) != len(slots)
        or any(value not in {2, 4, 8, 16} for value in slots)
    ):
        raise ValueError("distinct supported slot counts required")
    if not 1024 <= context <= 32768 or not 32 <= cap <= 256:
        raise ValueError("invalid capture context or output bound")
    if output.exists():
        raise ValueError("capture output must be new")
    owner = Store(output)
    tensors: Record = {}
    with device_lease("cuda"):
        began = now()
        for count in slots:
            cell = Store(output / str(count))
            port = port_number()
            command = arguments(
                servers[0].path,
                inputs["target"].path,
                inputs["drafter"].path,
                context,
                count - 1,
                port,
            )
            environment = kernel_capture_environment(
                servers[0].path.parent, helper.path, cell.root, count
            )
            cell.atomic(
                cell.root / "command.json",
                canonical(
                    {
                        "argv": command,
                        "capture_library": build["capture_library"],
                        "decode_consistent": 32,
                        "cuda_graphs": False,
                    }
                ),
            )
            with (cell.root / "server.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "benchmarks.server_child",
                        str(os.getpid()),
                        *command,
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    deadline = now() + 180
                    while not ready(port):
                        if process.poll() is not None or now() >= deadline:
                            raise ValueError("capture server startup failed; inspect server.log")
                        pause(0.25)
                    tokens = prompt_tokens(port, task, cap, context)
                    cell.atomic(cell.root / "armed", b"DEV request begins")
                    response = request(
                        port, "completion", payload(tokens, cap, 0.0, 1.0, 0, 0.0, 42)
                    )
                    cell.atomic(cell.root / "completion.json", canonical(response))
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
            rows = (cell.root / "capture.tsv").read_text().splitlines()
            if len(rows) != 12:
                raise ValueError("capture did not cover all six gate/down pairs")
            for index, line in enumerate(rows):
                layer, down = divmod(index, 2)
                group = "down" if down else "gate"
                key = f"blk.{layer}.ffn_{group}.{count}"
                columns = line.split("\t")
                if len(columns) != 3 or columns[0] != key or int(columns[2]) != (8 if down else 4):
                    raise ValueError("capture order or activation width differs")
                path = cell.root / (key + ".f32")
                if path.stat().st_size != count * (9216 if down else 2560) * 4:
                    raise ValueError("captured tensor size differs")
                tensors[key] = file_reference(path)
        gpu_seconds = now() - began
    result: Record = {
        "gpu_seconds": gpu_seconds,
        "schema": "kernel-activations-v1",
        "role": "dev",
        "specification": raw,
        "task": string(task["id"]),
        "tensors": tensors,
        "artifact_disposition": "Retain captured DEV inputs, source identities, request outputs and logs as benchmark inputs.",
    }
    owner.atomic(output / "result.json", canonical(result))
    return result
