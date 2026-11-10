import json
import os
import shutil
import signal
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, cast

from w1a1_eagle.capacity import backing_available
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.config import cpu_environment, numerical_python
from w1a1_eagle.contracts import Record, integer, number, record, string
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, Ownership
from w1a1_eagle.processes import ProcessIdentity


def execute(
    intent_path: Path, root: Path, scope: str, emit: Callable[[Record], None] | None = None
) -> Record:
    intent = Intent.load(intent_path)
    if scope != intent.raw["authority_scope"]:
        raise ValueError(
            "execution scope must match the externally authorized task; evidence is not authority"
        )
    code_root = Path(__file__).resolve().parent.parent
    if source_identity(code_root) != intent.raw["code_sha256"]:
        raise ValueError("dirty/generated source identity differs from the admitted intent")
    peak = intent.peak_new_bytes
    if shutil.disk_usage(root.parent).free - peak < intent.disk_reserve_bytes:
        raise ValueError("observed disk headroom cannot cover state plus reserve")
    if min(backing_available(), intent.backing_available_bytes) < peak + intent.disk_reserve_bytes:
        raise ValueError("declared backing-volume capacity cannot cover state plus reserve")
    ownership = Ownership(root)
    try:
        if any(
            path.name
            not in {
                ".controller",
                "events.jsonl",
                "control.json",
                "states",
                "exports",
                "incidents",
                "sources",
                "intents",
            }
            for path in root.iterdir()
        ):
            raise ValueError("output root contains foreign state; choose an isolated task root")
        journal = Journal(root)
        journal.recover_tail()
        events = journal.events()
        starts = [row for row in events if row.get("kind") == "started"]
        terminals = [row for row in events if row.get("kind") == "terminal"]
        for started in starts:
            if any(row["attempt"] == started["attempt"] for row in terminals):
                continue
            owners = [
                record(row["owner"])
                for row in events
                if row.get("kind") == "process" and row["attempt"] == started["attempt"]
            ]
            for value in [record(started["controller"]), *owners]:
                if ProcessIdentity(
                    integer(value["pid"]), string(value["boot"]), string(value["birth"])
                ).alive():
                    raise ValueError(
                        "prior attempt still owns live work; refuse competing execution"
                    )
            terminal: Record = {
                "kind": "terminal",
                "attempt": started["attempt"],
                "outcome": "interrupted",
                "exit": None,
                "wall_seconds": started["wall_budget"],
                "activation": "unknown; interrupted attempt charged its entire reserved wall budget",
            }
            journal.append(terminal)
            terminals.append(terminal)
        initialized = (root / "states/latest.json").exists() or any(
            row.get("kind") == "update-attempt" for row in events
        )
        if starts and (
            (intent.mode == "fresh" and initialized)
            or any(row["recipe"] != intent.recipe for row in starts)
        ):
            raise ValueError(
                "initialized output root requires exact-recipe resume; branch uses a new root"
            )
        spent = sum(number(row["wall_seconds"]) for row in terminals)
        remaining = intent.budget.wall_seconds - spent
        if remaining <= 0:
            raise ValueError("cumulative wall budget exhausted, including failed attempts")
        frozen_root = freeze_source(
            code_root, root / "sources", str(intent.raw["code_sha256"]), intent.disk_reserve_bytes
        )
        sealed_intent = intent.seal(root / "intents")
        attempt = uuid.uuid4().hex
        controller = ProcessIdentity.read(os.getpid())
        journal.append(
            {
                "kind": "started",
                "attempt": attempt,
                "recipe": intent.recipe,
                "intent": intent.sha256,
                "source": intent.raw["code_sha256"],
                "source_snapshot": str(frozen_root),
                "intent_receipt": str(sealed_intent),
                "controller": cast(Record, asdict(controller)),
                "wall_budget": remaining,
            }
        )
        clock = Monotonic()
        began = clock.now()
        environment = cpu_environment()
        environment["PYTHONPATH"] = str(frozen_root)
        process: subprocess.Popen[bytes] | None = None
        owner: ProcessIdentity | None = None
        outcome = "failed"
        exit_code: int | None = None
        try:
            readiness = subprocess.run(
                [
                    numerical_python(code_root),
                    "-B",
                    "-m",
                    "w1a1_eagle.bootstrap",
                    str(remaining),
                    "--sealed-preflight",
                    str(sealed_intent),
                ],
                cwd=frozen_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=remaining,
                check=False,
            )
            if emit is not None:
                emit(record(cast(object, json.loads(readiness.stdout))))
            if readiness.returncode:
                journal.append(
                    {
                        "kind": "failure",
                        "attempt": attempt,
                        "cause": "preflight",
                        "detail": readiness.stdout[:32768],
                    }
                )
                outcome = "refused"
                exit_code = readiness.returncode
            else:
                runtime_remaining = remaining - (clock.now() - began)
                if runtime_remaining <= 0:
                    outcome = "wall-budget"
                else:
                    process = subprocess.Popen(
                        [
                            numerical_python(code_root),
                            "-B",
                            "-m",
                            "w1a1_eagle.bootstrap",
                            str(runtime_remaining),
                            str(sealed_intent),
                            str(root.resolve()),
                            attempt,
                            str(runtime_remaining),
                        ],
                        cwd=frozen_root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    owner = ProcessIdentity.read(process.pid)
                    journal.append(
                        {
                            "kind": "process",
                            "attempt": attempt,
                            "owner": cast(Record, asdict(owner)),
                        }
                    )
                    try:
                        exit_code = process.wait(timeout=runtime_remaining)
                        outcome = "complete" if exit_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        outcome = "wall-budget"
                        if owner.alive():
                            owner.signal_group(signal.SIGTERM)
                        try:
                            exit_code = process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            if owner.alive():
                                owner.signal_group(signal.SIGKILL)
                            exit_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            outcome = "wall-budget"
        except (OSError, ValueError) as error:
            journal.append(
                {
                    "kind": "failure",
                    "attempt": attempt,
                    "cause": type(error).__name__,
                    "detail": str(error)[:2000],
                }
            )
        finally:
            if process is not None and process.poll() is None:
                if owner is None:
                    owner = ProcessIdentity.read(process.pid)
                if owner.alive():
                    owner.signal_group(signal.SIGKILL)
                exit_code = process.wait(timeout=5)
            journal.append(
                {
                    "kind": "terminal",
                    "attempt": attempt,
                    "outcome": outcome,
                    "exit": exit_code,
                    "wall_seconds": max(0, clock.now() - began),
                    "activation": "inspect worker record; process completion alone is not activation evidence",
                }
            )
        return journal.report()
    finally:
        ownership.close()
