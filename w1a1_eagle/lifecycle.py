import fcntl
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, string
from w1a1_eagle.json_stream import invalid_constant, unique
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import canonical, identity, read


class Journal:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.path = self.root / "events.jsonl"
        self.cache: list[Record] = []
        self.length = -1
        self.previous = "root"

    def events(self) -> list[Record]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size == self.length:
            return list(self.cache)
        if self.path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("journal exceeds supported size; preserve and migrate")
        result, previous = self.decode(self.path.read_bytes())
        self.cache = result
        self.length = self.path.stat().st_size
        self.previous = previous
        return list(result)

    @staticmethod
    def decode(data: bytes) -> tuple[list[Record], str]:
        import json

        result: list[Record] = []
        previous = "root"
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("partial journal tail requires explicit preserved recovery")
            raw = record(
                cast(
                    object,
                    json.loads(line, object_pairs_hook=unique, parse_constant=invalid_constant),
                )
            )
            if (
                set(raw) != {"schema", "sequence", "previous", "payload", "sha256"}
                or type(raw["schema"]) is not int
                or raw["schema"] != 1
            ):
                raise ValueError("unsupported event schema")
            payload = record(raw["payload"])
            envelope: Record = {
                "schema": 1,
                "sequence": len(result),
                "previous": previous,
                "payload": payload,
            }
            if (
                integer(raw["sequence"]) != len(result)
                or raw["previous"] != previous
                or raw["sha256"] != identity(envelope)
            ):
                raise ValueError("journal chain mismatch")
            result.append(payload)
            previous = string(raw["sha256"])
        return result, previous

    def recover_tail(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink() or self.path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("unsupported journal ownership or size")
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        cut = raw.rfind(b"\n") + 1
        prefix, tail = raw[:cut], raw[cut:]
        if len(tail) > 4 * 1024 * 1024:
            raise ValueError(
                "partial event exceeds supported size; preserve for explicit migration"
            )
        events, _ = self.decode(prefix)
        terminal = {string(row["attempt"]) for row in events if row.get("kind") == "terminal"}
        for row in events:
            if row.get("kind") not in {"started", "process"} or row.get("attempt") in terminal:
                continue
            value = record(row["controller"] if row["kind"] == "started" else row["owner"])
            if ProcessIdentity(
                integer(value["pid"]), string(value["boot"]), string(value["birth"])
            ).alive():
                raise ValueError("cannot reconcile a live owner's journal tail")
        digest = hashlib.sha256(raw).hexdigest()
        incident = Store(self.root / "incidents")
        incident.atomic(
            incident.root / (digest + ".json"),
            canonical(
                {
                    "schema": 1,
                    "original_sha256": digest,
                    "original_bytes": len(raw),
                    "intact_prefix_bytes": cut,
                    "tail_hex": tail.hex(),
                    "reason": "torn final append; intact records validated; no complete record discarded",
                }
            ),
        )
        Store(self.root).atomic(self.path, prefix)
        self.length = -1
        self.events()
        self.append({"kind": "tail-recovered", "incident": digest, "discarded_complete_records": 0})

    def append(self, payload: Record) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab", buffering=0) as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            if self.path.stat().st_size != self.length:
                self.events()
            envelope: Record = {
                "schema": 1,
                "sequence": len(self.cache),
                "previous": self.previous,
                "payload": payload,
            }
            data = canonical({**envelope, "sha256": identity(envelope)}) + b"\n"
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("event exceeds supported record size")
            if self.path.stat().st_size + len(data) > 64 * 1024 * 1024:
                raise ValueError("journal capacity exhausted; preserve and migrate")
            if cast(int, output.write(data)) != len(data):
                raise OSError("short journal write")
            os.fsync(output.fileno())
            self.cache.append(payload)
            self.previous = identity(envelope)
            self.length = self.path.stat().st_size

    def report(self) -> Record:
        events = self.events()
        terminal = [row for row in events if row.get("kind") == "terminal"]
        started = [row for row in events if row.get("kind") == "started"]
        attempts = len(started)
        cost = sum(number(row["wall_seconds"]) for row in terminal)
        names = [string(row["attempt"]) for row in started]
        ended = [string(row["attempt"]) for row in terminal]
        if (
            len(set(names)) != len(names)
            or len(set(ended)) != len(ended)
            or set(ended) - set(names)
        ):
            raise ValueError("ambiguous attempt/terminal accounting")
        if any(
            number(row["wall_seconds"]) < 0
            or row.get("outcome")
            not in {"complete", "failed", "interrupted", "wall-budget", "refused"}
            for row in terminal
        ):
            raise ValueError("invalid terminal outcome or cost")
        current = started[-1].get("attempt") if started else None
        current_events = [row for row in events if row.get("attempt") == current]
        admissions = [row for row in current_events if row.get("kind") == "admission"]
        workers = [row for row in current_events if row.get("kind") == "worker"]
        failures = [row for row in current_events if row.get("kind") == "failure"]
        diagnostics = [row for row in current_events if row.get("kind") == "development-diagnostic"]
        return {
            "schema": 1,
            "attempts": attempts,
            "completed_attempts": len(terminal),
            "cumulative_attempt_seconds": cost,
            "latest": terminal[-1] if terminal else None,
            "admission": admissions[-1] if admissions else None,
            "worker": workers[-1] if workers else None,
            "failure": failures[-1] if failures else None,
            "operational": "terminal" if terminal and len(terminal) == attempts else "incomplete",
            "scientific": "DEV teacher-forced diagnostics measured; native acceptance and usefulness remain separate"
            if diagnostics
            else "unknown: no supplied development evidence was evaluated",
            "development_diagnostic": diagnostics[-1] if diagnostics else None,
            "objective": "unknown: no benchmark result exists",
            "validity": "journal integrity verified; numerical/serving validity is separate",
            "selection": "unselected; no selection policy or evaluator is installed",
        }


class Ownership:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / ".controller").open("ab")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise

    def close(self) -> None:
        self.file.close()


def control(root: Path, attempt: str, owner: ProcessIdentity, sequence_number: int) -> bool:
    path = root / "control.json"
    if not path.exists():
        return False
    raw = read(path, 4096)
    if set(raw) != {"schema", "attempt", "owner", "sequence", "action"}:
        raise ValueError("malformed control message")
    if (
        type(raw["schema"]) is not int
        or raw["schema"] != 1
        or raw["attempt"] != attempt
        or raw["owner"] != cast(Record, asdict(owner))
    ):
        raise ValueError("foreign or stale control message")
    if integer(raw["sequence"]) != sequence_number or raw["action"] != "stop":
        raise ValueError("replayed or unsupported control message")
    journal = Journal(root)
    if any(
        row.get("kind") == "control"
        and row.get("attempt") == attempt
        and row.get("sequence") == sequence_number
        for row in journal.events()
    ):
        raise ValueError("replayed control message")
    journal.append(
        {"kind": "control", "attempt": attempt, "sequence": sequence_number, "action": "stop"}
    )
    return True
