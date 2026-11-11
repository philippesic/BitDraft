import json
import os
import shutil
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from w1a1_eagle.capacity import backing_available
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, number, record
from w1a1_eagle.data import Window
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import loaded_identity, source_identity
from w1a1_eagle.index import IndexReader
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, control
from w1a1_eagle.model import Geometry
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.processes import ProcessIdentity


def admission(intent: Intent, corpus: Corpus) -> Record:
    if intent.raw.get("target_sha256") is not None and corpus.target != intent.raw["target_sha256"]:
        raise ValueError("candidate intent and capture target identity differ")
    exposure = corpus.exposure(intent.budget.updates)
    policy = record(intent.raw["science_policy"])
    if integer(exposure["distinct_positions"]) < integer(policy["min_unique_positions"]):
        raise ValueError("supplied minimum distinct-data policy refuses launch")
    worst_mean = intent.budget.attempts * 15 / corpus.unique_positions
    exposure["maximum_attempt_label_exposures"] = intent.budget.attempts * 15
    exposure["maximum_attempt_mean_visits"] = worst_mean
    if worst_mean > number(policy["max_mean_visits"]):
        raise ValueError("supplied maximum data-reuse policy refuses launch")
    return {
        "kind": "admission",
        "intent": intent.sha256,
        "recipe": intent.recipe,
        "objective": intent.raw["objective"],
        "hypothesis": intent.raw["hypothesis"],
        "decision": intent.raw["decision"],
        "limiting_risk": intent.raw["risk"],
        "starting_identity": intent.parent.sha256 if intent.parent else intent.source.sha256,
        "starting_name": intent.parent.path.name if intent.parent else intent.source.path.name,
        "mode": intent.mode,
        "mechanism": "HE4 W1A4/A8/Q4_K fifteen-label weighted-CE updates",
        "exposure": exposure,
        "prior_use_warning": "Prior training is marked for "
        + str(exposure["historical_training_sequences"])
        + " sequences; prior use is unknown for "
        + str(exposure["unknown_prior_use_sequences"])
        + ". Visit estimates are additional to this run, not lifetime counts. Fresh initialization does not mean unused data.",
        "budget": intent.raw["budget"],
        "peak_new_bytes_upper_bound": intent.peak_new_bytes,
        "checkpoint_updates": intent.checkpoint_updates,
        "checkpoint_seconds": intent.checkpoint_seconds,
        "authority": "caller-provided scope; not established by this readiness report",
        "readiness": "CPU data-policy checks passed; model allocation and runtime still pending",
        "scientific": "only supplied data-admission policy checked; learning and usefulness unknown",
        "objective_health": "unknown: no benchmark implemented",
        "stop_action": "stop at complete update and publish recoverable state; supervisor bounds wall time",
        "evidence_scope": "CPU source path; native execution and performance unqualified",
    }


def preflight(path: Path, sealed: bool = False) -> Record:
    intent = Intent.load_sealed(path) if sealed else Intent.load(path)
    if source_identity(Path(__file__).resolve().parents[2]) != intent.raw["code_sha256"]:
        raise ValueError("source identity differs from intent")
    cfg = record(intent.raw["geometry"])
    corpus = read_corpus(intent, integer(cfg["hidden"]))
    intent.source.verify()
    intent.vocabulary.verify()
    return admission(intent, corpus)


def read_corpus(intent: Intent, hidden: int) -> Corpus:
    source = (
        IndexReader(intent.corpus, intent.index_spec, intent.memory_bytes).load()
        if intent.index_spec is not None
        else intent.corpus
    )
    return Corpus(source, intent.seed, intent.memory_bytes, hidden)


def run(intent_path: Path, root: Path, attempt: str) -> None:
    device = str(Intent.load_sealed(intent_path).raw["device"])
    with device_lease(device):
        run_leased(intent_path, root, attempt)


def run_leased(intent_path: Path, root: Path, attempt: str) -> None:
    intent = Intent.load_sealed(intent_path)
    code_root = Path(__file__).resolve().parents[2]
    if source_identity(code_root) != intent.raw["code_sha256"]:
        raise ValueError("source changed between controller admission and worker load")
    torch.set_num_threads(2)
    torch.manual_seed(intent.seed)
    geometry = record(intent.raw["geometry"])
    cfg = Geometry(**{key: integer(value) for key, value in geometry.items()})
    corpus = read_corpus(intent, cfg.hidden)
    journal = Journal(root)
    starts = [
        row
        for row in journal.events()
        if row.get("kind") == "started"
        and row.get("attempt") == attempt
        and row.get("intent") == intent.sha256
    ]
    if len(starts) != 1:
        raise ValueError("private worker requires one controller-created attempt")
    controller = record(starts[0]["controller"])
    if (
        controller.get("pid") != os.getppid()
        or not ProcessIdentity(
            integer(controller["pid"]), str(controller["boot"]), str(controller["birth"])
        ).alive()
    ):
        raise ValueError("controller process ownership mismatch")
    own_identity = ProcessIdentity.read(os.getpid())
    owners = [
        record(row["owner"])
        for row in journal.events()
        if row.get("kind") == "process" and row.get("attempt") == attempt
    ]
    if owners != [cast(Record, asdict(own_identity))]:
        raise ValueError("worker requires its durable controller-bound process identity")
    journal.append({**admission(intent, corpus), "attempt": attempt})
    model = load_he4(intent.source, intent.vocabulary, intent.memory_bytes, cfg)
    model.to(str(intent.raw["device"]))
    loaded_cache: dict[tuple[int, int, int, int, int], str] = {}
    initial_loaded = loaded_identity(loaded_cache)
    journal.append({"kind": "loaded", "attempt": attempt, "identity": initial_loaded})
    worker = TensorWorker(model, corpus, intent.recipe)
    states = Store(root / "states", intent.disk_reserve_bytes)
    if intent.parent:
        parent = Store(intent.parent.path.parent).describe(intent.parent)
        worker.restore(
            parent,
            intent.memory_bytes,
            "resume"
            if intent.mode == "resume"
            else "continue"
            if intent.mode == "continue"
            else "branch",
        )
    attempted_updates = [row for row in journal.events() if row.get("kind") == "update-attempt"]
    if attempted_updates and intent.mode == "resume":
        attempted = max(integer(row["attempts"]) for row in attempted_updates)
        extra = max(0, attempted - worker.progress.attempts)
        failures = [
            integer(record(row["progress"])["failed"])
            for row in journal.events()
            if row.get("kind") == "worker-failure"
        ]
        failed = max([worker.progress.failed, *failures])
        added_failed = min(extra, max(0, failed - worker.progress.failed))
        worker.progress.failed += added_failed
        worker.progress.discarded += extra - added_failed
        worker.progress.attempts += extra
    stopped = False
    stop_reason = ""

    def stop_signal(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    owner = ProcessIdentity.read(os.getpid())
    clock = Monotonic()
    last_checkpoint_updates = worker.updates
    last_checkpoint_seconds = clock.now()

    def stop() -> bool:
        nonlocal last_checkpoint_updates, last_checkpoint_seconds, stop_reason
        if journal.path.stat().st_size >= 56 * 1024 * 1024:
            stop_reason = "journal-capacity"
            return True
        if shutil.disk_usage(root).free - intent.state_bytes < intent.disk_reserve_bytes:
            stop_reason = "disk-reserve"
            return True
        if backing_available() - intent.state_bytes < intent.disk_reserve_bytes:
            stop_reason = "backing-reserve"
            return True
        if stopped:
            stop_reason = "signal-stop"
            return True
        if control(root, attempt, owner, 1):
            stop_reason = "owned-control-stop"
            return True
        if (
            worker.updates - last_checkpoint_updates >= intent.checkpoint_updates
            or clock.now() - last_checkpoint_seconds >= intent.checkpoint_seconds
        ):
            saved = worker.checkpoint(states, intent.state_bytes)
            journal.append(
                {
                    "kind": "checkpoint",
                    "attempt": attempt,
                    "updates": worker.updates,
                    "artifact": saved.artifact.sha256,
                }
            )
            last_checkpoint_updates = worker.updates
            last_checkpoint_seconds = clock.now()
        return False

    class RecordedSamples:
        def sample(self, cursor: int) -> tuple[Capture, Window]:
            journal.append(
                {
                    "kind": "update-attempt",
                    "attempt": attempt,
                    "attempts": worker.progress.attempts,
                    "cursor": cursor,
                }
            )
            capture, window = corpus.sample(cursor)
            journal.append(
                {
                    "kind": "exposure",
                    "attempt": attempt,
                    "attempts": worker.progress.attempts,
                    "cursor": cursor,
                    "trajectory": window.trajectory,
                    "anchor": window.anchor,
                }
            )
            return capture, window

    worker.samples = RecordedSamples()
    worker.checkpoint(states, intent.state_bytes)
    try:
        result = worker.run(clock, intent.budget, stop)
        if result == "stopped" and stop_reason:
            result = stop_reason
    except BaseException:
        journal.append(
            {
                "kind": "worker-failure",
                "attempt": attempt,
                "progress": cast(Record, asdict(worker.progress)),
            }
        )
        raise
    reference = worker.checkpoint(states, intent.state_bytes)
    exported = export_he4(
        worker,
        intent.source,
        intent.vocabulary,
        Store(root / "exports", intent.disk_reserve_bytes),
        intent.state_bytes,
        intent.memory_bytes,
    )
    journal.append(
        {
            "kind": "export",
            "attempt": attempt,
            "artifact": exported.artifact.sha256,
            "source": intent.source.sha256,
            "vocabulary": intent.vocabulary.sha256,
            "scope": "GGUF producer/reader conformance; native loading unqualified",
        }
    )
    if source_identity(code_root) != intent.raw["code_sha256"]:
        raise ValueError("executable source drifted during this attempt; qualification refused")
    final_loaded = loaded_identity(loaded_cache)
    final_files = record(final_loaded["loaded_code_files"])
    if any(
        final_files.get(path) != digest
        for path, digest in record(initial_loaded["loaded_code_files"]).items()
    ):
        raise ValueError("loaded dependency files drifted during this attempt")
    journal.append({"kind": "loaded-terminal", "attempt": attempt, "identity": final_loaded})
    journal.append(
        {
            "kind": "worker",
            "attempt": attempt,
            "outcome": result,
            "updates": worker.updates,
            "cursor": worker.cursor,
            "attempts": worker.progress.attempts,
            "state": reference.artifact.sha256,
            "export": exported.artifact.sha256,
            "activation": {
                "actual_optimizer_updates": worker.optimizer_updates,
                "training_device": model.vocabulary.device.type,
            },
            "scientific": "unknown: no development evaluation supplied",
            "objective": "unknown: no benchmark",
        }
    )


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] in {"--preflight", "--sealed-preflight"}:
        try:
            print(
                json.dumps(
                    preflight(Path(sys.argv[2]), sys.argv[1] == "--sealed-preflight"),
                    sort_keys=True,
                )
            )
            return 0
        except (ValueError, OSError, KeyError) as error:
            refusal: Record = {
                "outcome": "refused",
                "cause": type(error).__name__,
                "detail": str(error),
            }
            print(json.dumps(refusal))
            return 2
    if len(sys.argv) != 5:
        raise ValueError("private worker requires controller-bound intent, output root and attempt")

    def timeout(_signum: int, _frame: object) -> None:
        os._exit(124)

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, number(float(sys.argv[4])))
    root = Path(sys.argv[2])
    try:
        run(Path(sys.argv[1]), root, sys.argv[3])
        return 0
    except Exception as error:
        Journal(root).append(
            {
                "kind": "failure",
                "attempt": sys.argv[3],
                "cause": type(error).__name__,
                "detail": str(error)[:2000],
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
