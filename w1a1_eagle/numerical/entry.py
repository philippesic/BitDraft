import json
import os
import shutil
import signal
import sys
from pathlib import Path

import torch

from w1a1_eagle.capacity import backing_available
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, number, record
from w1a1_eagle.data import Window
from w1a1_eagle.identity import loaded_identity, source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, control
from w1a1_eagle.model import Geometry
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.processes import ProcessIdentity


def admission(intent: Intent, corpus: Corpus) -> Record:
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
        "mode": intent.mode,
        "mechanism": "HE4 W1A4/A8/Q4_K fifteen-label weighted-CE updates",
        "exposure": exposure,
        "budget": intent.raw["budget"],
        "authority": "caller-provided scope; not established by this readiness report",
        "readiness": "CPU data-policy checks passed; model allocation and runtime still pending",
        "scientific": "only supplied data-admission policy checked; learning and usefulness unknown",
        "objective_health": "unknown: no benchmark implemented",
        "stop_action": "stop at complete update and publish recoverable state; supervisor bounds wall time",
        "evidence_scope": "CPU source path; native execution and performance unqualified",
    }


def preflight(path: Path) -> Record:
    intent = Intent.load(path)
    if source_identity(Path(__file__).resolve().parents[2]) != intent.raw["code_sha256"]:
        raise ValueError("source identity differs from intent")
    cfg = record(intent.raw["geometry"])
    corpus = Corpus(intent.corpus, intent.seed, intent.memory_bytes, integer(cfg["hidden"]))
    intent.source.verify()
    intent.vocabulary.verify()
    return admission(intent, corpus)


def run(intent_path: Path, root: Path, attempt: str) -> None:
    intent = Intent.load(intent_path)
    code_root = Path(__file__).resolve().parents[2]
    if source_identity(code_root) != intent.raw["code_sha256"]:
        raise ValueError("source changed between controller admission and worker load")
    torch.set_num_threads(2)
    torch.manual_seed(intent.seed)
    geometry = record(intent.raw["geometry"])
    cfg = Geometry(**{key: integer(value) for key, value in geometry.items()})
    corpus = Corpus(intent.corpus, intent.seed, intent.memory_bytes, cfg.hidden)
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
    journal.append({**admission(intent, corpus), "attempt": attempt})
    model = load_he4(intent.source, intent.vocabulary, intent.memory_bytes, cfg)
    journal.append({"kind": "loaded", "attempt": attempt, "identity": loaded_identity()})
    worker = TensorWorker(model, corpus, intent.recipe)
    states = Store(root / "states", intent.disk_reserve_bytes)
    if intent.parent:
        parent = Store(intent.parent.path.parent).describe(intent.parent)
        worker.restore(
            parent, intent.memory_bytes, "resume" if intent.mode == "resume" else "branch"
        )
    exposures = [row for row in journal.events() if row.get("kind") == "exposure"]
    if exposures and intent.mode == "resume":
        attempted = max(integer(row["attempts"]) for row in exposures)
        extra = max(0, attempted - worker.progress.attempts)
        worker.progress.discarded += extra
        worker.progress.attempts += extra
    stopped = False

    def stop_signal(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    owner = ProcessIdentity.read(os.getpid())

    def stop() -> bool:
        if shutil.disk_usage(root).free - intent.state_bytes < intent.disk_reserve_bytes:
            return True
        if backing_available() - intent.state_bytes < intent.disk_reserve_bytes:
            return True
        return stopped or control(root, attempt, owner, 1)

    class RecordedSamples:
        def sample(self, cursor: int) -> tuple[Capture, Window]:
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
    result = worker.run(Monotonic(), intent.budget, stop)
    reference = worker.checkpoint(states, intent.state_bytes)
    journal.append(
        {
            "kind": "worker",
            "attempt": attempt,
            "outcome": result,
            "updates": worker.updates,
            "cursor": worker.cursor,
            "attempts": worker.progress.attempts,
            "state": reference.artifact.sha256,
            "activation": {
                "actual_optimizer_updates": worker.optimizer_updates,
                "training_device": "cpu",
            },
            "scientific": "unknown: no development evaluation supplied",
            "objective": "unknown: no benchmark",
        }
    )


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--preflight":
        try:
            print(json.dumps(preflight(Path(sys.argv[2])), sort_keys=True))
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
