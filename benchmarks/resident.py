import os
import signal
import sys
from pathlib import Path

from benchmarks.campaign_segments import initial_reference, prepare, retain, select
from benchmarks.comparison import Comparison
from benchmarks.measurement import deadline, now
from benchmarks.metrics import totals
from benchmarks.native import file_reference
from benchmarks.references import register
from benchmarks.reporting import controller_record, interruptions
from benchmarks.runner import compare_leased
from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.device import cuda_identity
from w1a1_eagle.identity import source_identity
from w1a1_eagle.lifecycle import Journal, Ownership
from w1a1_eagle.numerical.entry import preflight, run
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.supervisor import execute


def campaign(specification: Path, output: Path) -> Record:
    raw, template, comparison = prepare(specification)
    retention = record(template.raw.get("retention", {}))
    if integer(retention.get("states", 0)) < 4 or integer(retention.get("exports", 0)) < 3:
        raise ValueError("resident campaigns require retention of at least4 states and3 exports")
    if template.raw["device"] != "cuda":
        raise ValueError("resident native campaigns require CUDA training")
    if template.parent is None:
        raise ValueError("resident campaign requires a durable parent")
    initial = initial_reference(raw, comparison, specification.resolve().parent)
    if any(value["gpu"] != cuda_identity() for value in comparison.cached().values()):
        raise ValueError("saved controls require matching hardware/driver before training")
    code = source_identity(Path(__file__).resolve().parents[1])
    ownership = Ownership(output)
    try:
        store = Store(output)
        binding: Record = {"specification": file_reference(specification), "code_sha256": code}
        if (store.root / "binding.json").exists() and read(store.root / "binding.json") != binding:
            raise ValueError("campaign inputs/source changed; resume recorded source")
        store.atomic(store.root / "binding.json", canonical(binding))
        store.atomic(store.root / "controller.json", canonical(controller_record()))
        status_path = store.root / "status.json"
        if status_path.exists():
            status = read(status_path)
        else:
            admission = preflight(artifact(raw["intent"], specification.resolve().parent).path)
            store.atomic(store.root / "admission.json", canonical(admission))
            drafter = artifact(raw["initial_drafter"], specification.resolve().parent)
            status = {
                "status": "running",
                "mode": "resident",
                "segment": 1,
                "evaluation": 0,
                "phase": "train" if initial else "evaluate",
                "active_training": None,
                "latest_state": reference(template.parent),
                "latest_drafter": reference(drafter),
                "provisional_fast_leader": {
                    "state": reference(template.parent),
                    "drafter": reference(drafter),
                    "score": totals([record(row) for row in sequence(initial["rows"])])[
                        "acceptance"
                    ],
                    "metric": "acceptance",
                    "evidence": initial["result"],
                    "reference": initial["reference"],
                }
                if initial
                else None,
                "initial_evaluation": initial["reference"] if initial else None,
                "full_dev_incumbent": None,
                "model_update_seconds": 0.0,
                "completed_evaluations": [],
                "scope": "60-request DEV acceptance; no deployment or training-success claim",
            }
            store.atomic(status_path, canonical(status))
        if status["status"] == "complete":
            if read(store.root / "result.json") != status:
                raise ValueError("completed campaign status disagrees with result")
            return status
        inputs = Store(store.root / "inputs")
        while True:
            segment = integer(status["segment"])
            root = store.root / (str(segment).zfill(4) + "-training")
            parent = artifact(status["latest_state"], store.root)
            recipe = template.raw
            recipe.update(
                parent=reference(parent), mode=template.mode if segment == 1 else "resume"
            )
            if (root / "states/latest.json").exists():
                latest = read(root / "states/latest.json")
                recipe.update(
                    parent=file_reference(root / "states" / (string(latest["sha256"]) + ".bin")),
                    mode="resume",
                )
            path = inputs.root / (
                str(segment).zfill(4)
                + "-"
                + string(record(recipe["parent"])["sha256"])[:12]
                + ".json"
            )
            inputs.atomic(path, canonical(recipe))
            status.update(status="running", active_training=str(root))
            store.atomic(status_path, canonical(status))
            spent = sum(
                number(Journal(p).report()["cumulative_attempt_seconds"])
                for p in store.root.glob("[0-9][0-9][0-9][0-9]-training")
                if (p / "events.jsonl").exists()
            )
            result = execute(
                path,
                root,
                string(template.raw["authority_scope"]),
                worker_preflight=True,
                worker_module="benchmarks.resident",
                worker_arguments=(str(specification.resolve()), str(store.root)),
                maximum_seconds=template.budget.wall_seconds - spent,
            )
            status = read(status_path)
            if (
                result["operational"] != "terminal"
                or record(result["latest"])["outcome"] != "complete"
            ):
                raise ValueError("resident worker failed; preserve state and inspect owned attempt")
            worker = record(result["worker"])
            if worker["outcome"] == "journal-capacity":
                status["segment"] = segment + 1
                store.atomic(status_path, canonical(status))
                retain(store.root, status)
                continue
            if worker["outcome"] not in {"update-budget", "model-update-budget"}:
                raise ValueError(
                    "resident worker stopped before update guard: " + string(worker["outcome"])
                )
            status.update(status="complete", active_training=None, stop_reason=worker["outcome"])
            store.atomic(status_path, canonical(status))
            store.atomic(store.root / "result.json", canonical(status))
            return status
    except BaseException as error:
        failed_store = Store(output)
        if (output / "status.json").exists():
            failed_status = read(output / "status.json")
            failed_status["status"] = "failed"
            failed_store.atomic(output.resolve() / "status.json", canonical(failed_status))
        failed_store.atomic(
            output.resolve() / "failure.json",
            canonical(
                {
                    "error": str(error),
                    "recovery": "retain exact source/recipe/state; preserve incomplete native attempts",
                }
            ),
        )
        raise
    finally:
        ownership.close()


class Evaluations:
    def __init__(self, specification: Path, output: Path, training: Path, attempt: str) -> None:
        binding = read(output / "binding.json")
        ref = artifact(binding["specification"], output)
        ref.verify()
        if ref.path != specification.resolve() or binding["code_sha256"] != source_identity(
            Path(__file__).resolve().parents[1]
        ):
            raise ValueError("resident worker campaign binding differs")
        self.raw, self.template, self.comparison = prepare(specification)
        self.specification = specification
        self.store = Store(output)
        self.inputs = Store(output / "inputs")
        self.training = training
        self.attempt = attempt
        self.status = read(output / "status.json")
        self.next_limit = 0.0

    def save(self) -> None:
        self.store.atomic(self.store.root / "status.json", canonical(self.status))

    def finished(self, worker: TensorWorker) -> bool:
        return worker.updates >= self.template.budget.updates or (
            "model_update_seconds" in self.template.raw
            and worker.accounting.successful_update_seconds
            + worker.accounting.failed_update_seconds
            >= number(self.template.raw["model_update_seconds"])
        )

    def resumed(self) -> None:
        pauses = interruptions(self.store.root)
        self.store.atomic(
            self.store.root / "interruptions.json",
            canonical({"values": pauses, "budget_seconds": self.raw["maximum_pause_seconds"]}),
        )
        if any(
            row["reason"] in {None, "model-update-time"}
            and row["seconds"] is not None
            and number(row["seconds"]) > number(self.raw["maximum_pause_seconds"])
            for row in pauses
        ):
            raise ValueError("resident training interruption exceeded budget; recovery preserved")
        if "pending_selection" in self.status:
            self.status["provisional_fast_leader"] = self.status.pop("pending_selection")
            self.save()

    def start(self, worker: TensorWorker) -> None:
        current = (
            worker.accounting.successful_update_seconds + worker.accounting.failed_update_seconds
        )
        self.next_limit = number(
            self.status.get(
                "next_optimizer_evaluation",
                current + number(self.raw["optimizer_seconds_per_evaluation"]),
            )
        )
        self.status["next_optimizer_evaluation"] = self.next_limit
        self.save()
        if self.status["phase"] == "evaluate":
            publications = [
                row
                for row in Journal(self.training).events()
                if row.get("kind") == "boundary-publication"
            ]
            if publications:
                state = worker.checkpoint(
                    Store(self.training / "states"), self.template.state_bytes
                )
                exported = export_he4(
                    worker,
                    self.template.source,
                    self.template.vocabulary,
                    Store(self.training / "exports"),
                    self.template.state_bytes,
                    self.template.memory_bytes,
                )
                self.check(worker, state, exported.artifact, self.finished(worker))
            else:
                state = worker.checkpoint(
                    Store(self.training / "states"), self.template.state_bytes
                )
                self.check(
                    worker,
                    state,
                    artifact(self.status["latest_drafter"], self.store.root),
                    self.finished(worker),
                )

    def limit(self) -> float:
        return self.next_limit

    def protected(self, kind: str) -> tuple[str, ...]:
        return tuple(
            artifact(record(self.status[key])[kind], self.store.root).sha256
            for key in ("provisional_fast_leader", "full_dev_incumbent", "pending_selection")
            if self.status.get(key) is not None
        )

    def evaluate(
        self, worker: TensorWorker, state: StateRef, exported: StateRef, outcome: str
    ) -> None:
        self.status.update(
            latest_state=reference(state.artifact),
            latest_drafter=reference(exported.artifact),
            model_update_seconds=worker.accounting.successful_update_seconds
            + worker.accounting.failed_update_seconds,
        )
        if outcome in {"model-update-time", "update-budget", "model-update-budget"}:
            if (
                outcome in {"update-budget", "model-update-budget"}
                and self.status.get("last_evaluated_updates") == worker.updates
            ):
                self.save()
                return
            self.status["phase"] = "evaluate"
            self.save()
            self.check(
                worker,
                state,
                exported.artifact,
                outcome in {"update-budget", "model-update-budget"},
            )
        else:
            self.save()

    def check(
        self, worker: TensorWorker, state: StateRef, drafter: Artifact, final: bool = False
    ) -> None:
        evaluation = integer(self.status["evaluation"])
        prefix = str(evaluation).zfill(4)
        path = self.store.root / (prefix + "-evaluation")
        attempt = 0
        while path.exists() and not (path / "result.json").exists():
            attempt += 1
            path = self.store.root / (prefix + "-evaluation-retry-" + str(attempt))
        raw = read(artifact(self.raw["comparison"], self.specification.resolve().parent).path)
        raw["arms"] = [{"name": "candidate", "drafter": reference(drafter)}]
        leader = self.status["provisional_fast_leader"]
        if leader is not None:
            record(raw["references"])["prior-leader"] = record(leader)["reference"]
        input_path = self.inputs.root / (path.name + ".json")
        self.inputs.atomic(input_path, canonical(raw))
        began = now()
        try:
            spec = Comparison.load(input_path)
            spec.verify()
            with worker.offload(
                release_gradients=True, retain_frozen=True, publication=state
            ) as offload:
                self.status.update(phase="evaluate", offload=offload)
                self.save()
                with deadline(number(self.raw["maximum_pause_seconds"]), nested=True):
                    if (path / "result.json").exists():
                        if read(path / "specification.json") != raw:
                            raise ValueError("completed native attempt inputs differ")
                        result = read(path / "result.json", 64 * 1024 * 1024)
                    else:
                        result = compare_leased(input_path, path, spec, managed_group=True)
                evidence = file_reference(path / "result.json")
                selected = select(
                    result, leader, reference(state.artifact), reference(drafter), evidence
                )
                if selected is not leader:
                    registration = self.inputs.root / (path.name + "-reference.json")
                    self.inputs.atomic(
                        registration,
                        canonical(
                            {
                                "schema": 1,
                                "comparison": file_reference(input_path),
                                "result": evidence,
                                "arm": "candidate",
                                "rep": 0,
                            }
                        ),
                    )
                    record(selected)["reference"] = register(
                        registration, self.store.root / "references"
                    )["reference"]
            if final:
                boundaries = [
                    row
                    for row in Journal(self.training).events()
                    if row.get("kind") == "training-boundary" and row.get("attempt") == self.attempt
                ]
                final_began = number(boundaries[-1]["monotonic"]) if boundaries else began
                if now() - final_began > number(self.raw["maximum_pause_seconds"]):
                    raise ValueError("final training boundary exceeded budget; selection withheld")
                self.status["provisional_fast_leader"] = selected
                self.status.pop("pending_selection", None)
            else:
                self.status["pending_selection"] = selected
            self.status["last_evaluated_updates"] = worker.updates
            self.status["completed_evaluations"] = [
                *sequence(self.status["completed_evaluations"]),
                evidence,
            ]
        except (OSError, ValueError, KeyError) as error:
            self.inputs.atomic(
                self.inputs.root / (path.name + "-failure.json"),
                canonical(
                    {"error": str(error), "selection": "none; invalid/incomplete attempt preserved"}
                ),
            )
            if self.raw["evaluation_failure"] == "stop":
                raise
        self.next_limit = (
            worker.accounting.successful_update_seconds
            + worker.accounting.failed_update_seconds
            + number(self.raw["optimizer_seconds_per_evaluation"])
        )
        self.status.update(
            evaluation=evaluation + 1,
            phase="train",
            offload=None,
            next_optimizer_evaluation=self.next_limit,
            last_native_interruption_seconds=now() - began,
        )
        self.save()
        pauses = interruptions(self.store.root)
        self.store.atomic(
            self.store.root / "interruptions.json",
            canonical({"values": pauses, "budget_seconds": self.raw["maximum_pause_seconds"]}),
        )
        if any(
            row["seconds"] is not None
            and number(row["seconds"]) > number(self.raw["maximum_pause_seconds"])
            for row in pauses
        ):
            raise ValueError("resident training interruption exceeded budget; recovery preserved")


def main() -> int:
    if len(sys.argv) != 7:
        raise ValueError("private resident worker requires controller-bound arguments")
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]

    def timeout(_signal: int, _frame: object) -> None:
        os._exit(124)

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(
        signal.ITIMER_REAL, min(remaining, float(sys.argv[4])) if remaining else float(sys.argv[4])
    )
    root, attempt = Path(sys.argv[2]), sys.argv[3]
    try:
        run(
            Path(sys.argv[1]),
            root,
            attempt,
            Evaluations(Path(sys.argv[5]), Path(sys.argv[6]), root, attempt),
        )
        return 0
    except Exception as error:
        Journal(root).append(
            {
                "kind": "failure",
                "attempt": attempt,
                "cause": type(error).__name__,
                "detail": str(error)[:2000],
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
