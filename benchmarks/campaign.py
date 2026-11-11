from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.native import file_reference
from benchmarks.reporting import controller_record
from benchmarks.runner import compare
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, Ownership
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.supervisor import execute


def campaign(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) - {"pilot"}
        != {
            "schema",
            "intent",
            "comparison",
            "initial_drafter",
            "optimizer_seconds_per_evaluation",
            "evaluation_failure",
        }
        or raw["schema"] != "training-campaign-v1"
    ):
        raise ValueError("unsupported training campaign")
    if raw["evaluation_failure"] not in {"stop", "continue"}:
        raise ValueError("explicit evaluation failure policy required")
    pilot = raw.get("pilot", False)
    if type(pilot) is not bool:
        raise ValueError("pilot must be explicit boolean")
    cadence = number(raw["optimizer_seconds_per_evaluation"])
    if cadence <= 0:
        raise ValueError("positive optimizer-time evaluation cadence required")
    root = specification.resolve().parent
    intent_ref, comparison_ref = artifact(raw["intent"], root), artifact(raw["comparison"], root)
    initial_drafter = artifact(raw["initial_drafter"], root)
    for ref in (intent_ref, comparison_ref, initial_drafter):
        ref.verify()
    template = Intent.load(intent_ref.path)
    comparison = Comparison.load(comparison_ref.path)
    comparison.verify()
    if (
        comparison.tier != ("pilot" if pilot else "short")
        or comparison.role != "dev"
        or comparison.target.sha256 != template.raw.get("target_sha256")
    ):
        raise ValueError("campaign requires target-bound 120-request DEV comparison")
    if template.parent is None:
        raise ValueError("campaign requires an initial durable state and matching native export")
    code = source_identity(Path(__file__).resolve().parents[1])
    if template.raw["code_sha256"] != code:
        raise ValueError("campaign intent must bind the executing source")
    ownership = Ownership(output)
    try:
        store = Store(output)
        store.atomic(store.root / "controller.json", canonical(controller_record()))
        binding: Record = {"specification": file_reference(specification), "code_sha256": code}
        binding_path = store.root / "binding.json"
        if binding_path.exists() and read(binding_path) != binding:
            raise ValueError("campaign inputs/source changed; resume the recorded frozen source")
        store.atomic(binding_path, canonical(binding))
        inputs = Store(store.root / "inputs")
        status_path = store.root / "status.json"
        status = (
            read(status_path)
            if status_path.exists()
            else {
                "status": "running",
                "segment": 0,
                "phase": "evaluate",
                "latest_state": reference(template.parent),
                "latest_drafter": reference(initial_drafter),
                "provisional_fast_leader": None,
                "full_dev_incumbent": None,
                "model_update_seconds": 0.0,
                "completed_evaluations": [],
            }
        )
        if not status_path.exists():
            from w1a1_eagle.numerical.worker import checkpoint_accounting

            parent_state = Store(template.parent.path.parent).describe(template.parent)
            initial_accounting = checkpoint_accounting(parent_state, template.memory_bytes)
            if initial_accounting is not None:
                status["model_update_seconds"] = (
                    initial_accounting.successful_update_seconds
                    + initial_accounting.failed_update_seconds
                )
            status["scope"] = "pilot only" if pilot else "120-request DEV selection"
        while True:
            segment = integer(status["segment"])
            prefix = str(segment).zfill(4)
            parent = artifact(status["latest_state"], store.root)
            parent_ref = Store(parent.path.parent).describe(parent)
            if status["phase"] == "evaluate":
                evaluation_path = store.root / (prefix + "-evaluation")
                evaluation_input = inputs.root / (prefix + "-comparison.json")
                comparison_raw = read(comparison_ref.path)
                arms = sequence(comparison_raw["arms"])[:2]
                leader = status["provisional_fast_leader"]
                if leader is not None:
                    arms.append({"name": "prior-leader", "drafter": record(leader)["drafter"]})
                arms.append({"name": "candidate", "drafter": status["latest_drafter"]})
                comparison_raw["arms"] = arms
                if evaluation_input.exists() and read(evaluation_input) != comparison_raw:
                    raise ValueError("campaign evaluation input changed")
                inputs.atomic(evaluation_input, canonical(comparison_raw))
                try:
                    if (evaluation_path / "result.json").exists():
                        result = read(evaluation_path / "result.json", 64 * 1024 * 1024)
                        if read(evaluation_path / "specification.json") != comparison_raw:
                            raise ValueError("published campaign evaluation disagrees with inputs")
                    elif evaluation_path.exists():
                        raise ValueError(
                            "interrupted timing block retained; explicit whole-block rerun required"
                        )
                    else:
                        result = compare(evaluation_input, evaluation_path)
                    selected = select(
                        result,
                        status["provisional_fast_leader"],
                        status["latest_state"],
                        status["latest_drafter"],
                        file_reference(evaluation_path / "result.json"),
                    )
                    status["provisional_fast_leader"] = selected
                    status["completed_evaluations"] = [
                        *sequence(status["completed_evaluations"]),
                        file_reference(evaluation_path / "result.json"),
                    ]
                except (ValueError, OSError, KeyError) as error:
                    incident = {
                        "error": str(error),
                        "segment": segment,
                        "selection": "none; incomplete/invalid evaluation",
                    }
                    inputs.atomic(
                        inputs.root / (prefix + "-evaluation-failure.json"), canonical(incident)
                    )
                    if raw["evaluation_failure"] == "stop":
                        raise
                status["phase"] = "train"
                store.atomic(status_path, canonical(status))
            if parent_ref.updates >= template.budget.updates:
                status["status"] = "complete"
                store.atomic(status_path, canonical(status))
                store.atomic(store.root / "result.json", canonical(status))
                return status
            segment += 1
            prefix = str(segment).zfill(4)
            training_root = store.root / (prefix + "-training")
            intent_raw = template.raw
            intent_raw.update(
                {
                    "parent": reference(parent),
                    "mode": template.mode if segment == 1 else "resume",
                    "model_update_seconds": number(status["model_update_seconds"]) + cadence,
                }
            )
            if (training_root / "states/latest.json").exists():
                latest = read(training_root / "states/latest.json")
                path = training_root / "states" / (string(latest["sha256"]) + ".bin")
                intent_raw.update({"parent": file_reference(path), "mode": "resume"})
            training_input = inputs.root / (prefix + "-intent.json")
            if training_input.exists() and read(training_input) != intent_raw:
                training_input = inputs.root / (
                    prefix
                    + "-resume-"
                    + string(record(intent_raw["parent"])["sha256"])[:12]
                    + ".json"
                )
            inputs.atomic(training_input, canonical(intent_raw))
            status.update(
                {"status": "running", "active_training": str(training_root), "phase": "train"}
            )
            store.atomic(status_path, canonical(status))
            retain(store.root, status)
            prior = (
                Journal(training_root).report() if (training_root / "events.jsonl").exists() else {}
            )
            worker = record(prior["worker"]) if prior.get("worker") is not None else {}
            if (
                not worker
                or prior.get("operational") != "terminal"
                or worker.get("outcome") not in {"model-update-time", "update-budget"}
            ):
                report = execute(
                    training_input, training_root, string(template.raw["authority_scope"])
                )
                if (
                    report["operational"] != "terminal"
                    or record(report["latest"])["outcome"] != "complete"
                ):
                    raise ValueError(
                        "training segment failed; durable state and charged attempts preserved"
                    )
                worker = record(report["worker"])
            if integer(worker["updates"]) <= parent_ref.updates:
                raise ValueError(
                    "training segment made no progress; inspect its stop reason and budgets"
                )
            states, exports = training_root / "states", training_root / "exports"
            status.update(
                {
                    "segment": segment,
                    "phase": "evaluate",
                    "active_training": None,
                    "latest_state": file_reference(states / (string(worker["state"]) + ".bin")),
                    "latest_drafter": file_reference(exports / (string(worker["export"]) + ".bin")),
                }
            )
            accounting = record(worker["accounting"])
            status["model_update_seconds"] = number(
                accounting["successful_update_seconds"]
            ) + number(accounting["failed_update_seconds"])
            store.atomic(status_path, canonical(status))
    except BaseException as error:
        Store(output).atomic(
            output.resolve() / "failure.json",
            canonical(
                {
                    "error": str(error),
                    "recovery": "latest checkpoint remains separate from provisional selection; inspect status and attempt journals",
                }
            ),
        )
        raise
    finally:
        ownership.close()


def select(
    result: Record, leader: object, state: object, drafter: object, evidence: Record
) -> object:
    if (
        result.get("status") != "complete"
        or result.get("diagnostic", False) is True
        or result.get("sampled", False) is True
    ):
        raise ValueError("partial evaluations cannot select a checkpoint")
    summaries = record(result["summary"])
    if any(record(value)["correct"] is not True for value in summaries.values()):
        raise ValueError("invalid evaluation cannot select a checkpoint")
    candidate = record(summaries["candidate"])
    prior = record(summaries["prior-leader"]) if leader is not None else None
    if prior is not None and number(candidate["tokens_per_second"]) <= number(
        prior["tokens_per_second"]
    ):
        return leader
    return {
        "state": state,
        "drafter": drafter,
        "score": candidate["tokens_per_second"],
        "evidence": evidence,
        "status": "provisional DEV request leader; not a full-DEV incumbent or deployment qualification",
    }


def retain(root: Path, status: Record) -> None:
    recovery = artifact(status["latest_state"], root)
    protected = {recovery.path, artifact(status["latest_drafter"], root).path}
    for key in ("provisional_fast_leader", "full_dev_incumbent"):
        if status.get(key) is not None:
            selected = record(status[key])
            protected.update(artifact(selected[name], root).path for name in ("state", "drafter"))
    current = integer(status["segment"])
    for path in sorted(root.glob("[0-9][0-9][0-9][0-9]-training")):
        if int(path.name[:4]) >= current - 1:
            continue
        if Journal(path).report().get("operational") != "terminal":
            raise ValueError("campaign retention refuses active segment")
        for name in ("states", "exports"):
            Store(path / name).retire_closed(frozenset(protected), recovery)
