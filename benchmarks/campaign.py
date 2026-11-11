from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.native import file_reference
from benchmarks.references import register
from benchmarks.reporting import controller_record, interruptions
from benchmarks.runner import compare
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, Ownership
from w1a1_eagle.records import artifact, canonical, read, reference
from w1a1_eagle.supervisor import execute


def prepare(specification: Path) -> tuple[Record, Intent, Comparison]:
    raw = read(specification)
    if set(raw) - {"pilot", "maximum_pause_seconds"} != {
        "schema",
        "intent",
        "comparison",
        "initial_drafter",
        "optimizer_seconds_per_evaluation",
        "evaluation_failure",
    } or raw["schema"] not in {"training-campaign-v1", "training-campaign-v2"}:
        raise ValueError("unsupported training campaign")
    if raw["evaluation_failure"] not in {"stop", "continue"}:
        raise ValueError("explicit evaluation failure policy required")
    pilot = raw.get("pilot", False)
    if type(pilot) is not bool:
        raise ValueError("pilot must be explicit boolean")
    if raw["schema"] == "training-campaign-v1" and not pilot:
        raise ValueError("legacy campaigns are pilot-only; use candidate-only training-campaign-v2")
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
        comparison.tier
        != ("pilot" if pilot and raw["schema"] == "training-campaign-v1" else "screen")
        or comparison.role != "dev"
        or comparison.target.sha256 != template.raw.get("target_sha256")
    ):
        raise ValueError("campaign requires target-bound 60-request DEV screen")
    if raw["schema"] == "training-campaign-v2" and (
        set(dict(comparison.references)) != {"target", "baseline"}
        or len(comparison.arms) != 1
        or comparison.arms[0].name != "candidate"
        or comparison.repeats != 1
        or comparison.diagnostic
        or not 0 < number(raw.get("maximum_pause_seconds")) <= 120
        or not pilot
        and not 1800 <= cadence <= 3600
    ):
        raise ValueError(
            "candidate-only campaign requires saved controls, <=120s pause and 30–60min cadence"
        )
    if template.parent is None:
        raise ValueError("campaign requires an initial durable state and matching native export")
    if raw["schema"] == "training-campaign-v2":
        initial_arm = comparison.arms[0].drafter
        if initial_arm is None or initial_arm.sha256 != initial_drafter.sha256:
            raise ValueError("initial candidate and campaign drafter disagree")
        verify_initial_export(template.parent, initial_drafter)
    code = source_identity(Path(__file__).resolve().parents[1])
    if template.raw["code_sha256"] != code:
        raise ValueError("campaign intent must bind the executing source")
    return raw, template, comparison


def verify_initial_export(parent: Artifact, drafter: Artifact) -> None:
    state = Store(parent.path.parent).describe(parent)
    exported = Store(drafter.path.parent).describe(drafter)
    producer = drafter.path.parent.parent
    journal = Journal(producer)
    report = journal.report()
    worker = record(report["worker"]) if report.get("worker") is not None else {}
    if (
        report.get("operational") != "terminal"
        or not worker
        or worker.get("export") != drafter.sha256
        or worker.get("updates") != state.updates
        or exported.updates != state.updates
    ):
        raise ValueError("initial export lacks a complete matching producer/state binding")
    if worker["state"] != parent.sha256:
        attempt = worker["attempt"]
        start = next(
            row
            for row in journal.events()
            if row.get("kind") == "started" and row["attempt"] == attempt
        )
        recipe = Intent.load_sealed(Path(string(start["intent_receipt"])))
        if (
            recipe.parent is None
            or recipe.parent.sha256 != parent.sha256
            or any(
                row.get("kind") == "update-attempt" and row["attempt"] == attempt
                for row in journal.events()
            )
        ):
            raise ValueError(
                "initial export is neither the parent nor its verified zero-update export"
            )


def preflight(specification: Path) -> Record:
    raw, template, comparison = prepare(specification)
    from w1a1_eagle.numerical.entry import preflight as training_preflight

    return {
        "training": training_preflight(artifact(raw["intent"], specification.parent).path),
        "candidate_requests": len(comparison.tasks),
        "references": [name for name, _ in comparison.references],
        "selection": "acceptance"
        if raw["schema"] == "training-campaign-v2"
        else "legacy pilot throughput",
        "maximum_pause_seconds": raw.get("maximum_pause_seconds"),
        "scope": "CPU admission; runtime budget requires measured pause/export/eval/resume",
    }


def campaign(specification: Path, output: Path) -> Record:
    raw, template, comparison = prepare(specification)
    root = specification.resolve().parent
    intent_ref, comparison_ref = artifact(raw["intent"], root), artifact(raw["comparison"], root)
    initial_drafter = artifact(raw["initial_drafter"], root)
    pilot = raw.get("pilot", False) is True
    cached = raw["schema"] == "training-campaign-v2"
    cadence = number(raw["optimizer_seconds_per_evaluation"])
    code = source_identity(Path(__file__).resolve().parents[1])
    admission: Record | None = None
    if not (output / "binding.json").exists():
        from w1a1_eagle.numerical.entry import preflight

        admission = preflight(intent_ref.path)
    if template.parent is None:
        raise ValueError("initial durable parent required")
    ownership = Ownership(output)
    try:
        store = Store(output)
        if admission is not None:
            store.atomic(store.root / "admission.json", canonical(admission))
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
            status["scope"] = "pilot only" if pilot else "60-request DEV acceptance selection"
            store.atomic(status_path, canonical(status))
        if status.get("status") == "complete":
            if read(store.root / "result.json") != status:
                raise ValueError("completed campaign status disagrees with its result")
            return status
        while True:
            segment = integer(status["segment"])
            prefix = str(segment).zfill(4)
            parent = artifact(status["latest_state"], store.root)
            parent_ref = Store(parent.path.parent).describe(parent)
            if status["phase"] == "evaluate":
                evaluation_path = store.root / (prefix + "-evaluation")
                evaluation_input = inputs.root / (prefix + "-comparison.json")
                comparison_raw = read(comparison_ref.path)
                arms = [] if cached else sequence(comparison_raw["arms"])[:2]
                leader = status["provisional_fast_leader"]
                if leader is not None and cached:
                    record(comparison_raw["references"])["prior-leader"] = record(leader)[
                        "reference"
                    ]
                elif leader is not None:
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
                    if cached and number(result["elapsed_seconds"]) > number(
                        raw["maximum_pause_seconds"]
                    ):
                        raise ValueError(
                            "candidate evaluation alone exceeded the complete interruption budget"
                        )
                    selected = select(
                        result,
                        status["provisional_fast_leader"],
                        status["latest_state"],
                        status["latest_drafter"],
                        file_reference(evaluation_path / "result.json"),
                    )
                    if cached and selected is not leader:
                        registration: Record = {
                            "schema": 1,
                            "comparison": file_reference(evaluation_input),
                            "result": file_reference(evaluation_path / "result.json"),
                            "arm": "candidate",
                            "rep": 0,
                        }
                        registration_path = inputs.root / (prefix + "-reference.json")
                        inputs.atomic(registration_path, canonical(registration))
                        record(selected)["reference"] = register(
                            registration_path, store.root / "references"
                        )["reference"]
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
                    training_input,
                    training_root,
                    string(template.raw["authority_scope"]),
                    worker_preflight=cached,
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
            if cached:
                pauses = interruptions(store.root)
                store.atomic(
                    store.root / "interruptions.json",
                    canonical({"values": pauses, "budget_seconds": raw["maximum_pause_seconds"]}),
                )
                if any(
                    value["seconds"] is not None
                    and number(value["seconds"]) > number(raw["maximum_pause_seconds"])
                    for value in pauses
                ):
                    raise ValueError(
                        "complete training interruption exceeded its budget; recovery preserved"
                    )
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
    metric = "acceptance" if result.get("historical_references") else "tokens_per_second"
    if prior is not None and number(candidate[metric]) <= number(prior[metric]):
        return leader
    return {
        "state": state,
        "drafter": drafter,
        "score": candidate[metric],
        "metric": metric,
        "evidence": evidence,
        "status": "provisional DEV leader; acceptance is a progress diagnostic, not deployment qualification",
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
