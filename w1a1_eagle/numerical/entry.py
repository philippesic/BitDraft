import json
import os
import shutil
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol, cast

import torch

from w1a1_eagle.capacity import backing_available
from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, number, record
from w1a1_eagle.data import Supervision
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import loaded_identity, source_identity
from w1a1_eagle.index import IndexReader
from w1a1_eagle.intent import Intent
from w1a1_eagle.lifecycle import Journal, control
from w1a1_eagle.model import HE4, Geometry, bound_latents
from w1a1_eagle.numerical.calibration import calibrate, start_check
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.diagnostics import Diagnostics
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.mixture import mixture
from w1a1_eagle.numerical.weights import exported_tables, fusion_identity, load_he4
from w1a1_eagle.numerical.worker import TensorWorker, checkpoint_accounting
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact
from w1a1_eagle.round_states import RoundStateWeighting, load_map
from w1a1_eagle.training import MAXIMUM_SLOTS


class Boundary(Protocol):
    def start(self, worker: TensorWorker) -> None: ...
    def limit(self) -> float: ...
    def protected(self, kind: str) -> tuple[str, ...]: ...
    def evaluate(
        self, worker: TensorWorker, state: StateRef, exported: StateRef, outcome: str
    ) -> None: ...
    def resumed(self) -> None: ...


def round_state_weighting(intent: Intent, corpus: Corpus) -> RoundStateWeighting | None:
    setting = intent.round_states
    if setting is None:
        return None
    states = load_map(setting.map)
    if (states.corpus_sha256, states.corpus_bytes) != (intent.corpus.sha256, intent.corpus.size):
        raise ValueError("round-state map was built for another corpus than the intent's")
    if states.target != corpus.target:
        raise ValueError("round-state map and corpus target identities differ")
    return RoundStateWeighting(states, setting.weight)


def round_state_admission(weighting: RoundStateWeighting, corpus: Corpus, updates: int) -> Record:
    windows = starts = 0
    for cursor in range(updates):
        row, anchors = corpus.locate(cursor)
        for anchor in anchors:
            if not weighting.states.covers(row.capture.identity, anchor):
                raise ValueError(
                    "round-state map does not cover the window "
                    + row.capture.identity
                    + "@"
                    + str(anchor)
                    + " planned at cursor "
                    + str(cursor)
                )
            windows += 1
            starts += anchor in weighting.states.starts.get(row.capture.identity, frozenset())
    return {
        **weighting.states.summary(),
        "weight": weighting.weight,
        "planned_cursors": updates,
        "planned_windows": windows,
        "planned_round_starts": starts,
        "planned_share": starts / windows if windows else None,
        "normalization": "per update, window weights sum to one; a round-start window has"
        " weight times the weight of any other window in its update",
    }


def admission(intent: Intent, corpus: Corpus) -> Record:
    if intent.raw.get("target_sha256") is not None and corpus.target != intent.raw["target_sha256"]:
        raise ValueError("candidate intent and capture target identity differ")
    starting_updates = 0
    prior_labels = 0
    known_prior = False
    if intent.parent is not None:
        parent = Store(intent.parent.path.parent).describe(intent.parent)
        if intent.mode == "resume" and parent.config != intent.recipe:
            raise ValueError(
                "resume requires the recorded recipe; use an explicit continuation for changed inputs/source"
            )
        starting_updates = parent.updates
        accounting = checkpoint_accounting(parent, intent.memory_bytes)
        if (
            intent.mode == "resume"
            and accounting is not None
            and accounting.dataset != intent.corpus.sha256
        ):
            raise ValueError("resume dataset identity differs from the durable checkpoint")
        if accounting is not None and accounting.dataset == intent.corpus.sha256:
            prior_labels = accounting.dataset_labels
            known_prior = True
        elif intent.mode == "resume":
            prior_labels = starting_updates * 15
    schedule = intent.optimization.schedule
    if starting_updates < schedule.start_update:
        raise ValueError("training starts before its step-size schedule")
    if schedule.final_update is not None and intent.budget.updates > schedule.final_update:
        raise ValueError("update budget extends beyond the step-size schedule's final update")
    planned_updates = max(0, intent.budget.updates - starting_updates)
    planned_attempts = max(0, intent.budget.attempts - starting_updates)
    exposure = corpus.exposure(planned_updates)
    exposure["starting_updates"] = starting_updates
    exposure["known_prior_dataset_labels"] = prior_labels
    exposure["prior_dataset_exposure_known"] = int(known_prior)
    exposure["dataset_label_exposures_upper_bound"] = (
        prior_labels + planned_updates * corpus.labels_per_update
    )
    policy = record(intent.raw["science_policy"])
    if integer(exposure["distinct_positions"]) < integer(policy["min_unique_positions"]):
        raise ValueError("supplied minimum distinct-data policy refuses launch")
    attempt_labels = prior_labels + planned_attempts * corpus.labels_per_update
    worst_mean = attempt_labels / corpus.unique_positions
    exposure["maximum_attempt_label_exposures"] = attempt_labels
    exposure["maximum_attempt_mean_visits"] = worst_mean
    if worst_mean > number(policy["max_mean_visits"]):
        raise ValueError("supplied maximum data-reuse policy refuses launch")
    weighting = round_state_weighting(intent, corpus)
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
        "cpu_threads": intent.cpu_threads,
        "mechanism": "HE4 W1A4/A8/Q4_K weighted CE using each capture’s declared supervision policy",
        "grouping": cast(Record, asdict(intent.grouping)),
        "optimization": cast(Record, asdict(intent.optimization)),
        "blocks": intent.blocks.encode(),
        **({"trained_tables": True} if intent.trained_tables else {}),
        **({} if intent.features.default else {"features": intent.features.encode()}),
        **({"arithmetic": intent.arithmetic} if intent.arithmetic != "reference" else {}),
        **(
            {"calibration": {"sample": intent.calibration.encode()}}
            if intent.calibration is not None
            else {}
        ),
        **(
            {"gradual_precision": intent.gradual_precision.encode()}
            if intent.gradual_precision is not None
            else {}
        ),
        **({"telemetry": True} if intent.telemetry else {}),
        **(
            {"round_states": round_state_admission(weighting, corpus, intent.budget.updates)}
            if weighting is not None
            else {}
        ),
        "exposure": exposure,
        "prior_use_warning": "Prior training is marked for "
        + str(exposure["historical_training_sequences"])
        + " sequences; prior use is unknown for "
        + str(exposure["unknown_prior_use_sequences"])
        + ". Visit estimates are additional to this run, not lifetime counts. Fresh initialization does not mean unused data.",
        "budget": intent.raw["budget"],
        "model_update_seconds": intent.raw.get("model_update_seconds"),
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
    corpus = read_corpus(intent)
    if "development" in intent.raw:
        Diagnostics(artifact(intent.raw["development"], intent.base), corpus, intent.memory_bytes)
    intent.source.verify()
    intent.vocabulary.verify()
    if intent.initialization is not None:
        intent.initialization.verify()
    if intent.frozen_baseline is not None:
        intent.frozen_baseline.verify()
    return admission(intent, corpus)


def geometry(intent: Intent) -> Geometry:
    return Geometry(
        **{key: integer(value) for key, value in record(intent.raw["geometry"]).items()}
    )


def read_corpus(intent: Intent) -> Corpus:
    cfg = geometry(intent)
    source = (
        mixture(intent.corpus, intent.seed, intent.memory_bytes, cfg.hidden, intent.fused_input)
        if intent.mixture_input
        else IndexReader(intent.corpus, intent.index_spec, intent.memory_bytes).load()
        if intent.index_spec is not None
        else intent.corpus
    )
    return Corpus(
        source,
        intent.seed,
        intent.memory_bytes,
        cfg.hidden,
        intent.sampling,
        intent.grouping,
        cfg,
        intent.blocks,
        intent.fused_input,
    )


def calibration_slots(intent: Intent) -> int:
    fixed = intent.blocks.fixed
    return MAXIMUM_SLOTS if fixed is None else fixed


def initialize(
    intent: Intent, corpus: Corpus, model: HE4, publish: Callable[[Record], None]
) -> None:
    gradual = intent.gradual_precision
    if intent.calibration is not None:
        publish(
            {
                "kind": "calibration",
                **calibrate(
                    model,
                    corpus,
                    intent.calibration,
                    intent.features,
                    calibration_slots(intent),
                    intent.memory_bytes,
                    gradual is not None,
                ),
            }
        )
    if gradual is not None:
        check = start_check(
            model,
            corpus,
            gradual.start_check,
            gradual.start_temperature,
            calibration_slots(intent),
            intent.memory_bytes,
        )
        publish(check)
        if check["passed"] is not True:
            raise ValueError(
                "gradual precision start check exceeds its tolerance: max "
                + str(check["max_relative_error"])
                + ", worst table mean "
                + str(check["worst_mean_relative_error"])
            )
    elif intent.optimization.latent_multiple is not None:
        bound_latents(model, intent.optimization.latent_multiple)


def start_check_command(path: Path) -> Record:
    intent = Intent.load(path)
    gradual = intent.gradual_precision
    if gradual is None or intent.parent is not None:
        raise ValueError("the start check measures a fresh gradual precision intent")
    device = str(intent.raw["device"])
    with device_lease(device):
        torch.set_num_threads(intent.cpu_threads)
        torch.manual_seed(intent.seed)
        corpus = read_corpus(intent)
        model = load_he4(
            intent.source,
            intent.vocabulary,
            intent.memory_bytes,
            geometry(intent),
            intent.initialization,
            intent.frozen_format,
            intent.frozen_baseline,
            intent.trained_tables,
        )
        model.to(device)
        calibrated = None
        if intent.calibration is not None:
            calibrated = calibrate(
                model,
                corpus,
                intent.calibration,
                intent.features,
                calibration_slots(intent),
                intent.memory_bytes,
                True,
            )
        check = start_check(
            model,
            corpus,
            gradual.start_check,
            gradual.start_temperature,
            calibration_slots(intent),
            intent.memory_bytes,
        )
        return {
            **check,
            "intent": intent.sha256,
            "recipe": intent.recipe,
            "device": device,
            **({"calibration_values_sha256": calibrated["values_sha256"]} if calibrated else {}),
        }


def device_admission(intent: Intent, corpus: Corpus, model: HE4) -> int:
    frozen = sum(value.numel() * value.element_size() for value in model.buffers())
    trained = sum(
        value.numel() * value.element_size() for value in model.parameters() if value.requires_grad
    )
    context = max((max(row.anchors) for row in corpus.rows if row.anchors), default=0)
    retained = (
        4 * model.geometry.taps * model.geometry.hidden * context if model.trained_tables else 0
    )
    return intent.grouping.admit_device(frozen, trained, retained)


def run(intent_path: Path, root: Path, attempt: str, boundary: Boundary | None = None) -> None:
    device = str(Intent.load_sealed(intent_path).raw["device"])
    with device_lease(device):
        run_leased(intent_path, root, attempt, boundary)


def run_leased(
    intent_path: Path, root: Path, attempt: str, boundary: Boundary | None = None
) -> None:
    intent = Intent.load_sealed(intent_path)
    code_root = Path(__file__).resolve().parents[2]
    if source_identity(code_root) != intent.raw["code_sha256"]:
        raise ValueError("source changed between controller admission and worker load")
    torch.set_num_threads(intent.cpu_threads)
    torch.manual_seed(intent.seed)
    cfg = geometry(intent)
    corpus = read_corpus(intent)
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
    phase_clock = Monotonic()

    def phase(name: str) -> None:
        journal.append(
            {
                "kind": "worker-phase",
                "attempt": attempt,
                "phase": name,
                "monotonic": phase_clock.now(),
            }
        )

    phase("model-load-began")
    model = load_he4(
        intent.source,
        intent.vocabulary,
        intent.memory_bytes,
        cfg,
        intent.initialization,
        intent.frozen_format,
        intent.frozen_baseline,
        intent.trained_tables,
    )
    if corpus.fusion is not None and (
        model.trained_tables or fusion_identity(model.fusion.operand) != corpus.fusion
    ):
        raise ValueError("fused corpus and model fusion table identities differ")
    model.to(str(intent.raw["device"]))
    if model.vocabulary.device.type == "cuda":
        device_admission(intent, corpus, model)
        total = torch.cuda.get_device_properties(model.vocabulary.device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, intent.grouping.device_bytes / total), model.vocabulary.device
        )
    phase("model-load-complete")
    loaded_cache: dict[tuple[int, int, int, int, int], str] = {}
    initial_loaded = loaded_identity(loaded_cache)
    journal.append({"kind": "loaded", "attempt": attempt, "identity": initial_loaded})
    if intent.parent is None:
        phase("initialization-began")
        initialize(
            intent, corpus, model, lambda event: journal.append({**event, "attempt": attempt})
        )
        phase("initialization-complete")
    worker = TensorWorker(
        model,
        corpus,
        intent.recipe,
        intent.corpus.sha256,
        optimization=intent.optimization,
        blocks=intent.blocks,
        block_slots=corpus.slots,
        features=intent.features,
        precision=intent.gradual_precision,
        telemetry=intent.telemetry,
        arithmetic=intent.arithmetic,
        round_states=round_state_weighting(intent, corpus),
    )
    worker.fusion_sha256 = corpus.fusion
    diagnostics = (
        Diagnostics(artifact(intent.raw["development"], intent.base), corpus, intent.memory_bytes)
        if "development" in intent.raw
        else None
    )
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
    phase("restore-complete")
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

    def retain_states() -> None:
        if "retention" in intent.raw:
            keep = integer(record(intent.raw["retention"])["states"])
            pinned = ((intent.parent.sha256,) if intent.parent else ()) + (
                boundary.protected("state") if boundary else ()
            )
            retired = states.prune(keep, pinned)
            if retired:
                journal.append(
                    {"kind": "retention", "attempt": attempt, "retired_states": list(retired)}
                )

    last_checkpoint_updates = worker.updates
    last_checkpoint_seconds = clock.now()
    last_diagnostic_updates = -1
    last_progress_updates = -1
    last_loss_updates = worker.updates
    losses: list[float | None] = []

    def diagnose(diagnostics: Diagnostics, refresh: bool) -> None:
        nonlocal last_diagnostic_updates
        worker.verify_frozen()
        began = clock.now()
        precision = worker.precision_record()
        result = diagnostics.measure(model)
        exported = (
            diagnostics.measure(model, tables=exported_tables(model))
            if model.trained_tables
            else None
        )
        worker.verify_frozen()
        weights = (
            list(worker.refresh_position_weights(result))
            if refresh and intent.blocks.position_weights == "reach"
            else None
        )
        journal.append(
            {
                "kind": "development-diagnostic",
                "attempt": attempt,
                "updates": worker.updates,
                "seconds": clock.now() - began,
                "result": result,
                **({"exported_tables": exported} if exported is not None else {}),
                **({"position_weights": weights} if weights is not None else {}),
                **({"precision": precision} if precision is not None else {}),
            }
        )
        last_diagnostic_updates = worker.updates

    def stop() -> bool:
        nonlocal \
            last_checkpoint_updates, \
            last_checkpoint_seconds, \
            stop_reason, \
            last_diagnostic_updates, \
            last_progress_updates, \
            last_loss_updates
        if worker.updates != last_loss_updates:
            losses.append(worker.progress.last_loss)
            last_loss_updates = worker.updates
        if worker.updates != last_progress_updates and worker.cursor % 100 == 0:
            telemetry = worker.telemetry_record()
            round_states = worker.round_state_record()
            journal.append(
                {
                    "kind": "training-progress",
                    "attempt": attempt,
                    "updates": worker.updates,
                    "cursor": worker.cursor,
                    "loss": worker.progress.last_loss,
                    "losses": list(losses),
                    "gradient_norm": worker.last_gradient_norm,
                    "accounting": worker.accounting.encode(),
                    **({"telemetry": telemetry} if telemetry is not None else {}),
                    **({"round_states": round_states} if round_states is not None else {}),
                }
            )
            losses.clear()
            last_progress_updates = worker.updates
        update_limit = (
            boundary.limit()
            if boundary
            else number(intent.raw["model_update_seconds"])
            if "model_update_seconds" in intent.raw
            else None
        )
        if boundary is not None and "model_update_seconds" in intent.raw:
            update_limit = min(boundary.limit(), number(intent.raw["model_update_seconds"]))
        if (
            update_limit is not None
            and worker.accounting.successful_update_seconds
            + worker.accounting.failed_update_seconds
            >= update_limit
        ):
            stop_reason = (
                "model-update-budget"
                if boundary is not None
                and "model_update_seconds" in intent.raw
                and worker.accounting.successful_update_seconds
                + worker.accounting.failed_update_seconds
                >= number(intent.raw["model_update_seconds"])
                else "model-update-time"
            )
            return True
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
            diagnostics is not None
            and worker.cursor % integer(intent.raw.get("development_cadence_updates", 250)) == 0
            and worker.updates != last_diagnostic_updates
        ):
            diagnose(diagnostics, True)
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
            retain_states()
            last_checkpoint_updates = worker.updates
            last_checkpoint_seconds = clock.now()
        return False

    resumed = False

    class RecordedSamples:
        def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]:
            nonlocal resumed
            journal.append(
                {
                    "kind": "update-attempt",
                    "attempt": attempt,
                    "attempts": worker.progress.attempts,
                    "cursor": cursor,
                }
            )
            capture, windows = corpus.sample(cursor)
            slots = corpus.slots(cursor)
            journal.append(
                {
                    "kind": "exposure",
                    "attempt": attempt,
                    "attempts": worker.progress.attempts,
                    "cursor": cursor,
                    "trajectory": windows[0].trajectory,
                    "slots": slots,
                    "anchors": [window.anchor for window in windows],
                    "labels": [min(len(window.labels), slots - 1) for window in windows],
                }
            )
            if not resumed:
                journal.append(
                    {
                        "kind": "training-ready",
                        "attempt": attempt,
                        "monotonic": clock.now(),
                        "boot": owner.boot,
                    }
                )
                if boundary is not None:
                    boundary.resumed()
                resumed = True
            return capture, windows

    worker.samples = RecordedSamples()
    if intent.parent is None:
        initial = worker.checkpoint(states, intent.state_bytes)
        if intent.initial_export:
            initial_export = export_he4(
                worker,
                intent.source,
                intent.vocabulary,
                Store(root / "exports", intent.disk_reserve_bytes),
                intent.state_bytes,
                intent.memory_bytes,
            )
            journal.append(
                {
                    "kind": "initial-export",
                    "attempt": attempt,
                    "updates": worker.updates,
                    "state": initial.artifact.sha256,
                    "drafter": {
                        "path": str(initial_export.artifact.path),
                        "sha256": initial_export.artifact.sha256,
                        "bytes": initial_export.artifact.size,
                    },
                    **(
                        {"precision": precision_record}
                        if (precision_record := worker.precision_record()) is not None
                        else {}
                    ),
                    "scope": "untrained initialization; GGUF producer/reader conformance only",
                }
            )
    if boundary is not None:
        boundary.start(worker)
    while True:
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
        corpus.cache.clear()
        if (
            diagnostics is not None
            and (boundary is None or result != "model-update-time")
            and worker.updates != last_diagnostic_updates
        ):
            diagnose(diagnostics, False)
        journal.append(
            {
                "kind": "training-boundary",
                "attempt": attempt,
                "monotonic": clock.now(),
                "boot": owner.boot,
                "outcome": result,
            }
        )
        reference = worker.checkpoint(states, intent.state_bytes)
        phase("checkpoint-complete")
        retain_states()
        export_store = Store(root / "exports", intent.disk_reserve_bytes)
        exported = export_he4(
            worker,
            intent.source,
            intent.vocabulary,
            export_store,
            intent.state_bytes,
            intent.memory_bytes,
        )
        phase("export-complete")
        if source_identity(code_root) != intent.raw["code_sha256"]:
            raise ValueError("executable source drifted before boundary publication")
        if "retention" in intent.raw:
            retired_exports = export_store.prune(
                integer(record(intent.raw["retention"])["exports"]),
                boundary.protected("drafter") if boundary else (),
            )
            if retired_exports:
                journal.append(
                    {
                        "kind": "retention",
                        "attempt": attempt,
                        "retired_exports": list(retired_exports),
                    }
                )
        journal.append(
            {
                "kind": "export",
                "attempt": attempt,
                "artifact": exported.artifact.sha256,
                "source": intent.source.sha256,
                "vocabulary": intent.vocabulary.sha256,
                "scope": "GGUF producer/reader conformance; native loading unqualified",
                **(
                    {"precision": exported_precision}
                    if (exported_precision := worker.precision_record()) is not None
                    else {}
                ),
            }
        )
        journal.append(
            {
                "kind": "boundary-publication",
                "attempt": attempt,
                "updates": worker.updates,
                "state": {
                    "path": str(reference.artifact.path),
                    "sha256": reference.artifact.sha256,
                    "bytes": reference.artifact.size,
                },
                "drafter": {
                    "path": str(exported.artifact.path),
                    "sha256": exported.artifact.sha256,
                    "bytes": exported.artifact.size,
                },
            }
        )
        if boundary is not None:
            boundary.evaluate(worker, reference, exported, result)
        if boundary is None or result != "model-update-time":
            break
        resumed = False
        stop_reason = ""
        last_checkpoint_updates = worker.updates
        last_checkpoint_seconds = clock.now()
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
            "accounting": worker.accounting.encode(),
            "attempts": worker.progress.attempts,
            "state": reference.artifact.sha256,
            "export": exported.artifact.sha256,
            "activation": {
                "actual_optimizer_updates": worker.optimizer_updates,
                "training_device": model.vocabulary.device.type,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if model.vocabulary.device.type == "cuda"
                else None,
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
                if model.vocabulary.device.type == "cuda"
                else None,
            },
            "scientific": "DEV teacher-forced diagnostics measured; native acceptance is separate"
            if diagnostics is not None
            else "unknown: no development evaluation supplied",
            "objective": "unknown: no benchmark",
        }
    )


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--start-check":
        try:
            measured = start_check_command(Path(sys.argv[2]))
        except (ValueError, OSError, KeyError) as error:
            refused: Record = {
                "outcome": "refused",
                "cause": type(error).__name__,
                "detail": str(error),
            }
            print(json.dumps(refused))
            return 2
        print(json.dumps(measured, sort_keys=True))
        return 0 if measured["passed"] is True else 3
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
