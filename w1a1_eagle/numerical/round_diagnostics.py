import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Iterator, Sequence, cast

import torch
from torch import Tensor
from torch.nn import functional

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact, CaptureSpec, Supervision
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.drafting import (
    Pair,
    Ranked,
    Tally,
    context_band,
    generated_anchors,
    paired_interval,
    simulate,
    tally,
)
from w1a1_eagle.identity import source_identity
from w1a1_eagle.model import HE4, Geometry, anchor_bounds
from w1a1_eagle.numerical.baseline import load_baseline, table_rows, table_tensor
from w1a1_eagle.numerical.captures import Capture, load_capture
from w1a1_eagle.numerical.export import export_he4
from w1a1_eagle.numerical.labels import LabelFrequencies, label_frequencies, label_report
from w1a1_eagle.numerical.memory_plan import (
    DEVICE_MARGIN,
    FatalMemory,
    Guard,
    Shapes,
    host_available,
    plan,
    refusal,
)
from w1a1_eagle.numerical.probes import (
    ABLATIONS,
    EXPORTABLE,
    RangeProbe,
    Window,
    ablated,
    apply_ablation,
)
from w1a1_eagle.numerical.vocabulary_scoring import (
    Scoring,
    ScoringSettings,
    export_host_bytes,
    scoring_bytes,
    scoring_settings,
)
from w1a1_eagle.numerical.weights import load_he4
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import artifact, canonical, identity, read, reference
from w1a1_eagle.storage import output_path

SECTIONS = {
    "schema",
    "memory_bytes",
    "candidate",
    "baseline",
    "rounds",
    "windows",
    "ranges",
    "ablations",
    "exports",
    "labels",
}
EXPORT_BYTES = 6 * 1024**3
CANDIDATE_TABLE = ("output.weight", "token_embd.weight")
BASELINE_TABLE = ("token_embd.weight",)


class NoSamples:
    def sample(self, cursor: int) -> tuple[Capture, tuple[Supervision, ...]]:
        raise ValueError("round diagnostics request no training sample")


def protected_names(exclusions: Artifact) -> set[str]:
    exclusions.verify()
    return {"heldout24", *(string(v) for v in sequence(read(exclusions.path)["identities"]))}


def names(row: Record) -> set[str]:
    return {
        string(row["id"]),
        string(row["group"]),
        *(string(v) for v in sequence(row.get("aliases", []))),
    }


def dev_rows(corpus: Artifact, protected: set[str]) -> list[Record]:
    corpus.verify()
    rows = [record(value) for value in sequence(read(corpus.path, 64 * 1024 * 1024)["records"])]
    seen: set[str] = set()
    for row in rows:
        if row["role"] != "dev" or names(row) & protected:
            raise ValueError("round diagnostics require DEV captures outside protected sets")
        if string(row["id"]) in seen:
            raise ValueError("duplicate diagnostic capture")
        seen.add(string(row["id"]))
    if not rows:
        raise ValueError("empty DEV capture corpus")
    return rows


def capture_spec(row: Record, root: Path) -> CaptureSpec:
    return CaptureSpec(
        string(row["id"]),
        artifact(row["metadata"], root),
        tuple(artifact(value, root) for value in sequence(row["files"])),
    )


@dataclass(frozen=True)
class WindowSet:
    name: str
    windows: tuple[tuple[str, int], ...]
    specs: dict[str, CaptureSpec]
    categories: dict[str, str]


def window_sets(specification: Artifact) -> tuple[WindowSet, ...]:
    def load(reference_: Artifact, name: str, supplementary: bool) -> list[WindowSet]:
        reference_.verify()
        raw = read(reference_.path)
        schema = "training-diagnostics-group-v1" if supplementary else "training-diagnostics-v1"
        if (
            set(raw) - {"groups"} != {"schema", "corpus", "exclusions", "windows"}
            or raw["schema"] != schema
            or (supplementary and "groups" in raw)
        ):
            raise ValueError("unsupported fixed DEV windows")
        root = reference_.path.parent
        corpus = artifact(raw["corpus"], root)
        rows = dev_rows(corpus, protected_names(artifact(raw["exclusions"], root)))
        specs = {string(row["id"]): capture_spec(row, corpus.path.parent) for row in rows}
        categories = {string(row["id"]): string(row.get("topic") or "unclassified") for row in rows}
        windows = tuple(
            sorted(
                (string(record(v)["id"]), integer(record(v)["anchor"]))
                for v in sequence(raw["windows"])
            )
        )
        if (
            len(windows) != (20 if supplementary else 120)
            or len(set(windows)) != len(windows)
            or any(window[0] not in specs for window in windows)
        ):
            raise ValueError("fixed DEV windows differ from the 120 base or 20 group windows")
        result = [WindowSet(name, windows, specs, categories)]
        groups = record(raw.get("groups", {}))
        if set(groups) - {"agentic-coding", "thinking-on"}:
            raise ValueError("unsupported supplementary diagnostic group")
        for group, value in sorted(groups.items()):
            result.extend(load(artifact(value, root), group, True))
        return result

    sets = load(specification, "base", False)
    seen: set[tuple[str, int]] = set()
    for item in sets:
        if seen & set(item.windows):
            raise ValueError("diagnostic window sets overlap")
        seen.update(item.windows)
    return tuple(sets)


@dataclass(frozen=True)
class Run:
    memory: int
    geometry: Geometry
    guard: Guard


def captures(
    windows: tuple[tuple[str, int], ...], specs: dict[str, CaptureSpec], run: Run
) -> Iterator[tuple[int, str, int, Capture, Supervision]]:
    order = {name: index for index, name in enumerate(sorted({name for name, _ in windows}))}
    current: tuple[str, Capture] | None = None
    for name, anchor in windows:
        if current is None or current[0] != name:
            current = None
            gc.collect()
            current = (name, load_capture(specs[name], run.memory, run.geometry.hidden))
        run.guard.check()
        selected = [window for window in current[1].windows if window.anchor == anchor]
        if len(selected) != 1:
            raise ValueError("diagnostic anchor is not eligible in authenticated capture")
        yield order[name], name, anchor, current[1], selected[0]


def evaluate_windows(
    model: HE4,
    sets: tuple[WindowSet, ...],
    run: Run,
    depth: int,
    probe: RangeProbe | None = None,
) -> Record:
    result: Record = {}
    with torch.no_grad():
        for item in sets:
            hits = [0] * depth
            survivors = [0] * depth
            reached = [0] * depth
            categories: dict[str, list[int]] = {}
            for order, name, anchor, capture, window in captures(item.windows, item.specs, run):
                length = min(len(window.labels), depth)
                if probe is not None:
                    probe.window = Window(name, anchor, order % 2)
                logits = model.forward(capture.features, capture.tokens, anchor, depth + 1)
                predicted = cast(list[int], logits[1 : 1 + length].argmax(-1).tolist())
                alive = True
                cell = categories.setdefault(item.categories[name], [0, 0])
                for slot in range(length):
                    hit = predicted[slot] == window.labels[slot]
                    alive = alive and hit
                    reached[slot] += 1
                    hits[slot] += int(hit)
                    survivors[slot] += int(alive)
                    cell[0] += int(hit)
                    cell[1] += 1
            result[item.name] = {
                "windows": len(item.windows),
                "reached": reached,
                "top1": [hit / total if total else None for hit, total in zip(hits, reached)],
                "prefix_survival": [
                    alive / total if total else None for alive, total in zip(survivors, reached)
                ],
                "top1_by_category": {
                    name: values[0] / values[1] for name, values in sorted(categories.items())
                },
            }
    return {
        **result,
        "block_slots": depth + 1,
        "scope": "teacher-forced positions 1-3 of a four-slot block on the fixed DEV windows; not native acceptance",
    }


ScoringHook = Callable[[tuple[int, ...], Tensor, Tensor, Tensor], None]


def propose(
    model: HE4,
    capture: Capture,
    anchors: Sequence[int],
    depth: int,
    top: int,
    chunk: int,
    hook: ScoringHook | None = None,
) -> dict[int, Ranked]:
    result: dict[int, Ranked] = {}
    with torch.no_grad():
        for start in range(0, len(anchors), chunk):
            part = tuple(anchors[start : start + chunk])
            hidden = model.group_hidden(capture.features, capture.tokens, part, depth + 1)[:, 1:]
            logits = functional.linear(hidden, model.vocabulary)
            ranked = logits.topk(top, dim=-1).indices
            if hook is not None:
                hook(part, hidden, logits, ranked)
            del logits
            values = cast(list[list[list[int]]], ranked.cpu().tolist())
            for anchor, slots in zip(part, values, strict=True):
                result[anchor] = tuple(tuple(int(token) for token in slot) for slot in slots)
    return result


@dataclass
class Request:
    identity: str
    group: str
    category: str
    band: str
    rounds: int
    starts: Tally
    positions: Tally


def drafter_rounds(
    model: HE4,
    rows: list[Record],
    root: Path,
    settings: Record,
    run: Run,
    scoring: Scoring | None = None,
) -> list[Request]:
    depth, top, chunk = (integer(settings[key]) for key in ("depth", "top", "chunk"))
    requests: list[Request] = []
    for row in rows:
        run.guard.check()
        capture = load_capture(capture_spec(row, root), run.memory, run.geometry.hidden)
        windows = {window.anchor: window.labels for window in capture.windows}
        anchors = generated_anchors(windows)
        if not anchors:
            continue
        hook = None if scoring is None else scoring.request(capture, windows)
        proposals = propose(model, capture, anchors, depth, top, chunk, hook)
        rounds = simulate(windows, proposals, depth)
        request = Request(
            string(row["id"]),
            string(row["group"]),
            string(row.get("topic") or "unclassified"),
            context_band(min(windows) + 1),
            len(rounds),
            tally([item.anchor for item in rounds], windows, proposals, depth, top),
            tally(anchors, windows, proposals, depth, top),
        )
        requests.append(request)
        if scoring is not None and hook is not None:
            scoring.add(
                request.identity,
                request.group,
                request.category,
                hook,
                windows,
                proposals,
                tuple(item.anchor for item in rounds),
                request.starts,
            )
        del capture, hook
        gc.collect()
    return requests


def merged(requests: list[Request], depth: int, top: int) -> Record:
    starts, positions = Tally(depth, top), Tally(depth, top)
    for request in requests:
        starts.merge(request.starts)
        positions.merge(request.positions)
    return {
        "requests": len(requests),
        "round_starts": starts.record(),
        "all_generated_positions": positions.record(),
    }


def paired(requests: list[Request], depth: int, resamples: int, seed: int) -> Record:
    metrics: dict[str, list[Pair]] = {}
    for request in requests:
        a, b = request.starts, request.positions
        for slot in range(depth):
            metrics.setdefault(f"first_choice_slot_{slot + 1}", []).append(
                (
                    request.group,
                    a.first_choice[slot],
                    a.reached[slot],
                    b.first_choice[slot],
                    b.reached[slot],
                )
            )
            metrics.setdefault(f"top_recall_slot_{slot + 1}", []).append(
                (
                    request.group,
                    a.shortlist[slot],
                    a.reached[slot],
                    b.shortlist[slot],
                    b.reached[slot],
                )
            )
        metrics.setdefault("accepted_per_round", []).append(
            (request.group, a.accepted, a.positions, b.accepted, b.positions)
        )
    return {
        name: paired_interval(rows, resamples, Random(seed))
        for name, rows in sorted(metrics.items())
    }


def native_agreement(requests: list[Request], result: Record, arm: str) -> Record:
    native: dict[str, tuple[int, int]] = {}
    for value in sequence(result["rows"]):
        row = record(value)
        if row["arm"] == arm:
            counters = record(row["native"])
            native[string(row["task"])] = (
                integer(counters["accepted"]),
                integer(counters["proposed"]),
            )
    matched = [request for request in requests if request.identity in native]
    if not matched:
        return {"arm": arm, "requests": 0}
    accepted = sum(native[request.identity][0] for request in matched)
    proposed = sum(native[request.identity][1] for request in matched)
    simulated_accepted = sum(request.starts.accepted for request in matched)
    simulated_proposed = sum(request.starts.proposed for request in matched)
    return {
        "arm": arm,
        "requests": len(matched),
        "native_acceptance": accepted / proposed if proposed else None,
        "simulated_acceptance": simulated_accepted / simulated_proposed
        if simulated_proposed
        else None,
        "native_accepted": accepted,
        "simulated_accepted": simulated_accepted,
        "requests_with_equal_accepted": sum(
            native[request.identity][0] == request.starts.accepted for request in matched
        ),
        "mean_absolute_accepted_difference": sum(
            abs(native[request.identity][0] - request.starts.accepted) for request in matched
        )
        / len(matched),
    }


@dataclass(frozen=True)
class ScoringRun:
    settings: ScoringSettings
    frequencies: LabelFrequencies | None
    store: Store
    table: Record


def rounds_report(
    model: HE4,
    drafter: str,
    settings: Record,
    root: Path,
    run: Run,
    scoring_run: ScoringRun | None = None,
) -> Record:
    corpus = artifact(settings["corpus"], root)
    rows = dev_rows(corpus, protected_names(artifact(settings["exclusions"], root)))
    depth, top = integer(settings["depth"]), integer(settings["top"])
    scoring = (
        None
        if scoring_run is None
        else Scoring(scoring_run.settings, scoring_run.frequencies, model, depth)
    )
    requests = drafter_rounds(model, rows, corpus.path.parent, settings, run, scoring)
    by_category: dict[str, list[Request]] = {}
    by_band: dict[str, list[Request]] = {}
    for request in requests:
        by_category.setdefault(request.category, []).append(request)
        by_band.setdefault(request.band, []).append(request)
    report: Record = {
        "drafter": drafter,
        "corpus": corpus.sha256,
        **merged(requests, depth, top),
        "paired_round_starts_minus_positions": paired(
            requests, depth, integer(settings["resamples"]), integer(settings["seed"])
        ),
        "by_category": {
            name: merged(rows_, depth, top) for name, rows_ in sorted(by_category.items())
        },
        "by_input_band": {
            name: merged(rows_, depth, top) for name, rows_ in sorted(by_band.items())
        },
        "per_request": [
            {
                "id": request.identity,
                "group": request.group,
                "category": request.category,
                "band": request.band,
                "rounds": request.rounds,
                "accepted": request.starts.accepted,
                "proposed": request.starts.proposed,
            }
            for request in requests
        ],
    }
    if scoring is not None and scoring_run is not None:
        resamples, seed = integer(settings["resamples"]), integer(settings["seed"])
        report["scoring"] = scoring.report(resamples, seed)
        if scoring_run.settings.export_rounds:
            record(report["scoring"])["export"] = scoring.export(
                Store(scoring_run.store.root / ("scoring-" + drafter)),
                drafter,
                scoring_run.table,
            )
        del scoring
    if "native" in settings:
        native = artifact(settings["native"], root)
        native.verify()
        report["native_agreement"] = native_agreement(
            requests, read(native.path, 64 * 1024 * 1024), drafter
        )
    return report


def candidate_model(
    section: Record, root: Path, run: Run, device: str
) -> tuple[HE4, StateRef, dict[str, object], str, Artifact, Artifact]:
    allowed = {
        "source",
        "vocabulary",
        "initialization",
        "frozen_baseline",
        "state",
        "mode",
        "arm",
        "slot_rounding",
    }
    if not {"source", "vocabulary", "state", "mode"} <= set(section) <= allowed:
        raise ValueError("candidate needs source, vocabulary, state and mode")
    mode = string(section["mode"])
    if mode not in {"branch", "continue"}:
        raise ValueError("candidate mode must be branch or continue")
    if section.get("slot_rounding", "anchor-mask") != "anchor-mask":
        raise ValueError("candidate slot_rounding is anchor-mask when set")
    source, vocabulary, state = (
        artifact(section[key], root) for key in ("source", "vocabulary", "state")
    )
    initialization = (
        artifact(section["initialization"], root) if "initialization" in section else None
    )
    frozen = artifact(section["frozen_baseline"], root) if "frozen_baseline" in section else None
    model = load_he4(
        source,
        vocabulary,
        run.memory,
        run.geometry,
        initialization,
        "Q4_0" if frozen is None else "baseline-k",
        frozen,
    )
    if "slot_rounding" in section:
        anchor_bounds(model)
    if device == "cuda":
        model = model.cuda()
        gc.collect()
    parent = Store(state.path.parent).describe(state)
    if mode == "continue":
        recipe: dict[str, object] = {}
        config = parent.config
    else:
        recipe = {
            "parent_recipe": parent.config,
            "frozen_format": model.frozen_format,
            **({} if frozen is None else {"baseline": reference(frozen)}),
            **({"slot_rounding": section["slot_rounding"]} if "slot_rounding" in section else {}),
        }
        config = identity(recipe)
    return model, parent, recipe, config, source, vocabulary


def restored(model: HE4, parent: StateRef, config: str, memory: int, mode: str) -> TensorWorker:
    worker = TensorWorker(model, NoSamples(), config)
    worker.restore(parent, memory, "branch" if mode == "branch" else "continue")
    cast(dict[Tensor, object], getattr(worker.optimizer, "state")).clear()
    model.eval()
    return worker


def text_table(report: Record) -> str:
    lines: list[str] = []
    for drafter, value in record(report.get("rounds", {})).items():
        section = record(value)
        for kind in ("round_starts", "all_generated_positions"):
            row = record(section[kind])
            lines.append(
                f"rounds {drafter} {kind}: acceptance {row['acceptance']} per round "
                f"{row['accepted_per_round']} first-choice {row['first_choice_recall']} "
                f"top-16 {row['top_recall']} first wrong {row['first_wrong_slot']}"
            )
        if "native_agreement" in section:
            lines.append(f"rounds {drafter} native: {json.dumps(section['native_agreement'])}")
        if "scoring" in section:
            for name, value in record(record(section["scoring"])["variants"]).items():
                row = record(value)
                lines.append(
                    f"scoring {drafter} {name}: changed {row['first_choice_changed']} "
                    f"lost {row['correct_first_choice_lost']} "
                    f"outside {row['label_outside_candidates']} "
                    f"accepted/round difference {row['accepted_per_round_difference']}"
                )
    for name, value in record(report.get("ablations", {})).items():
        for group, measured in record(value).items():
            if isinstance(measured, dict) and "top1" in measured:
                lines.append(
                    f"ablation {name} {group}: top1 {measured['top1']} "
                    f"survival {measured['prefix_survival']}"
                )
    for layer, value in record(report.get("ranges", {})).items():
        for kind, measured in record(value).items():
            row = record(measured)
            lines.append(
                f"ranges {layer} {kind}: bound {row['bound']:.4g} clipped "
                f"{row['clipped_fraction']} anchor-share {row['clipped_share_anchor_slot']} "
                f"top1%-rows {row['clipped_share_top_1pct_rows']} "
                f"slot order repeats {row['slot_order_repeats']}"
            )
    for name, value in record(report.get("exports", {})).items():
        lines.append(f"export {name}: {record(value)['sha256']}")
    if "labels" in report:
        for name, value in record(record(report["labels"])["by_scenario"]).items():
            lines.append(f"labels {name}: {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def capture_length(spec: CaptureSpec) -> int:
    return len(sequence(read(spec.metadata.path, 64 * 1024 * 1024)["input_tokens"]))


def rounds_rows(rounds: Record, root: Path) -> tuple[list[Record], Path]:
    corpus = artifact(rounds["corpus"], root)
    return dev_rows(
        corpus, protected_names(artifact(rounds["exclusions"], root))
    ), corpus.path.parent


def scoring_run(
    raw: Record,
    root: Path,
    store: Store,
    frequencies: LabelFrequencies | None,
    drafter: str,
    geometry: Geometry,
) -> ScoringRun | None:
    rounds = record(raw.get("rounds", {}))
    if "scoring" not in rounds:
        return None
    names: tuple[str, ...]
    if drafter == "baseline":
        model = artifact(record(raw["baseline"])["target"], root)
        names = BASELINE_TABLE
    else:
        model = artifact(record(raw["candidate"])["vocabulary"], root)
        names = CANDIDATE_TABLE
    tensor, _ = table_tensor(model.path, names)
    return ScoringRun(
        scoring_settings(record(rounds["scoring"]), root, geometry.hidden),
        frequencies,
        store,
        {"model": reference(model), "tensor": tensor},
    )


def candidate_sections(
    raw: Record,
    root: Path,
    run: Run,
    device: str,
    sets: tuple[WindowSet, ...],
    store: Store,
    frequencies: LabelFrequencies | None = None,
) -> Record:
    rounds = record(raw.get("rounds", {}))
    ablations = tuple(string(v) for v in sequence(raw.get("ablations", [])))
    exports = tuple(string(v) for v in sequence(raw.get("exports", [])))
    section = record(raw["candidate"])
    result: Record = {}
    run.guard.begin()
    model, parent, recipe, config, source, vocabulary = candidate_model(section, root, run, device)
    run.guard.end("candidate load")
    run.guard.begin()
    mode = string(section["mode"])
    worker = restored(model, parent, config, run.memory, mode)
    run.guard.end("candidate restore")
    arm = string(section.get("arm", "candidate"))
    result["candidate"] = {
        "state": parent.artifact.sha256,
        "mode": mode,
        "config": config,
        "recipe": recipe,
        "updates": worker.updates,
        "frozen_transition": worker.frozen_transition,
    }
    if rounds:
        run.guard.begin()
        result["rounds"] = {
            arm: rounds_report(
                model,
                arm,
                rounds,
                root,
                run,
                scoring_run(raw, root, store, frequencies, arm, run.geometry),
            )
        }
        run.guard.end("candidate rounds")
    if sets:
        run.guard.begin()
        measured: Record = {}
        probe = RangeProbe(model, 4) if raw.get("ranges", False) else None
        names = ablations if "unchanged" in ablations else ("unchanged", *ablations)
        for name in names:
            with ablated(model, name):
                if probe is not None and name == "unchanged":
                    with probe.attached():
                        measured[name] = evaluate_windows(model, sets, run, 3, probe)
                else:
                    measured[name] = evaluate_windows(model, sets, run, 3)
        if ablations:
            result["ablations"] = measured
        if probe is not None:
            result["ranges"] = probe.record()
        del probe
        run.guard.end("candidate windows")
    if exports:
        run.guard.begin()
        exported: Record = {}
        base = recipe if mode == "branch" else {"parent_recipe": config}
        for name in exports:
            variant = config if name == "unchanged" else identity({**base, "ablation": name})
            del worker
            gc.collect()
            worker = restored(
                model, parent, variant, run.memory, mode if name == "unchanged" else "branch"
            )
            apply_ablation(model, name)
            state = export_he4(
                worker,
                source,
                vocabulary,
                Store(store.root / ("export-" + name)),
                EXPORT_BYTES,
                run.memory,
            )
            exported[name] = {**reference(state.artifact), "config": variant}
            run.guard.check()
        result["exports"] = exported
        run.guard.end("candidate exports")
    return result


def baseline_rounds(
    raw: Record,
    root: Path,
    run: Run,
    device: str,
    store: Store | None = None,
    frequencies: LabelFrequencies | None = None,
) -> Record:
    section = record(raw["baseline"])
    run.guard.begin()
    model = load_baseline(
        artifact(section["drafter"], root),
        artifact(section["target"], root),
        run.memory,
        run.geometry,
        device,
    )
    run.guard.end("baseline load")
    run.guard.begin()
    report = rounds_report(
        model,
        "baseline",
        record(raw["rounds"]),
        root,
        run,
        None
        if store is None
        else scoring_run(raw, root, store, frequencies, "baseline", run.geometry),
    )
    run.guard.end("baseline rounds")
    return report


def memory_shapes(
    raw: Record, root: Path, geometry: Geometry, sets: tuple[WindowSet, ...]
) -> Shapes:
    rounds = record(raw.get("rounds", {}))
    rounds_context = 0
    if rounds:
        rows, corpus_root = rounds_rows(rounds, root)
        rounds_context = max(capture_length(capture_spec(row, corpus_root)) for row in rows)
    windows_context = max(
        (
            capture_length(item.specs[name])
            for item in sets
            for name in {name for name, _ in item.windows}
        ),
        default=0,
    )
    candidate_rows = None
    if "candidate" in raw:
        vocabulary = artifact(record(raw["candidate"])["vocabulary"], root)
        candidate_rows = table_rows(vocabulary.path, CANDIDATE_TABLE)
    baseline_rows = None
    if "baseline" in raw:
        target = artifact(record(raw["baseline"])["target"], root)
        baseline_rows = table_rows(target.path, BASELINE_TABLE)
    scoring = scoring_host = 0
    if "scoring" in rounds:
        settings = scoring_settings(record(rounds["scoring"]), root, geometry.hidden)
        rows_ = max(candidate_rows or 0, baseline_rows or 0)
        scoring = scoring_bytes(
            settings, rows_, geometry.hidden, integer(rounds["chunk"]) * integer(rounds["depth"])
        )
        scoring_host = export_host_bytes(settings, rows_, geometry.hidden)
    return Shapes(
        geometry,
        candidate_rows,
        baseline_rows,
        rounds_context,
        windows_context,
        integer(rounds.get("chunk", 1)),
        integer(rounds.get("depth", 3)),
        len(sequence(raw.get("exports", []))),
        bool(raw.get("ranges", False)),
        artifact(record(raw["candidate"])["state"], root).size if "candidate" in raw else 0,
        scoring,
        scoring_host,
    )


def diagnose(specification: Path, output: Path, geometry: Geometry = Geometry()) -> Record:
    raw = read(specification)
    if raw.get("schema") != "round-diagnostics-v1" or not set(raw) <= SECTIONS:
        raise ValueError("unsupported round diagnostics specification")
    if output.exists():
        raise ValueError("round diagnostics output must be new")
    root = specification.resolve().parent
    memory = integer(raw["memory_bytes"])
    ablations = tuple(string(v) for v in sequence(raw.get("ablations", [])))
    exports = tuple(string(v) for v in sequence(raw.get("exports", [])))
    if set(ablations) - set(ABLATIONS) or set(exports) - set(EXPORTABLE):
        raise ValueError("unknown ablation or non-exportable variant")
    needs_windows = bool(raw.get("ranges", False)) or bool(ablations)
    if needs_windows and "windows" not in raw:
        raise ValueError("ranges and ablations need the fixed DEV windows")
    if (needs_windows or exports) and "candidate" not in raw:
        raise ValueError("window diagnostics and exports need a candidate")
    if "baseline" in raw and "rounds" not in raw:
        raise ValueError("the Baseline model is evaluated in the round simulation")
    if "rounds" in raw and "candidate" not in raw and "baseline" not in raw:
        raise ValueError("the round simulation needs a candidate or the Baseline")
    rounds = record(raw.get("rounds", {}))
    if rounds and not {"corpus", "exclusions", "depth", "top", "chunk", "seed", "resamples"} <= set(
        rounds
    ):
        raise ValueError("rounds need corpus, exclusions, depth, top, chunk, seed and resamples")
    uses_model = "candidate" in raw or "baseline" in raw
    cuda = uses_model and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    sets = window_sets(artifact(raw["windows"], root)) if needs_windows else ()
    shapes_ = memory_shapes(raw, root, geometry, sets)
    steps = plan(shapes_, cuda)
    free, total = torch.cuda.mem_get_info() if cuda else (0, 0)
    host_limit = min(memory, host_available())
    summary: Record = {
        "memory_plan": [step.record() for step in steps],
        "device_free_gib": round(free / 1024**3, 2) if cuda else None,
        "host_limit_gib": round(host_limit / 1024**3, 2),
    }
    print(json.dumps(summary), flush=True)
    refused = refusal(steps, free if cuda else None, host_limit)
    if refused is not None:
        raise ValueError("memory plan refused: " + refused)
    if cuda:
        torch.cuda.set_per_process_memory_fraction(max(0.05, (free - DEVICE_MARGIN) / total))
    run = Run(memory, geometry, Guard(cuda, host_limit))
    code_root = Path(__file__).resolve().parents[2]
    code = source_identity(code_root)
    clock = Monotonic()
    began = clock.now()
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    store.atomic(store.root / "memory-plan.json", canonical(summary))
    report: Record = {
        "schema": "round-diagnostics-report-v1",
        "code_sha256": code,
        "device": device,
        "gpu": cuda_identity() if cuda else None,
        "torch": str(torch.__version__),
        "specification": identity(raw),
    }
    frequencies = None
    if "scoring" in rounds:
        settings = scoring_settings(record(rounds["scoring"]), root, geometry.hidden)
        if settings.mixture is not None:
            frequencies = label_frequencies(settings.mixture)
    with device_lease(device):
        if "labels" in raw:
            labels = record(raw["labels"])
            report["labels"] = label_report(artifact(labels["mixture"], root))
        if "candidate" in raw:
            report.update(candidate_sections(raw, root, run, device, sets, store, frequencies))
            gc.collect()
            run.guard.release("the candidate")
        if "baseline" in raw:
            measured = baseline_rounds(raw, root, run, device, store, frequencies)
            record(report.setdefault("rounds", {}))["baseline"] = measured
            gc.collect()
            run.guard.release("the Baseline")
    if source_identity(code_root) != code:
        raise ValueError("round diagnostics source changed during execution")
    report["memory"] = {**summary, "measured": run.guard.measured}
    report["seconds"] = clock.now() - began
    store.atomic(store.root / "report.json", canonical(report))
    store.atomic(store.root / "report.txt", text_table(report).encode())
    return report


FATAL_MEMORY_EXIT = 3


def fatal(error: BaseException) -> bool:
    if isinstance(error, (torch.cuda.OutOfMemoryError, FatalMemory)):
        return True
    text = str(error)
    return isinstance(error, RuntimeError) and ("out of memory" in text or "ALLOC_FAILED" in text)


def run_diagnostics(specification: Path, output: Path, geometry: Geometry = Geometry()) -> int:
    try:
        report = diagnose(specification, output, geometry)
    except RuntimeError as error:
        if not fatal(error):
            raise
        message: Record = {"outcome": "fatal-memory", "detail": str(error)}
        print(json.dumps(message), file=sys.stderr, flush=True)
        return FATAL_MEMORY_EXIT
    summary: Record = {"output": str(output), "seconds": report["seconds"]}
    print(json.dumps(summary), flush=True)
    return 0


class Arguments(argparse.Namespace):
    specification: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m w1a1_eagle.numerical.round_diagnostics")
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    output = output_path(args.output, "benchmarks", "round-diagnostics")
    code = run_diagnostics(args.specification.resolve(), output)
    if code:
        os._exit(code)


if __name__ == "__main__":
    main()
