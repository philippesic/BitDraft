import argparse
import gc
import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, cast

import torch
from torch.nn import functional

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.clock import Monotonic
from w1a1_eagle.contracts import Record, integer, record, string
from w1a1_eagle.data import Artifact, CaptureRow
from w1a1_eagle.device import cuda_identity, device_lease
from w1a1_eagle.drafting import Ranked, generated_anchors, segments
from w1a1_eagle.identity import source_identity
from w1a1_eagle.intent import CORPUS_FORMATS, FUSED_FORMATS, Intent
from w1a1_eagle.model import HE4, Geometry, Projection
from w1a1_eagle.numerical.baseline import table_rows
from w1a1_eagle.numerical.captures import READERS, Capture, load_capture
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.entry import geometry as intent_geometry
from w1a1_eagle.numerical.entry import read_corpus
from w1a1_eagle.numerical.memory_plan import (
    DEVICE_MARGIN,
    FLOAT,
    PROCESS_HOST,
    Guard,
    Shapes,
    Step,
    capture_host,
    he4_elements,
    host_available,
    plan,
    refusal,
    working,
)
from w1a1_eagle.numerical.mixture import mixture
from w1a1_eagle.numerical.round_diagnostics import (
    Run,
    candidate_model,
    fatal,
    restored,
)
from w1a1_eagle.records import artifact, canonical, identity, read, reference
from w1a1_eagle.round_states import (
    MAP_SCHEMA,
    Span,
    covered_windows,
    needed_spans,
    parse_map,
    round_starts,
)
from w1a1_eagle.storage import output_path

SPEC_SCHEMA = "round-state-map-spec-v1"
SPEC_FIELDS = {
    "schema",
    "memory_bytes",
    "activation_bytes",
    "drafter",
    "corpus",
    "corpus_format",
    "restrict",
    "rows",
    "depth",
    "chunk",
}
FATAL_MEMORY_EXIT = 3


@dataclass(frozen=True)
class Planned:
    row: CaptureRow
    spans: tuple[Span, ...]

    @property
    def identity(self) -> str:
        return self.row.capture.identity


def proposal_count(anchors: tuple[int, ...], spans: tuple[Span, ...]) -> int:
    present = set(anchors)
    return sum(
        sum(1 for anchor in range(first + 1, last + 1) if anchor in present)
        for first, last in spans
    )


def full_proposals(corpus: Corpus) -> int:
    return sum(proposal_count(row.anchors, segments(row.anchors)) for row in corpus.rows)


def spec_corpus(
    raw: Record, root: Path, memory: int, cfg: Geometry
) -> tuple[Corpus, Artifact, str | None, Record]:
    corpus_ref = artifact(raw["corpus"], root)
    corpus_format = None if raw.get("corpus_format") is None else string(raw["corpus_format"])
    if corpus_format is not None and corpus_format not in CORPUS_FORMATS:
        raise ValueError("unsupported corpus format for a round-state map")
    if "restrict" in raw:
        intent_ref = artifact(raw["restrict"], root)
        intent_ref.verify()
        intent = Intent.load(intent_ref.path)
        if (intent.corpus.sha256, intent.corpus.size) != (corpus_ref.sha256, corpus_ref.size) or (
            intent.raw.get("corpus_format") != corpus_format
        ):
            raise ValueError("the restricting intent binds another corpus")
        if intent.index_spec is not None or intent_geometry(intent) != cfg:
            raise ValueError("the restricting intent needs a corpus manifest and this geometry")
        corpus = read_corpus(intent)
        return (
            corpus,
            corpus_ref,
            corpus_format,
            {
                "kind": "intent",
                "intent": intent.sha256,
                "recipe": intent.recipe,
                "seed": intent.seed,
                "cursors": intent.budget.updates,
                "sampling": identity(
                    {
                        "seed": intent.seed,
                        "grouping": intent.raw.get("grouping"),
                        "blocks": intent.raw.get("blocks"),
                        "sampling": intent.raw.get("sampling"),
                        "geometry": intent.raw["geometry"],
                    }
                ),
            },
        )
    fused = corpus_format in FUSED_FORMATS
    source = (
        mixture(corpus_ref, 0, memory, cfg.hidden, fused)
        if corpus_format in ("corpus-mixture-v1", "fused-mixture-v1")
        else corpus_ref
    )
    return (
        Corpus(source, 0, memory, cfg.hidden, shape=cfg, fused=fused),
        corpus_ref,
        corpus_format,
        {"kind": "full"},
    )


def planned_rows(corpus: Corpus, scope: Record, rows: int | None) -> list[Planned]:
    sampled: dict[str, set[int]] | None = None
    if scope["kind"] == "intent":
        sampled = {}
        for cursor in range(integer(scope["cursors"])):
            row, anchors = corpus.locate(cursor)
            sampled.setdefault(row.capture.identity, set()).update(anchors)
    result: list[Planned] = []
    for row in corpus.rows:
        if row.prompt.role != "train":
            raise ValueError("round-state maps are built from TRAIN rows only")
        if not row.anchors:
            continue
        chosen = None if sampled is None else sampled.get(row.capture.identity)
        if sampled is not None and chosen is None:
            continue
        spans = needed_spans(row.anchors, chosen)
        if spans:
            result.append(Planned(row, spans))
    if rows is not None:
        result = result[:: max(1, len(result) // rows)][:rows]
    return result


def anchors_per_call(
    cfg: Geometry, context: int, depth: int, rows: int, chunk: int, limit: int
) -> int:
    base = working(cfg, context, 0, depth + 1, rows, depth)
    each = working(cfg, context, 1, depth + 1, rows, depth) - base
    fitted = (limit - base) // each if limit > base else 0
    if fitted < 1:
        raise ValueError(
            f"a context of {context} tokens needs more than activation_bytes {limit}"
            " for one anchor; raise activation_bytes"
        )
    return min(chunk, fitted)


@contextmanager
def single_products(model: HE4) -> Iterator[None]:
    projections = [module for module in model.modules() if isinstance(module, Projection)]
    cores = model.vocabulary.device.type == "cuda"
    for module in projections:
        module.single_product = True
        module.tensor_cores = cores
    try:
        yield
    finally:
        for module in projections:
            module.single_product = False
            module.tensor_cores = False


def propose_first(
    model: HE4, capture: Capture, anchors: tuple[int, ...], depth: int, chunk: int
) -> dict[int, Ranked]:
    result: dict[int, Ranked] = {}
    if not anchors:
        return result
    fused = capture.fusion is not None
    with torch.no_grad(), single_products(model):
        contexts = model.group_contexts(capture.features, capture.tokens, max(anchors), fused)
        for start in range(0, len(anchors), chunk):
            part = anchors[start : start + chunk]
            hidden = model.group_hidden(
                capture.features, capture.tokens, part, depth + 1, fused, contexts
            )[:, 1:]
            chosen = functional.linear(hidden, model.vocabulary).argmax(-1)
            values = cast(list[list[int]], chosen.cpu().tolist())
            for anchor, slots in zip(part, values, strict=True):
                result[anchor] = tuple((int(token),) for token in slots)
        del contexts
    return result


def drafter_record(section: Record, root: Path, config: str, updates: int) -> Record:
    def sha(key: str) -> str | None:
        return artifact(section[key], root).sha256 if key in section else None

    state = artifact(section["state"], root)
    return {
        "state": {"sha256": state.sha256, "bytes": state.size},
        "mode": string(section["mode"]),
        "config": config,
        "updates": updates,
        "source": sha("source"),
        "vocabulary": sha("vocabulary"),
        "initialization": sha("initialization"),
        "frozen_baseline": sha("frozen_baseline"),
    }


def build(specification: Path, output: Path, cfg: Geometry = Geometry()) -> Record:
    raw = read(specification)
    if (
        raw.get("schema") != SPEC_SCHEMA
        or not {
            "schema",
            "memory_bytes",
            "drafter",
            "corpus",
        }
        <= set(raw)
        <= SPEC_FIELDS
    ):
        raise ValueError("unsupported round-state map specification")
    if output.exists():
        raise ValueError("round-state map output must be new")
    root = specification.resolve().parent
    memory = integer(raw["memory_bytes"])
    activation = integer(raw.get("activation_bytes", 3 * 1024**3))
    depth = integer(raw.get("depth", 3))
    chunk = integer(raw.get("chunk", 128))
    limit = None if raw.get("rows") is None else integer(raw["rows"])
    if depth < 1 or chunk < 1 or activation < 1 or (limit is not None and limit < 1):
        raise ValueError("positive depth, chunk, activation bytes and row limit required")
    if limit is not None and "restrict" in raw:
        raise ValueError("a rate sample spans the whole corpus")
    corpus, corpus_ref, corpus_format, scope = spec_corpus(raw, root, memory, cfg)
    if limit is not None:
        scope = {"kind": "sample", "rows": limit}
    planned = planned_rows(corpus, scope, limit)
    if not planned:
        raise ValueError("no TRAIN windows to map")
    section = record(raw["drafter"])
    vocabulary = artifact(section["vocabulary"], root)
    rows = table_rows(vocabulary.path, ("output.weight", "token_embd.weight"))
    state = artifact(section["state"], root)
    longest = max(item.spans[-1][1] + 1 for item in planned)
    sizes = {
        item.identity: anchors_per_call(cfg, item.spans[-1][1] + 1, depth, rows, chunk, activation)
        for item in planned
    }
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    steps = plan(Shapes(cfg, rows, None, 0, 0, chunk, depth, 0, False, state.size), cuda)
    resident = sum(he4_elements(cfg, rows)) * FLOAT
    mapping = Step(
        "round-state map",
        resident + activation if cuda else 0,
        PROCESS_HOST + 2 * capture_host(cfg, longest) + (0 if cuda else resident + activation),
    )
    steps = (*steps, mapping)
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
    run = Run(memory, cfg, Guard(cuda, host_limit))
    code_root = Path(__file__).resolve().parents[2]
    code = source_identity(code_root)
    clock = Monotonic()
    began = clock.now()
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    map_rows: list[Record] = []
    covered = proposals = starts = 0
    waited = computed = 0.0
    with device_lease(device):
        run.guard.begin()
        model, parent, _, config, _, _ = candidate_model(section, root, run, device)
        worker = restored(model, parent, config, memory, string(section["mode"]))
        drafter = drafter_record(section, root, config, worker.updates)
        del worker
        gc.collect()
        run.guard.end("drafter load and restore")
        run.guard.begin()
        with (
            ThreadPoolExecutor(READERS) as readers,
            ThreadPoolExecutor(1) as loader,
        ):

            def load(item: Planned) -> Capture:
                return load_capture(item.row.capture, memory, cfg.hidden, readers, corpus.fusion)

            upcoming: Future[Capture] = loader.submit(load, planned[0])
            for index, item in enumerate(planned):
                started = clock.now()
                capture = upcoming.result()
                if index + 1 < len(planned):
                    upcoming = loader.submit(load, planned[index + 1])
                waited += clock.now() - started
                run.guard.check()
                started = clock.now()
                if tuple(window.anchor for window in capture.windows) != item.row.anchors or any(
                    window.trajectory != item.identity for window in capture.windows
                ):
                    raise ValueError("admitted and materialized windows disagree")
                windows = covered_windows(
                    {window.anchor: window.labels for window in capture.windows}, item.spans
                )
                anchors = generated_anchors(windows)
                ranked = propose_first(model, capture, anchors, depth, sizes[item.identity])
                found = round_starts(windows, ranked, depth)
                computed += clock.now() - started
                covered += len(windows)
                proposals += len(anchors)
                starts += len(found)
                map_rows.append(
                    {
                        "id": item.identity,
                        "covered": [list(span) for span in item.spans],
                        "starts": list(found),
                    }
                )
                del capture
        run.guard.end("round-state map")
        del model
        gc.collect()
        run.guard.release("the reference drafter")
    if source_identity(code_root) != code:
        raise ValueError("round-state map source changed during execution")
    result: Record = {
        "schema": MAP_SCHEMA,
        "corpus": {"sha256": corpus_ref.sha256, "bytes": corpus_ref.size, "format": corpus_format},
        "target_sha256": corpus.target,
        "depth": depth,
        "scope": scope,
        "drafter": drafter,
        "code_sha256": code,
        "rows": map_rows,
        "totals": {
            "rows": len(map_rows),
            "covered_windows": covered,
            "proposals": proposals,
            "round_starts": starts,
        },
    }
    encoded = canonical(result)
    path = store.root / "map.json"
    store.atomic(path, encoded)
    written = Artifact(path, identity(result), len(encoded))
    parse_map(read(path, len(encoded)), written.sha256, written.size)
    every = full_proposals(corpus)
    rate = proposals / computed if computed > 0 else None
    report: Record = {
        "schema": "round-state-map-report-v1",
        "map": reference(written),
        "code_sha256": code,
        "device": device,
        "gpu": cuda_identity() if cuda else None,
        "torch": str(torch.__version__),
        "specification": identity(raw),
        "scope": scope,
        "rows": len(map_rows),
        "covered_windows": covered,
        "proposals": proposals,
        "round_starts": starts,
        "round_start_share": starts / covered if covered else None,
        "corpus_windows": corpus.count,
        "corpus_rows": len(corpus.rows),
        "full_corpus_proposals": every,
        "seconds": {
            "total": clock.now() - began,
            "compute": computed,
            "waiting_for_captures": waited,
        },
        "proposals_per_compute_second": rate,
        "forecast_full_compute_seconds": every / rate if rate else None,
        "memory": {**summary, "measured": run.guard.measured},
    }
    store.atomic(store.root / "report.json", canonical(report))
    return report


def run_build(specification: Path, output: Path, cfg: Geometry = Geometry()) -> int:
    try:
        report = build(specification, output, cfg)
    except RuntimeError as error:
        if not fatal(error):
            raise
        message: Record = {"outcome": "fatal-memory", "detail": str(error)}
        print(json.dumps(message), file=sys.stderr, flush=True)
        return FATAL_MEMORY_EXIT
    summary: Record = {
        "map": report["map"],
        "round_starts": report["round_starts"],
        "covered_windows": report["covered_windows"],
        "seconds": report["seconds"],
        "proposals_per_compute_second": report["proposals_per_compute_second"],
        "forecast_full_compute_seconds": report["forecast_full_compute_seconds"],
    }
    print(json.dumps(summary), flush=True)
    return 0


class Arguments(argparse.Namespace):
    specification: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m w1a1_eagle.numerical.round_state_map")
    parser.add_argument("specification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(namespace=Arguments())
    output = output_path(args.output, "datasets", "round-state-map")
    code = run_build(args.specification.resolve(), output, Geometry())
    if code:
        os._exit(code)


if __name__ == "__main__":
    main()
