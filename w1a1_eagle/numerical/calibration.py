import hashlib
import math
import random
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as functional

from w1a1_eagle.contracts import Record, number, record
from w1a1_eagle.model import HE4, Layer, Projection, named_projections, soft_weight
from w1a1_eagle.numerical.captures import CaptureCache
from w1a1_eagle.numerical.corpus import Corpus
from w1a1_eagle.numerical.weights import Array
from w1a1_eagle.records import canonical
from w1a1_eagle.training import Features, Sample, StartCheck

BOUND_CANDIDATES = 240
BOUND_DECADES = 3.0


@dataclass(frozen=True)
class SampledSequence:
    identity: str
    features: Tensor
    tokens: tuple[int, ...]
    anchors: tuple[int, ...]


@dataclass(frozen=True)
class Selection:
    sequences: tuple[SampledSequence, ...]
    record: Record


def select(corpus: Corpus, sample: Sample, byte_budget: int) -> Selection:
    generator = random.Random(sample.seed)
    eligible = [index for index, row in enumerate(corpus.rows) if row.anchors]
    chosen = sorted(generator.sample(eligible, min(sample.sequences, len(eligible))))
    cache = CaptureCache(byte_budget, corpus.shape.hidden)
    sequences: list[SampledSequence] = []
    rows: list[Record] = []
    loaded = 0
    for index in chosen:
        row = corpus.rows[index]
        positions = sorted(
            generator.sample(range(len(row.anchors)), min(sample.windows, len(row.anchors)))
        )
        anchors = tuple(row.anchors[position] for position in positions)
        capture = cache.load(row.capture)
        end = max(anchors) + 1
        features = capture.features[:end].clone()
        tokens = capture.tokens[:end]
        del capture
        loaded += features.numel() * features.element_size()
        if loaded > byte_budget // 4:
            raise ValueError("calibration sample features exceed a quarter of the memory budget")
        sequences.append(SampledSequence(row.prompt.identity, features, tokens, anchors))
        rows.append(
            {
                "id": row.prompt.identity,
                "capture": CaptureCache.key(row.capture),
                "anchors": list(anchors),
            }
        )
    cache.clear()
    if cache.readers is not None:
        cache.readers.shutdown()
    windows = sum(len(item.anchors) for item in sequences)
    identity: Record = {"sample": sample.encode(), "rows": rows}
    return Selection(
        tuple(sequences),
        {
            **identity,
            "sequences": len(sequences),
            "windows": windows,
            "sha256": hashlib.sha256(canonical(identity)).hexdigest(),
            "role": "train",
        },
    )


def layer_projections(model: HE4) -> list[tuple[str, tuple[Projection, Projection, Projection]]]:
    result: list[tuple[str, tuple[Projection, Projection, Projection]]] = []
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        result.append((f"blocks.{index}.", (block.gate, block.up, block.down)))
    return result


def recorded_inputs(
    model: HE4, selection: Selection, slots: int, targets: list[Projection]
) -> list[list[Tensor]]:
    for module in targets:
        module.recorded = []
    try:
        with torch.no_grad():
            for item in selection.sequences:
                model.group_hidden(item.features, item.tokens, item.anchors, slots)
        return [list(cast(list[Tensor], module.recorded)) for module in targets]
    finally:
        for module in targets:
            module.recorded = None


def rounded_values(projection: Projection, value: Tensor) -> Tensor:
    codes, step = projection.quantize(value)
    return codes * step


def rounding_error(gained: Tensor, bound: Tensor, maximum: int) -> Tensor:
    clipped = torch.maximum(torch.minimum(gained, bound), -bound)
    rounded = (clipped * (maximum / bound)).round().clamp(-maximum, maximum)
    return (gained - rounded * (bound / maximum)).double().square().sum()


def choose_bound(projection: Projection, inputs: list[Tensor]) -> float:
    gain = projection.log_gain.detach().exp()
    gained = [value.float() * gain for value in inputs]
    peak = max(float(value.abs().max()) for value in gained)
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("calibration inputs must be finite and not all zero")
    candidates = [
        peak * math.pow(10.0, BOUND_DECADES * (index / BOUND_CANDIDATES - 1.0))
        for index in range(BOUND_CANDIDATES + 1)
    ]
    errors: list[float] = []
    device = gained[0].device
    for candidate in candidates:
        bound = torch.tensor(candidate, dtype=torch.float32, device=device)
        errors.append(
            float(sum(rounding_error(value, bound, projection.maximum) for value in gained))
        )
    return candidates[errors.index(min(errors))]


def fit_rows(
    projection: Projection, inputs: list[Tensor], source: Tensor, fit_midpoint: bool
) -> Record:
    signs = torch.where(projection.latent_sign.detach() >= 0, 1.0, -1.0)
    rows = source.shape[0]
    device = source.device
    zero = torch.zeros(rows, dtype=torch.float64, device=device)
    aa, ac, cc, ay, cy = zero, zero, zero, zero, zero
    target = torch.zeros((), dtype=torch.float64, device=device)
    for value in inputs:
        values = rounded_values(projection, value).detach()
        flat = values.reshape(-1, values.shape[-1])
        expected = functional.linear(flat, source).double()
        product = functional.linear(flat, signs).double()
        total = flat.sum(-1, keepdim=True).double()
        aa = aa + product.square().sum(0)
        ac = ac + (product * total).sum(0)
        cc = cc + total.square().sum()
        ay = ay + (product * expected).sum(0)
        cy = cy + (total * expected).sum(0)
        target = target + expected.square().sum()
    initial = projection.scale.detach().double().clamp_min(0)
    scale = torch.where(aa > 0, ay / torch.where(aa > 0, aa, 1.0), initial)
    midpoint = torch.zeros_like(scale)
    if fit_midpoint:
        determinant = aa * cc - ac.square()
        solvable = determinant > 1e-12 * aa * cc
        safe = torch.where(solvable, determinant, 1.0)
        scale = torch.where(solvable, (ay * cc - ac * cy) / safe, scale)
        midpoint = torch.where(solvable, (aa * cy - ac * ay) / safe, midpoint)
    kept = scale <= 0
    scale = torch.where(kept, initial, scale)
    if fit_midpoint:
        refit = (cy - scale * ac) / torch.where(cc > 0, cc, 1.0)
        midpoint = torch.where(kept & (cc > 0), refit, midpoint)
    residual = (
        target
        - 2 * (scale * ay + midpoint * cy).sum()
        + (scale.square() * aa + 2 * scale * midpoint * ac + midpoint.square() * cc).sum()
    )
    with torch.no_grad():
        projection.scale.copy_(scale.float())
        projection.midpoint.copy_(midpoint.float())
    return {
        "relative_residual": float(residual.clamp_min(0) / target) if float(target) > 0 else None,
        "mean_scale": float(scale.mean()),
        "mean_abs_midpoint": float(midpoint.abs().mean()),
        "kept_initial_scale_rows": int(kept.sum()),
    }


def set_bound(projection: Projection, bound: float) -> None:
    with torch.no_grad():
        projection.log_clip.fill_(math.log(bound))


def calibrate(
    model: HE4,
    corpus: Corpus,
    sample: Sample,
    features: Features,
    slots: int,
    byte_budget: int,
    soft: bool,
) -> Record:
    selection = select(corpus, sample, byte_budget)
    fit_midpoint = features.midpoint == "learned"
    tables: Record = {}
    for prefix, (gate, up, down) in layer_projections(model):
        for stage in (((prefix + "gate", gate), (prefix + "up", up)), ((prefix + "down", down),)):
            first = stage[0][1]
            (inputs,) = recorded_inputs(model, selection, slots, [first])
            set_bound(first, choose_bound(first, inputs))
            for name, projection in stage:
                source = projection.latent_sign.detach()
                fitted: Record = {} if soft else fit_rows(projection, inputs, source, fit_midpoint)
                tables[name] = {"clip_bound": float(projection.log_clip.detach().exp()), **fitted}
            del inputs
    digest = hashlib.sha256()
    for name, projection in named_projections(model):
        for part, value in (
            ("scale", projection.scale),
            ("midpoint", projection.midpoint),
            ("log_clip", projection.log_clip),
        ):
            data = value.detach().float().cpu().contiguous()
            digest.update(canonical([name, part, list(data.shape)]))
            digest.update(cast(Array, data.numpy()).tobytes())
    return {
        "sample": selection.record,
        "slots": slots,
        "fitted": {
            "clip": "bound minimizing input rounding MSE on a geometric grid",
            "scale": not soft,
            "midpoint": fit_midpoint and not soft,
            "gain": False,
        },
        "values_sha256": digest.hexdigest(),
        "tables": tables,
    }


def relative_errors(module: Projection, inputs: list[Tensor], temperature: float) -> Record:
    errors: list[Tensor] = []
    skipped = 0
    with torch.no_grad():
        source = module.latent_sign.detach()
        weight = soft_weight(source, module.scale.detach().clamp_min(0), temperature)
        for value in inputs:
            values = rounded_values(module, value).reshape(-1, value.shape[-1])
            soft = (
                functional.linear(values, weight)
                + values.sum(-1, keepdim=True) * module.midpoint.detach()
            )
            public = functional.linear(values, source)
            reference = public.double().square().sum(-1).sqrt()
            difference = (soft - public).double().square().sum(-1).sqrt()
            nonzero = reference > 0
            skipped += int((~nonzero).sum())
            errors.append(difference[nonzero] / reference[nonzero])
    joined = torch.cat(errors)
    if not joined.numel():
        raise ValueError("start check has no nonzero public outputs")
    return {
        "max_relative_error": float(joined.max()),
        "mean_relative_error": float(joined.mean()),
        "rows": int(joined.numel()),
        "zero_rows": skipped,
    }


def start_check(
    model: HE4,
    corpus: Corpus,
    check: StartCheck,
    temperature: float,
    slots: int,
    byte_budget: int,
) -> Record:
    selection = select(corpus, check.sample, byte_budget)
    named = named_projections(model)
    for _, module in named:
        module.temperature = temperature
    tables: Record = {}
    worst = 0.0
    try:
        for prefix, (gate, up, down) in layer_projections(model):
            shared, separate = recorded_inputs(model, selection, slots, [gate, down])
            for name, module, inputs in (
                (prefix + "gate", gate, shared),
                (prefix + "up", up, shared),
                (prefix + "down", down, separate),
            ):
                measured = relative_errors(module, inputs, temperature)
                worst = max(worst, number(measured["max_relative_error"]))
                tables[name] = measured
            del shared, separate
    finally:
        for _, module in named:
            module.temperature = None
    means = [number(record(value)["mean_relative_error"]) for value in tables.values()]
    passed = all(
        number(record(value)["max_relative_error"]) <= check.max_relative_error
        and number(record(value)["mean_relative_error"]) <= check.mean_relative_error
        for value in tables.values()
    )
    return {
        "kind": "gradual-precision-start-check",
        "temperature": temperature,
        "sample": selection.record,
        "slots": slots,
        "metric": "per input row: L2 norm of (soft table - public table) output over L2 norm of"
        " public table output, same rounded input codes",
        "tolerance": {
            "max_relative_error": check.max_relative_error,
            "mean_relative_error": check.mean_relative_error,
        },
        "max_relative_error": worst,
        "worst_mean_relative_error": max(means),
        "tables": tables,
        "passed": passed,
    }
