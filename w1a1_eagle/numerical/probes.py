import heapq
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from w1a1_eagle.contracts import Record
from w1a1_eagle.drafting import CONTEXT_BANDS, context_band
from w1a1_eagle.model import HE4, Layer, Projection, hard_sign

FEED_FORWARD = ("gate", "up", "down")
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
EDGES = tuple(math.pow(2.0, step / 8) for step in range(-16 * 8, 8 * 8 + 1))
SPIKES = 10
ABOVE_BOUND = EDGES.index(1.0) + 1


@dataclass(frozen=True)
class Ablation:
    gains_one: bool = False
    midpoints_zero: bool = False
    unrounded: bool = False
    unclipped: bool = False


ABLATIONS = {
    "unchanged": Ablation(),
    "gains-one": Ablation(gains_one=True),
    "midpoints-zero": Ablation(midpoints_zero=True),
    "gains-one-midpoints-zero": Ablation(gains_one=True, midpoints_zero=True),
    "inputs-unrounded": Ablation(unrounded=True),
    "inputs-unrounded-unclipped": Ablation(unrounded=True, unclipped=True),
}
EXPORTABLE = ("unchanged", "gains-one", "midpoints-zero", "gains-one-midpoints-zero")


class AblatedProjection(nn.Module):
    def __init__(self, original: Projection, ablation: Ablation) -> None:
        super().__init__()
        self.original = original
        self.ablation = ablation

    def forward(self, value: Tensor, block: int | None = None) -> Tensor:
        projection, ablation = self.original, self.ablation
        gain = torch.zeros_like(projection.log_gain) if ablation.gains_one else projection.log_gain
        gained = value.float() * gain.exp()
        if not bool(torch.isfinite(gained).all()):
            raise ValueError("nonfinite quantizer input")
        bound = projection.bound(value.shape[0], block)
        maximum = (1 << (projection.bits - 1)) - 1
        clipped = (
            gained if ablation.unclipped else torch.maximum(torch.minimum(gained, bound), -bound)
        )
        normalized = clipped * (maximum / bound)
        if ablation.unrounded:
            codes = normalized
        else:
            rounded = normalized.round().clamp(-maximum, maximum)
            codes = normalized + (rounded - normalized)
        step = bound / maximum
        signs = hard_sign(projection.latent_sign)
        scale = projection.scale.clamp_min(0)
        midpoint = (
            torch.zeros_like(projection.midpoint)
            if ablation.midpoints_zero
            else projection.midpoint
        )
        return (functional.linear(codes, signs) * scale) * step + (
            codes.sum(-1, keepdim=True) * step
        ) * midpoint


def projections(model: HE4) -> Iterator[tuple[int, str, Projection]]:
    for index, module in enumerate(model.blocks):
        block = cast(Layer, module)
        for name in FEED_FORWARD:
            projection: object = getattr(block, name)
            if not isinstance(projection, Projection):
                raise ValueError("feed-forward projections are already substituted")
            yield index, name, projection


@contextmanager
def substituted(
    model: HE4, replacement: Callable[[int, str, Projection], nn.Module | None]
) -> Iterator[None]:
    originals: list[tuple[Layer, str, Projection]] = []
    try:
        for index, name, projection in list(projections(model)):
            module = replacement(index, name, projection)
            if module is not None:
                block = cast(Layer, model.blocks[index])
                originals.append((block, name, projection))
                setattr(block, name, module)
        yield
    finally:
        for block, name, projection in originals:
            setattr(block, name, projection)


@contextmanager
def ablated(model: HE4, name: str) -> Iterator[None]:
    if name not in ABLATIONS:
        raise ValueError("unknown ablation: " + name)
    ablation = ABLATIONS[name]
    if ablation == Ablation():
        yield
        return
    with substituted(model, lambda _, __, projection: AblatedProjection(projection, ablation)):
        yield


def apply_ablation(model: HE4, name: str) -> None:
    if name not in EXPORTABLE:
        raise ValueError("only parameter ablations can be exported: " + name)
    ablation = ABLATIONS[name]
    with torch.no_grad():
        for _, _, projection in projections(model):
            if ablation.gains_one:
                projection.log_gain.zero_()
            if ablation.midpoints_zero:
                projection.midpoint.zero_()


@dataclass
class Window:
    name: str
    anchor: int
    half: int


class InputRange:
    def __init__(self, slots: int, width: int, device: torch.device) -> None:
        self.slots = slots
        self.width = width
        self.bound = 0.0
        self.anchor_bound: float | None = None
        self.count = torch.zeros(slots, dtype=torch.int64, device=device)
        self.clipped = torch.zeros(slots, dtype=torch.int64, device=device)
        self.maximum = torch.zeros(slots, dtype=torch.float32, device=device)
        self.histogram = torch.zeros((2, slots, len(EDGES) + 1), dtype=torch.int64, device=device)
        self.channels = torch.zeros(width, dtype=torch.int64, device=device)
        self.bands = {band: [0, 0] for band in CONTEXT_BANDS}
        self.rows: list[int] = []
        self.spikes: list[tuple[float, int, str, int, int, int]] = []
        self.order = 0

    def observe(
        self, ratio: Tensor, bound: float, window: Window, anchor_bound: float | None = None
    ) -> None:
        slots = ratio.shape[0]
        if slots > self.slots or ratio.shape[1] != self.width:
            raise ValueError("probe geometry differs from the block")
        self.bound = bound
        self.anchor_bound = anchor_bound
        edges = torch.tensor(EDGES, dtype=torch.float32, device=ratio.device)
        over = ratio > 1
        per_row = over.sum(-1)
        self.count[:slots] += ratio.shape[1]
        self.clipped[:slots] += per_row
        self.maximum[:slots] = torch.maximum(self.maximum[:slots], ratio.max(-1).values)
        buckets = torch.bucketize(ratio, edges)
        for slot in range(slots):
            self.histogram[window.half, slot] += torch.bincount(
                buckets[slot], minlength=len(EDGES) + 1
            )
        self.channels += over.sum(0)
        band = context_band(window.anchor)
        self.bands[band][0] += int(per_row.sum())
        self.bands[band][1] += ratio.numel()
        self.rows.extend(cast(list[int], per_row.tolist()))
        peaks, channels = ratio.max(-1)
        peak_values = cast(list[float], peaks.tolist())
        peak_channels = cast(list[int], channels.tolist())
        for slot, (peak, channel) in enumerate(zip(peak_values, peak_channels, strict=True)):
            self.order += 1
            entry = (float(peak), -self.order, window.name, window.anchor, slot, int(channel))
            if len(self.spikes) < SPIKES:
                heapq.heappush(self.spikes, entry)
            elif entry > self.spikes[0]:
                heapq.heapreplace(self.spikes, entry)

    def record(self) -> Record:
        histogram = self.histogram.cpu()
        total_clipped = int(self.clipped.sum())
        total = int(self.count.sum())
        channels = sorted(cast(list[int], self.channels.cpu().tolist()), reverse=True)
        rows = sorted(self.rows, reverse=True)
        top_rows = max(1, len(rows) // 100)
        slot_names = ["anchor"] + [f"mask-{slot}" for slot in range(1, self.slots)]

        def slot_bound(slot: int) -> float:
            return self.anchor_bound if slot == 0 and self.anchor_bound is not None else self.bound

        def quantiles(counts: Tensor) -> list[float | None]:
            cumulative = counts.cumsum(0)
            size = int(cumulative[-1]) if cumulative.numel() else 0
            result: list[float | None] = []
            for quantile in QUANTILES:
                if not size:
                    result.append(None)
                    continue
                needed = torch.full((1,), math.ceil(quantile * size), dtype=cumulative.dtype)
                position = int(torch.searchsorted(cumulative, needed)[0])
                result.append(EDGES[min(position, len(EDGES) - 1)])
            return result

        per_slot = []
        for slot in range(self.slots):
            merged = histogram[:, slot].sum(0)
            count = int(self.count[slot])
            per_slot.append(
                {
                    "slot": slot_names[slot],
                    "values": count,
                    "clipped_fraction": int(self.clipped[slot]) / count if count else None,
                    "max_abs": float(self.maximum[slot]) * slot_bound(slot),
                    "max_over_bound": float(self.maximum[slot]),
                    "quantiles_over_bound": dict(
                        zip((str(q) for q in QUANTILES), quantiles(merged), strict=True)
                    ),
                    "halves_quantiles_over_bound": [
                        dict(
                            zip(
                                (str(q) for q in QUANTILES),
                                quantiles(histogram[half, slot]),
                                strict=True,
                            )
                        )
                        for half in range(2)
                    ],
                    "halves_clipped_values": [
                        int(histogram[half, slot][ABOVE_BOUND:].sum()) for half in range(2)
                    ],
                }
            )
        order = [
            [
                slot
                for _, slot in sorted(
                    (per_slot_value(per_slot[slot], half), slot) for slot in range(self.slots)
                )
            ]
            for half in range(2)
        ]
        return {
            "bound": self.bound,
            **({} if self.anchor_bound is None else {"anchor_bound": self.anchor_bound}),
            "values": total,
            "clipped": total_clipped,
            "clipped_fraction": total_clipped / total if total else None,
            "clipped_share_anchor_slot": int(self.clipped[0]) / total_clipped
            if total_clipped
            else None,
            "clipped_share_top_10_channels": sum(channels[:10]) / total_clipped
            if total_clipped
            else None,
            "clipped_share_top_1pct_rows": sum(rows[:top_rows]) / total_clipped
            if total_clipped
            else None,
            "clipped_by_context_band": {
                band: {
                    "clipped": values[0],
                    "values": values[1],
                    "fraction": values[0] / values[1] if values[1] else None,
                }
                for band, values in self.bands.items()
            },
            "slots": per_slot,
            "slot_order_by_p999_per_half": [[slot_names[slot] for slot in row] for row in order],
            "slot_order_repeats": order[0] == order[1],
            "largest_spikes": [
                {
                    "over_bound": peak,
                    "abs": peak * slot_bound(slot),
                    "window": name,
                    "anchor": anchor,
                    "slot": slot_names[slot],
                    "channel": channel,
                }
                for peak, _, name, anchor, slot, channel in sorted(self.spikes, reverse=True)
            ],
        }


def per_slot_value(row: Record, half: int) -> float:
    halves = cast(list[dict[str, float | None]], row["halves_quantiles_over_bound"])
    value = halves[half]["0.999"]
    return -1.0 if value is None else value


class ProbedProjection(nn.Module):
    def __init__(self, original: Projection, probe: "RangeProbe", key: tuple[int, str]) -> None:
        super().__init__()
        self.original = original
        self.probe = probe
        self.key = key

    def forward(self, value: Tensor, block: int | None = None) -> Tensor:
        self.probe.observe(self.key, self.original, value, block)
        return self.original.forward(value, block)


class RangeProbe:
    def __init__(self, model: HE4, slots: int) -> None:
        self.model = model
        self.slots = slots
        self.window: Window | None = None
        self.ranges: dict[tuple[int, str], InputRange] = {}
        device = model.vocabulary.device
        for index, name, projection in projections(model):
            if name != "up":
                self.ranges[(index, name)] = InputRange(
                    slots, projection.latent_sign.shape[1], device
                )

    def observe(
        self, key: tuple[int, str], projection: Projection, value: Tensor, block: int | None
    ) -> None:
        if self.window is None:
            raise ValueError("range probe requires a declared window")
        with torch.no_grad():
            bound = projection.bound(value.shape[0], block)
            ratio = (value.float() * projection.log_gain.exp()).abs() / bound
            anchor = projection.log_anchor_clip
            self.ranges[key].observe(
                ratio,
                float(projection.log_clip.exp()),
                self.window,
                None if anchor is None else float(anchor.exp()),
            )

    @contextmanager
    def attached(self) -> Iterator[None]:
        with substituted(
            self.model,
            lambda index, name, projection: (
                None if name == "up" else ProbedProjection(projection, self, (index, name))
            ),
        ):
            yield

    def record(self) -> Record:
        return {
            f"layer-{index}": {
                ("gate_up" if name == "gate" else "down"): self.ranges[(index, name)].record()
                for name in ("gate", "down")
            }
            for index in sorted({key[0] for key in self.ranges})
        }
