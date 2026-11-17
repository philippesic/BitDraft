from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Mapping, Sequence, TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.drafting import Pair, Ranked, Tally, paired_interval, simulate, tally
from w1a1_eagle.model import HE4
from w1a1_eagle.numerical.captures import Capture
from w1a1_eagle.numerical.labels import LabelFrequencies
from w1a1_eagle.records import artifact, canonical
from w1a1_eagle.records import reference as file_reference

KEYS = {"mixture", "frequent", "request_tokens", "one_bit", "low_rank", "tables", "export"}
TABLES = ("2-bit", "1-bit")
ROW_CHUNK = 16384
ACTIVATION_LIMIT = 127
REFINEMENTS = 8


Native: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.int32 | np.float32 | np.float16]]"


def payload(value: Tensor) -> bytes:
    return cast(Native, value.detach().cpu().contiguous().numpy()).tobytes()


def indices(values: list[int], device: torch.device) -> Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def positive(values: object, name: str) -> tuple[int, ...]:
    result = tuple(integer(value) for value in sequence(values))
    if not result or len(set(result)) != len(result) or any(value < 1 for value in result):
        raise ValueError(name + " needs distinct positive sizes")
    return tuple(sorted(result))


@dataclass(frozen=True)
class ScoringSettings:
    mixture: Artifact | None = None
    frequent: tuple[int, ...] = ()
    request_tokens: bool = False
    one_bit: tuple[int, ...] = ()
    rank: int = 0
    low_rank: tuple[int, ...] = ()
    tables: tuple[str, ...] = ()
    export_rounds: int = 0

    def variants(self) -> tuple[str, ...]:
        names = [f"frequent-{size}" for size in self.frequent]
        if self.request_tokens:
            names.extend(f"frequent-{size}+request" for size in self.frequent)
        names.extend(f"one-bit-{keep}" for keep in self.one_bit)
        names.extend(f"low-rank-{self.rank}-{keep}" for keep in self.low_rank)
        for table in self.tables:
            names.extend((f"table-{table}-scoring", f"table-{table}-drafter"))
        return tuple(names)


def scoring_settings(raw: Record, root: Path, hidden: int) -> ScoringSettings:
    if not set(raw) <= KEYS:
        raise ValueError("unsupported vocabulary scoring setting")
    frequent = positive(raw["frequent"], "frequent") if "frequent" in raw else ()
    if bool(frequent) != ("mixture" in raw):
        raise ValueError("frequent shortlists need the TRAIN mixture and the mixture needs sizes")
    request_tokens = raw.get("request_tokens", False)
    if type(request_tokens) is not bool or (request_tokens and not frequent):
        raise ValueError("request tokens extend the frequent shortlists")
    one_bit = positive(raw["one_bit"], "one_bit") if "one_bit" in raw else ()
    rank = 0
    low_rank: tuple[int, ...] = ()
    if "low_rank" in raw:
        section = record(raw["low_rank"])
        if set(section) != {"rank", "keep"}:
            raise ValueError("low_rank needs rank and keep")
        rank = integer(section["rank"])
        low_rank = positive(section["keep"], "low_rank keep")
        if not 1 <= rank <= hidden:
            raise ValueError("low-rank rank must be between one and the hidden width")
    tables = tuple(string(value) for value in sequence(raw.get("tables", [])))
    if len(set(tables)) != len(tables) or set(tables) - set(TABLES):
        raise ValueError("tables are distinct values from 2-bit and 1-bit")
    export_rounds = 0
    if "export" in raw:
        section = record(raw["export"])
        if set(section) != {"rounds_per_request"}:
            raise ValueError("export needs rounds_per_request")
        export_rounds = integer(section["rounds_per_request"])
        if export_rounds < 1:
            raise ValueError("export needs at least one round per request")
        if not one_bit or not low_rank or not frequent:
            raise ValueError("export needs the frequent, one-bit and low-rank settings")
    settings = ScoringSettings(
        artifact(raw["mixture"], root) if "mixture" in raw else None,
        frequent,
        request_tokens,
        one_bit,
        rank,
        low_rank,
        tables,
        export_rounds,
    )
    if not settings.variants():
        raise ValueError("vocabulary scoring needs at least one variant")
    return settings


def scoring_bytes(settings: ScoringSettings, rows: int, hidden: int, vectors: int) -> int:
    scores = 6 * vectors * rows * 4
    masks = 2 * vectors * rows
    chunk = 4 * ROW_CHUNK * hidden * 8
    factors = rows * settings.rank * 4 + hidden * hidden * 8 * 3
    per_row = 4 * rows * 8
    return (scores + masks + chunk + factors + per_row) * 5 // 4


def export_host_bytes(settings: ScoringSettings, rows: int, hidden: int) -> int:
    if not settings.export_rounds:
        return 0
    return 3 * rows * (hidden // 8 + settings.rank * 2 + 8)


def activation_codes(hidden: Tensor) -> Tensor:
    absmax = hidden.abs().amax(-1, keepdim=True)
    inverse = torch.where(
        absmax > 0,
        torch.full_like(absmax, float(ACTIVATION_LIMIT)) / absmax,
        torch.zeros_like(absmax),
    )
    normalized = (hidden * inverse).clamp(-ACTIVATION_LIMIT, ACTIVATION_LIMIT)
    return torch.round(normalized)


def signs(rows: Tensor) -> Tensor:
    return torch.where(rows >= 0, torch.ones_like(rows), -torch.ones_like(rows))


def two_bit_levels(rows: Tensor, scale: Tensor) -> Tensor:
    return torch.floor(rows / scale[:, None]).clamp(-2, 1) + 0.5


def two_bit_scale(rows: Tensor) -> Tensor:
    scale = rows.square().mean(-1).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    for _ in range(REFINEMENTS):
        levels = two_bit_levels(rows, scale)
        scale = ((rows * levels).sum(-1) / levels.square().sum(-1)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    return scale


def converted(rows: Tensor, kind: str, one_bit_scale: Tensor, two_bit: Tensor) -> Tensor:
    if kind == "1-bit":
        return signs(rows) * one_bit_scale[:, None]
    if kind == "2-bit":
        return two_bit_levels(rows, two_bit) * two_bit[:, None]
    raise ValueError("unknown table conversion")


class Tables:
    def __init__(
        self, vocabulary: Tensor, settings: ScoringSettings, frequencies: LabelFrequencies | None
    ) -> None:
        self.vocabulary = vocabulary
        self.settings = settings
        rows, width = int(vocabulary.shape[0]), int(vocabulary.shape[1])
        device = vocabulary.device
        self.rows = rows
        self.ranks = torch.full((rows,), rows + 1, dtype=torch.int64, device=device)
        if frequencies is not None:
            ranked = indices([token for token in frequencies.ranked if token < rows], device)
            self.ranks[ranked] = torch.arange(ranked.numel(), device=device)
        self.one_bit_scale = torch.empty(rows, device=device)
        self.two_bit_scale = torch.empty(rows, device=device)
        gram = torch.zeros((width, width), dtype=torch.float64, device=device)
        for start in range(0, rows, ROW_CHUNK):
            part = vocabulary[start : start + ROW_CHUNK]
            self.one_bit_scale[start : start + part.shape[0]] = part.abs().mean(-1)
            if "2-bit" in settings.tables:
                self.two_bit_scale[start : start + part.shape[0]] = two_bit_scale(part)
            if settings.rank:
                wide = part.double()
                gram += wide.T @ wide
        self.right = torch.empty((0, width), device=device)
        self.left = torch.empty((rows, 0), device=device)
        if settings.rank:
            _, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(gram))
            basis = vectors[:, -settings.rank :].flip(-1).float()
            self.right = basis.T.contiguous().half().float()
            self.left = torch.empty((rows, settings.rank), device=device)
            for start in range(0, rows, ROW_CHUNK):
                part = vocabulary[start : start + ROW_CHUNK]
                self.left[start : start + part.shape[0]] = (part @ basis).half().float()

    def one_bit_scores(self, hidden: Tensor) -> Tensor:
        codes = activation_codes(hidden)
        scores = torch.empty((hidden.shape[0], self.rows), device=hidden.device)
        for start in range(0, self.rows, ROW_CHUNK):
            part = self.vocabulary[start : start + ROW_CHUNK]
            scores[:, start : start + part.shape[0]] = (codes @ signs(part).T) * self.one_bit_scale[
                start : start + part.shape[0]
            ]
        return scores

    def low_rank_scores(self, hidden: Tensor) -> Tensor:
        return (hidden @ self.right.T) @ self.left.T

    def converted_argmax(self, hidden: Tensor, kind: str) -> Tensor:
        best = torch.full((hidden.shape[0],), -torch.inf, device=hidden.device)
        chosen = torch.zeros((hidden.shape[0],), dtype=torch.int64, device=hidden.device)
        for start in range(0, self.rows, ROW_CHUNK):
            end = min(start + ROW_CHUNK, self.rows)
            table = converted(
                self.vocabulary[start:end],
                kind,
                self.one_bit_scale[start:end],
                self.two_bit_scale[start:end],
            )
            values, indices = (hidden @ table.T).max(-1)
            better = values > best
            best = torch.where(better, values, best)
            chosen = torch.where(better, indices + start, chosen)
        return chosen

    def converted_rows(self, tokens: Tensor, kind: str) -> Tensor:
        return converted(
            self.vocabulary[tokens],
            kind,
            self.one_bit_scale[tokens],
            self.two_bit_scale[tokens],
        )


def restricted(logits: Tensor, allowed: Tensor, full: Tensor) -> Tensor:
    inside = allowed.gather(-1, full[:, None])[:, 0]
    masked = torch.where(allowed, logits, torch.full_like(logits, -torch.inf)).argmax(-1)
    return torch.where(inside, full, masked)


def rescored(logits: Tensor, candidates: Tensor, full: Tensor) -> Tensor:
    inside = (candidates == full[:, None]).any(-1)
    best = logits.gather(-1, candidates).argmax(-1)
    chosen = candidates.gather(-1, best[:, None])[:, 0]
    return torch.where(inside, full, chosen)


def contains(candidates: Tensor, labels: Tensor) -> Tensor:
    return (candidates == labels[:, None]).any(-1) & (labels >= 0)


@dataclass
class Choices:
    first: dict[int, tuple[int, ...]] = field(default_factory=dict)
    label_inside: dict[int, tuple[bool, ...]] = field(default_factory=dict)


class RequestScoring:
    def __init__(
        self,
        tables: Tables,
        model: HE4,
        capture: Capture,
        windows: Mapping[int, Sequence[int]],
        depth: int,
        export: bool,
    ) -> None:
        self.tables = tables
        self.model = model
        self.capture = capture
        self.windows = windows
        self.depth = depth
        self.export = export
        self.choices: dict[str, Choices] = {name: Choices() for name in tables.settings.variants()}
        self.hidden: dict[int, Tensor] = {}
        device = tables.vocabulary.device
        self.first_seen = torch.full(
            (tables.rows,), len(capture.tokens), dtype=torch.int64, device=device
        )
        if tables.settings.request_tokens:
            tokens = torch.tensor(capture.tokens, dtype=torch.int64, device=device)
            positions = torch.arange(len(capture.tokens), dtype=torch.int64, device=device)
            self.first_seen.scatter_reduce_(0, tokens, positions, "amin")

    def labels(self, part: tuple[int, ...]) -> Tensor:
        values = [
            self.windows[anchor][slot] if slot < len(self.windows[anchor]) else -1
            for anchor in part
            for slot in range(self.depth)
        ]
        return indices(values, self.tables.vocabulary.device)

    def store(self, name: str, part: tuple[int, ...], first: Tensor, inside: Tensor | None) -> None:
        values = cast(list[list[int]], first.view(len(part), self.depth).cpu().tolist())
        target = self.choices[name]
        for anchor, row in zip(part, values, strict=True):
            target.first[anchor] = tuple(int(token) for token in row)
        if inside is not None:
            flags = cast(list[list[bool]], inside.view(len(part), self.depth).cpu().tolist())
            for anchor, row_ in zip(part, flags, strict=True):
                target.label_inside[anchor] = tuple(bool(flag) for flag in row_)

    def __call__(
        self, part: tuple[int, ...], hidden: Tensor, logits: Tensor, ranked: Tensor
    ) -> None:
        settings = self.tables.settings
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat = logits.reshape(-1, logits.shape[-1])
        full = ranked[..., 0].reshape(-1)
        labels = self.labels(part)
        safe = labels.clamp_min(0)
        if self.export:
            for index, anchor in enumerate(part):
                self.hidden[anchor] = hidden[index].detach().cpu().clone()
        for size in settings.frequent:
            allowed = (self.tables.ranks < size)[None, :].expand(flat.shape[0], -1)
            self.store(
                f"frequent-{size}",
                part,
                restricted(flat, allowed, full),
                allowed.gather(-1, safe[:, None])[:, 0] & (labels >= 0),
            )
            if settings.request_tokens:
                ends = indices([anchor for anchor in part for _ in range(self.depth)], flat.device)
                seen = self.first_seen[None, :] <= ends[:, None]
                widened = allowed | seen
                self.store(
                    f"frequent-{size}+request",
                    part,
                    restricted(flat, widened, full),
                    widened.gather(-1, safe[:, None])[:, 0] & (labels >= 0),
                )
        coarse: list[tuple[str, Tensor, tuple[int, ...]]] = []
        if settings.one_bit:
            coarse.append(("one-bit", self.tables.one_bit_scores(flat_hidden), settings.one_bit))
        if settings.low_rank:
            coarse.append(
                (
                    f"low-rank-{settings.rank}",
                    self.tables.low_rank_scores(flat_hidden),
                    settings.low_rank,
                )
            )
        for prefix, scores, keeps in coarse:
            order = scores.topk(min(max(keeps), self.tables.rows), dim=-1).indices
            for keep in keeps:
                candidates = order[:, :keep]
                self.store(
                    f"{prefix}-{keep}",
                    part,
                    rescored(flat, candidates, full),
                    contains(candidates, labels),
                )
            del scores, order
        for kind in settings.tables:
            self.store(
                f"table-{kind}-scoring",
                part,
                self.tables.converted_argmax(flat_hidden, kind),
                None,
            )
            self.store(f"table-{kind}-drafter", part, self.drafter(part, kind), None)

    def drafter(self, part: tuple[int, ...], kind: str) -> Tensor:
        vocabulary = self.model.vocabulary
        tokens = indices(
            sorted({self.capture.tokens[anchor] for anchor in part} | {self.model.mask}),
            vocabulary.device,
        )
        saved = vocabulary[tokens].clone()
        try:
            vocabulary[tokens] = self.tables.converted_rows(tokens, kind)
            hidden = self.model.group_hidden(
                self.capture.features, self.capture.tokens, part, self.depth + 1
            )[:, 1:]
        finally:
            vocabulary[tokens] = saved
        return self.tables.converted_argmax(hidden.reshape(-1, hidden.shape[-1]), kind)


@dataclass
class VariantCounts:
    depth: int
    reached: list[int] = field(default_factory=list)
    changed: list[int] = field(default_factory=list)
    lost: list[int] = field(default_factory=list)
    outside: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reached = self.reached or [0] * self.depth
        self.changed = self.changed or [0] * self.depth
        self.lost = self.lost or [0] * self.depth
        self.outside = self.outside or [0] * self.depth

    def merge(self, other: "VariantCounts") -> None:
        for mine, theirs in (
            (self.reached, other.reached),
            (self.changed, other.changed),
            (self.lost, other.lost),
            (self.outside, other.outside),
        ):
            for index, value in enumerate(theirs):
                mine[index] += value


@dataclass
class VariantRequest:
    group: str
    category: str
    counts: VariantCounts
    rounds: Tally
    full_accepted: int
    full_rounds: int


def share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


class Scoring:
    def __init__(
        self,
        settings: ScoringSettings,
        frequencies: LabelFrequencies | None,
        model: HE4,
        depth: int,
    ) -> None:
        self.settings = settings
        self.frequencies = frequencies
        self.model = model
        self.depth = depth
        self.tables = Tables(model.vocabulary, settings, frequencies)
        self.requests: dict[str, list[VariantRequest]] = {name: [] for name in settings.variants()}
        self.exported: list[tuple[str, int, tuple[int, ...], Tensor]] = []

    def request(self, capture: Capture, windows: Mapping[int, Sequence[int]]) -> RequestScoring:
        return RequestScoring(
            self.tables, self.model, capture, windows, self.depth, bool(self.settings.export_rounds)
        )

    def add(
        self,
        identity: str,
        group: str,
        category: str,
        scoring: RequestScoring,
        windows: Mapping[int, Sequence[int]],
        proposals: Mapping[int, Ranked],
        starts: tuple[int, ...],
        full: Tally,
    ) -> None:
        for name, choices in scoring.choices.items():
            counts = VariantCounts(self.depth)
            for anchor in starts:
                labels = windows[anchor]
                for slot in range(min(self.depth, len(labels), len(proposals[anchor]))):
                    original = proposals[anchor][slot][0]
                    variant = choices.first[anchor][slot]
                    counts.reached[slot] += 1
                    counts.changed[slot] += int(variant != original)
                    counts.lost[slot] += int(original == labels[slot] and variant != labels[slot])
                    if choices.label_inside:
                        counts.outside[slot] += int(not choices.label_inside[anchor][slot])
            variant_proposals = {
                anchor: tuple((token,) for token in first)
                for anchor, first in choices.first.items()
            }
            rounds = simulate(windows, variant_proposals, self.depth)
            self.requests[name].append(
                VariantRequest(
                    group,
                    category,
                    counts,
                    tally(
                        [item.anchor for item in rounds],
                        windows,
                        variant_proposals,
                        self.depth,
                        1,
                    ),
                    full.accepted,
                    full.positions,
                )
            )
        if self.settings.export_rounds:
            for anchor in starts[: self.settings.export_rounds]:
                first = tuple(proposals[anchor][slot][0] for slot in range(self.depth))
                self.exported.append((identity, anchor, first, scoring.hidden[anchor]))

    def summary(self, rows: list[VariantRequest], sets: bool, resamples: int, seed: int) -> Record:
        counts = VariantCounts(self.depth)
        rounds = Tally(self.depth, 1)
        full_accepted = full_rounds = 0
        for row in rows:
            counts.merge(row.counts)
            rounds.merge(row.rounds)
            full_accepted += row.full_accepted
            full_rounds += row.full_rounds
        measured = rounds.record()
        variant_rate = share(rounds.accepted, rounds.positions)
        full_rate = share(full_accepted, full_rounds)
        pairs: list[Pair] = [
            (
                row.group,
                row.rounds.accepted,
                row.rounds.positions,
                row.full_accepted,
                row.full_rounds,
            )
            for row in rows
        ]
        return {
            "requests": len(rows),
            "reached": list(counts.reached),
            "first_choice_changed": [
                share(value, total) for value, total in zip(counts.changed, counts.reached)
            ],
            "correct_first_choice_lost": [
                share(value, total) for value, total in zip(counts.lost, counts.reached)
            ],
            "label_outside_candidates": [
                share(value, total) for value, total in zip(counts.outside, counts.reached)
            ]
            if sets
            else None,
            "rounds": measured,
            "full_accepted_per_round": full_rate,
            "accepted_per_round_difference": None
            if variant_rate is None or full_rate is None
            else variant_rate - full_rate,
            "paired_accepted_per_round": paired_interval(pairs, resamples, Random(seed)),
        }

    def report(self, resamples: int, seed: int) -> Record:
        variants: Record = {}
        for name, rows in self.requests.items():
            categories: dict[str, list[VariantRequest]] = {}
            for row in rows:
                categories.setdefault(row.category, []).append(row)
            sets = set_variant(name)
            variants[name] = {
                **self.summary(rows, sets, resamples, seed),
                "by_category": {
                    category: self.summary(items, sets, resamples, seed)
                    for category, items in sorted(categories.items())
                },
            }
        result: Record = {
            "variants": variants,
            "settings": {
                "frequent": list(self.settings.frequent),
                "request_tokens": self.settings.request_tokens,
                "one_bit": list(self.settings.one_bit),
                "low_rank": {"rank": self.settings.rank, "keep": list(self.settings.low_rank)},
                "tables": list(self.settings.tables),
            },
            "definitions": DEFINITIONS,
        }
        if self.frequencies is not None:
            result["train_labels"] = {
                "mixture": self.frequencies.mixture,
                "labels": self.frequencies.labels,
                "distinct_tokens": len(self.frequencies.ranked),
                "coverage": {
                    str(size): self.frequencies.coverage(size) for size in self.settings.frequent
                },
            }
        return result

    def export(self, store: Store, drafter: str, table: Record) -> Record:
        settings = self.settings
        tables = self.tables
        if not self.exported:
            raise ValueError("no round states to export")
        files: dict[str, bytes] = {}
        hidden = torch.stack([item[3] for item in self.exported]).float().contiguous()
        files["hidden.f32"] = payload(hidden)
        weights = 1 << torch.arange(32, dtype=torch.int64)
        packed: list[bytes] = []
        for start in range(0, tables.rows, ROW_CHUNK):
            part = tables.vocabulary[start : start + ROW_CHUNK].cpu()
            bits = (part >= 0).to(torch.int64).view(part.shape[0], -1, 32)
            words = (bits * weights).sum(-1)
            packed.append(payload((words - (words >= 1 << 31).long() * (1 << 32)).int()))
        files["signs.u32"] = b"".join(packed)
        files["scales.f32"] = payload(tables.one_bit_scale.float())
        files["low-rank-left.f16"] = payload(tables.left.half())
        files["low-rank-right.f16"] = payload(tables.right.half())
        if self.frequencies is None:
            raise ValueError("export needs the TRAIN label frequencies")
        ranked = torch.tensor(
            [token for token in self.frequencies.ranked if token < tables.rows][
                : max(settings.frequent)
            ],
            dtype=torch.int32,
        )
        files["frequent.i32"] = payload(ranked)
        for name, content in files.items():
            store.atomic(store.root / name, content)
        manifest: Record = {
            "schema": "vocabulary-scoring-export-v1",
            "drafter": drafter,
            "table": table,
            "rows": tables.rows,
            "hidden": int(tables.vocabulary.shape[1]),
            "depth": self.depth,
            "rank": settings.rank,
            "frequent_mixture": self.frequencies.mixture,
            "sets": [
                {"id": identity, "anchor": anchor, "first_choice": list(first)}
                for identity, anchor, first, _ in self.exported
            ],
            "files": {
                name: file_reference(Artifact.capture(store.root / name)) for name in sorted(files)
            },
            "definitions": EXPORT_DEFINITIONS,
        }
        store.atomic(store.root / "manifest.json", canonical(manifest))
        return file_reference(Artifact.capture(store.root / "manifest.json"))


def set_variant(name: str) -> bool:
    return name.startswith(("frequent-", "one-bit-", "low-rank-"))


DEFINITIONS: Record = {
    "round_states": "round starts of the unchanged drafter's simulated trajectory; per slot 1-3",
    "first_choice_changed": "share of round-start slots whose first choice differs from full scoring",
    "correct_first_choice_lost": "share of round-start slots where full scoring matched the target's token and the variant does not",
    "label_outside_candidates": "share of round-start slots whose target token is outside the variant's candidate set",
    "rounds": "the round simulation re-run with the variant's first choices, so its trajectory differs",
    "accepted_per_round_difference": "variant accepted tokens per round minus the unchanged drafter's",
    "paired_accepted_per_round": "variant minus unchanged accepted per round with a 95% bootstrap interval over source groups",
    "frequent": "the N most frequent target tokens at distinct supervised positions of the TRAIN mixture; ties by token id",
    "request": "adds every token of the request up to and including the round's anchor token",
    "one-bit": "8-bit per-token absmax activation codes against row signs (w >= 0 is +1) times mean |w| per row, the runtime W1A8 pass; top K rescored with the exact table",
    "low-rank": "best rank-r factor of the table from the eigenvectors of its Gram matrix, both factors rounded to F16; top K rescored with the exact table",
    "table-2-bit": "per-row levels {-1.5, -0.5, 0.5, 1.5} times a scale refined by least squares from the row RMS; no training",
    "table-1-bit": "per-row sign times mean |w|; no training",
    "scoring": "converted table used only for scoring the unchanged drafter's states",
    "drafter": "converted table used for the anchor and mask input rows and for scoring",
}

EXPORT_DEFINITIONS: Record = {
    "hidden.f32": "normed final states, sets x depth x hidden, little-endian F32, the scoring inputs",
    "signs.u32": "rows x ceil(hidden/32) words, bit j of word w is column 32w+j, 1 means w >= 0 (+1)",
    "scales.f32": "per-row mean |w|, the one-bit row scale",
    "low-rank-left.f16": "rows x rank, the table times the rank basis",
    "low-rank-right.f16": "rank x hidden, the transposed rank basis",
    "frequent.i32": "TRAIN label ranking, most frequent first, up to the largest shortlist",
}
