import math
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor

from w1a1_eagle.contracts import Record, integer, number, record, string
from w1a1_eagle.model import HE4, named_projections


@dataclass
class Accounting:
    dataset: str | None = None
    dataset_start_updates: int = 0
    dataset_labels: int = 0
    observed_labels: int = 0
    successful_update_seconds: float = 0.0
    failed_update_seconds: float = 0.0
    data_seconds: float = 0.0

    def validate(self) -> None:
        if self.dataset is not None and (
            len(self.dataset) != 64 or any(c not in "0123456789abcdef" for c in self.dataset)
        ):
            raise ValueError("explicit dataset SHA256 required")
        if any(
            type(value) is not int or value < 0
            for value in (self.dataset_start_updates, self.dataset_labels, self.observed_labels)
        ):
            raise ValueError("invalid dataset exposure counters")
        if self.dataset_labels > self.observed_labels or any(
            not math.isfinite(value) or value < 0
            for value in (
                self.successful_update_seconds,
                self.failed_update_seconds,
                self.data_seconds,
            )
        ):
            raise ValueError("invalid cumulative model/data accounting")

    def encode(self) -> Record:
        self.validate()
        return cast(Record, asdict(self))

    @classmethod
    def parse(cls, value: object) -> "Accounting":
        raw = record(value)
        if set(raw) != {
            "dataset",
            "dataset_start_updates",
            "dataset_labels",
            "observed_labels",
            "successful_update_seconds",
            "failed_update_seconds",
            "data_seconds",
        }:
            raise ValueError("incomplete dataset/clock accounting")
        result = cls(
            dataset=string(raw["dataset"]) if raw["dataset"] is not None else None,
            dataset_start_updates=integer(raw["dataset_start_updates"]),
            dataset_labels=integer(raw["dataset_labels"]),
            observed_labels=integer(raw["observed_labels"]),
            successful_update_seconds=number(raw["successful_update_seconds"]),
            failed_update_seconds=number(raw["failed_update_seconds"]),
            data_seconds=number(raw["data_seconds"]),
        )
        result.validate()
        return result


CLIP_LIMIT = 1.0


def packed_signs(value: Tensor) -> Tensor:
    bits = (value.detach() >= 0).reshape(-1)
    bits = torch.nn.functional.pad(bits, (0, (-bits.numel()) % 8)).view(-1, 8).to(torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8, device=value.device)
    return (bits << shifts).sum(-1, dtype=torch.uint8)


def unpacked_signs(value: Tensor, count: int) -> Tensor:
    shifts = torch.arange(8, dtype=torch.uint8, device=value.device)
    return ((value.unsqueeze(-1) >> shifts) & 1).reshape(-1)[:count].bool()


class Telemetry:
    def __init__(self, model: HE4) -> None:
        self.model = model
        self.projections = tuple(named_projections(model))
        self.device = model.vocabulary.device
        self.counts = {
            name: torch.zeros(2, dtype=torch.float64, device=self.device)
            for name, _ in self.projections
        }
        self.signs: dict[str, Tensor] | None = None
        self.started = 0
        self.labels = 0
        self.seconds = 0.0
        self.reset()

    def reset(self) -> None:
        self.updates = 0
        self.norm_sums: dict[str, Tensor] = {}
        self.norm_last: dict[str, Tensor] = {}
        self.factor_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.factor_last = torch.ones((), dtype=torch.float64, device=self.device)
        self.acted = torch.zeros((), dtype=torch.int64, device=self.device)
        for name, module in self.projections:
            self.counts[name].zero_()
            module.observed = 0

    def begin(self, updates: int, accounting: Accounting) -> None:
        if self.signs is None:
            self.signs = {
                name: packed_signs(module.latent_sign) for name, module in self.projections
            }
            self.started = updates
            self.labels = accounting.observed_labels
            self.seconds = accounting.successful_update_seconds
            self.reset()

    def attach(self) -> None:
        for name, module in self.projections:
            module.statistics = self.counts[name]

    def detach(self) -> None:
        for _, module in self.projections:
            module.statistics = None

    def gradients(self, families: dict[str, list[Tensor]]) -> None:
        with torch.no_grad():
            for family, gradients in families.items():
                if not gradients:
                    continue
                norm = (
                    torch.stack([value.square().sum() for value in gradients]).sum().sqrt().double()
                )
                self.norm_last[family] = norm
                previous = self.norm_sums.get(family)
                self.norm_sums[family] = norm if previous is None else previous + norm

    def clipping(self, total: Tensor) -> None:
        with torch.no_grad():
            factor = torch.clamp(CLIP_LIMIT / (total.double() + 1e-6), max=1.0)
            self.factor_last = factor
            self.factor_sum = self.factor_sum + factor
            self.acted = self.acted + (factor < 1).to(torch.int64)
        self.updates += 1

    def record(
        self, updates: int, accounting: Accounting, precision: Record | None, soft: bool
    ) -> Record:
        signs = self.signs
        if signs is None:
            raise ValueError("telemetry records follow at least one update")
        measured = max(1, self.updates)
        flips: dict[str, int] = {}
        bounded = 0
        total = 0
        current: dict[str, Tensor] = {}
        inputs: Record = {}
        with torch.no_grad():
            for name, module in self.projections:
                latent = module.latent_sign.detach()
                previous = unpacked_signs(signs[name], latent.numel())
                flips[name] = int(((latent.reshape(-1) >= 0) != previous).sum())
                bounded += int((latent.abs() >= 1).sum())
                total += latent.numel()
                current[name] = packed_signs(latent)
                clipped, zero = float(self.counts[name][0]), float(self.counts[name][1])
                observed = module.observed
                inputs[name] = {
                    "inputs": observed,
                    "clipped_share": clipped / observed if observed else None,
                    "zero_code_share": zero / observed if observed else None,
                }
        seconds = accounting.successful_update_seconds - self.seconds
        labels = accounting.observed_labels - self.labels
        cuda = self.device.type == "cuda"
        result: Record = {
            "since_updates": self.started,
            "updates": self.updates,
            "gradient_norms": {
                family: {
                    "last": float(self.norm_last[family]),
                    "mean": float(value) / measured,
                }
                for family, value in self.norm_sums.items()
            },
            "clipping": {
                "limit": CLIP_LIMIT,
                "last_factor": float(self.factor_last),
                "mean_factor": float(self.factor_sum) / measured if self.updates else 1.0,
                "acted": int(self.acted),
            },
            "signs": {
                "flipped": sum(flips.values()),
                "flipped_share": sum(flips.values()) / total,
                "by_projection": cast(Record, flips),
                "at_bound_share": None if soft else bounded / total,
            },
            "inputs": inputs,
            "precision": precision,
            "memory": {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            }
            if cuda
            else None,
            "labels_per_second": labels / seconds if seconds > 0 else None,
        }
        self.signs = current
        self.started = updates
        self.labels = accounting.observed_labels
        self.seconds = accounting.successful_update_seconds
        self.reset()
        return result
