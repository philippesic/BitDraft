import math
from dataclasses import asdict, dataclass
from typing import cast

from w1a1_eagle.contracts import Record, integer, number, record, string


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
