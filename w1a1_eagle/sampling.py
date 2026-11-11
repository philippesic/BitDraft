import math
import random
from collections import defaultdict


class TaskSampler:
    def __init__(
        self,
        counts: tuple[int, ...],
        strata: tuple[str, ...],
        weights: dict[str, float],
        seed: int,
        burst: int,
    ) -> None:
        if (
            len(counts) != len(strata)
            or not counts
            or type(seed) is not int
            or seed < 0
            or not 1 <= burst <= 64
        ):
            raise ValueError("invalid hierarchical sampler dimensions/seed/locality")
        if set(strata) != set(weights) or any(
            not math.isfinite(value) or value <= 0 for value in weights.values()
        ):
            raise ValueError("positive explicit weights required for every data stratum")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (count, name) in enumerate(zip(counts, strata, strict=True)):
            if type(count) is not int or count < 0:
                raise ValueError("invalid eligible position count")
            if count:
                grouped[name].append(index)
        if set(grouped) != set(weights):
            raise ValueError("a requested data stratum has no supervised positions")
        self.groups = {name: tuple(indices) for name, indices in grouped.items()}
        self.names = sorted(weights)
        self.weights = [weights[name] for name in self.names]
        self.counts = counts
        self.seed = seed
        self.burst = burst

    def locate(self, cursor: int) -> tuple[int, int]:
        if type(cursor) is not int or cursor < 0:
            raise ValueError("nonnegative sampler cursor required")
        generator = random.Random(self.seed + (cursor // self.burst) * 14029467366897019727)
        stratum = generator.choices(self.names, weights=self.weights, k=1)[0]
        row = generator.choice(self.groups[stratum])
        position = 0
        for _ in range(cursor % self.burst + 1):
            position = generator.randrange(self.counts[row])
        return row, position
