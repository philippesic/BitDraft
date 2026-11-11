import random
from collections import defaultdict
from typing import Callable

from w1a1_eagle.contracts import Record, integer, number, record, string


def measures(rows: list[Record]) -> tuple[float, float | None]:
    tokens = sum(integer(row["tokens"]) for row in rows)
    seconds = sum(number(row["wall_seconds"]) for row in rows)
    proposed = sum(integer(record(row["native"])["proposed"]) for row in rows)
    accepted = sum(integer(record(row["native"])["accepted"]) for row in rows)
    if seconds <= 0:
        raise ValueError("positive request time required")
    return tokens / seconds, accepted / proposed if proposed else None


def interval(values: list[float]) -> Record:
    if not values:
        return {"available": False}
    ordered = sorted(values)
    return {
        "lower": ordered[int((len(ordered) - 1) * 0.025)],
        "upper": ordered[int((len(ordered) - 1) * 0.975)],
    }


def paired(rows: list[Record], candidate: str, baseline: str, draws: int = 2000) -> Record:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    grouped: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        arm = string(row["arm"])
        if arm in {candidate, baseline}:
            grouped[string(row["group"])][arm].append(row)
    names = sorted(grouped)
    if not names or any(set(grouped[name]) != {candidate, baseline} for name in names):
        raise ValueError("paired comparison requires both arms in every source group")
    for name in names:
        left = {
            (string(row["task"]), integer(row["rep"]), integer(row.get("seed", 8101)))
            for row in grouped[name][candidate]
        }
        right = {
            (string(row["task"]), integer(row["rep"]), integer(row.get("seed", 8101)))
            for row in grouped[name][baseline]
        }
        if left != right:
            raise ValueError("unmatched requests or repeats in comparison")

    def draw(indices: list[str]) -> tuple[float, float | None]:
        a, aa = measures([row for name in indices for row in grouped[name][candidate]])
        b, ba = measures([row for name in indices for row in grouped[name][baseline]])
        return a / b, aa - ba if aa is not None and ba is not None else None

    estimate, acceptance = draw(names)
    if len(names) < 2:
        return {
            "groups": len(names),
            "speed_ratio": estimate,
            "acceptance_difference": acceptance,
            "interval_available": False,
        }
    generator = random.Random(8101)
    ratios: list[float] = []
    differences: list[float] = []
    for _ in range(draws):
        ratio, difference = draw(generator.choices(names, k=len(names)))
        ratios.append(ratio)
        if difference is not None:
            differences.append(difference)
    return {
        "groups": len(names),
        "speed_ratio": estimate,
        "acceptance_difference": acceptance,
        "speed_ratio_95_interval": interval(ratios),
        "acceptance_difference_95_interval": interval(differences),
        "bootstrap_draws": draws,
        "scope": "paired source-group bootstrap; repeats and harness variants stay together; small cohorts do not establish deployment transfer",
    }


def partitions(
    rows: list[Record], field: str, summarize: Callable[[list[Record]], Record]
) -> Record:
    groups: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}
