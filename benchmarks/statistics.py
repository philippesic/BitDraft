import random
from collections import Counter, defaultdict
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


def interval(values: list[float], alpha: float = 0.05) -> Record:
    if not 0 < alpha < 1:
        raise ValueError("interval alpha must be between zero and one")
    if not values:
        return {"available": False}
    ordered = sorted(values)
    return {
        "lower": ordered[int((len(ordered) - 1) * alpha / 2)],
        "upper": ordered[int((len(ordered) - 1) * (1 - alpha / 2))],
    }


def paired(
    rows: list[Record], candidate: str, baseline: str, draws: int = 2000, family: int = 1
) -> Record:
    if draws < 100 or type(family) is not int or family < 1:
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
        if (
            len(left) != len(grouped[name][candidate])
            or len(right) != len(grouped[name][baseline])
            or left != right
        ):
            raise ValueError("unmatched requests or repeats in comparison")

    def draw(indices: list[str]) -> tuple[float, float | None]:
        a, aa = measures([row for name in indices for row in grouped[name][candidate]])
        b, ba = measures([row for name in indices for row in grouped[name][baseline]])
        return a / b, aa - ba if aa is not None and ba is not None else None

    estimate, acceptance = draw(names)
    strata: dict[str, list[str]] = defaultdict(list)
    for name in names:
        composition = Counter(
            string(row.get("category", "unclassified")) for row in grouped[name][baseline]
        )
        strata[str(sorted(composition.items()))].append(name)
    if any(len(members) < 2 for members in strata.values()):
        return {
            "groups": len(names),
            "speed_ratio": estimate,
            "acceptance_difference": acceptance,
            "interval_available": False,
            "reason": "insufficient independent groups within a category-composition stratum",
            "strata": {key: len(value) for key, value in strata.items()},
        }
    generator = random.Random(8101)
    ratios: list[float] = []
    differences: list[float] = []
    for _ in range(draws):
        indices = [
            name
            for members in strata.values()
            for name in generator.choices(members, k=len(members))
        ]
        ratio, difference = draw(indices)
        ratios.append(ratio)
        if difference is not None:
            differences.append(difference)
    return {
        "groups": len(names),
        "speed_ratio": estimate,
        "acceptance_difference": acceptance,
        "speed_ratio_95_interval": interval(ratios),
        "acceptance_difference_95_interval": interval(differences),
        "simultaneous_speed_ratio_interval": interval(ratios, 0.05 / family),
        "simultaneous_acceptance_difference_interval": interval(differences, 0.05 / family),
        "comparison_family": family,
        "strata": {key: len(value) for key, value in strata.items()},
        "bootstrap_draws": draws,
        "interval_available": True,
        "scope": "paired source-group bootstrap within fixed category-composition strata; repeats/variants stay together; small cohorts do not establish deployment transfer",
    }


def partitions(
    rows: list[Record], field: str, summarize: Callable[[list[Record]], Record]
) -> Record:
    groups: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}
