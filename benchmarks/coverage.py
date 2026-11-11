import json
from collections import defaultdict, deque
from typing import Callable

from w1a1_eagle.contracts import Record, integer, string
from w1a1_eagle.records import identity

TRAIN = (1500, 1000, 1250, 1000, 750, 500)
FULL = (150, 100, 125, 100, 75, 50)
SHORT = (30, 20, 25, 20, 15, 10)


def selection_order(row: Record) -> str:
    return identity({"seed": 8101, "id": row["id"]})


def identifier(row: Record) -> str:
    return string(row["id"])


def length_band(tokens: int) -> int:
    if tokens < 512:
        return 0
    if tokens < 2048:
        return 1
    if tokens < 8192:
        return 2
    if tokens <= 16384:
        return 3
    raise ValueError("prompt exceeds admitted bank length")


def stratum(row: Record) -> str:
    category, subtype = string(row["category"]), string(row["subtype"])
    if category == "code":
        return "code-edit" if subtype == "editing" else "code-generate"
    if category == "reasoning":
        return "reasoning-other" if subtype == "non_arithmetic" else "reasoning-arithmetic"
    if category == "writing":
        return "writing-translation" if subtype == "translation" else "writing-other"
    return category


def quotas(counts: tuple[int, ...]) -> dict[str, int]:
    code, reasoning, documents, conversation, structured, writing = counts
    return {
        "code-edit": code // 2,
        "code-generate": code - code // 2,
        "reasoning-arithmetic": reasoning // 2,
        "reasoning-other": reasoning - reasoning // 2,
        "documents": documents,
        "conversation": conversation,
        "structured": structured,
        "writing-translation": writing // 5,
        "writing-other": writing - writing // 5,
    }


def select(rows: list[Record], counts: tuple[int, ...], bands: tuple[int, ...]) -> list[Record]:
    return apportion(rows, quotas(counts), dict(enumerate(bands)), stratum, bank_length)


def bank_length(row: Record) -> int:
    return integer(row["length_band"])


def apportion(
    rows: list[Record],
    demands: dict[str, int],
    columns: dict[int, int],
    row_kind: Callable[[Record], str],
    column_kind: Callable[[Record], int],
) -> list[Record]:
    if sum(demands.values()) != sum(columns.values()) or any(
        type(v) is not int or v < 0 for v in (*demands.values(), *columns.values())
    ):
        raise ValueError("nonnegative matching allocation margins required")
    pools: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for row in rows:
        pools[row_kind(row), column_kind(row)].append(row)
    capacities: dict[tuple[str, str], int] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)

    def edge(left: str, right: str, capacity: int) -> None:
        capacities[left, right] = capacity
        capacities[right, left] = 0
        adjacency[left].add(right)
        adjacency[right].add(left)

    for name, count in demands.items():
        edge("start", name, count)
        for band in columns:
            edge(name, "band" + str(band), len(pools[name, band]))
    for band, count in columns.items():
        edge("band" + str(band), "end", count)
    flow = 0
    while True:
        parents: dict[str, str] = {}
        pending = deque(["start"])
        visited = {"start"}
        while pending and "end" not in visited:
            left = pending.popleft()
            for right in sorted(adjacency[left]):
                if right not in visited and capacities[left, right] > 0:
                    visited.add(right)
                    parents[right] = left
                    pending.append(right)
        if "end" not in visited:
            break
        amount, node = sum(demands.values()), "end"
        while node != "start":
            amount = min(amount, capacities[parents[node], node])
            node = parents[node]
        node = "end"
        while node != "start":
            left = parents[node]
            capacities[left, node] -= amount
            capacities[node, left] += amount
            node = left
        flow += amount
    if flow != sum(demands.values()):
        coverage = {name + "/" + str(band): len(values) for (name, band), values in pools.items()}
        raise ValueError(
            "source pool cannot satisfy real category/length coverage: "
            + json.dumps(coverage, sort_keys=True)
        )
    selected: list[Record] = []
    for (name, band), values in sorted(pools.items()):
        count = capacities.get(("band" + str(band), name), 0)
        selected.extend(sorted(values, key=selection_order)[:count])
    return selected
