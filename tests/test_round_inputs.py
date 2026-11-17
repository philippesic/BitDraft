from collections import Counter
from pathlib import Path

import pytest

from benchmarks.round_inputs import BANDS, CATEGORIES, Cell, marginal_plan, seen_selection
from w1a1_eagle.contracts import Record

SCREEN = Counter(
    {
        ("code", 0): 9,
        ("code", 1): 3,
        ("code", 2): 3,
        ("conversation", 1): 8,
        ("conversation", 2): 2,
        ("documents", 0): 6,
        ("documents", 2): 3,
        ("documents", 3): 3,
        ("reasoning", 0): 10,
        ("structured", 1): 7,
        ("structured", 2): 1,
        ("writing", 0): 5,
    }
)
TRAIN_CELLS = {
    ("code", 0): 50,
    ("code", 1): 20,
    ("reasoning", 0): 50,
    ("documents", 0): 50,
    ("documents", 1): 10,
    ("documents", 2): 20,
    ("documents", 3): 10,
    ("structured", 1): 50,
    ("conversation", 0): 5,
    ("conversation", 1): 50,
    ("conversation", 2): 20,
    ("writing", 0): 30,
}


def margins(plan: dict[Cell, int]) -> tuple[list[int], list[int]]:
    return (
        [sum(plan[(c, b)] for b in BANDS) for c in CATEGORIES],
        [sum(plan[(c, b)] for c in CATEGORIES) for b in BANDS],
    )


def test_plan_moves_unavailable_cells_but_keeps_both_marginals() -> None:
    available = {(c, b): TRAIN_CELLS.get((c, b), 0) for c in CATEGORIES for b in BANDS}
    plan = marginal_plan(SCREEN, available)
    assert margins(plan) == margins({(c, b): SCREEN[(c, b)] for c in CATEGORIES for b in BANDS})
    assert all(plan[key] <= available[key] for key in plan)
    assert plan[("code", 2)] == 0 and plan[("structured", 2)] == 0
    assert plan[("reasoning", 0)] == 10 and plan[("documents", 3)] == 3
    starved = {**available, ("documents", 3): 2}
    with pytest.raises(ValueError, match="marginals"):
        marginal_plan(SCREEN, starved)


def test_selection_is_seeded_and_uses_each_source_group_once() -> None:
    rows: list[Record] = []
    for (category, band), count in TRAIN_CELLS.items():
        tokens = (100, 1000, 4000, 10000)[band]
        for index in range(count):
            rows.append(
                {
                    "id": f"{category}-{band}-{index}",
                    "group": f"{category}-{band}-{index // 2}",
                    "category": category,
                    "input_tokens": tokens,
                }
            )
    selected, plan = seen_selection(SCREEN, rows, 7)
    again, _ = seen_selection(SCREEN, list(reversed(rows)), 7)
    other, _ = seen_selection(SCREEN, rows, 8)
    assert [row["id"] for row in selected] == [row["id"] for row in again]
    assert [row["id"] for row in selected] != [row["id"] for row in other]
    assert len(selected) == 60 == sum(plan.values())
    assert len({row["group"] for row in selected}) == 60


def comparison_spec(tmp_path: Path, role: str, tier: str) -> Path:
    from benchmarks.native import file_reference
    from w1a1_eagle.records import canonical

    files = {}
    for name in ("server", "library.so", "target.gguf", "drafter.gguf", "build.json"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = file_reference(path)
    tasks = []
    for (category, band), count in SCREEN.items():
        for index in range(count):
            tasks.append(
                {
                    "id": f"{category}-{band}-{index}",
                    "group": f"{category}-{band}-{index}",
                    "role": role,
                    "category": category,
                    "messages": [{"role": "user", "content": "x"}],
                    "output_cap": 128,
                    "ancestry": {"input_tokens": (100, 1000, 4000, 10000)[band]},
                }
            )
    workload = tmp_path / "workload.json"
    workload.write_bytes(canonical({"schema": 1, "tasks": tasks}))
    spec = tmp_path / f"{role}-{tier}.json"
    spec.write_bytes(
        canonical(
            {
                "schema": "native-comparison-v1",
                "source": files["build.json"],
                "binary": files["server"],
                "libraries": [files["library.so"]],
                "target": files["target.gguf"],
                "arms": [
                    {"name": "target", "drafter": None},
                    {"name": "baseline", "drafter": files["drafter.gguf"]},
                ],
                "workload": file_reference(workload),
                "repeats": 1,
                "warmups": 1,
                "context": 32768,
                "depth": 3,
                "role": role,
                "tier": tier,
            }
        )
    )
    return spec


def test_train_role_is_opt_in_for_fresh_screen_shaped_comparisons(tmp_path: Path) -> None:
    from benchmarks.comparison import Comparison

    assert Comparison.load(comparison_spec(tmp_path, "dev", "screen")).role == "dev"
    assert Comparison.load(comparison_spec(tmp_path, "train", "screen")).role == "train"
    with pytest.raises(ValueError, match="TRAIN"):
        Comparison.load(comparison_spec(tmp_path, "train", "comparison"))
    with pytest.raises(ValueError, match="role"):
        Comparison.load(comparison_spec(tmp_path, "heldout", "screen"))
