from hypothesis import given, settings
from hypothesis import strategies as st

from tools.contracts import Record, best_serving, components, exposure, validate_contract

settings.register_profile(
    "foundation", derandomize=True, max_examples=100, deadline=None, database=None
)
settings.load_profile("foundation")


def test_two_nulls_and_namespace() -> None:
    rows: list[Record] = [
        {"id": "a", "source": "s", "group": "x", "topic": None},
        {"id": "b", "source": "s", "group": "y", "topic": None},
    ]
    assert components(rows) == [["a"], ["b"]]
    assert components(
        [
            {"id": "a", "source": "s", "group": "x", "topic": "z"},
            {"id": "b", "source": "s", "group": "y", "topic": "x"},
        ]
    ) == [["a"], ["b"]]


@given(st.lists(st.sampled_from(("a", "b", "c")), min_size=3, max_size=3, unique=True))  # type: ignore[misc]
def test_group_permutations(order: list[str]) -> None:
    rows: list[Record] = [
        {
            "id": identity,
            "source": "s",
            "group": "shared" if identity != "c" else "separate",
            "topic": None,
        }
        for identity in order
    ]
    assert components(rows) == [["a", "b"], ["c"]]
    canonical: list[Record] = [
        {"id": identity, "source": "s", "group": group[0], "topic": None}
        for group in components(rows)
        for identity in group
    ]
    assert components(canonical) == [["a", "b"], ["c"]]
    assert components(rows + [{"id": "d", "source": "s", "group": "other", "topic": ""}]) == [
        ["a", "b"],
        ["c"],
        ["d"],
    ]


def test_full_accounting() -> None:
    assert 10000 == 96 + 48 + 9856
    assert 9856 == 12 + 23 + 5505 + 4316
    assert 9856 - 12 - 23 == 9821
    rows: list[object] = [
        {"id": str(i), "reason": "dev" if i < 12 else "collision" if i < 35 else "admitted"}
        for i in range(9856)
    ]
    assert (
        validate_contract(
            "count_conservation", {"inputs": [str(i) for i in range(9856)], "terminal": rows}
        )
        == []
    )
    assert validate_contract(
        "count_conservation", {"inputs": [str(i) for i in range(9856)], "terminal": rows[:-1]}
    )


def test_unique_positions_and_overlap() -> None:
    assert exposure([["a", "b"], ["b", "c"]], [2, 1]) == (6, 2.0, 3, 1.5)
    labels: list[object] = [{"trajectory": "t", "position": 1}, {"trajectory": "t", "position": 1}]
    assert validate_contract("count_unique", {"labels": labels, "claimed": 1}) == []
    assert validate_contract("count_unique", {"labels": labels, "claimed": 2})


def test_role_alias_and_ancestry() -> None:
    assert validate_contract(
        "heldout_role",
        {
            "protected": ["heldout24"],
            "rows": [{"id": "alias", "role": "train", "aliases": ["heldout24"]}],
        },
    )
    assert validate_contract(
        "unknown_ancestry", {"rows": [{"role": "fresh-eval", "ancestry": "unknown"}]}
    )


def test_example_limits_are_inputs() -> None:
    assert (
        validate_contract(
            "count_intent", {"minimum": 1_000_000, "maximum": 2_000_000, "actual": 1_111_811}
        )
        == []
    )
    assert (
        validate_contract(
            "exposure_schedule",
            {
                "windows": [["a", "b"]],
                "visits": [2],
                "total": 4,
                "mean_limit": 30,
                "max_limit": 30,
                "pass_limit": 2,
            },
        )
        == []
    )
    assert (
        validate_contract(
            "disk_reserve",
            {"devices": [{"free": 10 * 2**30, "peak": 2 * 2**30, "reserve": 8 * 2**30}]},
        )
        == []
    )
    assert validate_contract(
        "disk_reserve", {"devices": [{"free": 9 * 2**30, "peak": 2 * 2**30, "reserve": 8 * 2**30}]}
    )


def test_label_offset_and_eog() -> None:
    assert (
        validate_contract(
            "capture_offset", {"tokens": [7, 8, 9], "labels": [{"position": 0, "label": 8}]}
        )
        == []
    )
    assert validate_contract(
        "capture_offset", {"tokens": [7, 8, 9], "labels": [{"position": 0, "label": 7}]}
    )
    assert (
        validate_contract(
            "capture_eog", {"tokens": [1, 99], "eog": 99, "positions": [0], "anchors": []}
        )
        == []
    )


def test_selection_peak_tie_and_invalid() -> None:
    scores: list[Record] = [
        {"id": "early", "kind": "serving", "complete": True, "durable": True, "score": 2},
        {"id": "later", "kind": "serving", "complete": True, "durable": True, "score": 2},
        {"id": "dev", "kind": "dev", "complete": True, "durable": True, "score": 4},
        {"id": "partial", "kind": "serving", "complete": False, "durable": True, "score": 5},
    ]
    assert best_serving(scores) == "early"


def test_complete_request_aggregate() -> None:
    data: Record = {
        "requests": [
            {"tokens": [1, 2], "reference_tokens": [1, 2], "proposed": 1, "wall": 1},
            {"tokens": [3], "reference_tokens": [3], "proposed": 1, "wall": 3},
        ],
        "throughput": 0.75,
    }
    assert validate_contract("compare_correctness", data) == []
    data["throughput"] = (2 + 1 / 3) / 2
    assert validate_contract("compare_correctness", data)


def test_verify_corruption() -> None:
    assert validate_contract("verify_label", {"logits": [[1, 4, 2]], "labels": [1]}) == []
    assert validate_contract("verify_label", {"logits": [[1, 4, 2]], "labels": [0]})
    assert validate_contract("verify_missing", {"scope": "runtime", "qualified": True, "raw": [1]})
