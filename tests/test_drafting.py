from random import Random

import pytest

from w1a1_eagle.drafting import (
    THINK_CLOSE,
    THINK_OPEN,
    Round,
    Tally,
    context_band,
    generated_anchors,
    paired_interval,
    segments,
    simulate,
    tally,
)

TEXT = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20)


def trajectory() -> dict[int, tuple[int, ...]]:
    return {anchor: TEXT[anchor - 4 :][:15] for anchor in range(4, 4 + len(TEXT))}


def drafts(
    windows: dict[int, tuple[int, ...]], wrong: dict[int, int]
) -> dict[int, tuple[tuple[int, ...], ...]]:
    proposals: dict[int, tuple[tuple[int, ...], ...]] = {}
    for anchor, labels in windows.items():
        slots = []
        for slot in range(3):
            truth = labels[slot] if slot < len(labels) else 0
            first = 99 if wrong.get(anchor) == slot + 1 else truth
            slots.append((first, truth, 98))
        proposals[anchor] = tuple(slots)
    return proposals


def test_rounds_follow_verified_text_from_the_first_generated_token() -> None:
    windows = trajectory()
    proposals = drafts(windows, {5: 2, 7: 1})
    rounds = simulate(windows, proposals, 3)
    assert rounds == (Round(5, 3, 1), Round(7, 3, 0), Round(8, 3, 3), Round(12, 2, 2))
    assert generated_anchors(windows) == tuple(range(5, 14))
    assert segments((3, 4, 5, 9, 10)) == ((3, 5), (9, 10))


def test_round_start_and_position_metrics() -> None:
    windows = trajectory()
    proposals = drafts(windows, {5: 2, 7: 1})
    rounds = simulate(windows, proposals, 3)
    starts = tally([item.anchor for item in rounds], windows, proposals, 3, 2)
    record = starts.record()
    assert record["accepted"] == 6 and record["proposed"] == 11 and record["positions"] == 4
    assert record["first_wrong_slot"] == {"1": 1, "2": 1, "3": 0, "none": 1, "ended": 1}
    assert record["accepted_distribution"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert record["reached"] == [4, 4, 3]
    assert record["first_choice_recall"] == [0.75, 0.75, 1.0]
    assert record["top_recall"] == [1.0, 1.0, 1.0]
    assert record["prefix_survival"] == [0.75, 0.5, 1 / 3]
    narrow = tally([item.anchor for item in rounds], windows, proposals, 3, 1).record()
    assert narrow["top_recall"] == narrow["first_choice_recall"]
    everywhere = tally(generated_anchors(windows), windows, proposals, 3, 16)
    merged = Tally(3, 16)
    merged.merge(everywhere)
    assert merged.record() == everywhere.record()
    with pytest.raises(ValueError, match="different"):
        merged.merge(Tally(2, 16))


def test_paired_interval_resamples_groups_and_needs_two() -> None:
    rows = [("a", 3, 4, 2, 4), ("b", 2, 4, 2, 4), ("b", 1, 2, 1, 2), ("c", 4, 4, 1, 4)]
    first = paired_interval(rows, 200, Random(5))
    assert first == paired_interval(rows, 200, Random(5))
    interval, difference = first["interval"], first["difference"]
    assert isinstance(interval, list) and isinstance(difference, float)
    low, high = (float(value) for value in interval)
    assert abs(difference - (10 / 14 - 6 / 14)) < 1e-12
    assert low <= difference <= high
    assert paired_interval(rows[:1], 200, Random(5))["interval"] is None


def test_context_bands_and_thinking_template() -> None:
    assert [context_band(n) for n in (0, 511, 512, 2047, 2048, 8191, 8192, 16384, 16385)] == [
        "<512",
        "<512",
        "512-2K",
        "512-2K",
        "2K-8K",
        "2K-8K",
        "8K-16K",
        "8K-16K",
        ">16K",
    ]
    from w1a1_eagle.drafting import thinking_mode

    assert thinking_mode((1, THINK_OPEN, 2), 3) == "on"
    assert thinking_mode((1, THINK_OPEN, 2, THINK_CLOSE, 3), 5) == "off"
    assert thinking_mode((1, THINK_OPEN, 2, THINK_CLOSE, 3), 2) == "on"
    assert thinking_mode((1, 2), 2) == "absent"
