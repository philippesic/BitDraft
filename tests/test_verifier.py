import json

import pytest

from benchmarks.verifier import reduce_verifier_trace
from w1a1_eagle.contracts import Record


def line(
    proposed: list[int],
    targets: list[int],
    accepted: list[int],
    replay: bool = False,
    committed: bool = True,
) -> str:
    raw: Record = {
        "proposed": proposed,
        "target_argmax": targets,
        "accepted_tokens": accepted,
        "eog_ids": [99],
        "replay": replay,
        "committed": committed,
        "rejection": False,
    }
    return "BINARY_VERIFIER_TRACE " + json.dumps(raw)


def test_verifier_oracle_handles_rollback_replay_correction_and_eog() -> None:
    result = reduce_verifier_trace(
        [
            line([5, 6], [5, 9, 10], [5, 9], committed=False),
            line([5, 9], [5, 9, 1], [5, 9, 1], replay=True),
            line([99, 7], [99, 7, 1], [99]),
        ]
    )
    assert (result["rounds"], result["proposed"], result["accepted"]) == (3, 4, 1)
    assert result["rolled_back_rounds"] == 1 and result["eog_rounds"] == 1
    with pytest.raises(ValueError, match="independent"):
        reduce_verifier_trace([line([5, 6], [5, 9, 10], [5, 6, 10])])
    with pytest.raises(ValueError, match="independent"):
        reduce_verifier_trace([line([99, 7], [99, 7, 1], [99, 7, 1])])
    with pytest.raises(ValueError, match="no activated"):
        reduce_verifier_trace([])
