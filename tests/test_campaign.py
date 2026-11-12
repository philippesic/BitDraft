from collections import Counter
from pathlib import Path

import pytest

from benchmarks.campaign import select
from benchmarks.comparison import Arm, Comparison
from benchmarks.runner import execution_blocks
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.data import Artifact, position_count


def test_selection_keeps_earlier_ties_and_refuses_partial_or_invalid() -> None:
    leader: Record = {"state": "earlier", "score": 200.0}
    result: Record = {
        "status": "complete",
        "summary": {
            "candidate": {"correct": True, "tokens_per_second": 200.0},
            "prior-leader": {"correct": True, "tokens_per_second": 200.0},
        },
    }
    assert select(result, leader, "new-state", "new-export", {}) is leader
    result["summary"] = {
        "candidate": {"correct": True, "tokens_per_second": 201.0},
        "prior-leader": {"correct": True, "tokens_per_second": 200.0},
    }
    selected = select(result, leader, "new-state", "new-export", {})
    assert record(selected)["state"] == "new-state"
    with pytest.raises(ValueError, match="partial"):
        select({"status": "incomplete"}, None, "state", "export", {})
    with pytest.raises(ValueError, match="invalid"):
        select(
            {"status": "complete", "summary": {"candidate": {"correct": False}}},
            None,
            "state",
            "export",
            {},
        )


def test_mixed_partial_windows_count_actual_positions() -> None:
    assert position_count((10, 11, 12), (3, 2, 1)) == 3
    assert position_count((10, 11, 12)) == 17
    with pytest.raises(ValueError, match="lengths"):
        position_count((10, 11), (3,))
    with pytest.raises(ValueError, match="lengths"):
        position_count((10,), (16,))


def test_cached_acceptance_selection_ignores_historical_speed() -> None:
    leader: Record = {"state": "earlier", "score": 0.5}
    result: Record = {
        "status": "complete",
        "historical_references": {"baseline": {}},
        "summary": {
            "candidate": {"correct": True, "tokens_per_second": 999.0, "acceptance": 0.4},
            "prior-leader": {"correct": True, "tokens_per_second": 1.0, "acceptance": 0.5},
        },
    }
    assert select(result, leader, "new", "export", {}) is leader
    record(record(result["summary"])["candidate"])["acceptance"] = 0.6
    selected = record(select(result, leader, "new", "export", {}))
    assert selected["score"] == 0.6 and selected["metric"] == "acceptance"
    record(record(result["summary"])["candidate"])["acceptance"] = None
    with pytest.raises(ValueError):
        select(result, leader, "new", "export", {})


def test_boundary_export_preserves_valid_state_after_later_failure(tmp_path: Path) -> None:
    from benchmarks.campaign_segments import verify_initial_export
    from w1a1_eagle.checkpoints.store import Store
    from w1a1_eagle.lifecycle import Journal
    from w1a1_eagle.records import reference

    state = Store(tmp_path / "states").commit(b"state", "recipe", 17)
    exported = Store(tmp_path / "exports").commit(b"export", "recipe", 17)
    journal = Journal(tmp_path)
    journal.append({"kind": "started", "attempt": "one"})
    journal.append(
        {
            "kind": "boundary-publication",
            "attempt": "one",
            "updates": 17,
            "state": reference(state.artifact),
            "drafter": reference(exported.artifact),
        }
    )
    journal.append({"kind": "terminal", "attempt": "one", "outcome": "failed", "wall_seconds": 1.0})
    verify_initial_export(state.artifact, exported.artifact)
    wrong = Store(tmp_path / "exports").commit(b"wrong export", "recipe", 17)
    with pytest.raises(ValueError, match="producer"):
        verify_initial_export(state.artifact, wrong.artifact)
    changed_counter = Store(tmp_path / "exports").commit(b"export", "recipe", 18)
    with pytest.raises(ValueError, match="producer"):
        verify_initial_export(state.artifact, changed_counter.artifact)


def test_interruption_clock_excludes_controller_recovery_gaps(tmp_path: Path) -> None:
    from benchmarks.reporting import interruptions
    from w1a1_eagle.lifecycle import Journal

    for segment, began, ended, controller in (
        (1, 10.0, 20.0, 1),
        (2, 95.0, 100.0, 1),
        (3, 1000.0, 1010.0, 2),
    ):
        journal = Journal(tmp_path / (str(segment).zfill(4) + "-training"))
        journal.append({"kind": "started", "attempt": "test", "controller": {"pid": controller}})
        journal.append(
            {"kind": "training-ready", "attempt": "test", "boot": "boot", "monotonic": began}
        )
        journal.append(
            {"kind": "training-boundary", "attempt": "test", "boot": "boot", "monotonic": ended}
        )
    values = interruptions(tmp_path)
    assert values[0]["seconds"] == 75.0 and values[1]["seconds"] is None


def test_live_campaign_cost_uses_worker_progress_before_next_evaluation(tmp_path: Path) -> None:
    from benchmarks.native import file_reference
    from benchmarks.reporting import report
    from w1a1_eagle.lifecycle import Journal
    from w1a1_eagle.records import canonical

    specification = tmp_path / "specification.json"
    specification.write_bytes(canonical({"schema": "fixture"}))
    (tmp_path / "binding.json").write_bytes(
        canonical({"specification": file_reference(specification)})
    )
    (tmp_path / "status.json").write_bytes(
        canonical({"status": "running", "completed_evaluations": [], "model_update_seconds": 0.0})
    )
    journal = Journal(tmp_path / "0001-training")
    journal.append({"kind": "started", "attempt": "test", "controller": {"pid": 1}})
    journal.append(
        {
            "kind": "training-progress",
            "attempt": "test",
            "updates": 100,
            "accounting": {"successful_update_seconds": 120.0, "failed_update_seconds": 3.0},
        }
    )
    summary = report(tmp_path)
    assert record(summary["cost"])["recorded_optimizer_seconds"] == 123.0


def test_native_deadline_reaps_context_without_replacing_other_timer() -> None:
    import signal

    from benchmarks.measurement import deadline

    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(ValueError, match="exceeded"):
        with deadline(60):
            signal.raise_signal(signal.SIGALRM)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        with pytest.raises(ValueError, match="existing"):
            with deadline(60):
                pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        with pytest.raises(ValueError, match="exceeded"):
            with deadline(1, nested=True):
                signal.raise_signal(signal.SIGALRM)
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 60
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def test_resident_interruption_clock_pairs_multiple_checks_in_one_worker(tmp_path: Path) -> None:
    from benchmarks.reporting import interruptions
    from w1a1_eagle.lifecycle import Journal

    journal = Journal(tmp_path / "0001-training")
    journal.append({"kind": "started", "attempt": "test", "controller": {"pid": 1}})
    for kind, timestamp in (
        ("training-ready", 10),
        ("training-boundary", 20),
        ("training-ready", 95),
        ("training-boundary", 100),
        ("training-ready", 180),
        ("training-boundary", 200),
        ("training-boundary", 240),
        ("training-ready", 280),
    ):
        journal.append({"kind": kind, "attempt": "test", "boot": "boot", "monotonic": timestamp})
    assert [row["seconds"] for row in interruptions(tmp_path)] == [75, 80, 80]


def test_native_server_dies_with_its_parent(tmp_path: Path) -> None:
    import select
    import signal
    import subprocess
    import sys

    from benchmarks.measurement import now, pause
    from w1a1_eagle.processes import ProcessIdentity

    code = "import subprocess,sys,os\np=subprocess.Popen([sys.executable,'-B','-m','benchmarks.server_child',str(os.getpid()),sys.executable,'-c',\"import time;print('ready',flush=True);time.sleep(30)\"],stdout=subprocess.PIPE,text=True,start_new_session=True)\nprint(p.pid,flush=True)\nprint(p.stdout.readline(),flush=True)\nsys.stdin.read()"
    parent = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    child: ProcessIdentity | None = None
    try:
        assert parent.stdout is not None
        poll = select.poll()
        poll.register(parent.stdout, select.POLLIN)
        assert poll.poll(5000)
        child = ProcessIdentity.read(int(parent.stdout.readline()))
        assert parent.stdout.readline().strip() == "ready"
        parent.terminate()
        parent.wait(timeout=5)
        until = now() + 5
        while child.alive() and now() < until:
            pause(0.01)
        assert not child.alive()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if parent.stdin is not None:
            parent.stdin.close()
        if parent.stdout is not None:
            parent.stdout.close()
        if child is not None and child.alive():
            child.signal_group(signal.SIGKILL)


def test_grouped_blocks_balance_arm_positions_and_preserve_every_request(tmp_path: Path) -> None:
    ref = Artifact(tmp_path / "input", "a" * 64, 1)
    arms = (Arm("target", None), Arm("baseline", ref), Arm("candidate", ref))
    tasks: tuple[Record, ...] = tuple({"id": "task-" + str(i)} for i in range(90))
    spec = Comparison(
        ref, ref, (ref,), ref, arms, ref, tasks, 2, 0, 16, 2, "dev", "harness", frozenset()
    )
    positions: Counter[tuple[str, int]] = Counter()
    requests: Counter[tuple[int, str, str]] = Counter()
    for rep, _, selected, ordered, seed in execution_blocks(spec):
        assert seed == 8101
        for position, arm in enumerate(ordered):
            positions[arm.name, position] += 1
            for task in selected:
                requests[rep, str(task["id"]), arm.name] += 1
    assert set(positions.values()) == {2}
    assert len(requests) == 90 * 3 * 2 and set(requests.values()) == {1}


def test_bootstrap_preserves_fixed_category_mix_and_grouped_repeats() -> None:
    from benchmarks.statistics import paired

    rows: list[Record] = []
    for category in ["code", "writing"]:
        for group in range(2):
            for rep in range(2):
                for arm in ["candidate", "baseline"]:
                    seconds = 10.0 if category == "writing" else 1.0 if arm == "candidate" else 2.0
                    name = category + str(group)
                    rows.append(
                        {
                            "category": category,
                            "arm": arm,
                            "group": name,
                            "task": name,
                            "rep": rep,
                            "tokens": 1,
                            "wall_seconds": seconds,
                            "native": {"accepted": 1, "proposed": 2},
                        }
                    )
    result = paired(rows, "candidate", "baseline", 100)
    interval = record(result["speed_ratio_95_interval"])
    assert interval["lower"] == interval["upper"] == result["speed_ratio"]
    assert result["groups"] == 4
