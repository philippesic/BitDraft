from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import cast

from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.lifecycle import Journal
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, read


def report(root: Path) -> Record:
    root = root.resolve()
    status = read(root / "status.json")
    binding = read(root / "binding.json")
    specification = artifact(binding["specification"], root)
    specification.verify()
    controller = read(root / "controller.json") if (root / "controller.json").exists() else None
    alive = False
    if controller is not None:
        alive = ProcessIdentity(
            integer(controller["pid"]), string(controller["boot"]), string(controller["birth"])
        ).alive()
    visits: Counter[str] = Counter()
    positions: dict[str, set[int]] = defaultdict(set)
    progress: list[Record] = []
    diagnostics: list[Record] = []
    segments: list[Record] = []
    missing_widths = 0
    for path in sorted(root.glob("[0-9][0-9][0-9][0-9]-training")):
        journal = Journal(path)
        summary = journal.report()
        segments.append({"path": str(path), "report": summary})
        for row in journal.events():
            if row.get("kind") == "exposure":
                task = string(row["trajectory"])
                visits[task] += 1
                if "labels" in row:
                    anchor, width = integer(row["anchor"]), integer(row["labels"])
                    positions[task].update(range(anchor, anchor + width))
                else:
                    missing_widths += 1
            if row.get("kind") == "training-progress":
                progress.append({"segment": path.name, **row})
            if row.get("kind") == "development-diagnostic":
                diagnostics.append({"segment": path.name, **row})
    evaluations: list[Record] = []
    for value in sequence(status["completed_evaluations"]):
        ref = artifact(value, root)
        ref.verify()
        raw = read(ref.path, 64 * 1024 * 1024)
        evaluations.append(
            {
                "evidence": value,
                "summary": raw["summary"],
                "paired": raw.get("paired_against_baseline", {}),
            }
        )
    return {
        "status": status,
        "controller_alive": alive,
        "controller": controller,
        "operational": "running"
        if alive
        else "complete"
        if status["status"] == "complete"
        else "controller absent; inspect before recovery",
        "segments": segments,
        "progress": progress,
        "development": diagnostics,
        "native_evaluations": evaluations,
        "exposure": {
            "selected_window_attempts": sum(visits.values()),
            "distinct_tasks": len(visits),
            "unique_label_positions": sum(len(value) for value in positions.values()),
            "attempts_without_recorded_label_width": missing_widths,
            "visits_by_task": dict(visits),
            "scope": "observed new-pipeline sample attempts; includes retries, not lifetime training exposure",
        },
        "cost": {
            "completed_training_attempt_seconds": sum(
                number(record(segment["report"])["cumulative_attempt_seconds"])
                for segment in segments
            ),
            "recorded_optimizer_seconds": status["model_update_seconds"],
            "clock_scope": "optimizer accounting excludes model/data loading, evaluation and checkpoint I/O; these are separate clocks",
        },
        "selection_scope": "latest recovery, provisional fast DEV and full DEV pointers have distinct roles; no automatic deployment promotion",
    }


def controller_record() -> Record:
    import os

    return cast(Record, asdict(ProcessIdentity.read(os.getpid())))
