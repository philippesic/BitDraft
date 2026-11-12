from pathlib import Path

from benchmarks.campaign_segments import campaign as segmented
from benchmarks.campaign_segments import preflight as preflight
from benchmarks.campaign_segments import prepare
from benchmarks.campaign_segments import select as select
from w1a1_eagle.contracts import Record


def campaign(specification: Path, output: Path) -> Record:
    raw, _, _ = prepare(specification)
    if raw["schema"] == "training-campaign-v1":
        return segmented(specification, output)
    from benchmarks.resident import campaign as resident

    return resident(specification, output)
