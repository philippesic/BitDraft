from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.native import file_reference
from benchmarks.runner import compare
from benchmarks.workflows import live
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, record
from w1a1_eagle.identity import source_identity
from w1a1_eagle.lifecycle import Ownership
from w1a1_eagle.records import artifact, canonical, read


def suite(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "targets"} or raw["schema"] != "full-suite-v1":
        raise ValueError("unsupported full suite specification")
    targets = record(raw["targets"])
    if not targets:
        raise ValueError("full suite requires explicit targets")
    ownership = Ownership(output)
    try:
        store = Store(output)
        code = source_identity(Path(__file__).resolve().parents[1])
        bound: Record = {"specification": file_reference(specification), "code_sha256": code}
        binding = store.root / "binding.json"
        if binding.exists() and read(binding) != bound:
            raise ValueError(
                "suite inputs/source differ; preserve this attempt and use a new output"
            )
        store.atomic(binding, canonical(bound))
        results: Record = {}
        for name, value in targets.items():
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("unsafe target name")
            specs = record(value)
            if set(specs) != {"requests", "live", "harness"}:
                raise ValueError("full suite requires request, live and harness panels")
            target = ""
            arms: object = None
            cells: Record = {}
            for panel in ("requests", "live", "harness"):
                ref = artifact(specs[panel], specification.resolve().parent)
                ref.verify()
                comparison = Comparison.load(ref.path)
                comparison.verify()
                expected = {"requests": "full", "live": "live", "harness": "harness"}[panel]
                if comparison.tier != expected or comparison.repeats < 2 or comparison.diagnostic:
                    raise ValueError(
                        "full suite requires declared full panels and at least two repeat blocks"
                    )
                arm_identity = [
                    (arm.name, arm.drafter.sha256 if arm.drafter else None)
                    for arm in comparison.arms
                ]
                if target and (target != comparison.target.sha256 or arms != arm_identity):
                    raise ValueError(
                        "full suite panels must use the same target and candidate arms"
                    )
                target, arms = comparison.target.sha256, arm_identity
                destination = store.root / (name + "-" + panel)
                result_path = destination / "result.json"
                if result_path.exists():
                    prior = read(destination / "specification.json")
                    if prior != read(ref.path):
                        raise ValueError("published panel inputs disagree with suite")
                    result = read(result_path, 64 * 1024 * 1024)
                elif destination.exists():
                    raise ValueError(
                        "incomplete timing block preserved; rerun the whole panel in a new suite attempt"
                    )
                else:
                    store.atomic(
                        store.root / "status.json",
                        canonical(
                            {
                                "status": "running",
                                "target": name,
                                "panel": panel,
                                "completed": results,
                            }
                        ),
                    )
                    result = (
                        live(ref.path, destination)
                        if panel == "live"
                        else compare(ref.path, destination)
                    )
                if result["status"] != "complete" or any(
                    record(summary)["correct"] is not True
                    for summary in record(result["summary"]).values()
                ):
                    raise ValueError("invalid full panel cannot select an incumbent")
                cells[panel] = file_reference(result_path)
            results[name] = cells
        result = {
            "status": "complete",
            "targets": results,
            "code_sha256": code,
            "scope": "complete greedy DEV panels; live outcome failures retained; sampled and unseen-confirmation claims remain separate",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        Store(output).atomic(output.resolve() / "failure.json", canonical({"error": str(error)}))
        raise
    finally:
        ownership.close()
