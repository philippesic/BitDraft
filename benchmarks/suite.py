from pathlib import Path

from benchmarks.comparison import Comparison
from benchmarks.native import file_reference
from benchmarks.runner import compare
from benchmarks.workflows import live
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, number, record
from w1a1_eagle.identity import source_identity
from w1a1_eagle.lifecycle import Journal, Ownership
from w1a1_eagle.records import artifact, canonical, read


def suite(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "targets"} or raw["schema"] != "full-suite-v1":
        raise ValueError("unsupported full suite specification")
    targets = record(raw["targets"])
    if not targets:
        raise ValueError("full suite requires explicit targets")
    prepared: dict[tuple[str, str], Comparison] = {}
    for name, value in targets.items():
        specs = record(value)
        if set(specs) - {"candidate_bindings", "completed_panels"} != {
            "requests",
            "live",
            "harness",
        }:
            raise ValueError("full suite requires request, live and harness panels")
        runtime: tuple[str, ...] | None = None
        arms: object = None
        for panel, tier in (("requests", "full"), ("live", "live"), ("harness", "harness")):
            ref = artifact(specs[panel], specification.resolve().parent)
            ref.verify()
            comparison = Comparison.load(ref.path)
            comparison.verify()
            current_arms = [
                (arm.name, arm.drafter.sha256 if arm.drafter else None) for arm in comparison.arms
            ]
            if comparison.tier != tier or comparison.repeats < 2 or comparison.diagnostic:
                raise ValueError(
                    "full suite requires primary full panels with at least two repeat blocks"
                )
            if runtime is not None and (
                runtime != comparison.serving_identity() or arms != current_arms
            ):
                raise ValueError("suite panels differ in runtime, settings, role or candidate arms")
            runtime, arms = comparison.serving_identity(), current_arms
            prepared[name, panel] = comparison
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
            if set(specs) - {"candidate_bindings", "completed_panels"} != {
                "requests",
                "live",
                "harness",
            }:
                raise ValueError("full suite requires request, live and harness panels")
            cells: Record = {}
            for panel in ("requests", "live", "harness"):
                ref = artifact(specs[panel], specification.resolve().parent)
                comparison = prepared[name, panel]
                destination = store.root / (name + "-" + panel)
                result_path = destination / "result.json"
                completed = record(specs.get("completed_panels", {}))
                if panel in completed:
                    result_ref = artifact(completed[panel], specification.resolve().parent)
                    result_ref.verify()
                    result_path = result_ref.path
                if result_path.exists():
                    prior = read(result_path.parent / "specification.json")
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
            if "candidate_bindings" in specs:
                if comparison.role != "dev":
                    raise ValueError("full-DEV incumbent selection requires DEV panels")
                request_result = read(
                    artifact(cells["requests"], store.root).path, 64 * 1024 * 1024
                )
                selections: Record = {}
                for arm, value in record(specs["candidate_bindings"]).items():
                    if arm in {"target", "baseline"} or arm not in record(
                        request_result["summary"]
                    ):
                        raise ValueError("incumbent must be an evaluated experimental arm")
                    ref = artifact(value, specification.resolve().parent)
                    ref.verify()
                    bound = read(ref.path)
                    state = artifact(bound["state"], ref.path.parent)
                    drafter = artifact(bound["drafter"], ref.path.parent)
                    execution = artifact(bound["execution"], ref.path.parent)
                    for item in (state, drafter, execution):
                        item.verify()
                    report = Journal(execution.path.parent).report()
                    worker = record(report["worker"])
                    if (
                        report["operational"] != "terminal"
                        or worker["state"] != state.sha256
                        or worker["export"] != drafter.sha256
                    ):
                        raise ValueError(
                            "candidate state/export binding disagrees with completed execution"
                        )
                    request_spec = Comparison.load(
                        artifact(specs["requests"], specification.resolve().parent).path
                    )
                    if not any(
                        item.name == arm
                        and item.drafter is not None
                        and item.drafter.sha256 == drafter.sha256
                        for item in request_spec.arms
                    ):
                        raise ValueError("incumbent drafter differs from evaluated arm")
                    selections[arm] = {
                        **bound,
                        "score": record(record(request_result["summary"])[arm])[
                            "tokens_per_second"
                        ],
                        "updates": Store(state.path.parent).describe(state).updates,
                    }
                if not selections:
                    raise ValueError("at least one state-bound candidate required")

                def score(arm: str) -> tuple[float, float]:
                    return number(record(selections[arm])["score"]), -number(
                        record(selections[arm])["updates"]
                    )

                winner = max(selections, key=score)
                cells["full_dev_incumbent"] = {
                    "arm": winner,
                    **record(selections[winner]),
                    "evidence": cells["requests"],
                    "scope": "full DEV request-selected experimental state; workflow/harness outcomes remain separate, no deployment promotion",
                }
        result = {
            "status": "complete",
            "targets": results,
            "code_sha256": code,
            "scope": "complete greedy panels with matching declared roles; live failures retained; no automatic unseen-confirmation or deployment claim",
        }
        store.atomic(store.root / "result.json", canonical(result))
        return result
    except BaseException as error:
        Store(output).atomic(output.resolve() / "failure.json", canonical({"error": str(error)}))
        raise
    finally:
        ownership.close()
