from pathlib import Path

from benchmarks.comparison import Comparison, protocol_covers
from benchmarks.native import file_reference
from benchmarks.runtime_policy import LIVE_TIERS, arguments, payload
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.lifecycle import Ownership
from w1a1_eagle.records import artifact, canonical, identity, read, reference


def register(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "comparison", "result", "arm", "rep"} or raw["schema"] != 1:
        raise ValueError("reference registration requires comparison/result/arm/rep")
    root = specification.resolve().parent
    desired, result_ref = artifact(raw["comparison"], root), artifact(raw["result"], root)
    desired.verify()
    result_ref.verify()
    spec = Comparison.load(desired.path)
    spec.verify()
    prior_path = result_ref.path.parent / "specification.json"
    prior = Comparison.load(prior_path)
    prior.verify()
    result = read(result_ref.path, 64 * 1024 * 1024)
    if "protocol" in result and result["protocol"] != prior.reference_protocol():
        raise ValueError("source result executed a different runtime/request policy")
    name, rep = string(raw["arm"]), integer(raw["rep"])
    current_protocol, old_protocol = spec.arm_protocol(name), prior.arm_protocol(name)
    old_tasks = {string(task["id"]): task for task in prior.tasks}
    if (
        not protocol_covers(old_protocol, current_protocol)
        or (spec.sampling.temperature and spec.sampling.qualification is None)
        or spec.diagnostic
        or result.get("status") != "complete"
        or result.get("diagnostic", False)
        or any(old_tasks.get(string(task["id"])) != task for task in spec.tasks)
        or any(
            record(summary).get("correct") is not True
            for summary in record(result["summary"]).values()
        )
    ):
        raise ValueError("only complete valid matching evidence can establish a reference")
    rows = [record(row) for row in sequence(result["rows"])]
    expected = {
        (arm.name, string(task["id"]), rep, seed)
        for arm in prior.arms
        for task in prior.tasks
        for rep in range(prior.repeats)
        for seed in prior.sampling.seeds
    }
    observed = {
        (
            string(row["arm"]),
            string(row["task"]),
            integer(row["rep"]),
            integer(row.get("seed", 8101)),
        )
        for row in rows
    }
    if (
        len(rows) != len(expected)
        or observed != expected
        or any(row["correct"] is not True for row in rows)
    ):
        raise ValueError("incomplete or duplicated source panel cannot become a reference")
    arm = next((arm for arm in prior.arms if arm.name == name), None)
    if arm is None or not 0 <= rep < prior.repeats:
        raise ValueError("reference arm/repeat not present in source panel")
    selected = [
        row
        for row in rows
        if row["arm"] == name
        and row["rep"] == rep
        and row["task"] in {task["id"] for task in spec.tasks}
        and row.get("seed", 8101) in spec.sampling.seeds
    ]
    signatures: Record = {}
    receipts: list[Record] = []

    def command(cell: Path) -> None:
        path = cell / "command.json"
        argv = [string(value) for value in sequence(read(path)["argv"])]
        port = int(argv[argv.index("--port") + 1])
        if argv != arguments(
            prior.binary.path,
            prior.target.path,
            arm.drafter.path if arm.drafter else None,
            prior.context,
            prior.depth_of(name),
            port,
        ):
            raise ValueError("executed server policy differs from requested reference protocol")
        receipts.append(file_reference(path))

    if prior.tier in LIVE_TIERS:
        command(result_ref.path.parent / (str(rep) + "-" + name))
        for row in selected:
            signatures[string(row["task"])] = row["state_sha256"]
            receipts.append(
                file_reference(
                    result_ref.path.parent
                    / (str(rep) + "-" + name)
                    / string(row["task"])
                    / "transcript.json"
                )
            )
    else:
        from benchmarks.coverage import selection_order
        from benchmarks.metrics import (
            native_counts,
            reconcile,
            signature,
            valid_sample,
            verify_generation,
        )

        wanted = {string(row["task"]) for row in selected}
        count = prior.block_count()
        ordered = tuple(sorted(prior.tasks, key=selection_order)) if count > 1 else prior.tasks
        for group, seed in (
            (group, seed) for group in range(count) for seed in spec.sampling.seeds
        ):
            tasks = ordered[group::count]
            cell = result_ref.path.parent / (
                str(rep)
                + "-"
                + str(group)
                + "-"
                + (str(seed) + "-" if prior.sampling.temperature else "")
                + name
            )
            command(cell)
            responses = [
                read(cell / (str(index) + "-response.json"))
                for index in range(prior.warmups + len(tasks))
            ]
            if arm.drafter is not None:
                reconcile((cell / "server.log").read_text(), responses)
            for index, task in enumerate(tasks, prior.warmups):
                key = string(task["id"])
                if key in wanted:
                    path = cell / (str(index) + "-response.json")
                    response = responses[index]
                    query = read(cell / (str(index) + "-request.json"))
                    if query != payload(
                        sequence(query["prompt"]),
                        integer(task["output_cap"]),
                        prior.sampling.temperature,
                        prior.sampling.top_p,
                        prior.sampling.top_k,
                        prior.sampling.min_p,
                        seed,
                    ):
                        raise ValueError(
                            "executed request policy differs from requested reference protocol"
                        )
                    verify_generation(
                        response,
                        read(cell / (str(index) + "-request.json")),
                        arm.drafter is not None,
                    )
                    row = next(
                        row
                        for row in selected
                        if row["task"] == key and row.get("seed", 8101) == seed
                    )
                    if not valid_sample(response, integer(task["output_cap"])) or record(
                        row["native"]
                    ) != native_counts(response, arm.drafter is not None):
                        raise ValueError("reference row disagrees with raw counters or stopping")
                    signatures[key + ("/" + str(seed) if prior.sampling.temperature else "")] = (
                        signature(response)
                    )
                    receipts.append(file_reference(path))
    if len(selected) != len(spec.tasks) * len(spec.sampling.seeds) or len(signatures) != len(
        selected
    ):
        raise ValueError("reference observation is incomplete")
    value: Record = {
        "schema": "benchmark-reference-v1",
        "status": "complete",
        "protocol": current_protocol,
        "drafter": reference(arm.drafter) if arm.drafter is not None else None,
        "result": reference(result_ref),
        "specification": file_reference(prior_path),
        "source_arm": name,
        "source_repeat": rep,
        "rows": selected,
        "signatures": signatures,
        "raw": receipts,
        "gpu": (result_ref.path.parent / "gpu.txt").read_text(),
    }
    key = identity(
        {
            "protocol": current_protocol,
            "model": arm.drafter.sha256 if arm.drafter else "target",
        }
    )
    ownership = Ownership(output)
    try:
        store = Store(output)
        observations = Store(store.root / "observations")
        observation = observations.root / (identity(value) + ".json")
        if not observation.exists():
            observations.atomic(observation, canonical(value))
        entry = store.root / (key + ".json")
        reused = entry.exists()
        if not reused:
            store.atomic(entry, canonical(value))
        elif read(entry, 16 * 1024 * 1024)["protocol"] != value["protocol"]:
            raise ValueError("reference registry collision; preserve and inspect")
        return {
            "reference": file_reference(entry),
            "reused": reused,
            "observation": file_reference(observation),
        }
    finally:
        ownership.close()
