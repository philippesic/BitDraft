import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

from benchmarks.measurement import now, pause
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.config import native_environment
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.processes import ProcessIdentity
from w1a1_eagle.records import artifact, canonical, read


def verify_sampler(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if (
        set(raw) - {"device"} != {"schema", "source", "binary", "libraries", "target", "trials"}
        or raw["schema"] != 1
    ):
        raise ValueError("unsupported sampler qualification")
    root = specification.resolve().parent
    source, binary, target = (artifact(raw[key], root) for key in ("source", "binary", "target"))
    libraries = [artifact(value, root) for value in sequence(raw["libraries"])]
    for ref in (source, binary, target, *libraries):
        ref.verify()
    trials = integer(raw["trials"])
    if not 10000 <= trials <= 100000 or not libraries or output.exists():
        raise ValueError("bounded trials, authenticated libraries and a new output required")
    device = string(raw.get("device", "cuda"))
    with device_lease(device):
        store = Store(output)
        store.atomic(store.root / "specification.json", canonical(raw))
        with (
            (store.root / "stdout.json").open("wb") as stdout,
            (store.root / "stderr.log").open("wb") as stderr,
        ):
            process: subprocess.Popen[bytes] = subprocess.Popen(
                [
                    str(binary.path),
                    str(target.path),
                    str(trials),
                    *([device] if "device" in raw else []),
                ],
                stdout=stdout,
                stderr=stderr,
                env=native_environment(libraries[0].path.parent, device),
                start_new_session=True,
            )
            try:
                owner = ProcessIdentity.read(process.pid)
                store.atomic(store.root / "owner.json", canonical(cast(Record, asdict(owner))))
                deadline = now() + 30
                while process.poll() is None and now() < deadline:
                    maps = Path(f"/proc/{process.pid}/maps").read_bytes()
                    if all(str(ref.path).encode() in maps for ref in libraries):
                        store.atomic(store.root / "library-maps.txt", maps)
                        break
                    pause(0.05)
                process.wait(timeout=600)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
        result = read(store.root / "stdout.json")
        result["functions_tested"] = sorted(
            {
                string(record(case).get("algorithm", "rejection"))
                for case in sequence(result["distribution_cases"])
            }
        )
        result.update(
            {
                "returncode": process.returncode,
                "process_reaped": True,
                "target_sha256": target.sha256,
                "device": device,
                "runtime_source_sha256": source.sha256,
            }
        )
        if not (store.root / "library-maps.txt").exists():
            raise ValueError("actual sampler library mappings were not established")
        store.atomic(store.root / "result.json", canonical(result))
        if cast(int, process.returncode) != 0 or result["status"] != "passed":
            raise ValueError(
                "native sampler qualification failed; retained result identifies the gate"
            )
        return result


def reduce_verifier_trace(lines: list[str]) -> Record:
    rounds = proposed_total = accepted_total = rolled_back = eog_rounds = 0
    for line in lines:
        marker = "BINARY_VERIFIER_TRACE "
        if marker not in line:
            continue
        raw = record(cast(object, json.loads(line.split(marker, 1)[1])))
        proposed = [integer(v) for v in sequence(raw["proposed"])]
        targets = [integer(v) for v in sequence(raw["target_argmax"])]
        observed = [integer(v) for v in sequence(raw["accepted_tokens"])]
        eos = {integer(v) for v in sequence(raw["eog_ids"])}
        if raw["rejection"] is not False or len(targets) != len(proposed) + 1 or not proposed:
            raise ValueError(
                "trace must contain actual greedy verification and complete target argmax rows"
            )
        expected: list[int] = []
        for index, token in enumerate(proposed):
            expected.append(targets[index])
            if targets[index] != token or targets[index] in eos and index + 1 < len(proposed):
                break
        else:
            expected.append(targets[-1])
        if observed != expected:
            raise ValueError(
                "native accepted prefix/correction disagrees with independent target argmax oracle"
            )
        if type(raw["replay"]) is not bool or type(raw["committed"]) is not bool:
            raise ValueError("explicit replay and commit flags required")
        rounds += 1
        eog_rounds += any(token in eos for token in observed)
        proposed_total += 0 if raw["replay"] else len(proposed)
        if raw["committed"]:
            accepted = len(observed) - 1
            accepted_total += accepted - int(raw["replay"] is True and accepted > 0)
        else:
            rolled_back += 1
    if not rounds or not proposed_total:
        raise ValueError("no activated greedy verifier trace")
    return {
        "rounds": rounds,
        "proposed": proposed_total,
        "accepted": accepted_total,
        "rolled_back_rounds": rolled_back,
        "eog_rounds": eog_rounds,
        "scope": "independent greedy target-argmax prefix, correction/bonus and EOG oracle; verified-token counters can include work beyond the emitted cap",
    }


def verify_greedy(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != {"schema", "comparison", "logs"} or raw["schema"] != 1 or output.exists():
        raise ValueError("explicit comparison, trace logs and a new output required")
    comparison = artifact(raw["comparison"], specification.resolve().parent)
    comparison.verify()
    result = read(comparison.path, 64 * 1024 * 1024)
    if result.get("diagnostic") is not True or result["status"] != "complete":
        raise ValueError("greedy oracle requires a complete diagnostic comparison")
    summaries = record(result["summary"])
    logs = record(raw["logs"])
    expected_arms = set(summaries) - {"target"}
    if set(logs) != expected_arms:
        raise ValueError("trace logs must cover every speculative arm")
    verified: Record = {}
    for name, value in logs.items():
        ref = artifact(value, specification.resolve().parent)
        ref.verify()
        trace = reduce_verifier_trace(ref.path.read_text().splitlines())
        summary = record(summaries[name])
        if summary["correct"] is not True or (trace["proposed"], trace["accepted"]) != (
            summary["proposed"],
            summary["accepted"],
        ):
            raise ValueError("oracle counters and native comparison disagree")
        verified[name] = trace
    store = Store(output)
    store.atomic(store.root / "specification.json", canonical(raw))
    final: Record = {"status": "passed", "arms": verified, "comparison": raw["comparison"]}
    store.atomic(store.root / "result.json", canonical(final))
    return final
