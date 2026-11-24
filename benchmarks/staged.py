import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from benchmarks.measurement import now, pause
from benchmarks.native import file_reference
from w1a1_eagle.checkpoints.store import Store
from w1a1_eagle.contracts import Record, integer, record, sequence, string
from w1a1_eagle.device import device_lease
from w1a1_eagle.identity import source_identity
from w1a1_eagle.records import canonical, read

FIELDS = {
    "schema",
    "writer",
    "replay",
    "fuse",
    "intent",
    "first_row",
    "stage_rows",
    "stages",
    "updates_per_stage",
    "scope",
}
PYTHON = ("uv", "run", "--locked", "--no-sync", "--offline", "python")
FEATURE_SUFFIXES = (".f16", ".f32")
ATTEMPT_SECONDS = 6 * 3600


def completed(path: Path, field: str, value: object) -> bool:
    return path.exists() and read(path, 64 * 1024 * 1024).get(field) == value


def wait_for_gpu(log: Path, limit: float = 6 * 3600, delay: float = 30) -> None:
    began = now()
    while True:
        try:
            with device_lease("cuda"):
                return
        except ValueError as error:
            if now() - began > limit:
                raise ValueError("the GPU stayed busy; inspect its processes") from error
            with log.open("a") as sink:
                sink.write("waiting for the GPU: " + str(error) + "\n")
            pause(delay)


def run(argv: list[str], log: Path, root: Path) -> None:
    wait_for_gpu(log)
    with log.open("a") as sink:
        status = subprocess.run(argv, cwd=root, stdout=sink, stderr=sink, check=False).returncode
    if status != 0:
        raise ValueError("stage step failed; inspect " + str(log))


def write_spec(store: Store, name: str, value: Record) -> Path:
    path = store.root / name
    if path.exists():
        if read(path, 64 * 1024 * 1024) != value:
            raise ValueError("stage specification changed; start a new staged root")
    else:
        store.atomic(path, canonical(value))
    return path


def latest_state(train: Path) -> Record:
    pointer = read(train / "states" / "latest.json")
    sha = string(pointer["sha256"])
    return {
        "path": str(train / "states" / (sha + ".bin")),
        "sha256": sha,
        "bytes": integer(pointer["bytes"]),
    }


def trained_updates(train: Path) -> int:
    pointer = train / "states" / "latest.json"
    return integer(read(pointer)["updates"]) if pointer.exists() else 0


def exported_updates(train: Path) -> int:
    pointer = train / "exports" / "latest.json"
    if not pointer.exists():
        return 0
    value = read(pointer)
    path = train / "exports" / (string(value["sha256"]) + ".bin")
    if not path.exists() or path.stat().st_size != integer(value["bytes"]):
        return 0
    if file_reference(path)["sha256"] != value["sha256"]:
        raise ValueError("exported drafter differs from its pointer: " + str(path))
    return integer(value["updates"])


def files_present(corpus: Path) -> bool:
    for row in sequence(read(corpus, 1024 * 1024 * 1024)["records"]):
        for value in sequence(record(row)["files"]):
            ref = record(value)
            path = Path(string(ref["path"]))
            if not path.exists() or path.stat().st_size != integer(ref["bytes"]):
                return False
    return True


def train_stage(
    intent: Path,
    stage: Path,
    train: Path,
    scope: str,
    log: Path,
    root: Path,
    runner: Callable[[list[str], Path, Path], None] = run,
) -> None:
    try:
        runner(training_argv(intent, train, scope), log, root)
    except ValueError:
        if intent.name == "intent.json":
            raise
        with log.open("a") as sink:
            sink.write("resume refused after an interruption: restarting the stage\n")
        set_aside(train)
        runner(training_argv(stage / "intent.json", train, scope), log, root)


def training_argv(intent: Path, train: Path, scope: str) -> list[str]:
    return [
        *PYTHON,
        "-m",
        "w1a1_eagle",
        "run",
        str(intent),
        str(train),
        "--scope",
        scope,
        "--attempt-seconds",
        str(ATTEMPT_SECONDS),
    ]


def checked_state(train: Path, updates: int) -> bool:
    pointer = train / "states" / "latest.json"
    if not pointer.exists():
        return False
    value = read(pointer)
    path = train / "states" / (string(value["sha256"]) + ".bin")
    return (
        path.exists()
        and path.stat().st_size == integer(value["bytes"])
        and integer(value["updates"]) == updates
        and file_reference(path)["sha256"] == value["sha256"]
    )


def verified_state(train: Path, updates: int) -> Record:
    if not checked_state(train, updates):
        raise ValueError("the final stage's state is missing or differs from its pointer")
    return latest_state(train)


def state_present(train: Path) -> bool:
    pointer = train / "states" / "latest.json"
    return pointer.exists() and Path(string(latest_state(train)["path"])).exists()


def restart_stage(root: Path, log: Path) -> None:
    with log.open("a") as sink:
        sink.write("features missing after a restore: rebuilding the stage from its streams\n")
    for name in ("replay", "fused"):
        if (root / name).exists():
            shutil.rmtree(root / name)
    for path in [root / "fuse.json", *root.glob("intent*.json")]:
        path.unlink(missing_ok=True)
    if (root / "train").exists():
        set_aside(root / "train")


def wait_for_backup(root: Path, index: int, log: Path, limit: float = 12 * 3600) -> None:
    lag_path = root / "backup-lag"
    if not lag_path.exists():
        return
    lag = int(lag_path.read_text().strip())
    began = now()
    while True:
        ack = root / "backup-ack.json"
        acknowledged = integer(read(ack)["stage"]) if ack.exists() else -1
        if acknowledged >= index - 1 - lag:
            return
        if now() - began > limit:
            raise ValueError("home backups fell behind; inspect pull.sh")
        with log.open("a") as sink:
            sink.write(f"waiting for the home backup of stage {index - 1 - lag}\n")
        pause(60)


def retire_states(train: Path) -> int:
    freed = 0
    for path in sorted((train / "states").glob("*.bin")):
        freed += path.stat().st_size
        path.unlink()
    return freed


def set_aside(train: Path) -> None:
    index = 0
    while train.with_name(train.name + "-unstarted-" + str(index)).exists():
        index += 1
    train.rename(train.with_name(train.name + "-unstarted-" + str(index)))


def discard_features(root: Path) -> int:
    freed = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in FEATURE_SUFFIXES:
            freed += path.stat().st_size
            path.unlink()
    return freed


def stage_ids(writer: Path) -> tuple[list[Record], list[str], list[str]]:
    shards = sorted((writer / "shards").glob("streams-*.json"))
    references = [file_reference(path) for path in shards]
    ids: list[str] = []
    skipped: list[str] = []
    for path in shards:
        shard = read(path, 1024 * 1024 * 1024)
        ids += [string(record(row)["id"]) for row in sequence(shard["records"])]
        skipped += [string(value) for value in sequence(shard["skipped"])]
    expected = [string(value) for value in sequence(read(writer / "result.json")["row_ids"])]
    excluded = set(skipped)
    if len(set(ids)) != len(ids) or ids != [value for value in expected if value not in excluded]:
        raise ValueError("written streams do not cover the stage's bank slice exactly")
    return references, ids, skipped


def staged_train(specification: Path, output: Path) -> Record:
    raw = read(specification)
    if set(raw) != FIELDS or raw["schema"] != "staged-training-v1":
        raise ValueError("unsupported staged training specification")
    root = Path(__file__).resolve().parents[1]
    stages, stage_rows = integer(raw["stages"]), integer(raw["stage_rows"])
    first_row, per_stage = integer(raw["first_row"]), integer(raw["updates_per_stage"])
    if stages < 1 or stage_rows < 1 or first_row < 0 or per_stage < 1:
        raise ValueError("positive stage counts required")
    template = record(raw["intent"])
    if template["mode"] != "fresh" or template["parent"] is not None:
        raise ValueError("the intent template describes the fresh first stage")
    store = Store(output)
    write_spec(store, "specification.json", raw)
    expected_source = store.root / "expected-source.txt"
    if expected_source.exists() and expected_source.read_text().strip() != source_identity(root):
        raise ValueError("executable source differs from expected-source.txt")
    summary: list[Record] = []
    previous: Path | None = None
    parent_sha: str | None = None
    for index in range(stages):
        stage = Store(store.root / ("stage-" + str(index).zfill(4)))
        log = stage.root / "stage.log"
        writer, replay = stage.root / "writer", stage.root / "replay"
        fused, train = stage.root / "fused", stage.root / "train"
        target_updates = per_stage * (index + 1)
        wait_for_gpu(log)
        receipt = stage.root / "complete.json"
        if receipt.exists():
            done = read(receipt)
            if (
                integer(done["updates"]) != target_updates
                or exported_updates(train) != target_updates
                or string(read(train / "exports" / "latest.json")["sha256"]) != done["export"]
                or done["parent"] != parent_sha
            ):
                raise ValueError("stage " + str(index) + " completion receipt or ancestry differs")
            if previous is not None and checked_state(train, target_updates):
                retire_states(previous)
            for aside in sorted(stage.root.glob("train-unstarted-*")):
                retire_states(aside)
            summary.append({"stage": index, "status": "trained", "updates": target_updates})
            previous, parent_sha = train, string(done["state"])
            continue
        if index > 0 and (store.root / "hold").exists():
            held: Record = {
                "schema": "staged-training-result-v1",
                "status": "held",
                "held_before_stage": index,
                "stages": summary,
            }
            store.atomic(store.root / "held.json", canonical(held))
            return held
        if previous is not None and not checked_state(previous, target_updates - per_stage):
            raise ValueError("the previous stage's state is missing; restore it before continuing")
        wait_for_backup(store.root, index, log)
        writing = record(raw["writer"]) | {
            "schema": "bank-writing-v1",
            "first": first_row + index * stage_rows,
            "count": stage_rows,
        }
        path = write_spec(stage, "writer.json", writing)
        if not completed(writer / "result.json", "status", "complete"):
            run([*PYTHON, "-m", "benchmarks", "write-bank", str(path), str(writer)], log, root)
        streams, ids, skipped = stage_ids(writer)
        replaying = record(raw["replay"]) | {
            "schema": "token-replay-v1",
            "streams": streams,
            "ids": ids,
        }
        path = write_spec(stage, "replay.json", replaying)
        lost = ((fused / "corpus.json").exists() and not files_present(fused / "corpus.json")) or (
            not (fused / "corpus.json").exists()
            and (replay / "progress.json").exists()
            and not files_present(replay / "progress.json")
        )
        if lost:
            restart_stage(stage.root, log)
        if not (fused / "corpus.json").exists():
            if not completed(replay / "result.json", "status", "complete"):
                run(
                    [*PYTHON, "-m", "benchmarks", "replay-tokens", str(path), str(replay)],
                    log,
                    root,
                )
            if fused.exists():
                shutil.rmtree(fused)
            fusing = record(raw["fuse"]) | {
                "schema": "fused-conversion-v1",
                "corpus": file_reference(replay / "corpus.json"),
                "corpus_format": "corpus-v1",
            }
            path = write_spec(stage, "fuse.json", fusing)
            run(
                [
                    ".venv/numerical-clean/bin/python",
                    "-B",
                    "-m",
                    "w1a1_eagle.numerical.fusing",
                    str(path),
                    str(fused),
                ],
                log,
                root,
            )
        discard_features(replay)
        budget = record(template["budget"]) | {
            "updates": target_updates,
            "attempts": target_updates + per_stage + 1,
        }
        intent = template | {
            "corpus": file_reference(fused / "corpus.json"),
            "corpus_format": "fused-corpus-v1",
            "budget": budget,
            "code_sha256": source_identity(root),
        }
        if previous is not None:
            intent = {key: value for key, value in intent.items() if key != "calibration"}
            intent |= {"mode": "continue", "parent": latest_state(previous)}
        if train.exists() and not state_present(train):
            set_aside(train)
        if (train / "states" / "latest.json").exists():
            intent = read(stage.root / "intent.json") | {
                "mode": "resume",
                "parent": latest_state(train),
            }
            path = write_spec(
                stage, "intent-resume-" + str(trained_updates(train)) + ".json", intent
            )
        else:
            path = write_spec(stage, "intent.json", intent)
        train_stage(path, stage.root, train, string(raw["scope"]), log, root)
        if exported_updates(train) < target_updates or not checked_state(train, target_updates):
            raise ValueError("stage training stopped before a verified export and state")
        state = latest_state(train)
        if index > 0 and record(read(stage.root / "intent.json")["parent"])["sha256"] != parent_sha:
            raise ValueError("stage " + str(index) + " did not continue from the previous stage")
        store_receipt: Record = {
            "stage": index,
            "updates": target_updates,
            "export": string(read(train / "exports" / "latest.json")["sha256"]),
            "state": string(state["sha256"]),
            "parent": parent_sha,
        }
        stage.atomic(receipt, canonical(store_receipt))
        freed = discard_features(fused)
        if previous is not None:
            freed += retire_states(previous)
        for aside in sorted(stage.root.glob("train-unstarted-*")):
            freed += retire_states(aside)
        summary.append(
            {
                "stage": index,
                "status": "trained",
                "updates": target_updates,
                "records": len(ids),
                "skipped": skipped,
                "freed_feature_bytes": freed,
            }
        )
        previous, parent_sha = train, string(state["sha256"])
    result: Record = {
        "schema": "staged-training-result-v1",
        "stages": summary,
        "final_state": None if previous is None else verified_state(previous, per_stage * stages),
        "rule": "token streams, specifications, exports and the latest stage's training state are kept; features and earlier stages' states are discarded after use",
    }
    store.atomic(store.root / "result.json", canonical(result))
    return result
