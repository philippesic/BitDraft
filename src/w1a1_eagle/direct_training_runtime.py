"""Direct entry and native supervision for the existing DSpark lane.

The training journal, budget, checkpoints and timed evaluations remain the
original consumers' files. This module does not create a campaign database.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_SHA = "e647966ad6405499446330ab70f2a25c458428f9deae94c4998ec635367c040e"
PIPELINE_SHA = "a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af"


class GPUOwned(RuntimeError):
    """The canonical native lane lock is held; no launch was attempted."""


class MetadataChanged(ValueError):
    """A concurrent atomic publication changed a mutable status snapshot."""


def metadata(path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024**2:
        raise ValueError(f"bounded regular metadata required: {path}")
    raw = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise MetadataChanged(f"metadata changed during read: {path}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def live_metadata(path):
    """Bounded fresh reads for live progress, retaining every immutable check."""
    for attempt in range(5):
        try:
            return metadata(path)
        except MetadataChanged:
            if attempt == 4:
                raise
            time.sleep(0.05)


def supervisor_publication(path):
    """Read the original supervisor's bounded in-place/atomic publication.

    Only publication races are retried. Persistent malformed JSON remains its
    original exception; invalid identity/schema/source is never a retry class.
    """
    for attempt in range(5):
        try:
            return metadata(path)[0]
        except (MetadataChanged, json.JSONDecodeError, FileNotFoundError):
            if attempt == 4:
                raise
            time.sleep(0.05)


def capture_direct_authorization(original, api, lane, availability, lane_sha256, run_dir, supervisor_path):
    """Original admission/execution guards plus separate cleanup-only custody.

    No original global is changed. The sole source adaptation replaces the two
    direct reads of the mutable supervisor with the bounded owned reader.
    Cleanup custody cannot run a stage or issue a lease; only the exception
    handler gives it to the unchanged resource-return observer.
    """
    import ast
    import types

    source_path = Path(original.__file__)
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != "5d5c390b0f1a6f8e5347b4afb2ff2625033bbc9388fb0f2dda9227b72dc9511a":
        raise ValueError("original launch authorization source differs")
    function = original.capture_launch_authorization
    if function.__name__ != "capture_launch_authorization":
        raise ValueError("original launch authorization callable differs")
    compiled = compile(source, str(source_path), "exec", dont_inherit=True)
    expected = next(c for c in compiled.co_consts
                    if isinstance(c, types.CodeType) and c.co_name == function.__name__)
    if (function.__code__ != expected
            or function.__globals__ is not original.__dict__
            or any(function.__globals__.get(name) is not value for name, value in
                   (("os", os), ("json", json), ("Path", Path),
                    ("require", api.require), ("require_available", api.require_available)))):
        raise ValueError("original launch authorization callable/globals differ")
    availability, run_dir, supervisor_path = map(Path, (availability, run_dir, supervisor_path))
    ids = (os.getpid(), os.getppid())
    births = tuple(api.process_identity(pid) for pid in ids)
    lease_bytes = availability.read_bytes()

    def owned_supervisor():
        current = supervisor_publication(supervisor_path)
        if ((os.getpid(), os.getppid()) != ids
                or tuple(api.process_identity(pid) for pid in ids) != births
                or (current.get("pid"), current.get("supervisor_pid")) != ids
                or current.get("stop_grace_seconds", 0) < 90
                or not os.environ.get("TMUX")):
            raise ValueError("direct supervisor/controller custody identity differs")
        return current

    def stopped(current):
        receipt = current.get("received_signal")
        return (current.get("status") in {"stop_requested", "interrupted"}
                and isinstance(receipt, dict) and type(receipt.get("number")) is int
                and receipt["number"] in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
                and receipt.get("name") == signal.Signals(receipt["number"]).name
                and isinstance(receipt.get("received_at_utc"), str)
                and bool(receipt["received_at_utc"]))

    def execution_supervisor():
        current = owned_supervisor()
        if stopped(current):
            raise InterruptedError(f"campaign signal {current['received_signal']['number']}")
        return current

    tree = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "capture_launch_authorization")
    class PublicationRead(ast.NodeTransformer):
        count = 0
        def visit_Call(self, node):
            if ast.unparse(node) == "json.loads(supervisor_path.read_text())":
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id="_direct_supervisor", ctx=ast.Load()), args=[], keywords=[]), node)
            return self.generic_visit(node)
    rewrite = PublicationRead()
    derived = ast.fix_missing_locations(ast.Module(body=[rewrite.visit(tree)], type_ignores=[]))
    if rewrite.count != 2:
        raise ValueError("original supervisor publication read contract differs")
    namespace = dict(function.__globals__, _direct_supervisor=execution_supervisor)
    exec(compile(derived, str(source_path), "exec", dont_inherit=True), namespace)
    lease, original_authorize = namespace["capture_launch_authorization"](
        lane, availability, lane_sha256, run_dir, supervisor_path)
    admitted = owned_supervisor()

    def custody():
        current_control(lane)
        api.require(availability.read_bytes() == lease_bytes, "availability ownership changed")
        current = owned_supervisor()
        api.require(current["stop_grace_seconds"] == admitted["stop_grace_seconds"],
                    "supervisor stop grace changed")
        api.require((current.get("status") == "running" and current.get("received_signal") is None)
                    or stopped(current), "unrecognized supervisor cleanup state")

    def authorization():
        current_control(lane)
        original_authorize()

    authorization()
    custody()
    return lease, authorization, custody


def record_controller_failure(state, error, cleanup_release):
    """Keep the primary failure even when fresh cleanup observation also fails."""
    state.update(status="stopped" if isinstance(error, InterruptedError) else "failed",
                 failure={"type": type(error).__name__, "reason": str(error)})
    if cleanup_release is not None:
        try:
            cleanup_release()
        except BaseException as cleanup:
            state["cleanup_failure"] = {"type": type(cleanup).__name__, "reason": str(cleanup)}


def release_controller_student_cache(cache, proof, *, canonical_root):
    """Release only this controller's unchanged canonical student marker."""
    if cache is None or cache.state.get('owner') != {'owner':'lane-controller','phase':'student'}:
        return False
    root = Path(canonical_root)
    if (not root.is_absolute() or root.resolve() != root or cache.root != root
            or cache.state_path != root / 'cache-state.json'):
        raise ValueError('controller cleanup canonical cache scope differs')
    current, _ = metadata(cache.state_path)
    if current != cache.state:
        raise ValueError('controller cleanup cache changed after owned student capture')
    cache.release('lane-controller', released_proof=proof)
    return True


def owned_process_record(path, run_directory, *, max_lineage_bytes=256 * 1024**2):
    """Read one original owned process record; large lineage stays bounded.

    Original runners publish cumulative lineage. Historical files remain
    untouched; only this exact known role can exceed generic metadata's16MiB.
    """
    path, root = Path(path), Path(run_directory)
    if (not root.is_absolute() or root != root.resolve()
            or not path.is_absolute() or path != path.resolve()
            or not path.is_relative_to(root)
            or path.name not in {"process.json", "process-lineage.json"}
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("canonical run-owned original process record required")
    def identity(info):
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("single-link regular original process record required")
        return tuple(getattr(info, key) for key in
                     ("st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_gid", "st_nlink", "st_mtime_ns", "st_ctime_ns"))
    before = identity(path.lstat())
    limit = max_lineage_bytes if path.name == "process-lineage.json" else 16 * 1024**2
    if type(limit) is not int or limit <= 0 or before[2] > limit:
        raise ValueError("original process record exceeds bounded role size")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if identity(os.fstat(stream.fileno())) != before:
            raise MetadataChanged("original process record changed before read")
        raw = stream.read(limit + 1)
        if len(raw) > limit or identity(os.fstat(stream.fileno())) != before or identity(path.lstat()) != before:
            raise MetadataChanged("original process record changed during read")
    return json.loads(raw)


def recover_owned_process_inventory(run_directory, *, max_lineage_bytes=256 * 1024**2):
    """Original recovery inventory with ordered distinct complete identities.

    Deduplication never collapses different births or boots of a reused PID.
    This returns observations only; original release/context/resource gates still
    decide availability. No process is stopped, launched or adopted from outside
    the original run-owned process/process-lineage paths.
    """
    root = Path(run_directory)
    if (not root.is_absolute() or root != root.resolve() or not root.is_dir()
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError("canonical owned run directory required")
    groups, identities, seen = [], [], set()
    def add(value):
        if (not isinstance(value, dict) or set(value) != {"boot_id", "pid", "start_ticks"}
                or not isinstance(value["boot_id"], str) or not value["boot_id"]
                or type(value["pid"]) is not int or value["pid"] <= 0
                or type(value["start_ticks"]) is not int or value["start_ticks"] < 0):
            raise ValueError("original complete kernel identity required")
        key = (value["boot_id"], value["pid"], value["start_ticks"])
        if key not in seen:
            seen.add(key)
            identities.append(value)
    for path in sorted(root.glob("**/process.json")):
        process = owned_process_record(path, root, max_lineage_bytes=max_lineage_bytes)
        add(process.get("kernel_identity"))
        if type(process.get("pgid")) is not int or process["pgid"] <= 0 or process.get("pid") != process["kernel_identity"]["pid"] or process["pgid"] != process["pid"]:
            raise ValueError("original owned process group identity differs")
        if process["pgid"] not in groups:
            groups.append(process["pgid"])
    for path in sorted(root.glob("**/process-lineage.json")):
        lineage = owned_process_record(path, root, max_lineage_bytes=max_lineage_bytes)
        if not isinstance(lineage.get("kernel_identities"), list):
            raise ValueError("original process lineage inventory required")
        for value in lineage["kernel_identities"]:
            add(value)
    return groups, identities


def checked(record):
    value, digest = metadata(record["path"])
    if digest != record["sha256"]:
        raise ValueError(f"source metadata identity differs: {record['path']}")
    return value


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_result(path, value):
    """Exclusive atomic JSON publication for a concurrently waiting eval CLI."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex + ".tmp")
    write_new(temporary, value)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if metadata(path)[0] != value:
                raise ValueError("a different immutable evaluation result already exists")
    finally:
        temporary.unlink()


def current_control(lane):
    control, _ = metadata(lane["gpu_control_path"])
    live = control.get("rtx5080", {})
    if live.get("pause_requested") is not False or live.get("stop_all_requested") is not False:
        raise InterruptedError("5080 pause/stop requested")
    if control.get("rtx2080ti", {}).get("pause_requested") is not True:
        raise ValueError("2080 must remain paused")
    return control


def keeper_environment(record):
    """Authenticate the actual keeper before selecting its explicit socket."""
    value = checked(record)
    pid = value["pid"]
    process = Path("/proc") / str(pid)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ticks = int((process / "stat").read_text().split(") ", 1)[1].split()[19])
    expected_ticks = value["birth_ticks"]
    endpoint = Path(value.get("wsl_interop", value.get("socket", "")))
    if (
        value.get("boot_id") != boot
        or ticks != expected_ticks
        or time.time() >= value["deadline_unix"]
        or not stat.S_ISSOCK(endpoint.stat().st_mode)
        or endpoint.stat().st_dev != value["wsl_interop_socket"]["device"]
        or endpoint.stat().st_ino != value["wsl_interop_socket"]["inode"]
        or Path(value["stop_file"]).exists()
    ):
        raise ValueError("keeper process, deadline or socket differs")
    if process.stat().st_uid != os.getuid():
        raise ValueError("keeper UID differs")
    return {"WSL_INTEROP": str(endpoint), "WSL_DISTRO_NAME": "Ubuntu"}


def archive_evaluation_retry_failure(directory, operation):
    """The new exclusive owner preserves the prior attempt before child startup.

    Called inside launch's canonical flock, after source/keeper validation and
    before supervision starts. A previous failure must not abort a caller that
    just started a retry. New startup failures retain the normal failure path.
    """
    directory, operation = Path(directory), Path(operation)
    if not manual_pending(directory / "training"):
        return None
    request_path = directory / "evaluation-request.json"
    request, request_sha = metadata(request_path)
    output = directory / "evaluations" / request["request_id"]
    failure = output / "failure.json"
    if not failure.exists() or (output / "receipt.json").exists():
        return None
    if (operation.resolve() != operation
            or operation.parent != directory / "direct-runtime"
            or output.resolve() != output):
        raise ValueError("canonical owned evaluation retry paths required")
    value, digest = metadata(failure)
    if value.get("request_id") != request["request_id"]:
        raise ValueError("previous evaluation failure belongs to another request")
    if metadata(request_path) != (request, request_sha):
        raise MetadataChanged("evaluation request changed before retry")
    preserved = output / ("previous-failure-" + uuid.uuid4().hex + ".json")
    failure.rename(preserved)
    record = {
        "schema": "direct_native_evaluation_retry_v1",
        "request": {"path": str(request_path), "sha256": request_sha},
        "previous_failure": {"path": str(preserved), "sha256": digest},
        "operation": str(operation),
    }
    write_new(operation / "evaluation-retry.json", record)
    return record


OPERATIONAL_SOURCE_FILES = (
    "train.py", "eval.py", "src/w1a1_eagle/__init__.py",
    "src/w1a1_eagle/direct_training_runtime.py",
    "src/w1a1_eagle/sustained_block_handoff.py",
    "src/w1a1_eagle/direct_native_evaluation.py",
    "src/w1a1_eagle/direct_checkpoint_retention.py",
    "src/w1a1_eagle/direct_export_retention.py",
    "src/w1a1_eagle/checkpoint_digest_proofs.py",
    "src/w1a1_eagle/direct_prepared_recovery.py",
    "src/w1a1_eagle/direct_admission_recovery.py",
    "src/w1a1_eagle/direct_environment_launch.py",
    "src/w1a1_eagle/direct_ensure_environment.py",
    "src/w1a1_eagle/direct_environment_inputs.py",
    "src/w1a1_eagle/direct_startup_byte_proofs.py",
    "scripts/refresh_block_replay_publication_admission.py",
    "scripts/wsl_interop_keeper_20261011.py",
    "scripts/start_wsl_interop_keeper_20261011.ps1",
)
OPERATIONAL_FIXED_SOURCES = {
    "scripts/wsl_interop_keeper_20261011.py": "e16455fdad78bd3bbabf61f79ed3cd8b86c295aba846416a8326a5d96b684d7c",
    "scripts/start_wsl_interop_keeper_20261011.ps1": "7ae86393a2b31af8108cafcc38de2031db917d72b52b8d854f10d84b21ab6aa5",
}


def freeze_operational_sources(operation):
    """Snapshot the small operational closure before starting supervision.

    Scientific code remains at its separately authenticated native source root.
    Public command updates cannot change an already running job's source files.
    """
    operation = Path(operation)
    if not operation.is_absolute() or operation.resolve() != operation:
        raise ValueError("canonical operation directory required")
    fields = ("st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_gid",
              "st_nlink", "st_mtime_ns", "st_ctime_ns")
    identity = lambda value: tuple(getattr(value, key) for key in fields)
    contents = {}
    for name in OPERATIONAL_SOURCE_FILES:
        path = ROOT / name
        before = path.lstat()
        if (path.resolve() != path or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1 or before.st_size > 1024**2):
            raise ValueError("bounded canonical operational source required: " + name)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if identity(os.fstat(stream.fileno())) != identity(before):
                raise MetadataChanged("operational source changed at open: " + name)
            raw = stream.read(1024**2 + 1)
            if (len(raw) > 1024**2
                    or identity(os.fstat(stream.fileno())) != identity(before)
                    or identity(path.lstat()) != identity(before)):
                raise MetadataChanged("operational source changed during snapshot: " + name)
        fixed = OPERATIONAL_FIXED_SOURCES.get(name)
        if fixed is not None and hashlib.sha256(raw).hexdigest() != fixed:
            raise ValueError("original operational source changed: " + name)
        if path.suffix == ".py":
            compile(raw, str(path), "exec", dont_inherit=True)
        elif fixed is None:
            raise ValueError("unsupported operational source: " + name)
        contents[name] = (raw, identity(before))
    for name, (_, before) in contents.items():
        if identity((ROOT / name).lstat()) != before:
            raise MetadataChanged("operational source changed before publication: " + name)
    frozen = operation / "source"
    frozen.mkdir(mode=0o700, exist_ok=False)
    pins = {}
    for name, (raw, _) in contents.items():
        path = frozen / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(path, 0o444)
        digest = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("operational source snapshot differs: " + name)
        pins[str(path)] = digest
    return frozen, pins


def launch(config, *, run_dir=None, detach=False, evaluation_only=False, config_path=None):
    """One command obtains native supervision; an owned GPU is never duplicated."""
    if not all(
        callable(globals().get(name))
        for name in (
            "controller",
            "trainer",
            "prepare_controller_window",
            "evaluate_controller_manual",
        )
    ):
        raise RuntimeError(
            "Direct native controller integration is incomplete; the existing run is unchanged."
        )
    config_path = Path(config_path or ROOT / "configs/training.json").resolve(strict=True)
    lane = checked(config["lane"])
    current_control(lane)
    directory = Path(run_dir or config["run_dir"]).resolve(strict=True)
    lock = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GPUOwned("The GPU already has an owner; no duplicate training was started.")
        latest, _ = metadata(directory / "training/checkpoints/latest.json")
        ledger, _ = metadata(directory / "training/budget-used.json")
        if not latest.get("committed") or ledger["max_seconds"] != 43200:
            raise ValueError("a complete latest checkpoint and original allocation are required")
        from w1a1_eagle.direct_environment_launch import settle_original_budget

        prelaunch_accounting = settle_original_budget(
            config, lane, directory, config_path=config_path, owner_fd=owner.fileno(),
            metadata=metadata, checked=checked, current_control=current_control)
        ledger, _ = metadata(directory / "training/budget-used.json")
        # Original TrainingBudget.load owns dead-attempt reconciliation and its
        # conservative accounting. Rejecting an active journal here would make
        # the normal train command unable to recover a killed trainer at all.
        if evaluation_only and (
            ledger.get("active_attempt") is not None
            or latest["cursor"]["elapsed_seconds"] != ledger["training_seconds"]
        ):
            raise ValueError(
                "evaluation requires settled checkpoint accounting; run python train.py to recover first"
            )
        if not evaluation_only and ledger["training_seconds"] >= ledger["max_seconds"]:
            if prelaunch_accounting["allocation_exhausted_by_recovery"]:
                print(json.dumps({"accounting_recovery_blocked": prelaunch_accounting}), flush=True)
                raise ValueError(
                    "Original crash recovery charged the remaining reservation; "
                    "the saved checkpoint does not prove a completed 12-hour TRAIN run. "
                    "Original accounting and checkpoint counters are preserved."
                )
            raise ValueError(
                "The full training allocation is complete; use python eval.py for evaluation."
            )
        checked(config["evaluation_plan"])
        checked(config["original_admission"])
        supervisor = config["supervisor"]
        source = Path(supervisor["path"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != supervisor["sha256"]:
            raise ValueError("original supervisor source differs")
        if supervisor["sha256"] != SUPERVISOR_SHA:
            raise ValueError("original 90-second supervisor required")
        from w1a1_eagle import direct_admission_recovery
        from w1a1_eagle.direct_ensure_environment import ensure_environment

        config, keeper_values, environment_operation = ensure_environment(
            config_path, repo=ROOT, runtime=sys.modules[__name__],
            recovery=direct_admission_recovery, lock_fd=owner.fileno(), run_dir=directory,
            expected_config=prelaunch_accounting["canonical_config"])
        lane = checked(config["lane"])
        environment = os.environ.copy()
        environment.update(lane["environment"])
        environment.update(keeper_values)
        environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
        run_id = "direct-dspark-" + uuid.uuid4().hex
        operation = directory / "direct-runtime" / run_id
        (operation / "supervisor/scripts").mkdir(parents=True, exist_ok=False)
        driver = operation / "supervisor/scripts/remote_job.py"
        with driver.open("xb") as stream:
            stream.write(source)
        snapshot = dict(config, run_dir=str(directory), evaluation_only=evaluation_only)
        write_new(operation / "config.json", snapshot)
        source_root, source_pins = freeze_operational_sources(operation)
        private = Path("/dev/shm/binary-eagle-decoding") / run_id
        private.mkdir(mode=0o700, parents=True, exist_ok=False)
        old_umask = os.umask(0o077)
        try:
            lease_path = private / "availability.json"
            granted = time.time()
            write_new(
                lease_path,
                {
                    "schema": "nine_model_gpu_lease_v1",
                    "host": "rtx5080",
                    "user_announced_available": True,
                    "sole_owner": True,
                    "pause_requested": False,
                    "bundle_sha256": config["lane"]["sha256"],
                    "gpu_uuid": lane["gpu_uuid"],
                    "granted_unix": granted,
                    "expires_unix": granted + 300,
                },
            )
        finally:
            os.umask(old_umask)
        runtime = {
            "schema": "direct_dspark_runtime_v1",
            "config": {
                "path": str(operation / "config.json"),
                "sha256": hashlib.sha256((operation / "config.json").read_bytes()).hexdigest(),
            },
            "supervisor_state": str(operation / "supervisor/runs" / run_id / "state.json"),
            "availability": str(lease_path),
            "operation": str(operation),
            "source_origin": str(ROOT),
            "runtime_sources": source_pins,
            "prelaunch_accounting": prelaunch_accounting,
            "environment_operation": str(environment_operation),
        }
        runtime_path = operation / "runtime.json"
        write_new(runtime_path, runtime)
        archive_evaluation_retry_failure(directory, operation)
        command = [
            config["python"],
            "-B",
            str(driver),
            run_id,
            "--stop-grace-seconds",
            "90",
            "--",
            config["python"],
            "-B",
            str(source_root / "train.py"),
            "--config",
            str(operation / "config.json"),
            "--_controller",
            str(runtime_path),
        ]
        # The child takes the same canonical lock and rechecks the fresh lease.
        # No GPU or training process is launched while the previous owner holds it.
    explicit = {
        key: value
        for key, value in environment.items()
        if key in lane["environment"]
        or key in {"WSL_INTEROP", "WSL_DISTRO_NAME", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS"}
    }
    shell_command = shlex.join(
        ["env", *[key + "=" + value for key, value in explicit.items()], *command]
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", run_id, "-c", str(source_root), shell_command],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    print(f"Training supervised in tmux {run_id}; state: {runtime['supervisor_state']}", flush=True)
    deadline = time.monotonic() + 30
    while not (operation / "owned-identity.json").exists():
        path = Path(runtime["supervisor_state"])
        if path.exists():
            starting, _ = live_metadata(path)
            if starting["status"] in {"finished", "failed", "interrupted"}:
                print(
                    json.dumps({"startup_failed": starting, "operation": str(operation)}),
                    flush=True,
                )
                return starting.get("exit_code") or 1
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Supervisor startup is unverified; preserved operation: {operation}"
            )
        time.sleep(0.25)
    if detach:
        return 0
    return watch_training(runtime, directory)


def watch_training(runtime, directory):
    """A plain train command stays attached to progress; Ctrl-C stops its exact job."""
    last = None
    last_print = 0.0
    started = time.monotonic()
    try:
        while True:
            path = Path(runtime["supervisor_state"])
            if not path.exists():
                if time.monotonic() - started > 30:
                    raise RuntimeError(
                        "supervisor did not publish state; inspect the preserved operation"
                    )
                time.sleep(1)
                continue
            supervisor, _ = live_metadata(path)
            if supervisor["status"] in {"finished", "failed", "interrupted"}:
                print(
                    json.dumps(
                        {
                            "supervisor": supervisor["status"],
                            "state": str(path),
                            "exit_code": supervisor.get("exit_code"),
                        }
                    ),
                    flush=True,
                )
                return (
                    0
                    if supervisor["status"] == "finished" and supervisor.get("exit_code") == 0
                    else 1
                )
            if time.monotonic() - last_print >= 60:
                snapshot = status_snapshot({"run_dir": str(directory)})
                progress = (snapshot["current_updates"], snapshot["current_trainer_seconds"], snapshot["progress_source"], snapshot["live_status"])
                if progress != last:
                    print(
                        json.dumps(
                            {
                                key: snapshot[key] for key in (
                                    "current_updates", "current_trainer_seconds", "progress_source",
                                    "saved_updates", "saved_trainer_seconds", "live_status",
                                    "live_heartbeat_age_seconds", "remaining_trainer_seconds")
                            }
                        ),
                        flush=True,
                    )
                    last = progress
                last_print = time.monotonic()
            time.sleep(1)
    except KeyboardInterrupt:
        # STOP also handles an interrupt during the supervisor's startup, before
        # the authenticated controller has published its kernel birth receipt.
        owned_stop = None
        stop_path = directory / "STOP"
        stop_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
                       "st_size", "st_mtime_ns", "st_ctime_ns")
        try:
            with stop_path.open("x") as marker:
                item = os.fstat(marker.fileno())
                owned_stop = tuple(getattr(item, key) for key in stop_fields)
        except FileExistsError:
            pass
        supervisor, _ = live_metadata(runtime["supervisor_state"])
        identity_path = Path(runtime["operation"]) / "owned-identity.json"
        if identity_path.exists() and supervisor.get("status") == "running":
            expected = metadata(identity_path)[0]["supervisor"]
            pid = expected["pid"]

            def identity():
                fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
                return {
                    "pid": pid,
                    "start_ticks": int(fields[19]),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                }

            if supervisor["supervisor_pid"] != pid or identity() != expected:
                raise RuntimeError("supervisor birth differs; STOP retained and no signal sent")
            os.kill(pid, signal.SIGTERM)
        print("Stop requested; waiting for the owned supervisor and original cleanup.", flush=True)
        while True:
            result, _ = live_metadata(runtime["supervisor_state"])
            if result["status"] in {"finished", "failed", "interrupted"}:
                # This CLI's one-shot interruption is fulfilled. Independent or
                # changed STOP files and the shared human pause remain intact.
                if owned_stop is not None:
                    try:
                        item = stop_path.lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        if (stat.S_ISREG(item.st_mode) and item.st_size == 0
                                and tuple(getattr(item, key) for key in stop_fields) == owned_stop):
                            stop_path.unlink()
                return 130
            time.sleep(1)


def authenticate_runtime(path):
    runtime, _ = metadata(path)
    if runtime.get("schema") != "direct_dspark_runtime_v1":
        raise ValueError("direct runtime schema differs")
    for name, digest in runtime["runtime_sources"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"prepared runtime source changed: {name}")
    return runtime, checked(runtime["config"])


def status_snapshot(config, *, run_dir=None):
    """Read existing journals only; no GPU query, model read or process launch."""
    import math
    directory = Path(run_dir or config["run_dir"]).resolve(strict=True)
    sources = {}

    def read(role, path):
        value, digest = live_metadata(path)
        sources[role] = {"path": str(path), "sha256": digest}
        return value

    latest = read("latest", directory / "training/checkpoints/latest.json")
    ledger = read("budget", directory / "training/budget-used.json")
    state = read("lane", directory / "state.json")
    now = time.time()
    saved = latest["cursor"]
    def seconds(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    if (ledger.get("schema") != "continuous_training_budget_v1"
            or type(saved.get("step")) is not int or saved["step"] < 0
            or not seconds(saved.get("elapsed_seconds"))
            or not seconds(ledger.get("training_seconds")) or not seconds(ledger.get("max_seconds"))):
        raise ValueError("invalid saved cursor/training budget metadata")
    active = ledger.get("active_attempt")
    if active is not None and (not isinstance(active, dict) or type(active.get("pid")) is not int
            or active["pid"] <= 0 or not isinstance(active.get("boot_id"), str)
            or not active["boot_id"] or not isinstance(active.get("process_birth"), str)
            or not active["process_birth"] or not seconds(active.get("started_unix"))):
        raise ValueError("invalid active training budget identity")
    live, live_status, age = None, "unavailable", None
    status_path = directory / "training/status.json"
    if status_path.exists():
        training = read("training_status", status_path)
        if not isinstance(training, dict) or training.get("schema") != "nine_model_qat_training_v1":
            raise ValueError("original training status schema differs")
        live_status = "inactive_budget" if active is None else "status_not_running"
        if active is not None and training.get("status") == "running":
            cursor, heartbeat = training.get("cursor"), training.get("heartbeat_unix")
            if (not isinstance(cursor, dict) or type(cursor.get("step")) is not int
                    or cursor["step"] < 0 or not seconds(cursor.get("elapsed_seconds"))
                    or not seconds(heartbeat)):
                raise ValueError("invalid original completed-step status/heartbeat")
            age = now - heartbeat
            live_status = "bound_completed_step"
            if age < -5 or age > 120:
                live_status = "stale_or_future_heartbeat"
            elif heartbeat < active["started_unix"]:
                live_status = "heartbeat_before_active_attempt"
            elif cursor["step"] < saved["step"] or cursor["elapsed_seconds"] < max(saved["elapsed_seconds"], ledger["training_seconds"]):
                live_status = "counter_regression"
            elif cursor["elapsed_seconds"] > ledger["training_seconds"] + heartbeat - active["started_unix"] + 5:
                live_status = "elapsed_exceeds_active_interval"
            else:
                # These are completed optimizer transactions, never extrapolated
                # wall time. Journals alone do not establish process/GPU health.
                live = cursor
    budget_after, _ = live_metadata(directory / "training/budget-used.json")
    budget_consistent = budget_after == ledger
    if not budget_consistent:
        live, live_status = None, "budget_changed_during_observation"
    current = live if live is not None else saved
    accounted = max(current["elapsed_seconds"], ledger["training_seconds"])
    phase = state.get("status")
    if phase == "training" and ledger.get("active_attempt") is None:
        phase = "preparing_or_loading"
    elif phase == "restoring_shards":
        phase = "data_preparation"
    if not budget_consistent:
        phase = "transitioning"
    result = {
        "observed_unix": now,
        "phase": phase,
        "controller_phase": state.get("status"),
        "failure": state.get("failure"),
        "saved_updates": latest["cursor"]["step"],
        "saved_trainer_seconds": latest["cursor"]["elapsed_seconds"],
        "remaining_trainer_seconds": max(0, ledger["max_seconds"] - accounted),
        "current_updates": current["step"],
        "current_trainer_seconds": current["elapsed_seconds"],
        "accounted_trainer_seconds": accounted,
        "progress_source": "live_completed_step" if live is not None else "saved_checkpoint",
        "live_updates": live["step"] if live is not None else None,
        "live_trainer_seconds": live["elapsed_seconds"] if live is not None else None,
        "live_status": live_status,
        "live_heartbeat_age_seconds": age,
        "live_freshness_limit_seconds": 120,
        "budget_observation_consistent": budget_consistent,
        "process_health_verified": False,
        "budget_active_attempt": ledger.get("active_attempt"),
        "checkpoint": {key: latest[key] for key in ("path", "sha256")},
        "source_metadata": sources,
        "gpu_queried": False,
    }
    record = state.get("direct_training_runtime")
    if record:
        current = checked(record)
        supervisor = read("supervisor", Path(current["supervisor_state"]))
        result["supervisor"] = {
            key: supervisor.get(key)
            for key in (
                "status",
                "pid",
                "supervisor_pid",
                "received_signal",
                "stop_grace_seconds",
                "exit_code",
            )
        }
        owned = Path(current["operation"]) / "owned-identity.json"
        if owned.exists():
            result["owned_identities"] = read("owned_identity", owned)
    return result


def bind_scientific_sources(config):
    """Bind original scientific imports without consuming model/data admissions."""
    import w1a1_eagle

    root = Path(config["native_source_root"]).resolve(strict=True)
    expected = {
        "src/w1a1_eagle/nine_model_pipeline.py": PIPELINE_SHA,
        "src/w1a1_eagle/block_shard_lifecycle.py": "a2ad87283999fe1268cd933dce59388c0c0ab544a7a27dcc3797967f305ece4e",
        "src/w1a1_eagle/continuous_budget.py": "9cbc775931b1f2739c8a9b7cad9e98f746449000cce06dd8b9b6792b74eb655e",
        "scripts/train_nine_model_qat.py": "f6b2535d3d18c1f4df8b5e6abd65d988a17a08dbc80208ca45bd4cfd1f640a58",
        "scripts/run_nine_model_lane.py": "5d5c390b0f1a6f8e5347b4afb2ff2625033bbc9388fb0f2dda9227b72dc9511a",
    }
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("original source role differs: " + name)
    w1a1_eagle.__path__.insert(0, str(root / "src/w1a1_eagle"))
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    for name, loaded in list(sys.modules.items()):
        if (
            name.startswith("w1a1_eagle.")
            and not name.startswith("w1a1_eagle.direct_")
            and name != "w1a1_eagle.sustained_block_handoff"
        ):
            origin = getattr(loaded, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise ValueError("scientific module was imported before its frozen source: " + name)
    return root


def frozen_runtime(config):
    """Load the authenticated original scientific closure before optional hooks."""
    import importlib
    import importlib.util

    root = bind_scientific_sources(config)
    pipeline = importlib.import_module("w1a1_eagle.nine_model_pipeline")
    lifecycle = importlib.import_module("w1a1_eagle.block_shard_lifecycle")
    spec = importlib.util.spec_from_file_location(
        "direct_original_lane_runner", root / "scripts/run_nine_model_lane.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    record = config["operational_runtime"]["helper"]
    raw = Path(record["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("accepted operational memo helper differs")
    helper_spec = importlib.util.spec_from_file_location(
        "direct_original_memo_helper", record["path"]
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    if config.get("startup_byte_proof") is not None:
        from w1a1_eagle.direct_startup_byte_proofs import install_from_config

        install_from_config(helper, config, metadata=metadata, checked=checked,
                            fingerprint=pipeline.Files.fingerprint)
    manifest = checked(config["operational_runtime"]["manifest"])
    selection = json.loads(
        (
            Path(config["operational_runtime"]["manifest"]["path"]).parent / "selection.json"
        ).read_text()
    )
    expected_selection = manifest["files"]["selection.json"]
    if (
        hashlib.sha256(
            (
                Path(config["operational_runtime"]["manifest"]["path"]).parent / "selection.json"
            ).read_bytes()
        ).hexdigest()
        != expected_selection
    ):
        raise ValueError("accepted memo source selection differs")
    lane = checked(config["lane"])
    timed = importlib.import_module("evaluate_nine_model_timed_checkpoint")
    helper.install_op04_replay_memo(manifest, selection, lane, pipeline)
    helper.install_op04_stop_lifecycle(selection, lane)
    import runpy

    helper_path = (
        Path(config["operational_runtime"]["manifest"]["path"]).parent / "resource-helper-source.py"
    )
    if (
        hashlib.sha256(helper_path.read_bytes()).hexdigest()
        != manifest["files"]["resource-helper-source.py"]
    ):
        raise ValueError("accepted resource-return waiter source differs")
    pipeline.wait_for_resource_return = runpy.run_path(str(helper_path))["wait_for_resource_return"]
    runner.wait_for_resource_return = pipeline.wait_for_resource_return
    timed_source = Path(selection["original_timed_evaluator"]["path"]).read_bytes()
    if hashlib.sha256(timed_source).hexdigest() != selection["original_timed_evaluator"]["sha256"]:
        raise ValueError("original evaluator source changed")
    timed_tree, _ = helper.derive_op04_timed(timed_source)
    timed.__dict__["_op04_timed_handoff"] = lambda *a: helper.op04_timed_handoff(
        pipeline.wait_for_resource_return, pipeline.atomic_json, *a
    )
    timed.__dict__["_op04_runtime_provenance"] = {
        "profile": helper.OP04_PROFILE,
        "manifest": config["operational_runtime"]["manifest"],
        "original_scientific_source": selection["original_timed_evaluator"],
        "active_evaluation_cap_seconds": 1200,
        "released_handoff_wait_wall_separate": True,
    }
    exec(compile(timed_tree, timed.__file__, "exec"), timed.__dict__)
    # Reinstall source-bound scoped memos after the original evaluator definition.
    helper.install_op04_policy_memo(manifest, selection, lane, pipeline, timed)
    helper.install_op04_native_stage_memo(manifest, selection, lane, pipeline)
    recovery = config.get("physical_admission_recovery")
    if recovery is not None:
        if (not isinstance(recovery, dict)
                or set(recovery) != {"profile", "view", "existing_helper"}
                or recovery["profile"] != "original_completed_admission_current_boot_view_v1"):
            raise ValueError("explicit original-reader physical admission recovery required")
        from w1a1_eagle import block_data
        from w1a1_eagle.direct_admission_recovery import install

        install(block_data, recovery["view"], recovery["existing_helper"])
    return root, lane, pipeline, lifecycle, runner, helper, manifest, selection


def prior_active_window(config, state):
    """Authenticate one prior operation's existing active window, without executing it."""
    record = state.get("direct_training_runtime")
    if record is None:
        return None
    previous = checked(record)
    directory = Path(config["run_dir"]).resolve()
    operation = Path(previous["operation"]).resolve()
    if (
        previous.get("schema") != "direct_dspark_runtime_v1"
        or operation.parent != directory / "direct-runtime"
        or Path(record["path"]).resolve() != operation / "runtime.json"
    ):
        raise ValueError("prior runtime is outside this run's canonical operation")
    before = checked(previous["config"])
    if Path(before["run_dir"]).resolve() != directory or any(
        before.get(key) != config.get(key)
        for key in ("lane", "evaluation_plan", "original_admission", "native_source_root")
    ):
        raise ValueError("prior runtime scientific run bindings differ")
    active_path = operation / "active-window.json"
    if not active_path.exists():
        # A previous controller may have stopped before adopting its predecessor.
        # Preserve the one already authenticated predecessor reference in state.
        carried = state.get("prepared_window_predecessor")
        if carried is None:
            return None
        if carried["runtime"] == record:
            raise ValueError("prior prepared-window reference points to itself")
        recovered = prior_active_window(config, {"direct_training_runtime": carried["runtime"]})
        if recovered != carried:
            raise ValueError("carried prepared-window predecessor changed")
        return recovered
    active, active_sha = metadata(active_path)
    catalog = checked(active["catalog"])
    if not Path(active["catalog"]["path"]).resolve().is_relative_to(directory / "direct-runtime"):
        raise ValueError("prior active catalog is outside this run's owned operations")
    source_roots = {
        str(Path(row["path"]).resolve().parent) for row in catalog["source_cache_states"]
    }
    for pair in catalog["owned_pair_roots"]:
        path = Path(pair)
        if (
            not path.is_absolute()
            or str(path.resolve()) != pair
            or not path.is_relative_to(directory / "direct-runtime")
            or not path.name.startswith("pair-")
            or pair not in source_roots
        ):
            raise ValueError("prior prepared pair ownership differs from its original catalog")
    producer = catalog["producer_source"]
    bound = [
        digest
        for path, digest in previous["runtime_sources"].items()
        if Path(path).parts[-3:] == ("src", "w1a1_eagle", "sustained_block_handoff.py")
    ]
    info = Path(producer["path"]).lstat()
    if (
        len(bound) != 1
        or producer["sha256"] != bound[0]
        or not stat.S_ISREG(info.st_mode)
        or info.st_size > 1024**2
        or hashlib.sha256(Path(producer["path"]).read_bytes()).hexdigest() != bound[0]
    ):
        raise ValueError("prior catalog producer source differs from authenticated runtime")
    return {
        "schema": "prior_prepared_window_v1",
        "runtime": record,
        "active_window": {"path": str(active_path), "sha256": active_sha},
        "catalog": active["catalog"],
        "producer_source": producer,
    }


def prepare_controller_window(
    runtime, config, api, lane, state, run_dir, attempt, runner, release, authorization, cache
):
    """Prepare outside TRAIN and reuse a ready window across native evaluations."""
    import copy

    from w1a1_eagle import sustained_block_handoff as prepared

    if cache is None:
        raise ValueError("direct DSpark requires the admitted rotating block provider")
    proof = release()
    if cache.state["owner"] is not None:
        cache.release(cache.state["owner"]["owner"], released_proof=proof)
    spec = checked(lane["config"])
    plan = api.FrozenShardPlan.load({key: spec["data"][key] for key in ("path", "sha256")})
    latest, _ = metadata(Path(run_dir) / "training/checkpoints/latest.json")
    cursor = latest["cursor"]["data_cursor"]
    if config.get("evaluation_only") is True:
        # Native export needs the original saved model and source/consumed-seal
        # identity, not another training-data capture or optimizer update.
        catalog = prepared.build_catalog(
            plan,
            [api.pin(cache.state_path)],
            Path(runtime["operation"]) / ("evaluation-source-" + uuid.uuid4().hex + ".json"),
            max_bytes=config["prepared_data"]["max_window_bytes"],
            publication_shards=[],
            api=api,
        )
        api.atomic_json(
            Path(runtime["operation"]) / "active-window.json",
            {
                "catalog": catalog,
                "phase_costs": [],
                "trainer_seconds_charged": 0,
                "evaluation_only": True,
            },
        )
        return
    active_path = Path(runtime["operation"]) / "active-window.json"
    active = metadata(active_path)[0] if active_path.exists() else None
    predecessor = state.get("prepared_window_predecessor")
    if active is None and predecessor is not None:
        deferred_adoption = False
        verified = prior_active_window(config, {"direct_training_runtime": predecessor["runtime"]})
        if verified != predecessor:
            raise ValueError("prepared-window predecessor changed before adoption")
        active = checked(predecessor["active_window"])
        if predecessor["producer_source"]["sha256"] != api.file_sha256(Path(prepared.__file__)):
            previous_catalog = checked(predecessor["catalog"])
            previous_roots = {str(Path(row["path"]).parent)
                              for row in previous_catalog["source_cache_states"]}
            pending_roots = set(state.get("prepared_pair_registrations", {})) - previous_roots
            completed = any(row.get("original_catalog") == predecessor["catalog"]
                            and row.get("cursor") == cursor
                            for row in state.get("prepared_window_retirements", []))
            if completed and pending_roots:
                deferred_adoption = True
                # Old publications may already be evicted. The exact completed
                # retirement branch below authenticates its archives, owned
                # states and canonical union before full current re-admission.
                # Do not migrate those stale pre-retirement publication pins.
                import inspect
                for field, cls in (("original_provider_source", api.ShardCache),
                                   ("original_data_source", api.BlockDataset)):
                    api.checked(previous_catalog[field])
                    api._require(Path(previous_catalog[field]["path"]).resolve()
                                 == Path(inspect.getfile(cls)).resolve(),
                                 "retired predecessor original scientific source role differs")
            else:
                output = (Path(runtime["operation"]) / "catalog-migrations"
                          / uuid.uuid4().hex / "catalog.json")
                migration = prepared.migrate_catalog(
                    plan, predecessor["catalog"], output,
                    expected_producer_source=predecessor["producer_source"], api=api,
                )
                active = dict(active, catalog=migration["catalog"])
                state.setdefault("prepared_catalog_migrations", []).append(migration)
        # Until full current admission succeeds, keep the old producer bound
        # only through its authenticated predecessor. A failed recovery must
        # not publish an old catalog under the new runtime's source identity.
        if not deferred_adoption:
            api.atomic_json(active_path, active)
        state.setdefault("prepared_window_adoptions", []).append(predecessor)
        api.atomic_json(Path(run_dir) / "state.json", state)
    owned_roots = []

    def merge_seals(locator):
        incoming = json.loads(api.checked(locator).read_text())
        api._require(
            incoming.get("plan_sha256") == plan.sha256 and incoming.get("owner") is None,
            "only a genuinely released original publication may extend history",
        )
        for shard, value in incoming["seals"].items():
            old = cache.state["seals"].get(shard)
            if old is None:
                cache.state["seals"][shard] = copy.deepcopy(value)
            else:
                api._require(
                    prepared._same_original_seal(old, value),
                    "immutable historical tensor or original metadata seal changed",
                )
                if old.get("original_metadata") is None and value.get("original_metadata"):
                    old["original_metadata"] = copy.deepcopy(value["original_metadata"])
                if old.get("recovery") is None and value.get("recovery"):
                    # The original publication/retirement consumer validates its
                    # recovery before invoking this private serialized callback.
                    old["recovery"] = copy.deepcopy(value["recovery"])
        cache._save()
        return [api.pin(cache.state_path)]

    def validate_registration(record):
        # Reuse prior-runtime run/scientific binding authentication. This does
        # not execute that runtime or assume that it published an active window.
        prior_active_window(config, {"direct_training_runtime": record["runtime"]})
        previous = checked(record["runtime"])
        window = Path(record["window_root"])
        if window.parent != Path(previous["operation"]) / "prepared-data":
            raise ValueError("registered pair window differs from authenticated runtime")
        producer = record["producer_source"]
        expected = [
            digest
            for path, digest in previous["runtime_sources"].items()
            if Path(path).parts[-3:] == ("src", "w1a1_eagle", "sustained_block_handoff.py")
        ]
        if (
            expected != [producer["sha256"]]
            or hashlib.sha256(Path(producer["path"]).read_bytes()).hexdigest() != producer["sha256"]
        ):
            raise ValueError("registered pair producer source differs from authenticated runtime")

    def register_cache(locator, requested_shards):
        authorization()
        proof = release()
        api._require(
            proof.get("owned_process_groups_absent") is True
            and proof.get("owned_cuda_pids_absent") is True,
            "registration requires actual released original producer/student",
        )
        path = Path(locator["path"])
        pair = path.parent
        empty = checked(locator)
        if (
            path.resolve() != path
            or path.name != "cache-state.json"
            or pair.parent != window_root
            or window_root.parent != Path(runtime["operation"]) / "prepared-data"
            or Path(runtime["operation"]).resolve().parent
            != Path(run_dir).resolve() / "direct-runtime"
            or not pair.name.startswith("pair-")
            or len(pair.name) != 37
            or any(c not in "0123456789abcdef" for c in pair.name[5:])
            or empty.get("schema") != "block_shard_cache_v1"
            or empty.get("plan_sha256") != plan.sha256
            or empty.get("owner") is not None
            or empty.get("resident") != {}
            or empty.get("seals") != cache.state["seals"]
        ):
            raise ValueError("fresh registered pair scope/original empty history differs")
        snapshot = pair / "registered-empty-state.json"
        with snapshot.open("xb") as stream:
            stream.write(path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        initial = api.pin(snapshot)
        api._require(
            initial["sha256"] == locator["sha256"], "registered initial state copy differs"
        )
        registration = {
            "schema": "prepared_pair_registration_v1",
            "window_root": str(window_root),
            "initial_state": initial,
            "requested_shards": list(requested_shards),
            "runtime": api.pin(Path(runtime["operation"]) / "runtime.json"),
            "producer_source": api.pin(Path(prepared.__file__)),
        }
        validate_registration(registration)
        state.setdefault("prepared_pair_registrations", {})[str(pair)] = registration
        state["prepared_cache_roots"] = sorted({*state.get("prepared_cache_roots", []), str(pair)})
        state["prepared_owned_pair_roots"] = sorted(
            {*state.get("prepared_owned_pair_roots", []), str(pair)}
        )
        api.atomic_json(Path(run_dir) / "state.json", state)

    roots = {str(cache.root), *state.get("prepared_cache_roots", [])}
    registrations = state.get("prepared_pair_registrations", {})
    old_catalog = checked(active["catalog"]) if active is not None else None
    old_sources = {str(Path(row["path"]).parent) for row in old_catalog["source_cache_states"]} if old_catalog else set()
    pending_registrations = {root: row for root, row in registrations.items() if root not in old_sources}
    completed_retirement = None
    if active is not None and pending_registrations:
        # New producers can start only after an ORIGINAL complete retirement.
        # Its durable receipt and unchanged owned states distinguish that case
        # from an interrupted retirement, which still uses reconcile_retirement.
        matches = [row for row in state.get("prepared_window_retirements", [])
                   if row.get("original_catalog") == active["catalog"] and row.get("cursor") == cursor]
        api._require(bool(matches), "new registered PREP lacks completed original retirement")
        completed_retirement = matches[-1]
        api._require(completed_retirement.get("schema") == "prepared_block_window_retirement_v1"
                     and set(completed_retirement["owned_pair_roots"]) == set(old_catalog["owned_pair_roots"])
                     and {str(Path(row["path"]).parent) for row in completed_retirement["owned_cache_states"]}
                     == set(old_catalog["owned_pair_roots"]), "completed retirement owned inventory differs")
        archives = completed_retirement["source_state_archives"]
        api._require([row["original"] for row in archives] == old_catalog["source_cache_states"],
                     "completed retirement source inventory differs")
        receipt_directory = Path(archives[0]["snapshot"]["path"]).parent
        api._require(receipt_directory.parent == Path(active["catalog"]["path"]).parent
                     and receipt_directory.name.startswith("retirement-")
                     and metadata(receipt_directory / "receipt.json")[0] == completed_retirement,
                     "completed retirement immutable receipt differs")
        context = metadata(receipt_directory / "context.json")[0]
        api._require(context["catalog"] == active["catalog"] and context["cursor"] == cursor
                     and context["reservation"] == latest["cursor"].get("batch_reservation"),
                     "completed retirement exact cursor/reservation differs")
        for row in archives:
            api._require(row["snapshot"]["sha256"] == row["original"]["sha256"], "retirement source archive identity differs")
            api.checked(row["snapshot"])
        for locator in completed_retirement["owned_cache_states"]:
            api.checked(locator)
        api._require(set(pending_registrations) == roots - old_sources - {str(cache.root)}
                     and set(pending_registrations) <= set(state.get("prepared_owned_pair_roots", [])),
                     "unfinished new PREP roots lack exact registered ownership")
    if active is None or pending_registrations:
        recover = registrations if active is None else pending_registrations
        owned_roots = state.get("prepared_owned_pair_roots", [])
        if active is None and (set(owned_roots) != set(registrations) or roots - {str(cache.root)} != set(owned_roots)):
            raise ValueError("unfinished preparation roots lack exact registered ownership")
        if recover:
            from w1a1_eagle.direct_prepared_recovery import preserve_unpublished_registered_targets
            prefix = preserve_unpublished_registered_targets(
                plan, recover, run_dir=run_dir, shard_config=spec["shard_lifecycle"],
                runner=runner, source=lane["source"], release=release,
                authorization=authorization, validate_registration=validate_registration, api=api,
            )
            if prefix["preserved"]:
                state.setdefault("prepared_interrupted_prefix_recoveries", []).append(prefix)
                api.atomic_json(Path(run_dir) / "state.json", state)
            recovered = prepared.recover_registered_preparation(
                plan,
                recover,
                Path(runtime["operation"])
                / "prepared-recovery"
                / uuid.uuid4().hex
                / "catalog.json",
                run_dir=run_dir,
                cursor_payload=cursor,
                release=release,
                authorization=authorization,
                merge_seals=merge_seals,
                validate_registration=validate_registration,
                max_bytes=config["prepared_data"]["max_window_bytes"],
                api=api,
            )
            state.setdefault("prepared_preparation_recoveries", []).append(recovered)
            api.atomic_json(Path(run_dir) / "state.json", state)
    histories = []
    for root in sorted(roots):
        original_cache = api.ShardCache(
            plan, root, replay_admission=spec["data"].get("replay_admission")
        )
        if original_cache.state["owner"] is not None:
            original_cache.release(original_cache.state["owner"]["owner"], released_proof=release())
        histories.append(api.pin(original_cache.state_path))
        original_cache.close()
    if active is None:
        # Re-admit all ready canonical/publication bytes before deciding whether
        # more DATA is necessary. The original pending-batch check below owns READY.
        ready_catalog = prepared.build_catalog(
            plan, histories, Path(runtime["operation"]) / "prepared-recovery"
            / uuid.uuid4().hex / "ready-catalog.json",
            max_bytes=config["prepared_data"]["max_window_bytes"],
            owned_pair_roots=owned_roots, api=api,
        )
        active = {"catalog": ready_catalog, "phase_costs": [], "trainer_seconds_charged": 0,
                  "registered_preparation_reused": bool(registrations),
                  "existing_publications_reused": True}
        api.atomic_json(active_path, active)
    required_shards = []
    if active is not None:
        if completed_retirement is not None:
            # Full original data admission rechecks every surviving old/new
            # publication; no old payload/source pin is silently reset.
            canonical_archive = next(row for row in completed_retirement["source_state_archives"]
                                     if Path(row["original"]["path"]) == cache.state_path)
            expected = json.loads(api.checked(canonical_archive["snapshot"]).read_text())
            for locator in completed_retirement["owned_cache_states"] + recovered["cache_states"]:
                for shard, seal in json.loads(api.checked(locator).read_text())["seals"].items():
                    old = expected["seals"].get(shard)
                    api._require(old is None or prepared._same_original_seal(old, seal), "retired/new immutable canonical seal differs")
                    expected["seals"][shard] = copy.deepcopy(seal)
            api._require(cache.state == expected, "canonical history differs from original retirement and registered publications")
            carried = sorted(set(completed_retirement["owned_pair_roots"]) | set(recovered["owned_pair_roots"]))
            fresh = prepared.build_catalog(plan, histories,
                Path(runtime["operation"]) / "prepared-recovery" / uuid.uuid4().hex / "carried-catalog.json",
                max_bytes=config["prepared_data"]["max_window_bytes"], owned_pair_roots=carried, api=api)
            active = dict(active, catalog=fresh)
            api.atomic_json(active_path, active)
        recovered = prepared.reconcile_retirement(
            plan,
            cursor,
            active["catalog"],
            release=release,
            authorization=authorization,
            merge_seals=merge_seals,
            replay_admission=spec["data"].get("replay_admission"),
            reservation=latest["cursor"].get("batch_reservation"),
            api=api,
        )
        if recovered["changed"]:
            active["catalog"] = recovered["catalog"]
            api.atomic_json(active_path, active)
            state.setdefault("prepared_window_recoveries", []).append(recovered)
            api.atomic_json(Path(run_dir) / "state.json", state)
        catalog = json.loads(api.checked(active["catalog"]).read_text())
        provider = prepared.provider_from_catalog(plan, active["catalog"], api=api)
        try:
            restored = provider.restore_cursor(cursor)
            reservation, _ = ready_reservation(
                provider, restored, latest["cursor"].get("batch_reservation"), api
            )
            if set(reservation["shard_ids"]) <= set(catalog["publications"]):
                return
        except api.ShardRequired as required:
            required_shards = list(required.shard_ids)
        finally:
            provider.close()
        retired = prepared.retire_consumed_window(
            plan,
            cursor,
            active["catalog"],
            release=release,
            authorization=authorization,
            merge_seals=merge_seals,
            replay_admission=spec["data"].get("replay_admission"),
            reservation=latest["cursor"].get("batch_reservation"),
            api=api,
        )
        owned_roots = retired["owned_pair_roots"]
        histories = retired["canonical_history"] + retired["owned_cache_states"]
        state.setdefault("prepared_window_retirements", []).append(retired)
    limits = config["prepared_data"]
    maximum = limits["max_window_bytes"]
    # max_window_bytes remains the hard storage admission, never a compulsory
    # prefill target. Reuse ready data above the smaller target unchanged.
    target = min(maximum, limits.get("initial_window_bytes", 24 * 1024**3))
    window_root = Path(runtime["operation"]) / "prepared-data" / uuid.uuid4().hex
    result = prepared.prepare_window(
        plan,
        cursor,
        spec["shard_lifecycle"],
        histories,
        window_root,
        run_dir=run_dir,
        attempt=attempt,
        runner=runner,
        release=release,
        authorization=authorization,
        source=lane["source"],
        max_window_bytes=maximum,
        target_window_bytes=target,
        required_shards=required_shards,
        batch_reservation=latest["cursor"].get("batch_reservation"),
        reserve_free_bytes=limits["reserve_free_bytes"],
        merge_seals=merge_seals,
        register_cache=register_cache,
        owned_pair_roots=owned_roots,
        api=api,
    )
    catalog = json.loads(api.checked(result["catalog"]).read_text())
    checker = prepared.provider_from_catalog(plan, result["catalog"], api=api)
    try:
        reserved, _ = ready_reservation(
            checker, checker.restore_cursor(cursor), latest["cursor"].get("batch_reservation"), api
        )
        api._require(
            set(reserved["shard_ids"]) <= set(catalog["publications"]),
            "prepared window cannot supply the actual pending optimizer reservation",
        )
    finally:
        checker.close()
    new_roots = [str(Path(value["path"]).parent) for value in catalog["source_cache_states"]]
    state["prepared_cache_roots"] = sorted(
        set(roots) | set(new_roots) | set(catalog["owned_pair_roots"])
    )
    state.setdefault("prepared_windows", []).append(result)
    api.atomic_json(active_path, result)
    api.atomic_json(Path(run_dir) / "state.json", state)


def ready_reservation(provider, cursor, pending, api):
    """Ask the original provider for its exact reservation before array loading."""
    from types import SimpleNamespace

    # The frozen provider has no reserve_metadata method. Its genuine missing-
    # resident boundary publishes the complete reservation before loading arrays.
    # This local view changes no cache file, seal, cursor or student provider.
    view = SimpleNamespace(state=dict(provider.cache.state, resident={}))
    original = api.RotatingBlockProvider(provider.plan, view)
    try:
        original.reserve_batch(cursor, 2, reservation=pending)
    except api.ShardRequired as required:
        if required.reservation.get("kind") != "batch" or not provider.cache.activate(
            required.shard_ids
        ):
            raise
        return required.reservation, api.ShardCursor(**required.reservation["next_cursor"])
    raise RuntimeError("original missing-resident reservation boundary changed")


def evaluate_controller_manual(
    runtime,
    config,
    lane_locator,
    state,
    state_path,
    supervisor_path,
    training,
    plan,
    run_dir,
    files,
    resources,
    runner,
    release,
    authorization,
):
    """A manual evaluation is distinct from the original mandatory milestones."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("direct_evaluation_entry", ROOT / "eval.py")
    frontend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frontend)
    request, _ = metadata(Path(run_dir) / "evaluation-request.json")
    output = Path(run_dir) / "evaluations" / request["request_id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "attempts").mkdir(exist_ok=True)
    failure = output / "failure.json"
    if failure.exists():
        failure.rename(output / ("previous-failure-" + uuid.uuid4().hex + ".json"))
    for retry in range(2):
        attempt = output / "attempts" / uuid.uuid4().hex
        try:
            result = frontend.evaluate_manual_released_boundary(
                config,
                lane_locator=lane_locator,
                state_locator={
                    "path": str(state_path),
                    "sha256": hashlib.sha256(Path(state_path).read_bytes()).hexdigest(),
                },
                supervisor_locator={
                    "path": str(supervisor_path),
                    "sha256": hashlib.sha256(Path(supervisor_path).read_bytes()).hexdigest(),
                },
                training_locator=training,
                plan_locator=plan,
                directory=attempt,
                files=files,
                observer=resources,
                runner=runner,
                release=release,
                authorization=authorization,
                stop_path=Path(run_dir) / "STOP",
            )
            from w1a1_eagle.direct_checkpoint_retention import archive_completed_history

            # This calls the complete original manual scientific validator while
            # its immutable released context and full payloads still exist.
            archive = archive_completed_history(
                result,
                output / ("validated-history-" + uuid.uuid4().hex + ".json"),
                manual_config=config,
            )
            from w1a1_eagle.nine_model_pipeline import atomic_json

            state.setdefault("manual_evaluations", []).append(result)
            state.setdefault("manual_evaluation_history", []).append(archive)
            state.update(pending_training=None, status="training", pending_shard_resume=False)
            atomic_json(state_path, state)
            timed, _ = metadata(Path(run_dir) / "training/timed-evaluation-state.json")
            refresh_best(config, timed["completed"], Path(run_dir) / "training")
            request_path = Path(run_dir) / "evaluation-request.json"
            if metadata(request_path)[0] != request:
                raise ValueError("evaluation request changed during its owned native operation")
            request_path.rename(output / "submitted-request.json")
            publish_result(output / "receipt.json", checked(result))
            return
        except BaseException as error:
            error_record = {
                "type": type(error).__name__,
                "reason": str(error),
                "request_id": request["request_id"],
                "attempt": str(attempt),
            }
            write_new(output / ("failure-attempt-" + uuid.uuid4().hex + ".json"), error_record)
            if retry == 0 and isinstance(error, (OSError, ConnectionError, TimeoutError)):
                authorization()
                release()
                time.sleep(1)
                continue
            publish_result(failure, error_record)
            raise


def scheduled_complete(runtime, config, evaluated, state, state_path, run_dir):
    """Join an outstanding user request to an already validated latest milestone."""
    from w1a1_eagle.direct_export_retention import refresh_scheduled_best

    # The final milestone has no later trainer startup to refresh best.json.
    # Original evaluation validation/release and journal publication precede
    # this hook; its ranking still revalidates the original scientific result.
    refresh_scheduled_best(config, evaluated)
    request_path = Path(run_dir) / "evaluation-request.json"
    if not request_path.exists():
        return
    request, _ = metadata(request_path)
    if not manual_pending(Path(run_dir) / "training"):
        return
    result = checked(evaluated)
    latest, _ = metadata(Path(run_dir) / "training/checkpoints/latest.json")
    if result["checkpoint"] != {key: latest[key] for key in ("path", "sha256")}:
        raise ValueError("scheduled result cannot satisfy a different latest checkpoint request")
    output = Path(run_dir) / "evaluations" / request["request_id"]
    output.mkdir(parents=True, exist_ok=True)
    write_new(
        output / "joined-scheduled-evaluation.json",
        {
            "schema": "direct_joined_scheduled_evaluation_v1",
            "request_id": request["request_id"],
            "original_evaluation": evaluated,
            "checkpoint": result["checkpoint"],
        },
    )
    request_path.rename(output / "submitted-request.json")
    publish_result(
        output / "receipt.json",
        {
            "schema": "direct_joined_native_evaluation_receipt_v1",
            "request_id": request["request_id"],
            "completed": True,
            "checkpoint": result["checkpoint"],
            "original_evaluation": evaluated,
        },
    )


def publish_early_evaluation_failure(config, runtime, error):
    if config.get("evaluation_only") is not True:
        return
    path = Path(config["run_dir"]) / "evaluation-request.json"
    if not path.exists() or not manual_pending(Path(config["run_dir"]) / "training"):
        return
    request, _ = metadata(path)
    output = Path(config["run_dir"]) / "evaluations" / request["request_id"]
    output.mkdir(parents=True, exist_ok=True)
    failure = output / "failure.json"
    if not failure.exists():
        publish_result(
            failure,
            {
                "schema": "direct_native_evaluation_failure_v1",
                "request_id": request["request_id"],
                "type": type(error).__name__,
                "reason": str(error),
                "operation": runtime["operation"],
                "source": runtime["runtime_sources"],
            },
        )


def controller(runtime, config):
    """Own one native lane; use the existing lifecycle and its scientific gates."""
    import copy

    root, lane, api, lifecycle, original, helper, manifest, selection = frozen_runtime(config)
    from w1a1_eagle import continuous_budget as budget

    # Original recover_boundary_receipt imports this class at call time. Only
    # completed, hash-bound archived entries may omit historical model payloads.
    install_compact_history_reader(budget)
    directory = Path(config["run_dir"])
    operation = Path(runtime["operation"])
    attempt = directory / "attempts" / operation.name
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = directory / "state.json"
    state, state_sha = metadata(state_path)
    prepared_predecessor = prior_active_window(config, state)
    write_new(operation / "predecessor-state.json", state)
    state = copy.deepcopy(state)
    state.pop("failure", None)
    state.pop("cleanup_failure", None)
    if prepared_predecessor is not None:
        state["prepared_window_predecessor"] = prepared_predecessor
    state["direct_training_runtime"] = {
        "path": str(operation / "runtime.json"),
        "sha256": hashlib.sha256((operation / "runtime.json").read_bytes()).hexdigest(),
    }
    state["resume_history"] = [
        *state.get("resume_history", []),
        {"path": str(operation / "predecessor-state.json"), "sha256": state_sha},
    ]
    admission = checked(config["original_admission"])
    if (
        admission.get("schema") != "nine_model_training_admission_v1"
        or admission.get("status") != "PASS"
        or admission.get("candidate") != lane["candidate"]
    ):
        raise ValueError("genuine original training admission required")
    _, authorization, cleanup_custody = capture_direct_authorization(
        original,
        api,
        lane,
        Path(runtime["availability"]),
        config["lane"]["sha256"],
        directory,
        Path(runtime["supervisor_state"]),
    )
    write_new(
        operation / "owned-identity.json",
        {
            "controller": api.process_identity(os.getpid()),
            "supervisor": api.process_identity(os.getppid()),
        },
    )
    files = api.Files()
    observer = api.LinuxResources(lane["gpu_uuid"])
    runner = api.SubprocessRunner(root, lane.get("environment"), authorization=authorization)
    baseline = None
    controller_cache = None

    def persist(event):
        state.setdefault("resource_handoff_observations", []).append(event)
        api.atomic_json(
            directory / "resource-handoff-progress.json",
            {
                "schema": "nine_model_resource_handoff_progress_v1",
                "controller_pid": os.getpid(),
                "bundle_sha256": config["lane"]["sha256"],
                "status": state.get("status"),
                "observations": state["resource_handoff_observations"],
            },
        )

    def observe_release(wait, resource_authorization):
        proof, _ = api.wait_for_resource_return(
            observer,
            runner.process_groups,
            runner.process_identities,
            baseline,
            lane["resource_policy"],
            authorization=resource_authorization,
            persist=persist,
            wait=wait,
        )
        return proof

    def release(*, wait=True):
        return observe_release(wait, authorization)

    def cleanup_release():
        proof = observe_release(False, cleanup_custody)
        release_controller_student_cache(controller_cache, proof,
            canonical_root=checked(lane['config'])['data']['cache_root'])
        return proof

    def authorize():
        authorization()
        current_control(lane)

    original_run = runner.run

    def run(argv, **kwargs):
        expected = str(root / "scripts/train_nine_model_qat.py")
        if len(argv) > 1 and argv[1] == expected:
            argv = [
                config["python"],
                "-B",
                str(ROOT / "train.py"),
                "--config",
                runtime["config"]["path"],
                "--_trainer",
                str(operation / "runtime.json"),
                "--",
                *argv,
            ]
        return original_run(argv, **kwargs)

    runner.run = run
    namespace = dict(original.__dict__)
    def prepare_owned_window(*args):
        nonlocal controller_cache
        controller_cache = args[-1]
        return prepare_controller_window(runtime, config, lifecycle, *args)

    namespace["direct_prepare_window"] = prepare_owned_window
    def exclusive_exports():
        authorize()
        owned, _ = metadata(operation / "owned-identity.json")
        supervisor = supervisor_publication(runtime["supervisor_state"])
        lock_stat = owner.fileno()
        held = os.fstat(lock_stat)
        current = lock.lstat()
        return (
            (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
            and api.process_identity(os.getpid()) == owned["controller"]
            and api.process_identity(os.getppid()) == owned["supervisor"]
            and supervisor.get("pid") == os.getpid()
            and supervisor.get("supervisor_pid") == os.getppid()
            and supervisor.get("status") == "running"
            and supervisor.get("received_signal") is None
        )

    def retire_exports():
        from w1a1_eagle.direct_export_retention import cleanup_released_exports

        return cleanup_released_exports(
            config,
            operation=operation,
            budget_path=directory / "training/budget-used.json",
            authorization=authorize,
            release=release,
            exclusive=exclusive_exports,
        )

    def complete_manual(*args):
        evaluate_controller_manual(runtime, config, *args)
        retire_exports()

    def complete_scheduled(*args):
        scheduled_complete(runtime, config, *args)
        retire_exports()

    namespace["direct_evaluate_manual"] = complete_manual
    namespace["direct_evaluation_only"] = lambda: config.get("evaluation_only") is True
    namespace["direct_scheduled_complete"] = complete_scheduled

    def restore_window(_config, _request, **kwargs):
        nonlocal controller_cache
        controller_cache = kwargs['cache']
        prepare_controller_window(
            runtime,
            config,
            lifecycle,
            lane,
            state,
            kwargs["run_dir"],
            kwargs["attempt"],
            kwargs["runner"],
            kwargs["release"],
            kwargs["authorization"],
            kwargs["cache"],
        )
        return metadata(operation / "active-window.json")[0]["phase_costs"]

    namespace["direct_restore_window"] = restore_window
    exec(
        compile(
            derive_lifecycle((root / "scripts/run_nine_model_lane.py").read_text()),
            str(root / "scripts/run_nine_model_lane.py"),
            "exec",
        ),
        namespace,
    )
    lock = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    with lock.open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with api.stop_signals():
                authorize()
                # Reuse the original recovery inventory, including CPU-only
                # producers left behind by an interrupted prior supervisor.
                groups, identities = recover_owned_process_inventory(directory)
                runner.process_groups.extend(groups)
                runner.process_identities.extend(identities)
                observer.require_released(runner.process_groups, runner.process_identities)
                baseline = observer.snapshot()
                proof = observer.require_released([], [])
                api.require(
                    proof.get("owned_process_groups_absent") is True
                    and proof.get("owned_cuda_pids_absent") is True
                    and proof.get("other_context_pids") == []
                    and baseline.get("dxg_holders") == [],
                    "exclusive released GPU required",
                )
                api.resource_gate(baseline, baseline, lane["resource_policy"])
                state.update(
                    status="training", admission=config["original_admission"], attempt=str(attempt)
                )
                api.atomic_json(state_path, state)
                namespace["run_training_lifecycle"](
                    lane,
                    config["lane"],
                    directory,
                    attempt,
                    Path(config["original_admission"]["path"]),
                    Path(runtime["supervisor_state"]),
                    state,
                    state_path,
                    files,
                    runner,
                    observer,
                    release,
                    authorize,
                    resume=True,
                )
                if config.get("evaluation_only") is True:
                    settled, _ = metadata(directory / "training/budget-used.json")
                    state.update(
                        status="training_complete"
                        if settled["training_seconds"] >= settled["max_seconds"]
                        else "stopped",
                        evaluation_only_complete=True,
                    )
                    release()
                    api.atomic_json(state_path, state)
        except BaseException as error:
            with api.deferred_termination():
                record_controller_failure(state, error, cleanup_release if baseline is not None else None)
                api.atomic_json(state_path, state)
            raise
    return 0


def install_compact_history_reader(budget):
    """Select the completed-history reader before original controller recovery.

    This reader never creates an archive. Archive publication remains behind the
    child's original full request/result validation. Pending requests and the
    actual latest checkpoint retain the original full consumers.
    """
    import types

    from w1a1_eagle.direct_checkpoint_retention import compact_timed_evaluation_class

    expected_sha = "9cbc775931b1f2739c8a9b7cad9e98f746449000cce06dd8b9b6792b74eb655e"
    source = Path(budget.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ValueError("compact reader requires original scientific budget source")
    selected = budget.TimedEvaluation
    if getattr(selected, "_direct_compact_history_reader", None) == expected_sha:
        return selected
    compiled = compile(source, str(budget.__file__), "exec", dont_inherit=True)
    original_class = next(
        code
        for code in compiled.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "TimedEvaluation"
    )
    for name in ("__init__", "_request", "_result", "authorize_resume", "require_checkpoint"):
        expected = next(
            code
            for code in original_class.co_consts
            if isinstance(code, types.CodeType) and code.co_name == name
        )
        method = getattr(selected, name)
        if (
            not isinstance(method, types.FunctionType)
            or method.__globals__ is not budget.__dict__
            or method.__code__ != expected
        ):
            raise ValueError("original timed reader callable differs: " + name)
    reader = compact_timed_evaluation_class(selected)
    reader._direct_compact_history_reader = expected_sha
    budget.TimedEvaluation = reader
    return reader


def install_checkpoint_policy(budget, config):
    """Compact validated completed entries; current/pending states stay complete."""
    base = install_compact_history_reader(budget)

    class DirectTimedEvaluation(base):
        def authorize_resume(self, receipt=None):
            result = super().authorize_resume(receipt)
            history_root = self.path.parent / "completed-history"
            history_root.mkdir(parents=True, exist_ok=True)
            for index, entry in enumerate(self.state["completed"]):
                if not entry.get("compact"):
                    output = history_root / (
                        "milestone-" + str(index) + "-" + uuid.uuid4().hex + ".json"
                    )
                    self.compact_completed(index, output)
            refresh_best(config, self.state["completed"], self.path.parent)
            return result

    budget.TimedEvaluation = DirectTimedEvaluation


def refresh_best(config, timed_entries, training_directory):
    """Only genuinely measured checkpoints can become best; quality remains visible."""
    from w1a1_eagle.direct_checkpoint_retention import best_evaluated_checkpoint
    from w1a1_eagle.nine_model_pipeline import atomic_json

    path = Path(training_directory) / "best.json"
    prior = metadata(path)[0] if path.exists() else None
    receipts = [entry["receipt"] for entry in timed_entries]
    journal, _ = metadata(Path(config["run_dir"]) / "state.json")
    receipts.extend(journal.get("manual_evaluations", []))
    identity = hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest()
    if prior is not None and prior.get("evaluations_sha256") == identity:
        return prior
    histories = [entry["compact"] for entry in timed_entries if entry.get("compact")]
    histories.extend(journal.get("manual_evaluation_history", []))
    evaluations = []
    for locator in receipts:
        receipt = checked(locator)
        evaluations.append((locator, checked(receipt["evaluation"])))
    selected = best_evaluated_checkpoint(
        evaluations, candidate="dspark_a8", manual_config=config, completed_history=histories
    )
    result = {
        "schema": "direct_best_checkpoint_v1",
        "evaluations_sha256": identity,
        "best": selected,
    }
    atomic_json(path, result)
    return result


def install_save_retention(namespace, config, runtime):
    """Reuse each original publication's real hash proofs for its immediate prune."""
    import types

    from w1a1_eagle.checkpoint_digest_proofs import CheckpointDigestProofs
    from w1a1_eagle.direct_checkpoint_retention import apply, inventory

    original_save = namespace["save_block_checkpoint"]
    original_policy = namespace["BlockCheckpointRetention"]

    def policy(**values):
        return original_policy(**dict(values, keep_recent=1))

    namespace["BlockCheckpointRetention"] = policy

    def save(*args, **kwargs):
        checkpoint_directory = Path(args[4] if len(args) > 4 else kwargs["directory"])
        training_directory = checkpoint_directory.parent
        timed = metadata(training_directory / "timed-evaluation-state.json")[0]
        protected = []
        if timed.get("pending"):
            protected.append(checked(timed["pending"])["checkpoint"])
        for entry in timed["completed"]:
            if not entry.get("compact"):
                protected.append(checked(entry["request"])["checkpoint"])
        best_path = training_directory / "best.json"
        selected = metadata(best_path)[0].get("best") if best_path.exists() else None
        keep = {str(Path(row["path"]).resolve()) for row in protected}
        if selected:
            keep.add(str(Path(selected["checkpoint"]["path"]).resolve()))
        histories = [entry["compact"] for entry in timed["completed"] if entry.get("compact")]
        journal, _ = metadata(Path(config["run_dir"]) / "state.json")
        histories.extend(journal.get("manual_evaluation_history", []))
        # Completed history also needs its small provenance sidecar. Let only
        # the second, archive-aware pruner retire these payloads; the original
        # rolling pruner would remove their sidecars along with the weights.
        keep.update(str(Path(checked(row)["checkpoint"]["path"]).resolve()) for row in histories)
        proofs = CheckpointDigestProofs()
        globals_copy = dict(original_save.__globals__, CheckpointDigestProofs=lambda: proofs)
        original_plan = original_save.__globals__["_retention_plan"]

        def rolling_plan(*values, **options):
            # Original rolling deletion runs before our completed-history prune.
            # A manual best can be an ordinary shard checkpoint with no original
            # milestone protection, so it must also survive that first consumer.
            return [
                (path, receipt)
                for path, receipt in original_plan(*values, **options)
                if str(path.resolve()) not in keep
            ]

        globals_copy["_retention_plan"] = rolling_plan
        original = types.FunctionType(
            original_save.__code__,
            globals_copy,
            original_save.__name__,
            original_save.__defaults__,
            original_save.__closure__,
        )
        original.__kwdefaults__ = original_save.__kwdefaults__
        receipt = original(*args, **kwargs)
        checkpoint_directory = Path(receipt["path"]).parent
        training_directory = checkpoint_directory.parent
        state_file = training_directory / "timed-evaluation-state.json"
        timed = metadata(state_file)[0]
        histories = [entry["compact"] for entry in timed["completed"] if entry.get("compact")]
        journal, _ = metadata(Path(config["run_dir"]) / "state.json")
        histories.extend(journal.get("manual_evaluation_history", []))
        plan = inventory(
            checkpoint_directory,
            best=selected["checkpoint"] if selected else None,
            protected=protected,
            completed_history=histories,
            digest_proofs=proofs,
        )

        def exclusive():
            from w1a1_eagle.nine_model_pipeline import process_identity

            owned, _ = metadata(Path(runtime["operation"]) / "owned-identity.json")
            supervisor, _ = metadata(runtime["supervisor_state"])
            if (
                os.getppid() != owned["controller"]["pid"]
                or process_identity(os.getppid()) != owned["controller"]
                or process_identity(owned["supervisor"]["pid"]) != owned["supervisor"]
                or supervisor.get("pid") != owned["controller"]["pid"]
                or supervisor.get("supervisor_pid") != owned["supervisor"]["pid"]
                or supervisor.get("status") != "running"
                or supervisor.get("received_signal") is not None
            ):
                return False
            current_control(checked(config["lane"]))
            lock_path = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
            with lock_path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
                fcntl.flock(lock, fcntl.LOCK_UN)
                return False

        removed = apply(plan, exclusive=exclusive)
        # This callback executes only in the one original admitted trainer under
        # its controller's campaign lock, after a fully verified publication.
        evidence = checkpoint_directory / "retention"
        evidence.mkdir(exist_ok=True)
        namespace["atomic_json"](
            evidence / (str(time.time_ns()) + ".json"),
            {
                "schema": "direct_checkpoint_prune_receipt_v1",
                "latest": {k: receipt[k] for k in ("path", "sha256")},
                "best": selected,
                "removed": removed,
                "kept": [
                    {"path": e["path"], "bytes": e["bytes"], "reason": e["reason"]}
                    for e in plan["entries"]
                    if e["action"] == "keep"
                ],
            },
        )
        return receipt

    namespace["save_block_checkpoint"] = save


def trainer(runtime, config, argv):
    """Load the original model once per prepared window and retain its step loop."""
    import faulthandler
    import runpy
    import types

    faulthandler.enable()
    root, lane, api, lifecycle, original, helper, manifest, selection = frozen_runtime(config)
    argv = argv[1:] if argv[:1] == ["--"] else argv
    if (
        len(argv) < 2
        or argv[0] != config["python"]
        or Path(argv[1]).resolve() != root / "scripts/train_nine_model_qat.py"
    ):
        raise ValueError("only the original admitted trainer command may be bootstrapped")
    helper.validate_op04_trainer_argv(argv, lane, selection)
    window = metadata(Path(runtime["operation"]) / "active-window.json")[0]
    from w1a1_eagle.sustained_block_handoff import install_child_provider

    spec = checked(lane["config"])
    install_child_provider(lifecycle, spec["data"], window["catalog"])
    from w1a1_eagle import continuous_budget as budget

    install_checkpoint_policy(budget, config)
    namespace = runpy.run_path(
        str(root / "scripts/train_nine_model_qat.py"), run_name="direct_original_trainer"
    )
    install_save_retention(namespace, config, runtime)
    namespace["direct_manual_pending"] = manual_pending
    namespace["direct_manual_startup"] = manual_startup
    namespace["direct_manual_endpoint"] = lambda *args: manual_endpoint(namespace, *args)
    exec(
        compile(
            derive_trainer((root / "scripts/train_nine_model_qat.py").read_text()),
            str(root / "scripts/train_nine_model_qat.py"),
            "exec",
        ),
        namespace,
    )
    sys.argv = argv[1:]
    # run_path returns a COPY of the execution dictionary. Its original main
    # otherwise dispatches to the original run_block and misses installed hooks.
    original_main = namespace["main"]
    main = types.FunctionType(original_main.__code__, namespace, original_main.__name__,
                              original_main.__defaults__, original_main.__closure__)
    main.__kwdefaults__ = original_main.__kwdefaults__
    namespace["main"] = main
    main()
    return 0


def derive_trainer(source):
    """Change only safe manual-stop scheduling, not optimizer/data mathematics."""
    import ast

    tree = ast.parse(source)
    block = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_block")
    loops = [
        n
        for n in ast.walk(block)
        if isinstance(n, ast.While) and ast.unparse(n.test) == "not capped() and (not stopped)"
    ]
    if not loops:
        loops = [
            n
            for n in ast.walk(block)
            if isinstance(n, ast.While) and ast.unparse(n.test) == "not capped() and not stopped"
        ]
    if len(loops) != 1:
        raise ValueError("original safe completed-step loop differs")
    loops[0].test = ast.BoolOp(
        ast.And(),
        [loops[0].test, ast.parse("not direct_manual_pending(args.run_dir)", mode="eval").body],
    )
    publish = [
        n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "publish_endpoint"
    ]
    if len(publish) != 1:
        raise ValueError("original endpoint publication differs")
    pending_branch = next(
        n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "shard_boundary"
    )
    block.body.insert(
        block.body.index(pending_branch),
        ast.parse(
            "if direct_manual_startup(args, cursor, boundary, dataset):\n"
            "    return publish_endpoint(receipt, boundary is not None and boundary.due(cursor.elapsed_seconds))"
        ).body[0],
    )
    publish[0].body.insert(
        0,
        ast.parse(
            "if not milestone_due and direct_manual_pending(args.run_dir):\n"
            "    return direct_manual_endpoint(model, source, latest, cursor, args, spec, budget, hardware, dataset)"
        ).body[0],
    )
    ast.fix_missing_locations(block)
    timing = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "timed_evaluation"
    )
    gates = [
        n
        for n in timing.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "len(boundary.state['completed']) == len(boundary.milestones)"
    ]
    if len(gates) != 1:
        raise ValueError("original completed evaluation guard differs")
    gates[0].test = ast.BoolOp(
        ast.And(),
        [gates[0].test, ast.parse("not direct_manual_pending(args.run_dir)", mode="eval").body],
    )
    ast.fix_missing_locations(timing)
    return ast.Module(body=[block, timing], type_ignores=[])


def derive_lifecycle(source):
    """Retain original receipt checks and mandatory evaluations around two hooks."""
    import ast

    tree = ast.parse(source)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_training_lifecycle"
    )
    loop = next(n for n in function.body if isinstance(n, ast.While))
    changed = 0
    restored = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Set) and {
            n.value for n in node.elts if isinstance(n, ast.Constant)
        } == {"approved_budget_complete", "timed_evaluation_boundary"}:
            node.elts.append(ast.Constant("evaluation_requested"))
            changed += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "restore_requested_shards"
        ):
            node.func.id = "direct_restore_window"
            restored += 1
    # Locate the original pre-trainer cache ownership transaction by its calls.
    launch_else = next(
        n
        for n in loop.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "pending"
    ).orelse
    before = [
        n
        for n in launch_else
        if isinstance(n, ast.If) and ast.unparse(n.test) == "cache is not None"
    ]
    if len(before) != 1 or changed != 1 or restored != 1:
        raise ValueError("original lifecycle receipt/ownership structure differs")
    launch_else.insert(
        launch_else.index(before[0]),
        ast.parse(
            "direct_prepare_window(lane, state, run_dir, attempt, runner, release, authorization, cache)"
        ).body[0],
    )
    branches = [
        n
        for n in loop.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "is_shard_boundary"
    ]
    if len(branches) != 1:
        raise ValueError("original released boundary branch differs")
    index = loop.body.index(branches[0])
    manual = ast.parse(
        "if endpoint.get('completion_reason') == 'evaluation_requested':\n"
        "    direct_evaluate_manual(lane_locator, state, state_path, supervisor_path, training, "
        "timed, run_dir, files, resources, runner, release, authorization)\n"
        "    if direct_evaluation_only():\n"
        "        return\n"
        "    resume = True\n"
        "    continue"
    ).body[0]
    loop.body.insert(index, manual)
    finals = [
        n
        for n in loop.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "final"
    ]
    if len(finals) != 1:
        raise ValueError("original completed native evaluation branch differs")
    loop.body.insert(
        loop.body.index(finals[0]),
        ast.parse(
            "direct_scheduled_complete(evaluated, state, state_path, run_dir)\n"
            "if direct_evaluation_only():\n"
            "    return"
        ).body[0],
    )
    # Separate the explicit evaluation-only return from the original final gate.
    loop.body.insert(
        loop.body.index(finals[0]), ast.parse("if direct_evaluation_only():\n    return").body[0]
    )
    ast.fix_missing_locations(function)
    return ast.Module(body=[function], type_ignores=[])


def manual_pending(training_directory):
    path = Path(training_directory).parent / "evaluation-request.json"
    if not path.exists():
        return False
    value, _ = metadata(path)
    if (
        value.get("schema") != "direct_native_evaluation_request_v1"
        or value.get("selection") != "latest"
        or len(value.get("request_id", "")) != 32
        or any(c not in "0123456789abcdef" for c in value["request_id"])
    ):
        raise ValueError("manual evaluation request differs")
    return True


def manual_startup(args, cursor, boundary, dataset):
    if not manual_pending(args.run_dir):
        return False
    if not args.resume or cursor.step <= 0:
        raise ValueError("manual native evaluation requires genuine saved training")
    dataset.restore_cursor(cursor.data_cursor)
    return True


def manual_endpoint(original, model, source, latest, cursor, args, spec, budget, hardware, dataset):
    """Export the genuine saved completed step with an explicitly manual request."""
    from dataclasses import asdict

    dataset.restore_cursor(cursor.data_cursor)
    requested, _ = metadata(Path(args.run_dir).parent / "evaluation-request.json")
    request_id = requested["request_id"]
    output = args.run_dir / "manual-evaluation" / request_id
    output.mkdir(parents=True, exist_ok=False)
    exported = original["export_block_checkpoint"](model, source, output / "export")
    locator = original["artifact_locator"]
    counters = json.loads(json.dumps(asdict(cursor)))
    checkpoint = {key: latest[key] for key in ("path", "sha256")}
    boundary = {
        "schema": "direct_native_evaluation_boundary_v1",
        "request_id": request_id,
        "candidate": spec["candidate"],
        "bundle_sha256": args.bundle_sha256,
        "config_sha256": original["sha256"](args.config),
        "checkpoint": checkpoint,
        "counters": counters,
        "elapsed_seconds": cursor.elapsed_seconds,
        "protocol": spec["evaluation_protocol"],
        "budget_ledger": locator(budget.path),
    }
    boundary_path = output / "request.json"
    original["atomic_json"](boundary_path, boundary)
    return {
        "schema": "nine_model_stage_receipt_v1",
        "stage": args.stage_name,
        "status": "PASS",
        "artifact_kind": "production",
        "bundle_sha256": args.bundle_sha256,
        "config_sha256": boundary["config_sha256"],
        "committed": True,
        "run_purpose": spec.get("run_purpose", "finalist"),
        "allocation_seconds": budget.maximum,
        "completion_reason": "evaluation_requested",
        "checkpoint": checkpoint,
        "manual_evaluation_request": locator(boundary_path),
        "counters": counters,
        "exports": {
            spec["candidate"]: {
                "checkpoint": locator(Path(exported["npz"])),
                "manifest": locator(Path(exported["manifest"])),
                "base_gguf_sha256": source["base_gguf_sha256"],
            }
        },
        "hardware": hardware,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--status", action="store_true", help="show saved progress without GPU queries"
    )
    parser.add_argument(
        "--check", action="store_true", help="check source bindings without GPU/model access"
    )
    parser.add_argument(
        "--detach", action="store_true", help="start supervised training and return"
    )
    parser.add_argument("--_controller", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_trainer", type=Path, help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args(argv)
    if args._controller:
        runtime, config = authenticate_runtime(args._controller)
        try:
            return controller(runtime, config)
        except BaseException as error:
            publish_early_evaluation_failure(config, runtime, error)
            raise
    if args._trainer:
        runtime, config = authenticate_runtime(args._trainer)
        return trainer(runtime, config, remaining)
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    config = metadata(args.config)[0]
    if args.status:
        print(json.dumps(status_snapshot(config, run_dir=args.run_dir), indent=2, sort_keys=True))
        return 0
    if args.check:
        root, *_ = frozen_runtime(config)
        from w1a1_eagle import continuous_budget as budget

        install_compact_history_reader(budget)
        compile(
            derive_trainer((root / "scripts/train_nine_model_qat.py").read_text()),
            "direct-trainer-preflight",
            "exec",
        )
        compile(
            derive_lifecycle((root / "scripts/run_nine_model_lane.py").read_text()),
            "direct-controller-preflight",
            "exec",
        )
        print(
            json.dumps(
                {
                    "source_bindings": "PASS",
                    "controller_trainer_derivations": "PASS",
                    "gpu_queried": False,
                    "training_started": False,
                    "native_runtime_verified": False,
                }
            )
        )
        return 0
    return launch(config, run_dir=args.run_dir, detach=args.detach, config_path=args.config.resolve(strict=True))
