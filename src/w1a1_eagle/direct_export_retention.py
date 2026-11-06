"""Candidate: optional export cleanup at an already released controller boundary.

No launch/save hook, no verdict cache. Caller retains the original authorization,
canonical lock and all-owned resource census. Malformed evidence raises after an
error record; only explicitly unsettled/changed live references defer cleanup.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

from w1a1_eagle.checkpoint_digest_proofs import CheckpointDigestProofs, regular_file_identity


class DeferredCleanup(RuntimeError):
    pass


def _read(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("cleanup metadata cannot traverse symlinks")
    before = regular_file_identity(path)
    if before[2] > 16 * 1024 * 1024:
        raise ValueError("cleanup metadata exceeds 16 MiB")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if regular_file_identity(path) != before:
            raise ValueError("cleanup metadata changed before open")
        raw = stream.read(16 * 1024 * 1024 + 1)
        opened = os.fstat(stream.fileno())
        observed = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mode,
                    opened.st_uid, opened.st_gid, opened.st_nlink,
                    opened.st_mtime_ns, opened.st_ctime_ns)
        if observed != before or regular_file_identity(path) != before:
            raise ValueError("cleanup metadata changed during read")
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def cleanup_released_exports(config, *, operation, budget_path, authorization,
                             release, exclusive, retention=None):
    """Return a cleanup receipt or raise the original malformed-evidence error.

    Invoke only after timed resume authorization/compaction or manual completion
    persistence and refresh_best. Never call from a paid trainer/save operation.
    ``exclusive`` must verify the existing exact controller/supervisor identities,
    canonical lock and controls. ``release`` is the original all-owned census.
    A fresh proof pool lives only for this complete inventory/apply transaction.
    """
    if retention is None:
        from w1a1_eagle import direct_checkpoint_retention as retention
    run = Path(config["run_dir"]).resolve(strict=True)
    if Path(budget_path) != run / "training/budget-used.json":
        raise ValueError("exact current-run original TRAIN ledger required")
    operation = Path(operation)
    if operation != operation.resolve(strict=True) or not operation.is_relative_to(run / "direct-runtime"):
        raise ValueError("owned direct-runtime operation required")
    if any(p.is_symlink() for p in (operation, *operation.parents)):
        raise ValueError("operation cannot traverse symlinks")
    output = operation / ("export-retention-" + uuid.uuid4().hex)
    output.mkdir()
    plan = None
    paths = {}
    absent = set()
    named_paths = set()

    def save(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")

    def observe(path, *, optional=False):
        path = Path(path)
        if optional and not os.path.lexists(path):
            absent.add(path)
            return None
        value, locator = _read(path)
        paths[path] = locator
        return value

    def checked(locator):
        value = observe(locator["path"])
        if paths[Path(locator["path"])] != locator:
            raise ValueError("cleanup immutable metadata digest differs")
        return value

    def verify_owner():
        authorization()
        if exclusive() is not True:
            raise DeferredCleanup("exclusive released controller ownership unsettled")
        for path, locator in paths.items():
            if not path.exists() or _read(path)[1] != locator:
                if path in named_paths:
                    raise DeferredCleanup("named reference changed: " + str(path))
                raise ValueError("immutable cleanup evidence changed: " + str(path))
        if any(os.path.lexists(path) for path in absent):
            raise DeferredCleanup("previously absent named reference appeared")
        return True

    try:
        authorization()
        if exclusive() is not True:
            raise DeferredCleanup("exclusive released controller ownership unsettled")
        proof = release()
        if (proof.get("owned_process_groups_absent") is not True
                or proof.get("owned_cuda_pids_absent") is not True
                or proof.get("other_context_pids", [])):
            raise DeferredCleanup("all-owned GPU release incomplete")
        ledger = observe(budget_path)
        if (ledger.get("schema") != "continuous_training_budget_v1"
                or ledger.get("source_sha256") != config["lane"]["sha256"]
                or ledger.get("max_seconds") != 43200):
            raise ValueError("original training budget required")
        if ledger.get("active_attempt") is not None:
            raise DeferredCleanup("TRAIN budget active; cleanup deferred")
        latest = observe(run / "training/checkpoints/latest.json")
        if latest.get("schema") != "block_qat_checkpoint_v1" or latest.get("committed") is not True:
            raise ValueError("committed original latest checkpoint required")
        if latest["cursor"]["elapsed_seconds"] != ledger["training_seconds"]:
            raise DeferredCleanup("latest checkpoint and settled TRAIN clock differ")
        timed = observe(run / "training/timed-evaluation-state.json")
        state = observe(run / "state.json")
        best = observe(run / "training/best.json", optional=True)
        frontend = observe(run / "evaluation-request.json", optional=True)
        if frontend is not None and (frontend.get("schema") != "direct_native_evaluation_request_v1"
                                     or frontend.get("selection") != "latest"):
            raise ValueError("original latest evaluation request required")
        if best is not None and best.get("schema") != "direct_best_checkpoint_v1":
            raise ValueError("original best selection required")
        named_paths.update(paths)
        # The entire journals are pinned by verify_owner, while their current
        # and pending payload references are collected explicitly below. Passing
        # full ancestral journals to the generic locator walker would falsely
        # protect obsolete payloads from historical previous_attempt records.
        live_references = [locator for path, locator in paths.items()
                           if path not in {run / "state.json", run / "training/timed-evaluation-state.json"}]
        histories = [row["compact"] for row in timed["completed"] if row.get("compact")]
        histories.extend(state.get("manual_evaluation_history", []))
        histories = list({(row["path"], row["sha256"]): row for row in histories}.values())
        if not histories:
            result = {"status": "NOOP", "reason": "no committed compact histories", "removed": []}
            save("receipt.json", result)
            return result
        if best is None:
            raise DeferredCleanup("completed histories await genuine refresh_best")
        protected = [{key: latest[key] for key in ("path", "sha256")}]
        archived_receipts, archived_training = set(), set()
        for locator in histories:
            archive = checked(locator)
            # Structural indexing only; export_inventory performs full original
            # native/raw/source/quality validation before granting deletion.
            archived_receipts.add(tuple(archive["receipt"][k] for k in ("path", "sha256")))
            archived_training.add(tuple(archive["training_receipt"][k] for k in ("path", "sha256")))
        def protect_training(locator):
            receipt = checked(locator)
            protected.append(receipt["checkpoint"])
            for item in receipt["exports"].values():
                protected.append(item["checkpoint"])
        def protect_evaluation(locator):
            if tuple(locator[k] for k in ("path", "sha256")) in archived_receipts:
                return
            receipt = checked(locator)
            protected.append(receipt["checkpoint"])
            protect_training(receipt["training_receipt"])
            report = checked(receipt["evaluation"])
            exported = checked(report["export_receipt"])
            protected.append(exported["model"])
        for row in timed["completed"]:
            if not row.get("compact"):
                protected.append(checked(row["request"])["checkpoint"])
                protect_evaluation(row["receipt"])
        if timed.get("pending"):
            protected.append(checked(timed["pending"])["checkpoint"])
        for key in ("train_receipt", "pending_training"):
            locator = state.get(key)
            if locator and (key == "pending_training" or tuple(locator[k] for k in ("path", "sha256")) not in archived_training):
                protect_training(locator)
        for locator in state.get("evaluations", []) + state.get("manual_evaluations", []):
            protect_evaluation(locator)
        if state.get("evaluation_receipt"):
            protect_evaluation(state["evaluation_receipt"])
        verify_owner()
        plan = retention.export_inventory(
            run, completed_history=histories, evaluation_plan=config["evaluation_plan"],
            best=best["best"]["checkpoint"] if best.get("best") else None,
            protected=protected, reference_files=live_references,
            digest_proofs=CheckpointDigestProofs())
        save("plan.json", plan)
        removed = retention.apply_exports(plan, exclusive=verify_owner)
        result = {"status": "COMPLETE", "removed": removed,
                  "reclaimable_bytes": plan["reclaimable_bytes"], "release": proof}
        save("receipt.json", result)
        return result
    except BaseException as error:
        record = {"status": "DEFERRED" if isinstance(error, DeferredCleanup) else "ERROR",
                  "type": type(error).__name__, "reason": str(error),
                  "evidence_directory": str(output)}
        if plan is not None:
            record["planned_payloads_now_absent"] = [row["locator"]["path"] for row in plan["entries"]
                if row["action"] == "delete" and not Path(row["locator"]["path"]).exists()]
        try:
            save("failure.json", record)
        except BaseException:
            pass  # Never replace the original error with an observer failure.
        if isinstance(error, DeferredCleanup):
            return record
        raise


def refresh_scheduled_best(config, evaluated, *, retention=None):
    from w1a1_eagle.continuous_budget import checked_locator
    from w1a1_eagle.nine_model_pipeline import atomic_json

    if retention is None:
        from w1a1_eagle import direct_checkpoint_retention as retention
    run = Path(config["run_dir"])
    timed = retention._read(run / "training/timed-evaluation-state.json")
    journal = retention._read(run / "state.json")
    latest = retention._read(run / "training/checkpoints/latest.json")
    current = retention._read(checked_locator(evaluated))
    if (evaluated not in journal.get("evaluations", [])
            or current["checkpoint"] != {key: latest[key] for key in ("path", "sha256")}):
        raise ValueError("scheduled best requires the persisted genuine latest evaluation")
    receipts = [row["receipt"] for row in timed["completed"]]
    receipts.extend(journal.get("manual_evaluations", []))
    receipts.append(evaluated)
    unique = {}
    for item in receipts:
        prior = unique.get(item["path"])
        if prior is not None and prior != item:
            raise ValueError("conflicting evaluation receipt identities")
        unique[item["path"]] = item
    receipts = list(unique.values())
    histories = [row["compact"] for row in timed["completed"] if row.get("compact")]
    histories.extend(journal.get("manual_evaluation_history", []))
    histories = list({(row["path"], row["sha256"]): row for row in histories}.values())
    evaluations = [(locator, retention._read(checked_locator(
        retention._read(checked_locator(locator))["evaluation"]))) for locator in receipts]
    proofs = CheckpointDigestProofs()
    selected = retention.best_evaluated_checkpoint(
        evaluations, candidate="dspark_a8", manual_config=config,
        completed_history=histories, digest_proofs=proofs)
    result = {"schema": "direct_best_checkpoint_v1",
              "evaluations_sha256": hashlib.sha256(json.dumps(receipts, sort_keys=True).encode()).hexdigest(),
              "best": selected}
    atomic_json(run / "training/best.json", result)
    return result
