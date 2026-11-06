"""Single-run checkpoint retention; metadata never substitutes for resume state.

Plans are dry-run inventories. Applying requires an exclusive controller callback
and revalidates every payload before deletion. Existing timed-history references
must be supplied as protected: this module does not bypass their readers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular metadata required: {path}")
    before = _identity(path)
    if before[2] > 16 * 1024 * 1024:
        raise ValueError("checkpoint metadata exceeds 16 MiB")
    value = json.loads(path.read_text())
    if _identity(path) != before:
        raise ValueError("metadata changed while reading")
    return value


def _identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class RetentionPlan(dict):
    """JSON inventory plus process-local actual-byte proofs for this transaction."""


_RETIRABLE_PROTECTIONS = {"initial", "milestone", "endpoint", "stop"}


def retirement_proof(archive_locator):
    """Bind explicit retirement to one fully validated completed checkpoint.

    This does not retire a different initial/source or a transition checkpoint.
    Those may still be scientific dependencies and need their own real evidence.
    Latest, best and pending references always override this proof. Empty
    protections are valid: that case only authenticates completed history and
    grants no additional deletion authority to the ordinary unprotected payload.
    """
    from .continuous_budget import artifact_locator

    history = read_completed_history(archive_locator)
    checkpoint = history["checkpoint_identity"]
    sidecar = Path(checkpoint["path"]).with_suffix(".receipt.json")
    proof = {
        "schema": "completed_checkpoint_retirement_proof_v1",
        "archive": archive_locator,
        "checkpoint": checkpoint,
        "sidecar": artifact_locator(sidecar),
    }
    _validate_retirement_proof(proof, _history=history)
    return proof


def _validate_retirement_proof(proof, *, _history=None):
    from .continuous_budget import checked_locator

    if proof.get("schema") != "completed_checkpoint_retirement_proof_v1":
        raise ValueError("explicit completed-checkpoint retirement proof required")
    history = read_completed_history(proof["archive"]) if _history is None else _history
    receipt = _read(checked_locator(proof["sidecar"]))
    checkpoint = {key: receipt.get(key) for key in ("path", "sha256")}
    protections = receipt.get("retention", {}).get("protected")
    if (
        checkpoint != history["checkpoint_identity"]
        or proof["checkpoint"] != checkpoint
        or proof["sidecar"]["path"] != str(Path(checkpoint["path"]).with_suffix(".receipt.json"))
        or receipt.get("schema") != "block_qat_checkpoint_v1"
        or receipt.get("committed") is not True
        or receipt.get("cursor") != history["counters"]
        or not isinstance(protections, list)
        or not set(protections) <= _RETIRABLE_PROTECTIONS
    ):
        raise ValueError("retirement archive/sidecar/checkpoint/protection differs")
    return receipt


def inventory(
    directory,
    *,
    best=None,
    protected=(),
    completed_history=(),
    retirement_proofs=(),
    digest_proofs=None,
):
    """Authenticate one original checkpoint family and produce a deletion plan.

    ``best`` is the checkpoint locator selected by genuine completed evaluation;
    the caller owns scientific metric/quality selection. Pending evaluations and
    every old-reader checkpoint/model reference belong in ``protected``. Foreign,
    malformed and unretired protected artifacts are retained, never adopted.
    A retirement proof only releases protection on its exact completed checkpoint;
    it cannot override latest, best or pending references.
    """
    from .block_training import _owned_checkpoint
    from .checkpoint_digest_proofs import CheckpointDigestProofs
    from .continuous_budget import checked_locator

    proofs = digest_proofs if digest_proofs is not None else CheckpointDigestProofs()

    directory = Path(directory)
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("retention root cannot traverse symlinks")
    directory = directory.resolve(strict=True)
    latest = _read(directory / "latest.json")
    family = latest.get("retention", {}).get("family_sha256")
    if (
        not family
        or _owned_checkpoint(Path(latest["path"]), family, digest_proofs=proofs) != latest
    ):
        raise ValueError("latest must be a complete authenticated owned checkpoint")
    if Path(latest["path"]).parent != directory:
        raise ValueError("latest checkpoint outside this run")
    keep = {str(Path(latest["path"]).resolve())}
    if best is not None:
        path = Path(best["path"])
        receipt = _read(path.with_suffix(".receipt.json"))
        if (
            receipt.get("retention", {}).get("family_sha256") != family
            or receipt["sha256"] != best["sha256"]
            or path.parent != directory
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError("best checkpoint must belong to this run and family")
        keep.add(str(path.resolve()))
    keep.update(
        str(Path(item["path"] if isinstance(item, dict) else item).resolve()) for item in protected
    )
    retirement_proofs = list(retirement_proofs)
    retired = [_validate_retirement_proof(proof) for proof in retirement_proofs]
    for locator in completed_history:
        # Keeping a required best/latest/reference or an already absent payload
        # grants no deletion authority. Do not re-run the entire native history
        # merely to preserve it at every periodic save. The execution/resume
        # reader still validates completed history independently.
        archive = _read(checked_locator(locator))
        archived_path = Path(archive["checkpoint"]["path"])
        if str(archived_path.resolve()) in keep or not archived_path.exists():
            continue
        proof = retirement_proof(locator)
        retirement_proofs.append(proof)
        retired.append(_read(proof["sidecar"]["path"]))
    entries = []
    for path in sorted(directory.glob("*.pt")):
        # Retention of a protected payload grants no model validity; do not
        # repeatedly stream initial/best/history weights merely to keep them.
        try:
            sidecar = _read(path.with_suffix(".receipt.json"))
        except (OSError, ValueError):
            sidecar = {}
        retired_protection = bool(sidecar.get("retention", {}).get("protected")) and sidecar in retired
        retain_only = str(path.resolve()) in keep or (
            bool(sidecar.get("retention", {}).get("protected")) and not retired_protection
        )
        receipt = sidecar if retain_only else _owned_checkpoint(path, family, digest_proofs=proofs)
        reason = "foreign_or_unverified"
        action = "keep"
        if receipt is not None:
            if str(path.resolve()) in keep:
                reason = "latest_best_or_live_reference"
            elif receipt.get("retention", {}).get("protected") and not retired_protection:
                reason = "original_protection"
            else:
                reason, action = "obsolete_same_run_checkpoint", "delete"
        entries.append(
            {
                "path": str(path),
                "bytes": path.lstat().st_size,
                "identity": _identity(path),
                "receipt": receipt,
                "retired_protection": retired_protection,
                "action": action,
                "reason": reason,
            }
        )
    plan = RetentionPlan(
        {
            "schema": "direct_checkpoint_retention_plan_v1",
            "directory": str(directory),
            "latest": latest,
            "best": best,
            "protected": sorted(keep),
            "completed_history": list(completed_history),
            "retirement_proofs": retirement_proofs,
            "entries": entries,
            "reclaimable_bytes": sum(e["bytes"] for e in entries if e["action"] == "delete"),
        }
    )
    plan.digest_proofs = proofs
    return plan


def apply(plan, *, exclusive):
    """Apply only a reviewed unchanged plan under a trusted exclusive writer.

    Keep the tiny original sidecar as provenance after payload deletion. The
    sidecar's presence grants no resume permission: original loader still hashes
    and loads the actual retained latest/best payload with optimizer/RNG state.
    """
    from .block_training import _owned_checkpoint

    if not callable(exclusive) or exclusive() is not True:
        raise ValueError("exclusive checkpoint/evaluation ownership required")
    directory = Path(plan["directory"])
    if _read(directory / "latest.json") != plan["latest"]:
        raise ValueError("latest changed since inventory; make a new plan")
    if not any(entry["action"] == "delete" for entry in plan["entries"]):
        # An empty deletion transaction cannot retire a payload. Preserve the
        # exclusive/latest checks without another full historical validation.
        return []
    retired = [_validate_retirement_proof(proof) for proof in plan.get("retirement_proofs", [])]
    removed = []
    for entry in plan["entries"]:
        if entry["action"] != "delete":
            continue
        path = Path(entry["path"])
        receipt = entry["receipt"]
        if (
            path.parent != directory
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or _identity(path) != entry["identity"]
            or _owned_checkpoint(
                path,
                receipt["retention"]["family_sha256"],
                digest_proofs=getattr(plan, "digest_proofs", None),
            )
            != receipt
            or (
                receipt["retention"]["protected"]
                and not (entry.get("retired_protection") and receipt in retired)
            )
            or str(path) in plan["protected"]
        ):
            raise ValueError("retention candidate changed; deletion stopped")
        if exclusive() is not True:
            raise ValueError("checkpoint ownership changed; deletion stopped")
        path.unlink()
        removed.append(str(path))
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return removed


def _canonical_export_path(path, root):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("canonical owned export path required")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("owned export cannot traverse symlinks")
    if not path.is_relative_to(root) or path == root:
        raise ValueError("export outside owned run")
    return path


def _locator_paths(value):
    """Exact references only; neither hashes nor basename matches grant scope."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and "sha256" in value:
            yield str(Path(value["path"]).absolute())
        for item in value.values():
            yield from _locator_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _locator_paths(item)


def export_inventory(
    run_directory, *, completed_history, evaluation_plan, best=None,
    protected=(), reference_files=(), digest_proofs=None,
):
    """Plan exact obsolete completed GGUF/NPZ payloads, preserving all metadata.

    The controller supplies every pending/current reference and their mutable
    metadata files under exclusive ownership. ``best`` is a checkpoint locator;
    protecting a checkpoint protects both of its archived exports. The original
    plan's control/source locators are always protected. No directory is scanned,
    no incomplete evaluation is adopted, and no archive is manufactured here.
    """
    from .continuous_budget import checked_locator
    from .checkpoint_digest_proofs import CheckpointDigestProofs

    root = Path(run_directory)
    protected = list(protected)
    if best is not None and not (isinstance(best, dict) and {"path", "sha256"} <= set(best)):
        raise ValueError("best checkpoint locator required")
    if any(not isinstance(item, dict) or not {"path", "sha256"} <= set(item) for item in protected):
        raise ValueError("protected checkpoint/export locators required")
    _canonical_export_path(root / "training", root)
    latest_path = root / "training/checkpoints/latest.json"
    latest = _read(latest_path)
    latest_checkpoint = {key: latest[key] for key in ("path", "sha256")}
    checkpoint_root = root / "training/checkpoints"
    _canonical_export_path(latest_checkpoint["path"], checkpoint_root)
    family = latest.get("retention", {}).get("family_sha256")
    if not family or _read(Path(latest["path"]).with_suffix(".receipt.json")) != latest:
        raise ValueError("original owned latest sidecar required")
    plan_value = _read(checked_locator(evaluation_plan))
    # Protect every locator in the admitted plan, including initial/Q4/target,
    # source NPZs and any explicit historical control references.
    keep = set(_locator_paths(plan_value)) | set(_locator_paths(protected))
    protected_checkpoints = [latest_checkpoint]
    if best is not None:
        protected_checkpoints.append(best)
    protected_checkpoints.extend(item for item in protected if isinstance(item, dict))
    keep.update(_locator_paths(protected_checkpoints))
    references = [{"locator": item, "value": _read(checked_locator(item))}
                  for item in reference_files]
    keep.update(path for item in references for path in _locator_paths(item["value"]))
    proofs = digest_proofs if digest_proofs is not None else CheckpointDigestProofs()
    entries = {}
    histories = list(completed_history)
    validated = []
    for archive_locator in histories:
        history = read_completed_history(archive_locator, digest_proofs=proofs)
        archive = _read(checked_locator(archive_locator))
        receipt = history["receipt"]
        if receipt.get("research_continuation_policy") is None:
            raise ValueError("export retirement requires archived original native/raw/quality evidence")
        keep.update(_locator_paths(receipt["research_bindings"]))
        validated.append((archive_locator, archive, history))
    # A later archive's controls must also protect an earlier archive's payload.
    for archive_locator, archive, history in validated:
        receipt = history["receipt"]
        checkpoint = history["checkpoint_identity"]
        checkpoint_path = _canonical_export_path(checkpoint["path"], checkpoint_root)
        sidecar = _read(checkpoint_path.with_suffix(".receipt.json"))
        if ({key: sidecar.get(key) for key in ("path", "sha256")} != checkpoint
                or sidecar.get("retention", {}).get("family_sha256") != family):
            raise ValueError("archived export checkpoint belongs to another run/family")
        exported = _read(checked_locator(archive["export_receipt"]))
        training = _read(checked_locator(archive["training_receipt"]))
        if (exported.get("schema") != "nine_model_lane_endpoint_export_v1"
                or exported.get("artifact_kind") != "production"
                or exported.get("candidate") != receipt["candidate"]
                or exported.get("training_receipt") != archive["training_receipt"]):
            raise ValueError("original production candidate export required")
        inputs = training["exports"][receipt["candidate"]]
        manifest = _read(checked_locator(inputs["manifest"]))
        gguf, npz = archive["payload_identities"][1:]
        gguf_path = _canonical_export_path(gguf["path"], root)
        npz_path = _canonical_export_path(npz["path"], root / "training")
        if (gguf_path.name != "trained.gguf"
                or gguf_path.parent != Path(archive["export_receipt"]["path"]).parent
                or npz_path.name not in {"binary.npz", "joint.npz"}
                or Path(inputs["manifest"]["path"]).parent != npz_path.parent
                or manifest.get("checkpoint_sha256") != npz["sha256"]):
            raise ValueError("original candidate GGUF/joint NPZ layout or manifest differs")
        for role, locator in (("trained_gguf", gguf), ("joint_npz", npz)):
            path = Path(locator["path"])
            retain = checkpoint["path"] in keep or str(path) in keep
            context = (archive_locator["sha256"], role)
            identity = None
            size = 0
            if path.exists():
                from .checkpoint_digest_proofs import regular_file_identity

                identity = list(regular_file_identity(path))
                size = path.stat().st_size
                if not retain and not proofs.matches(path, locator["sha256"], context=context):
                    raise ValueError("owned export bytes differ from archived identity")
            entry = {"locator": locator, "role": role, "archive": archive_locator,
                     "checkpoint": checkpoint, "identity": identity, "bytes": size,
                     "action": "keep" if retain or identity is None else "delete",
                     "reason": "latest_best_pending_or_control" if retain else
                               "already_absent" if identity is None else "obsolete_completed_export"}
            prior = entries.get(str(path))
            if prior is not None and prior["locator"] != locator:
                raise ValueError("conflicting archived export identities")
            if prior is None or retain:
                entries[str(path)] = entry
    result = RetentionPlan(schema="direct_export_retention_plan_v1", directory=str(root),
                           latest=latest, evaluation_plan=evaluation_plan,
                           completed_history=histories, best=best, protected=list(protected),
                           reference_files=references, entries=list(entries.values()))
    result["reclaimable_bytes"] = sum(e["bytes"] for e in result["entries"] if e["action"] == "delete")
    result.digest_proofs = proofs
    return result


def apply_exports(plan, *, exclusive):
    """Revalidate archive/native/raw/quality and exact bytes before each unlink.

    Absence is idempotent after an interrupted deletion; reappearing or changed
    payloads are refused. Tiny receipts, manifests, raw rows and sources survive.
    This callback must exclude trainer, exporter, evaluator and reference writers.
    """
    if not callable(exclusive) or exclusive() is not True:
        raise ValueError("exclusive checkpoint/evaluation ownership required")
    from .continuous_budget import checked_locator
    from .checkpoint_digest_proofs import regular_file_identity

    if plan.get("schema") != "direct_export_retention_plan_v1":
        raise ValueError("owned export retention plan required")
    for reference in plan["reference_files"]:
        if _read(checked_locator(reference["locator"])) != reference["value"]:
            raise ValueError("export live references changed")
    if _read(Path(plan["directory"]) / "training/checkpoints/latest.json") != plan["latest"]:
        raise ValueError("latest changed since export inventory")
    # Rebuild scope/protections and re-run all historical scientific readers.
    fresh = export_inventory(plan["directory"], completed_history=plan["completed_history"],
                             evaluation_plan=plan["evaluation_plan"], best=plan["best"],
                             protected=plan["protected"],
                             reference_files=[r["locator"] for r in plan["reference_files"]],
                             digest_proofs=getattr(plan, "digest_proofs", None))
    by_path = {e["locator"]["path"]: e for e in fresh["entries"]}
    removed = []
    for entry in plan["entries"]:
        if entry["action"] != "delete":
            continue
        path = Path(entry["locator"]["path"])
        current = by_path[str(path)]
        if current["reason"] == "already_absent":
            continue
        if (current != entry or list(regular_file_identity(path)) != entry["identity"]
                or exclusive() is not True):
            raise ValueError("export candidate or ownership changed; deletion stopped")
        _canonical_export_path(path, Path(plan["directory"]))
        if list(regular_file_identity(path)) != entry["identity"]:
            raise ValueError("export changed during ownership check")
        path.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        removed.append(str(path))
    return removed


def best_evaluated_checkpoint(evaluations, *, candidate, manual_config=None,
                             completed_history=(), digest_proofs=None):
    """Select highest matched full-request throughput; preserve quality label.

    Reauthenticate the report locator and original scientific continuation before
    ranking. Status/quality are never renamed to make a checkpoint best.
    """
    from .continuous_budget import checked_locator, validate_evaluation_continuation
    from .checkpoint_digest_proofs import CheckpointDigestProofs

    proofs = digest_proofs if digest_proofs is not None else CheckpointDigestProofs()
    histories = [read_completed_history(locator, digest_proofs=proofs)
                 for locator in completed_history]
    ranked = []
    protocol = None
    for receipt_locator, report in evaluations:
        receipt = _read(checked_locator(receipt_locator))
        if protocol is not None and receipt["protocol"] != protocol:
            raise ValueError("best checkpoints require the same matched protocol")
        protocol = receipt["protocol"]
        authenticated_report = _read(checked_locator(receipt["evaluation"]))
        if authenticated_report != report:
            raise ValueError("authenticated original evaluation required")
        archived = next(
            (history for history in histories if history["receipt_locator"] == receipt_locator),
            None,
        )
        if archived is not None:
            if archived["receipt"] != receipt or archived["report"] != report:
                raise ValueError("archived best evaluation differs")
        elif receipt.get("schema") == "direct_native_evaluation_receipt_v1":
            if manual_config is None:
                raise ValueError("manual best selection requires execution configuration")
            from .direct_native_evaluation import validate_manual_completed
            from .nine_model_pipeline import Files

            validate_manual_completed(receipt_locator, manual_config, files=Files())
        elif not validate_evaluation_continuation(receipt):
            raise ValueError("authenticated original evaluation required")
        if (
            receipt.get("completed") is not True
            or receipt.get("candidate") != candidate
            or report.get("schema") != "nine_model_timed_matched_report_v1"
        ):
            raise ValueError("completed original matched evaluation required")
        throughput = report["cells"][candidate]["request_tokens_per_second"]
        import math

        if type(throughput) not in (float, int) or not math.isfinite(throughput) or throughput <= 0:
            raise ValueError("positive finite full-request throughput required")
        if report.get("fixture") or report.get("strict_quality_status", "PASS") not in {
            "PASS",
            "FAILED",
        }:
            raise ValueError("genuine native quality verdict required")
        if receipt.get("strict_quality_status", "PASS") != report.get(
            "strict_quality_status", "PASS"
        ):
            raise ValueError("evaluation quality labels disagree")
        ranked.append(
            {
                "checkpoint": receipt["checkpoint"],
                "request_tokens_per_second": throughput,
                "strict_quality_status": report.get("strict_quality_status", "PASS"),
                "protocol": receipt["protocol"],
                "research_only": receipt.get("research_only", False),
            }
        )
    return _select_validated_best(ranked)


def _select_validated_best(ranked):
    """Ranking only; callers must first authenticate completed matched results."""
    if not ranked:
        return None
    best = dict(
        max(
            ranked,
            key=lambda item: (
                item["strict_quality_status"] == "PASS",
                item["request_tokens_per_second"],
            ),
        )
    )
    best["selection"] = (
        "best_strict" if best["strict_quality_status"] == "PASS" else "best_research"
    )
    if best["selection"] == "best_research":
        best["research_only"] = True
    return best


def archive_completed_history(
    receipt_locator, output, *, validate_continuation=None, manual_config=None
):
    """Archive an already fully validated completed result before retiring weights.

    This is invoked while payloads still exist. Only the original continuation
    validator may establish the serialized-checkpoint/native-stage proof. The
    archive is bound into the existing completed entry as ``compact``; no other
    current-state file is introduced. No deletion is performed here.
    """
    from . import continuous_budget as budget
    from .continuous_budget import checked_locator, validate_evaluation_continuation
    from .nine_model_pipeline import Files

    if (
        validate_continuation is not None
        and validate_continuation is not validate_evaluation_continuation
    ):
        raise ValueError("original continuation validator required")
    receipt = _read(checked_locator(receipt_locator))
    if receipt.get("schema") == "direct_native_evaluation_receipt_v1":
        if manual_config is None:
            raise ValueError("manual history requires exact execution configuration")
        from .direct_native_evaluation import validate_manual_completed

        validate_manual_completed(receipt_locator, manual_config, files=Files())
    elif receipt.get("completed") is not True or not validate_evaluation_continuation(receipt):
        raise ValueError("complete original evaluation must validate before compacting")
    report = _read(checked_locator(receipt["evaluation"]))
    request = _read(checked_locator(receipt["request"]))
    training = _read(checked_locator(receipt["training_receipt"]))
    if (
        receipt["checkpoint"] != request["checkpoint"]
        or request["checkpoint"] != training["checkpoint"]
        or request["counters"] != training["counters"]
    ):
        raise ValueError("completed history checkpoint/counters differ")
    exported = _read(checked_locator(report["export_receipt"]))
    payloads = [
        receipt["checkpoint"],
        exported["model"],
        training["exports"][receipt["candidate"]]["checkpoint"],
    ]
    archive = {
        "schema": "validated_completed_evaluation_history_v1",
        "receipt": receipt_locator,
        "request": receipt["request"],
        "report": receipt["evaluation"],
        "training_receipt": receipt["training_receipt"],
        "export_receipt": report["export_receipt"],
        "checkpoint": receipt["checkpoint"],
        "counters": request["counters"],
        "strict_quality_status": report.get("strict_quality_status", "PASS"),
        "payload_identities": payloads,
        "validation_source": {
            "path": budget.__file__,
            "sha256": hashlib.sha256(Path(budget.__file__).read_bytes()).hexdigest(),
        },
    }
    if manual_config is not None:
        from . import direct_native_evaluation as manual

        archive["manual_validation_source"] = {
            "path": manual.__file__,
            "sha256": hashlib.sha256(Path(manual.__file__).read_bytes()).hexdigest(),
        }
    output = Path(output).resolve()
    for key in ("validation_source", "manual_validation_source"):
        if key not in archive:
            continue
        source = checked_locator(archive[key])
        snapshot = output.with_name(output.stem + "-" + key + ".py")
        with snapshot.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        archive[key] = {
            "path": str(snapshot),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        }
    with output.open("x") as stream:
        json.dump(archive, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def read_completed_history(locator, *, digest_proofs=None):
    """Metadata-only historical evidence reader; never authorizes a new update.

    The caller must hash-bind this locator in the existing completed entry and
    separately validate the actual latest resume payload with the original
    loader. Pending requests always use the original full continuation path.
    Retained raw rows are rechecked against the original quality policy; this
    function neither turns a failed quality verdict PASS nor restores a model.
    """
    from .continuous_budget import checked_locator, research_continuation_decision

    archive = _read(checked_locator(locator))
    if archive.get("schema") != "validated_completed_evaluation_history_v1":
        raise ValueError("validated completed history required")
    if (
        archive["validation_source"]["sha256"]
        != "9cbc775931b1f2739c8a9b7cad9e98f746449000cce06dd8b9b6792b74eb655e"
    ):
        raise ValueError("compact history original validation source differs")
    checked_locator(archive["validation_source"])
    if archive.get("manual_validation_source"):
        checked_locator(archive["manual_validation_source"])
    receipt = _read(checked_locator(archive["receipt"]))
    request = _read(checked_locator(archive["request"]))
    report = _read(checked_locator(archive["report"]))
    training = _read(checked_locator(archive["training_receipt"]))
    exported = _read(checked_locator(archive["export_receipt"]))
    if (
        receipt.get("completed") is not True
        or receipt["request"] != archive["request"]
        or receipt["evaluation"] != archive["report"]
        or report["export_receipt"] != archive["export_receipt"]
        or receipt["training_receipt"] != archive["training_receipt"]
        or receipt.get("owned_release", {}).get("owned_process_groups_absent") is not True
        or receipt.get("owned_release", {}).get("owned_cuda_pids_absent") is not True
        or receipt["checkpoint"] != request["checkpoint"]
        or request["checkpoint"] != training["checkpoint"]
        or request["checkpoint"] != archive["checkpoint"]
        or request["counters"] != training["counters"]
        or request["counters"] != archive["counters"]
        or exported["checkpoint"] != archive["checkpoint"]
        or archive["payload_identities"]
        != [
            receipt["checkpoint"],
            exported["model"],
            training["exports"][receipt["candidate"]]["checkpoint"],
        ]
        or archive["strict_quality_status"] != report.get("strict_quality_status", "PASS")
    ):
        raise ValueError("compact history receipt/report/checkpoint/quality joins differ")
    if (
        receipt["candidate"] != training["stage"].removesuffix("/train")
        or receipt["bundle_sha256"] != training["bundle_sha256"]
        or receipt["config_sha256"] != training["config_sha256"]
        or receipt.get("research_continuation_policy") != report.get("research_continuation_policy")
        or receipt.get("research_bindings") != report.get("research_bindings")
    ):
        raise ValueError("historical source/quality bindings differ")
    if receipt.get("research_continuation_policy") is not None:
        raw = _read(checked_locator(report["measurements"]))
        # Re-run the ORIGINAL process/dispatch/resource/raw joins. Only already
        # validated, archived payload identities may be absent; metadata cannot.
        import types

        from . import continuous_budget as original
        from .checkpoint_digest_proofs import CheckpointDigestProofs

        allowed = archive["payload_identities"]
        # The private original validator checks the same target/control in
        # every stage. Its ordinary checker would stream those weights again
        # for each row. Each expected digest and strict before/open/after
        # identity still checks. Independent reads use a fresh proof set;
        # one timed-reader instance can reuse unchanged actual-byte proofs.
        # Every metadata/raw/quality validation still runs on every read.
        if digest_proofs is not None and type(digest_proofs) is not CheckpointDigestProofs:
            raise ValueError("original process-local byte proof pool required")
        proofs = digest_proofs if digest_proofs is not None else CheckpointDigestProofs()

        def historical_payload_check(record):
            if record in allowed:
                return Path(record["path"])
            # Preserve the original checked_locator schema, regular-file and
            # expected-SHA guards. Only its repeated digest IO is shared.
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError("exact path/SHA256 locator required")
            path = Path(record["path"])
            if not path.is_file() or path.is_symlink():
                raise ValueError("timed evaluation artifact absent or symlinked")
            # The checkpoint proof is deliberately stricter about hard links.
            # Preserve the original reader's accepted files via full validation
            # rather than inventing reuse authority for linked payloads.
            if path.stat().st_nlink != 1:
                return checked_locator(record)
            if not proofs.matches(path, record["sha256"]):
                raise ValueError("timed evaluation artifact SHA256 differs")
            return path

        source_path = checked_locator(archive["validation_source"])
        # Authenticated9cbc has only stdlib imports, a literal policy constant,
        # and function/class definitions at top level; no decorators or runtime
        # execution. A private exec preserves ALL original bodies/defaults and
        # their shared global aliases, including policy/nested native validators.
        historical_payload_check({"path": str(Path(original.__file__)),
                                  "sha256": archive["validation_source"]["sha256"]})
        private = {"__name__": original.__name__, "__package__": original.__package__,
                   "__file__": original.__file__, "__builtins__": __builtins__}
        exec(compile(source_path.read_text(), str(source_path), "exec", dont_inherit=True), private)
        private["checked_locator"] = historical_payload_check
        private["validate_native_stages"](
            receipt["research_bindings"],
            raw["clean_records"],
            raw["diagnostic_records"],
            report["native_stages"],
            report["model_ancestry"],
            report["producer_sources"],
            candidate_audit=exported["audit"],
            execution_context=report["execution_context"],
        )
        decision = private["research_continuation_decision"](
            receipt["research_continuation_policy"],
            receipt["research_bindings"],
            raw["clean_records"],
            raw["diagnostic_records"],
        )
        if any(
            receipt.get(key) != decision[key] or report.get(key) != decision[key]
            for key in ("measurement_status", "strict_quality_status", "continuation_status")
        ):
            raise ValueError("historical scientific continuation decision differs")
    elif receipt.get("status") != "PASS" or report.get("strict_quality_status", "PASS") != "PASS":
        raise ValueError("failed history requires original research continuation")
    return {
        "receipt_locator": archive["receipt"],
        "receipt": receipt,
        "request": request,
        "report": report,
        "checkpoint_identity": archive["checkpoint"],
        "counters": archive["counters"],
    }


def compact_timed_evaluation_class(original_class):
    """Exact original timed reader with explicit completed-entry archive support.

    Current/pending checkpoint authorization always takes the original full path.
    Historical skip authority is only an authenticated archive's exact checkpoint
    locator; a wrong expected digest never aliases a pruned checkpoint.
    """
    import types

    from .continuous_budget import TimedEvaluation, checked_locator

    if original_class is not TimedEvaluation:
        raise ValueError("original timed evaluation reader required")

    class CompactTimedEvaluation(original_class):
        def _history(self, *, request=None, receipt=None):
            for entry in self.state.get("completed", []):
                if (
                    (request is None or entry["request"] == request)
                    and (receipt is None or entry["receipt"] == receipt)
                    and entry.get("compact")
                ):
                    from .checkpoint_digest_proofs import CheckpointDigestProofs

                    if not hasattr(self, "_history_digest_proofs"):
                        self._history_digest_proofs = CheckpointDigestProofs()
                    history = read_completed_history(
                        entry["compact"], digest_proofs=self._history_digest_proofs
                    )
                    if history["receipt"]["request"] != entry["request"] or history[
                        "receipt"
                    ] != _read(checked_locator(entry["receipt"])):
                        raise ValueError("compact completed entry differs")
                    return history
            return None

        def _request(self, locator):
            history = self._history(request=locator)
            if history is None:
                return super()._request(locator)
            checkpoint = history["checkpoint_identity"]

            def historical_check(record):
                return Path(record["path"]) if record == checkpoint else checked_locator(record)

            function = original_class._request
            private = types.FunctionType(
                function.__code__,
                dict(function.__globals__, checked_locator=historical_check),
                function.__name__,
                function.__defaults__,
                function.__closure__,
            )
            value = private(self, locator)
            if value != history["request"]:
                raise ValueError("compact request changed")
            return value

        def _result(self, locator, request_locator, request):
            history = self._history(request=request_locator, receipt=locator)
            if history is None:
                return super()._result(locator, request_locator, request)

            def validated_history(result):
                if result != history["receipt"]:
                    raise ValueError("compact historical receipt differs")
                return True

            function = original_class._result
            private = types.FunctionType(
                function.__code__,
                dict(function.__globals__, validate_evaluation_continuation=validated_history),
                function.__name__,
                function.__defaults__,
                function.__closure__,
            )
            return private(self, locator, request_locator, request)

        def compact_completed(self, index, output):
            entry = self.state["completed"][index]
            if entry.get("compact"):
                read_completed_history(entry["compact"])
                return entry["compact"]
            # Validate through the original reader while full payloads exist.
            request = original_class._request(self, entry["request"])
            original_class._result(self, entry["receipt"], entry["request"], request)
            locator = archive_completed_history(entry["receipt"], output)
            updated = json.loads(json.dumps(self.state))
            updated["completed"][index]["compact"] = locator
            self.write(self.path, updated)
            self.state = updated
            return locator

    return CompactTimedEvaluation
