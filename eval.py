#!/usr/bin/env python3
"""Request a safely released native evaluation and print its matched throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
REQUEST_SCHEMA = "direct_native_evaluation_request_v1"


def _json(path, *, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"bounded regular metadata required: {path}")
    identity = lambda p: (
        p.stat().st_dev,
        p.stat().st_ino,
        p.stat().st_size,
        p.stat().st_mtime_ns,
        p.stat().st_ctime_ns,
    )
    before = identity(path)
    data = path.read_bytes()
    if identity(path) != before:
        raise ValueError("metadata changed while reading")
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("native original receipt hash mismatch")
    return json.loads(data)


def request_evaluation(run_dir):
    """Idempotently join an outstanding request; never touch a GPU process."""
    run_dir = Path(run_dir).resolve(strict=True)
    path = run_dir / "evaluation-request.json"
    request = {
        "schema": REQUEST_SCHEMA,
        "request_id": uuid.uuid4().hex,
        "created_unix": time.time(),
        "selection": "latest",
    }
    temporary = run_dir / (".evaluation-request-" + request["request_id"] + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(request, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary, path
            )  # exclusive atomic publication; readers never see a partial JSON
        except FileExistsError:
            request = _json(path)
    finally:
        temporary.unlink(missing_ok=True)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("selection") != "latest"
        or not isinstance(request.get("request_id"), str)
        or len(request["request_id"]) != 32
        or any(c not in "0123456789abcdef" for c in request["request_id"])
    ):
        raise ValueError("invalid existing evaluation request; preserve it for diagnosis")
    return request


def evaluate_released_boundary(*args, **kwargs):
    """Controller-only backend; original export/native/quality/release consumers.

    The direct controller calls this while holding exclusive lane ownership with
    settled TRAIN accounting. All original checkpoint/export/source/protocol and
    fresh release callbacks remain mandatory; no JSON flag authorizes a GPU.
    """
    from evaluate_nine_model_timed_checkpoint import evaluate_checkpoint

    return evaluate_checkpoint(*args, **kwargs)


def evaluate_manual_released_boundary(config, **runtime):
    """Direct controller backend; runtime objects come from its exclusive lane."""
    from w1a1_eagle.direct_native_evaluation import evaluate_manual_boundary, load_native_consumers

    exporter, evaluator = load_native_consumers(config)
    return evaluate_manual_boundary(
        exporter=exporter, evaluator=evaluator, config=config, **runtime
    )


def summarize(receipt_path, *, request_id=None):
    receipt = _json(receipt_path)
    if request_id is not None and receipt.get("request_id") != request_id:
        raise ValueError("native receipt does not match the requested evaluation")
    joined_path = None
    if receipt.get("schema") == "direct_joined_native_evaluation_receipt_v1":
        joined_path = Path(receipt_path).resolve()
        nonce = receipt.get("request_id")
        if (
            receipt.get("completed") is not True
            or not isinstance(nonce, str)
            or len(nonce) != 32
            or any(c not in "0123456789abcdef" for c in nonce)
        ):
            raise ValueError("completed joined evaluation with valid request nonce required")
        original = receipt["original_evaluation"]
        if set(original) != {"path", "sha256"}:
            raise ValueError("original scheduled evaluation locator required")
        source = _json(original["path"], expected_sha256=original["sha256"])
        if (
            source.get("schema") != "nine_model_timed_evaluation_receipt_v1"
            or source.get("completed") is not True
            or source.get("checkpoint") != receipt.get("checkpoint")
        ):
            raise ValueError("joined original evaluation/checkpoint differs")
        receipt, receipt_path = source, Path(original["path"])
    if receipt.get("completed") is not True:
        raise ValueError("native evaluation did not complete")
    locator = receipt["evaluation"]
    report_path = Path(locator["path"])
    if report_path.is_symlink() or report_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("bounded regular native report required")
    before = report_path.stat()
    data = report_path.read_bytes()
    after = report_path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("native report changed during read")
    if hashlib.sha256(data).hexdigest() != locator["sha256"]:
        raise ValueError("native report hash mismatch")
    report = json.loads(data)
    if report.get("schema") != "nine_model_timed_matched_report_v1":
        raise ValueError("original full matched native report required")
    candidate = receipt["candidate"]
    summary = {
        "checkpoint": receipt["checkpoint"],
        "candidate": candidate,
        "native_tokens_per_second": report["cells"][candidate]["request_tokens_per_second"],
        "eagle_q4_tokens_per_second": report["cells"]["eagle_q4"]["request_tokens_per_second"],
        "acceptance_rate": report["cells"][candidate].get("acceptance_rate"),
        "strict_quality_status": report.get("strict_quality_status", receipt["status"]),
        "receipt": str(Path(receipt_path).resolve()),
    }
    if joined_path is not None:
        summary["request_receipt"] = str(joined_path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=24 * 3600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    config = _json(args.config)
    if config.get("evaluation_request_protocol") != REQUEST_SCHEMA:
        raise ValueError("this controller has no deployed safe evaluation-request consumer")
    run_dir = (args.run_dir or Path(config["run_dir"])).resolve(strict=True)
    request = request_evaluation(run_dir)
    receipt_path = run_dir / "evaluations" / request["request_id"] / "receipt.json"
    # Launch is lock-arbitrated by the native runtime. A healthy trainer keeps
    # ownership and consumes the same request at its next complete checkpoint.
    # An idle/completed lane instead gets a dedicated zero-update evaluator.
    from w1a1_eagle.direct_training_runtime import GPUOwned, launch

    try:
        launched = launch(config, run_dir=run_dir, evaluation_only=True, detach=True,
                          config_path=args.config.resolve(strict=True))
    except GPUOwned:
        pass
    else:
        if launched != 0:
            raise RuntimeError(
                f"evaluation-only runtime launch returned {launched!r}; pending request preserved at "
                f"{run_dir / 'evaluation-request.json'}"
            )
    print(
        "Evaluation requested; waiting for a complete checkpoint and exclusive GPU release.",
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    while not receipt_path.exists():
        failure = receipt_path.parent / "failure.json"
        if failure.exists():
            raise RuntimeError(f"native evaluation failed; preserved diagnosis: {failure}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"request remains pending, preserved at {run_dir / 'evaluation-request.json'}"
            )
        time.sleep(1)
    print(
        json.dumps(
            summarize(receipt_path, request_id=request["request_id"]), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
