"""Explicit manual-boundary adapter around the admitted exporter/native benchmark.

Only the scheduling boundary differs. Scientific checkpoint/export/quality code
and the four matched native arms are the original consumers.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import types
from pathlib import Path

EXPORTER_SHA = "10ba61f9269256c048ccfac82369a1f49fe569165ffbf80dd2c9548eb3cb9fab"
EVALUATOR_SHA = "37fad07fa309d1be92203fefda303d3b370acb0a8cd8156327a8b12f5a2f8bdf"
OPERATIONAL_HELPER_SHA = "5b0ffcbe472fa25275c49c610333777972a8681de24732f616f445e65e17bc10"
OPERATIONAL_MANIFEST_SHA = "1dc5a9f86942ee19b735c8173edb47352aa75431e576887530223b69aaab4d2a"
PIPELINE_SHA = "a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af"


def validate_manual_request(receipt, lane, lane_locator, spec, limits, files):
    """Cheap provenance/accounting checks before opening any model payload."""
    request = json.loads(files.check(receipt["manual_evaluation_request"]).read_text())
    elapsed = receipt["counters"]["elapsed_seconds"]
    ledger = json.loads(files.check(request["budget_ledger"]).read_text())
    request_id = request.get("request_id")
    if (
        not isinstance(request_id, str)
        or len(request_id) != 32
        or any(c not in "0123456789abcdef" for c in request_id)
        or request.get("schema") != "direct_native_evaluation_boundary_v1"
        or request.get("candidate") != lane["candidate"]
        or request.get("bundle_sha256") != lane_locator["sha256"]
        or request.get("config_sha256") != lane["config"]["sha256"]
        or request.get("checkpoint") != receipt["checkpoint"]
        or request.get("counters") != receipt["counters"]
        or request.get("protocol") != spec.get("evaluation_protocol")
        or request.get("elapsed_seconds") != elapsed
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or ledger.get("schema") != "continuous_training_budget_v1"
        or ledger.get("max_seconds") != limits["max_seconds"]
        or ledger.get("training_seconds") != elapsed
        or ledger.get("active_attempt") is not None
        or receipt.get("completion_reason") != "evaluation_requested"
    ):
        raise ValueError("manual request/checkpoint/protocol/settled accounting differs")
    files.check(request["protocol"])
    return request


def validate_manual_export_inputs(receipt, lane, files):
    """Reject malformed joint metadata before target/base/checkpoint model IO."""
    exports = receipt.get("exports", {})
    name = lane["candidate"]
    if set(exports) != {name}:
        raise ValueError("manual boundary requires genuine single-candidate exports")
    exported = exports[name]
    joint = json.loads(files.check(exported["manifest"]).read_text())
    if (
        joint.get("checkpoint_sha256") != exported["checkpoint"]["sha256"]
        or joint.get("base_gguf_sha256") != exported.get("base_gguf_sha256")
        or joint.get("activation_bits") != int(name.split("_a")[1])
    ):
        raise ValueError("manual joint checkpoint/base/precision differs")
    return joint


def _source(module, expected):
    path = Path(module.__file__).resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original native consumer source differs")
    source = raw.decode()
    compiled = compile(source, str(module.__file__), "exec", dont_inherit=True)
    names = (
        ("validate_endpoint", "export_endpoint")
        if expected == EXPORTER_SHA
        else ("validate_plan", "evaluate")
    )
    for name in names:
        expected_code = next(
            code
            for code in compiled.co_consts
            if isinstance(code, types.CodeType) and code.co_name == name
        )
        actual = getattr(module, name)
        if not isinstance(actual, types.FunctionType) or actual.__code__ != expected_code:
            raise ValueError("live original native consumer differs: " + name)
    return source


def _verified_small(record):
    path = Path(record["path"])
    if (
        not path.is_absolute()
        or path != path.resolve(strict=True)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 8 * 1024**2
    ):
        raise ValueError("bounded canonical operational source required")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        any(getattr(before, field) != getattr(after, field) for field in fields)
        or hashlib.sha256(raw).hexdigest() != record["sha256"]
    ):
        raise ValueError("operational source/metadata digest differs")
    return raw


def private_timed_evaluator(evaluator, config):
    """Raw37fa authentication precedes the exact accepted5b0 wait derivation."""
    import importlib.util
    import runpy
    import sys

    sys.dont_write_bytecode = True
    source = _source(evaluator, EVALUATOR_SHA)
    selected = config["operational_runtime"]
    helper_record, manifest_record = selected["helper"], selected["manifest"]
    if (
        helper_record["sha256"] != OPERATIONAL_HELPER_SHA
        or manifest_record["sha256"] != OPERATIONAL_MANIFEST_SHA
    ):
        raise ValueError("accepted operational helper/manifest selection differs")
    manifest = json.loads(_verified_small(manifest_record))
    package = Path(manifest_record["path"]).parent
    if (
        Path(helper_record["path"]) != package / "scripts/run_frozen_dspark_hash_memo_20261011.py"
        or manifest["files"]["scripts/run_frozen_dspark_hash_memo_20261011.py"]
        != OPERATIONAL_HELPER_SHA
        or manifest.get("active_evaluation_cap_seconds") != 1200
        or manifest.get("released_handoff_wait_wall_separate") is not True
        or manifest.get("original_scientific_source_changed") is not False
    ):
        raise ValueError("accepted operational package namespace/cap differs")
    _verified_small(helper_record)
    selection_record = {
        "path": str(package / "selection.json"),
        "sha256": manifest["files"]["selection.json"],
    }
    selection = json.loads(_verified_small(selection_record))
    if selection["original_timed_evaluator"]["sha256"] != EVALUATOR_SHA:
        raise ValueError("original timed evaluator selection differs")
    waiter_record = {
        "path": str(package / "resource-helper-source.py"),
        "sha256": manifest["files"]["resource-helper-source.py"],
    }
    _verified_small(waiter_record)
    spec = importlib.util.spec_from_file_location(
        "_direct_accepted_resource_memo", helper_record["path"]
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    waiter = runpy.run_path(waiter_record["path"])["wait_for_resource_return"]
    # Authenticate the unchanged original gate imported by the accepted waiter.
    for name in ("require", "resource_gate"):
        function = waiter.__globals__[name]
        pipeline_path = Path(function.__globals__["__file__"]).resolve()
        raw = _verified_small({"path": str(pipeline_path), "sha256": PIPELINE_SHA})
        module_code = compile(raw, str(pipeline_path), "exec", dont_inherit=True)
        expected = next(
            code
            for code in module_code.co_consts
            if isinstance(code, types.CodeType) and code.co_name == name
        )
        if function.__code__ != expected:
            raise ValueError("original waiter gate callable differs: " + name)
    tree, counts = helper.derive_op04_timed(source)
    if counts != manifest["timed_changes"] or counts != {
        "released_stage_wait": 1,
        "runtime_provenance": 1,
    }:
        raise ValueError("accepted timed wait AST seam count differs")
    provenance = {
        "profile": helper.OP04_PROFILE,
        "manifest": manifest_record,
        "original_scientific_source": selection["original_timed_evaluator"],
        "operational_helper": helper_record,
        "resource_waiter": waiter_record,
        "active_evaluation_cap_seconds": 1200,
        "released_handoff_wait_wall_separate": True,
    }
    namespace = dict(evaluator.__dict__)
    namespace["_op04_timed_handoff"] = lambda *args: helper.op04_timed_handoff(
        waiter, evaluator.atomic_json, *args
    )
    namespace["_op04_runtime_provenance"] = provenance
    exec(compile(tree, str(evaluator.__file__), "exec"), namespace)
    return types.SimpleNamespace(**namespace), provenance


def manual_exporter(exporter):
    """Private function copies; never mutate imported/frozen module globals."""
    source = _source(exporter, EXPORTER_SHA)
    tree = ast.parse(source)
    function = copy.deepcopy(
        next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_endpoint"
        )
    )
    replacements = 0
    for node in ast.walk(function):
        # The two scheduling-mode collections have distinct original sizes.
        if isinstance(node, ast.Set):
            values = {
                n.value
                for n in node.elts
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
            if values == {"final", "timed", "timed_evidence"} or values == {
                "timed",
                "timed_evidence",
            }:
                node.elts.append(ast.Constant("manual"))
                replacements += 1
            elif values == {"timed_evaluation_boundary", "approved_budget_complete"}:
                node.elts.append(ast.Constant("evaluation_requested"))
                replacements += 1
        if isinstance(node, ast.Compare) and ast.unparse(node) == "checkpoint_mode == 'timed'":
            node.comparators[0] = ast.Constant("manual")
            # Both require identical current controller/parent supervision.
            node.ops = [ast.In()]
            node.comparators = [ast.Set([ast.Constant("timed"), ast.Constant("manual")])]
            replacements += 1
    # Replace only the original milestone request block. All later checkpoint,
    # source/data/precision, optimizer/RNG, NPZ and audit checks stay literal.
    blocks = [
        n
        for n in function.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "timed"
    ]
    if len(blocks) != 1 or replacements != 4:
        raise ValueError("original endpoint scheduling AST differs")
    block = blocks[0]
    block.test = ast.parse("timed and checkpoint_mode != 'manual'", mode="eval").body
    index = function.body.index(block)
    function.body.insert(
        index + 1,
        ast.parse(
            "if checkpoint_mode == 'manual':\n    validate_manual_request(receipt, lane, lane_locator, spec, limits, files)\n    validate_manual_export_inputs(receipt, lane, files)"
        ).body[0],
    )
    ast.fix_missing_locations(function)
    namespace = dict(
        exporter.__dict__,
        validate_manual_request=validate_manual_request,
        validate_manual_export_inputs=validate_manual_export_inputs,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(exporter.__file__), "exec"),
        namespace,
    )
    validator = namespace["validate_endpoint"]
    export_globals = dict(exporter.export_endpoint.__globals__, validate_endpoint=validator)
    export = types.FunctionType(
        exporter.export_endpoint.__code__,
        export_globals,
        exporter.export_endpoint.__name__,
        exporter.export_endpoint.__defaults__,
        exporter.export_endpoint.__closure__,
    )
    export.__kwdefaults__ = exporter.export_endpoint.__kwdefaults__
    return validator, export


def evaluate_manual_boundary(
    *,
    exporter,
    evaluator,
    config,
    lane_locator,
    state_locator,
    supervisor_locator,
    training_locator,
    plan_locator,
    directory,
    files,
    observer,
    runner,
    release,
    authorization,
    stop_path,
):
    """One in-process controller operation, holding its original exclusive lane."""
    evaluator, operational = private_timed_evaluator(evaluator, config)
    validator, export = manual_exporter(exporter)
    context = validator(
        lane_locator,
        state_locator,
        supervisor_locator,
        training_locator,
        checkpoint_mode="manual",
        files=files,
    )
    plan, protocol = evaluator.validate_plan(
        plan_locator, context["lane"], context["admission_plan"], files
    )
    plan = dict(
        plan, frozen_lane=lane_locator, remaining_candidates=context["lane"]["remaining_candidates"]
    )
    request_locator = context["receipt"]["manual_evaluation_request"]
    request = json.loads(files.check(request_locator).read_text())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    def released(_context):
        authorization()
        release()
        return True

    def cpu_admission(_context, _directory):
        import shutil

        snapshot = observer.snapshot()
        evaluator.resource_gate(snapshot, snapshot, plan["resource_policy"])
        size = sum(
            Path(context[key]["path"]).stat().st_size for key in ("checkpoint", "base_model")
        )
        return (
            snapshot["host_available_bytes"] >= size * 2
            and shutil.disk_usage(directory).free >= size * 2
        )

    def cpu_run(command, **kwargs):
        previous = runner.environment
        runner.environment = kwargs["env"]
        try:
            return runner.run(
                command,
                directory=directory / "cpu-process",
                stop_path=stop_path,
                wall_seconds=kwargs["timeout"],
            )
        finally:
            runner.environment = previous

    exported = export(
        context,
        directory / "export",
        release_check=released,
        cpu_admission=cpu_admission,
        wall_seconds=plan["export_wall_seconds"],
        run=cpu_run,
    )
    authorization()
    release()
    # Preserve immutable source snapshots exactly as the timed producer does.
    immutable = dict(
        exported,
        snapshot_ancestry={},
        source_run_directory=str(Path(state_locator["path"]).parent.resolve()),
    )
    for key in ("lane_state", "supervisor_state"):
        destination = directory / "export" / ("immutable-" + key + ".json")
        destination.write_bytes(files.check(exported[key]).read_bytes())
        immutable[key] = evaluator.pin(destination)
        immutable["snapshot_ancestry"][key] = {
            "original": exported[key],
            "snapshot": immutable[key],
        }
    immutable["original_export_receipt"] = evaluator.pin(directory / "export/receipt.json")
    evaluator.atomic_json(directory / "export/research-receipt.json", immutable)
    plan["research_export_receipt"] = evaluator.pin(directory / "export/research-receipt.json")
    plan["execution_context"] = {
        "schema": "native_timed_evaluation_attempt_v1",
        "request": request_locator,
        "training_receipt": training_locator,
        "checkpoint": context["checkpoint"],
        "model": exported["model"],
        "attempt_nonce": directory.name,
        "owner_identity": evaluator.process_identity(os.getpid()),
        "evaluation_directory": str((directory / "evaluation").resolve()),
    }
    report = evaluator.evaluate(
        plan,
        files,
        protocol,
        immutable,
        directory / "evaluation",
        stop_path,
        authorization,
        observer,
        runner,
    )
    authorization()
    proof = release()
    value = json.loads(files.check(report).read_text())
    receipt = {
        "schema": "direct_native_evaluation_receipt_v1",
        "completed": True,
        "status": "FAILED" if value.get("strict_quality_status") == "FAILED" else "PASS",
        "candidate": context["lane"]["candidate"],
        "bundle_sha256": lane_locator["sha256"],
        "config_sha256": context["lane"]["config"]["sha256"],
        "checkpoint": context["checkpoint"],
        "request": request_locator,
        "request_id": request["request_id"],
        "protocol": plan["protocol"],
        "evaluation": report,
        "evaluation_plan": plan_locator,
        "training_receipt": training_locator,
        "owned_release": proof,
        "elapsed_seconds": request["elapsed_seconds"],
        "campaign_complete": False,
        "operational_resource_handoff": operational,
    }
    for key in (
        "measurement_status",
        "strict_quality_status",
        "continuation_status",
        "research_only",
        "research_continuation_policy",
        "research_bindings",
        "execution_context",
    ):
        if key in value:
            receipt[key] = value[key]
    files.check(request_locator)
    evaluator.validate_plan(plan_locator, context["lane"], context["admission_plan"], files)
    evaluator.atomic_json(directory / "receipt.json", receipt)
    return evaluator.pin(directory / "receipt.json")


def load_native_consumers(config):
    """Load only exact admitted source files; no model payloads or GPU query."""
    import importlib.util
    import sys

    sys.dont_write_bytecode = True
    result = []
    for role, expected in (("exporter", EXPORTER_SHA), ("timed_evaluator", EVALUATOR_SHA)):
        locator = config["native_sources"][role]
        path = Path(locator["path"])
        if (
            path != path.resolve(strict=True)
            or locator["sha256"] != expected
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("native consumer source pin differs: " + role)
        spec = importlib.util.spec_from_file_location("_direct_original_" + role, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return tuple(result)


def validate_manual_completed(receipt_locator, config, *, files):
    """Reauthenticate a just-completed manual result before same-state resume.

    Call while the released controller is still at the immutable evaluation
    boundary. Historical compact evidence is a separate reader, never permission
    to restore a pruned payload or bypass a newly pending evaluation.
    """
    exporter, evaluator = load_native_consumers(config)
    result = json.loads(files.check(receipt_locator).read_text())
    if (
        result.get("schema") != "direct_native_evaluation_receipt_v1"
        or result.get("completed") is not True
    ):
        raise ValueError("completed direct native result required")
    report = json.loads(files.check(result["evaluation"]).read_text())
    exported = json.loads(files.check(report["export_receipt"]).read_text())
    originals = json.loads(files.check(exported["original_export_receipt"]).read_text())
    validator, _ = manual_exporter(exporter)
    context = validator(
        exported["frozen_lane"],
        originals["lane_state"],
        originals["supervisor_state"],
        result["training_receipt"],
        checkpoint_mode="manual",
        files=files,
    )
    exporter.validate_export(context, report["export_receipt"], files=files)
    request = json.loads(files.check(result["request"]).read_text())
    if (
        result.get("operational_resource_handoff") != report.get("operational_resource_handoff")
        or result.get("operational_resource_handoff", {}).get("manifest")
        != config["operational_runtime"]["manifest"]
        or result.get("operational_resource_handoff", {}).get("operational_helper")
        != config["operational_runtime"]["helper"]
        or result["checkpoint"] != context["checkpoint"]
        or request["checkpoint"] != result["checkpoint"]
        or request["request_id"] != result["request_id"]
        or context["receipt"]["manual_evaluation_request"] != result["request"]
        or report["execution_context"] != result["execution_context"]
        or report["trained_export"] != exported
        or result.get("owned_release", {}).get("owned_process_groups_absent") is not True
        or result.get("owned_release", {}).get("owned_cuda_pids_absent") is not True
    ):
        raise ValueError("manual evaluated checkpoint/source/release differs")
    from .continuous_budget import research_continuation_decision, validate_native_stages

    raw = json.loads(files.check(report["measurements"]).read_text())
    validate_native_stages(
        result["research_bindings"],
        raw["clean_records"],
        raw["diagnostic_records"],
        report["native_stages"],
        report["model_ancestry"],
        report["producer_sources"],
        candidate_audit=exported["audit"],
        execution_context=report["execution_context"],
    )
    decision = research_continuation_decision(
        result["research_continuation_policy"],
        result["research_bindings"],
        raw["clean_records"],
        raw["diagnostic_records"],
    )
    if any(
        result.get(key) != decision[key] or report.get(key) != decision[key]
        for key in ("measurement_status", "strict_quality_status", "continuation_status")
    ):
        raise ValueError("manual scientific continuation decision differs")
    return result
