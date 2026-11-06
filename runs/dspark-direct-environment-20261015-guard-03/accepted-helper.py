#!/usr/bin/env python3
"""Selected operational replay-audit memo; frozen scientific validators still run.

Only one validate_replay_admission invocation shares byte hashes. Nothing persists
between validations/processes. Every original expected-digest comparison remains.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import os
import importlib.util
import json
from pathlib import Path
import runpy
import stat
import sys
import time
import traceback
import threading
import types

PROFILE = 'replay_audit_scoped_hash_memo_v1'
RESOURCE_DIAGNOSTICS = {
    'profile': 'original_resource_gate_failure_arguments_v1',
    'scope': 'original resource_gate failures; already observed arguments only',
    'extra_sampling': False,
    'thresholds_baselines_retries_changed': False,
    'original_result_exception_retained': True,
    'observer_failures_best_effort': True,
    'observer_error_scope': 'BaseException after original gate failure',
}


class ReplayAuditHashes:
    def __init__(self, read_hash, fingerprint):
        self.read_hash, self.fingerprint = read_hash, fingerprint
        self.entries = {}
        self.reads = self.hits = self.bytes_read = 0

    def __call__(self, path):
        path = Path(path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError('canonical replay audit file required')
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('regular replay audit file required')
        identity = self.fingerprint(before)
        previous = self.entries.get(str(path))
        if previous is not None:
            if previous[0] != identity:
                raise ValueError('replay audit artifact identity changed')
            digest = previous[1]
            self.hits += 1
        else:
            digest = self.read_hash(path)
            if self.fingerprint(path.lstat()) != identity or path != path.resolve():
                raise ValueError('replay audit artifact changed during hash')
            self.entries[str(path)] = (identity, digest)
            self.reads += 1
            self.bytes_read += before.st_size
        if self.fingerprint(path.lstat()) != identity or path != path.resolve():
            raise ValueError('replay audit artifact changed during cache check')
        return digest


def install(lifecycle, fingerprint, observe=None):
    original_validation = lifecycle.validate_replay_admission
    original_hash = lifecycle.file_sha256

    def validate(*args, **kwargs):
        # Fresh full-byte validation every invocation, including after resume.
        started = time.monotonic()
        memo = ReplayAuditHashes(original_hash, fingerprint)
        if lifecycle.file_sha256 is not original_hash:
            raise ValueError('unexpected replay audit hash implementation')
        lifecycle.file_sha256 = memo
        try:
            return original_validation(*args, **kwargs)
        finally:
            lifecycle.file_sha256 = original_hash
            if observe is not None:
                try:
                    observe({'profile': PROFILE, 'hash_reads': memo.reads, 'hash_reuses': memo.hits,
                             'bytes_hashed': memo.bytes_read, 'validation_wall_seconds': time.monotonic() - started})
                except Exception:
                    # Diagnostic observer failures cannot replace an original scientific
                    # error or interrupt healthy validation. Metrics then unavailable.
                    pass
    lifecycle.validate_replay_admission = validate




POLICY_PROFILE = 'research_policy_scoped_actual_digest_memo_v1'
POLICY_BUDGET_SHA256 = 'c10e0c7b6a3c8589a6040af994cbb1e30c280e4dc1a364b3b44f567fd96819bf'
OP04_POLICY_BUDGET_SHA256 = '9cbc775931b1f2739c8a9b7cad9e98f746449000cce06dd8b9b6792b74eb655e'
POLICY_PROVENANCE = {
    'profile': POLICY_PROFILE,
    'scope': 'one original validate_research_policy invocation',
    'original_budget_sha256': POLICY_BUDGET_SHA256,
    'replacement': 'source-pinned checked_locator AST digest calculation only',
    'key': 'canonical path and strict identity; actual digest; original expected-SHA comparison every call',
    'fingerprint': 'original Files nine fields before/fd/after read and each cache hit',
    'persistent_hash_cache': False,
    'original_expected_digest_checks_retained': True,
    'frozen_scientific_source_changed': False,
    'runtime_implementation_changed': True,
}


class PolicyDigest:
    def __init__(self, value):
        self.value = value

    def hexdigest(self):
        return self.value


class PolicyHashes:
    """Cache actual bytes' digest, so even invalid/unhashable expected SHA compares."""
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.entries = {}
        self.reads = self.hits = self.bytes_read = 0

    def __call__(self, path):
        path = Path(path)
        if not path.is_absolute() or path != path.resolve():
            raise ValueError('canonical policy artifact required')
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('regular policy artifact required')
        identity = self.fingerprint(before)
        previous = self.entries.get(str(path))
        if previous is not None:
            if previous[0] != identity:
                raise ValueError('policy artifact identity changed')
            result = previous[1]
            self.hits += 1
        else:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                if self.fingerprint(os.fstat(stream.fileno())) != identity:
                    raise ValueError('policy artifact replaced while opening')
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                if self.fingerprint(os.fstat(stream.fileno())) != identity:
                    raise ValueError('policy artifact changed during read')
            if path != path.resolve() or self.fingerprint(path.lstat()) != identity:
                raise ValueError('policy artifact changed during hash')
            result = PolicyDigest(digest.hexdigest())
            self.entries[str(path)] = (identity, result)
            self.reads += 1
            self.bytes_read += before.st_size
        if path != path.resolve() or self.fingerprint(path.lstat()) != identity:
            raise ValueError('policy artifact changed during cache check')
        return result


def policy_checker(source, memo, *, expected_source_sha256=POLICY_BUDGET_SHA256):
    """Preserve exact original guards/comparison/return; replace only SHA reading."""
    if (expected_source_sha256 not in (POLICY_BUDGET_SHA256, OP04_POLICY_BUDGET_SHA256)
            or hashlib.sha256(source).hexdigest() != expected_source_sha256):
        raise ValueError('original policy budget source SHA differs')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checked_locator')
    original = ast.dump(function, include_attributes=False)
    if len(function.body) != 8 or not isinstance(function.body[4], ast.Assign) or not isinstance(function.body[5], ast.With):
        raise ValueError('original checked_locator digest AST structure differs')
    # The pinned source places digest calculation at statements4/5, followed by
    # its untouched hexdigest expected-SHA comparison and exact Path return.
    expected = ast.parse("digest = hashlib.sha256()\nwith path.open('rb') as stream:\n    for block in iter(lambda: stream.read(1024 * 1024), b''):\n        digest.update(block)").body
    if [ast.dump(n, include_attributes=False) for n in function.body[4:6]] != [ast.dump(n, include_attributes=False) for n in expected]:
        raise ValueError('original inline SHA digest block differs')
    replacement = ast.parse('digest = _policy_digest(path)').body[0]
    ast.copy_location(replacement, function.body[4])
    function.body[4:6] = [replacement]
    ast.fix_missing_locations(function)
    scope = {'Path': Path, '_policy_digest': memo}
    filename = '<' + POLICY_PROFILE + ':checked_locator AST from ' + expected_source_sha256 + '>'
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), scope)
    return scope['checked_locator'], original


def install_policy_memo(budget, fingerprint, *, observe=None, timed_module=None,
                        expected_source_sha256=POLICY_BUDGET_SHA256):
    source = Path(budget.__file__).read_bytes()
    # Authenticate/derive before altering any live module binding.
    # Preserve the legacy call shape for existing admitted packages.
    def derive(memo):
        if expected_source_sha256 == POLICY_BUDGET_SHA256:
            return policy_checker(source, memo)
        return policy_checker(source, memo, expected_source_sha256=expected_source_sha256)
    derive(None)
    original_validator, original_checker = budget.validate_research_policy, budget.checked_locator
    if getattr(original_validator, '_policy_memo_profile', None) is not None:
        raise ValueError('policy memo already installed')
    # Authenticate the live callables too: a pinned file alone must not authorize
    # replacing an already modified in-memory validator or locator guard.
    def signature(code):
        return (code.co_code, tuple(signature(value) if isinstance(value, types.CodeType)
                    else value for value in code.co_consts), code.co_names,
                code.co_varnames, code.co_argcount, code.co_kwonlyargcount,
                code.co_posonlyargcount, code.co_flags, code.co_freevars, code.co_cellvars)
    codes = {value.co_name: value for value in compile(source, str(budget.__file__), 'exec').co_consts
             if isinstance(value, types.CodeType)}
    for name, function in (('checked_locator', original_checker),
                           ('validate_research_policy', original_validator)):
        if (not isinstance(function, types.FunctionType) or function.__globals__ is not budget.__dict__
                or signature(function.__code__) != signature(codes[name])):
            raise ValueError('original policy callable implementation differs: ' + name)
    if timed_module is not None and timed_module.validate_research_policy is not original_validator:
        raise ValueError('known timed policy validator alias differs')

    active = threading.Lock()

    def validate(*args, **kwargs):
        # Guard acquisition precedes AST construction/global inspection: no two
        # concurrent calls can share or restore another invocation's memo.
        acquired = False
        try:
            acquired = active.acquire(blocking=False)
            if not acquired:
                raise ValueError('overlapping policy validation invocation')
            if budget.checked_locator is not original_checker:
                raise ValueError('overlapping policy checked_locator implementation')
            memo = PolicyHashes(fingerprint)
            checker, _ = derive(memo)
            started = time.monotonic()
            try:
                budget.checked_locator = checker
                return original_validator(*args, **kwargs)
            finally:
                budget.checked_locator = original_checker
                if observe is not None:
                    try:
                        observe({'profile': POLICY_PROFILE, 'hash_reads': memo.reads,
                                 'hash_reuses': memo.hits, 'bytes_hashed': memo.bytes_read,
                                 'validation_wall_seconds': time.monotonic() - started})
                    except BaseException:
                        pass
        finally:
            if acquired:
                active.release()
    validate._policy_memo_profile = POLICY_PROFILE
    budget.validate_research_policy = validate
    if timed_module is not None:
        timed_module.validate_research_policy = validate
    return validate


def install_resource_diagnostics(pipeline, observe):
    """Retain a failed gate's existing sample; never query or alter admission."""
    original_gate = pipeline.resource_gate

    def resource_gate(observed, baseline, policy):
        try:
            return original_gate(observed, baseline, policy)
        except BaseException as error:
            # The raw snapshot is otherwise lost when callers publish only after
            # the gate. Keep the original argument objects and bare reraising;
            # diagnostic construction/serialization/sink failures are secondary.
            try:
                caller = sys._getframe(1)
                observe({
                    'profile': RESOURCE_DIAGNOSTICS['profile'],
                    'observed': observed, 'baseline': baseline, 'policy': policy,
                    'callsite': {'path': caller.f_code.co_filename,
                                 'function': caller.f_code.co_name,
                                 'line': caller.f_lineno},
                    'failure': {'type': type(error).__name__, 'reason': str(error)},
                    'original_traceback': [
                        {'path': frame.f_code.co_filename, 'function': frame.f_code.co_name, 'line': line}
                        for frame, line in traceback.walk_tb(error.__traceback__)
                    ],
                })
            except BaseException:
                # An observer's SystemExit/KeyboardInterrupt is secondary too.
                # The original gate failure (including a real stop) wins.
                pass
            raise
    pipeline.resource_gate = resource_gate


def historical_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('original', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original = args.original[1:] if args.original[:1] == ['--'] else args.original
    wrapper_path = Path(__file__).resolve().with_name('exec_frozen_dspark_lane_20261010.py')
    spec = importlib.util.spec_from_file_location('continuity_package', wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    root, selection = wrapper.load_package(args.manifest, args.manifest_sha256)
    wrapper.need(selection.get('operational_validation_profile') == PROFILE, 'memo profile not selected')
    effective = wrapper.small_json(args.manifest)['effective_validation']
    wrapper.need(effective == {'profile': PROFILE, 'scope': 'one original validate_replay_admission invocation',
        'original_expected_digest_checks_retained': True, 'persistent_hash_cache': False,
        'frozen_scientific_source_changed': False, 'runtime_implementation_changed': True},
        'effective validation provenance differs')
    wrapper.need(wrapper.small_json(args.manifest).get('resource_diagnostics') == RESOURCE_DIAGNOSTICS
                 and selection.get('resource_diagnostics') == RESOURCE_DIAGNOSTICS,
                 'resource failure diagnostics provenance differs')
    wrapper.need(wrapper.small_json(args.manifest).get('policy_hash_memo') == selection.get('policy_hash_memo')
                 and selection.get('policy_hash_memo') in (None, POLICY_PROVENANCE),
                 'policy hash memo selection/provenance differs')
    wrapper.need(Path(__file__).resolve() == root / 'scripts/run_frozen_dspark_hash_memo_20261011.py', 'invoke packaged memo bootstrap')
    wrapper.need(original[:1] == [selection['runner']['path']], 'original frozen runner selection differs')
    for record in selection['memo_source'].values():
        wrapper.authenticate(record['path'], record['sha256'])
    wrapper.authenticate(selection['runner']['path'], selection['runner']['sha256'])
    source = Path(selection['runner']['path']).parents[1]
    sys.path[:0] = [str(source / 'src'), str(source / 'scripts')]
    from w1a1_eagle import block_shard_lifecycle as lifecycle
    from w1a1_eagle.nine_model_pipeline import Files
    wrapper.need(Path(lifecycle.__file__).resolve() == Path(selection['memo_source']['lifecycle']['path']), 'wrong lifecycle imported')
    wrapper.need(Path(sys.modules[Files.__module__].__file__).resolve() == Path(selection['memo_source']['pipeline']['path']), 'wrong pipeline imported')
    if selection.get('device_admission_view'):
        wrapper.need(wrapper.small_json(args.manifest).get('device_admission_view') == {
            'profile': 'original_trial_admission_device_view_v1', 'input': selection['device_admission_view'],
            'scope': 'original0410 trial only; parent processes', 'original_digest_comparisons_retained': True,
            'fingerprint_function_changed': False, 'fresh_original_reader_audit_required': True, 'child_bootstrap': False},
            'explicit trial device view provenance differs')
        view_path = root / 'scripts/refresh_block_replay_publication_admission.py'
        spec = importlib.util.spec_from_file_location('original_trial_device_view', view_path)
        view_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(view_module)
        from w1a1_eagle import block_data
        view_module.install_device_admission_view(block_data, selection['device_admission_view'])
    if selection.get('policy_hash_memo') is not None:
        wrapper.need(selection['policy_hash_memo'] == POLICY_PROVENANCE
                     and wrapper.small_json(args.manifest).get('policy_hash_memo') == POLICY_PROVENANCE,
                     'policy hash memo provenance differs')
        from w1a1_eagle import continuous_budget as budget
        wrapper.need(Path(budget.__file__).resolve() == Path(selection['memo_source']['budget']['path']),
                     'wrong policy budget imported')
        install_policy_memo(budget, Files.fingerprint,
            timed_module=sys.modules.get('evaluate_nine_model_timed_checkpoint'),
            observe=lambda event: print(json.dumps({'policy_hash_memo':event}),file=sys.stderr,flush=True))
    pipeline = sys.modules[Files.__module__]
    install_resource_diagnostics(pipeline, observe=lambda event: print(
        json.dumps({'original_resource_gate_failure': event}, allow_nan=False), file=sys.stderr, flush=True))
    install(lifecycle, Files.fingerprint, observe=lambda event: print(json.dumps({'replay_audit': event}), file=sys.stderr, flush=True))
    # Kernel argv explicitly selects this bootstrap; Python argv is the exact
    # original runner/flags. exec by the launch wrapper preserves PID/PPID/PGID.
    sys.argv = original
    runpy.run_path(original[0], run_name='__main__')


OP04_PROFILE = "same_op04_pending1800_resource_handoff_v1"
OP04_LANE = "478f3292f8fe2a7b31f9960937816be4a565658cd39dbb72adf0a23a9cd0a155"
OP04_RUNNER = "5d5c390b0f1a6f8e5347b4afb2ff2625033bbc9388fb0f2dda9227b72dc9511a"
OP04_PIPELINE = "a27e9121261938175d88fb78651b2f02fa0a641d4b5afccc2197e561399979af"
OP04_POLICY_PROVENANCE = dict(POLICY_PROVENANCE, original_budget_sha256=OP04_POLICY_BUDGET_SHA256)
OP04_COMPLETED_PHASE = "completed1800_train_resume_v1"
OP04_SHARD_PHASE = "shard_checkpoint_resume_v1"
OP04_TRAINER_SHA256 = "f6b2535d3d18c1f4df8b5e6abd65d988a17a08dbc80208ca45bd4cfd1f640a58"
OP04_CHILD_PROVENANCE = {
    "profile": "op04_trainer_child_scoped_policy_memo_v1",
    "policy_hash_memo": OP04_POLICY_PROVENANCE,
    "trainer_sha256": OP04_TRAINER_SHA256,
    "original_python_argv_retained": True,
    "kernel_argv_explicit_bootstrap": True,
    "environment_changed": False,
    "additional_process": False,
    "persistent_hash_cache": False,
}


def validate_op04_child_selection(manifest, bindings, lane):
    selected = bindings.get("trainer_policy_hash_memo")
    if selected != manifest.get("trainer_policy_hash_memo") or selected not in (None, OP04_CHILD_PROVENANCE):
        raise ValueError("OP04 trainer child memo selection differs")
    if selected is not None:
        if (bindings.get("policy_hash_memo") != OP04_POLICY_PROVENANCE
                or lane["source"].get("scripts/train_nine_model_qat.py", {}).get("sha256") != OP04_TRAINER_SHA256):
            raise ValueError("OP04 trainer child source/profile differs")
    return selected


def validate_op04_trainer_argv(argv, lane, bindings):
    """Only the original resumed production trainer command can select bootstrap."""
    trainer = lane["source"]["scripts/train_nine_model_qat.py"]
    if argv[:2] != [sys.executable, trainer["path"]]:
        raise ValueError("original OP04 trainer interpreter/script differs")
    paired = {"--config", "--run-dir", "--bundle-sha256", "--stage-name",
              "--completion-output", "--admission", "--evaluation-receipt", "--evaluation-receipt-sha256"}
    options, flags, index = {}, set(), 2
    while index < len(argv):
        key = argv[index]
        if key in paired and key not in options and index + 1 < len(argv):
            options[key] = argv[index + 1]
            index += 2
        elif key in {"--allow-cuda", "--resume"} and key not in flags:
            flags.add(key)
            index += 1
        else:
            raise ValueError("original OP04 trainer argument schema differs")
    original = bindings["original_argv"]
    run_dir = original[original.index("--run-dir") + 1]
    output = Path(options.get("--completion-output", ""))
    if (set(options) != paired or flags != {"--allow-cuda", "--resume"}
            or options["--config"] != lane["config"]["path"]
            or options["--run-dir"] != str(Path(run_dir) / "training")
            or options["--bundle-sha256"] != OP04_LANE
            or options["--stage-name"] != "dspark_a8/train"
            or options["--admission"] != bindings["metadata"]["admission"]["path"]
            or not output.is_absolute() or output.name != "train-receipt.json"
            or Path(run_dir) / "attempts" not in output.parents):
        raise ValueError("original OP04 resumed trainer scope differs")
    op04_read(trainer)
    op04_read(lane["config"])
    op04_read(bindings["metadata"]["admission"])
    result = json.loads(op04_read({"path": options["--evaluation-receipt"],
                                   "sha256": options["--evaluation-receipt-sha256"]}))
    if result.get("bundle_sha256") != OP04_LANE or result.get("candidate") != "dspark_a8":
        raise ValueError("original OP04 trainer evaluation scope differs")


def install_op04_trainer_bootstrap(manifest, bindings, lane, pipeline, manifest_path, manifest_sha):
    if validate_op04_child_selection(manifest, bindings, lane) is None:
        return
    original = pipeline.SubprocessRunner.run
    if getattr(original, "_op04_child_bootstrap", False):
        raise ValueError("OP04 trainer bootstrap already installed")
    if (getattr(original, "__globals__", None) is not pipeline.__dict__
            or original.__qualname__ != "SubprocessRunner.run"
            or Path(original.__code__.co_filename).resolve() != Path(pipeline.__file__).resolve()):
        raise ValueError("original OP04 subprocess consumer differs")
    compiled = compile(op04_read(bindings["original_pipeline"]), str(pipeline.__file__), "exec")
    consumer = next(c for c in compiled.co_consts if isinstance(c, types.CodeType)
                    and c.co_name == "SubprocessRunner")
    expected = next(c for c in consumer.co_consts if isinstance(c, types.CodeType) and c.co_name == "run")
    if original.__code__ != expected:
        raise ValueError("original OP04 subprocess implementation differs")
    trainer_path = lane["source"]["scripts/train_nine_model_qat.py"]["path"]

    def run(self, argv, **kwargs):
        if len(argv) > 1 and argv[1] == trainer_path:
            validate_op04_trainer_argv(argv, lane, bindings)
            argv = [argv[0], str(Path(__file__).resolve()), "--op04-trainer-manifest",
                    str(manifest_path), "--manifest-sha256", manifest_sha, "--", *argv]
        return original(self, argv, **kwargs)
    run._op04_child_bootstrap = True
    pipeline.SubprocessRunner.run = run


def op04_trainer_main(argv):
    parser = argparse.ArgumentParser(description="Source-pinned original OP04 trainer child bootstrap")
    parser.add_argument("--op04-trainer-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("original", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    manifest = json.loads(op04_read({"path": str(args.op04_trainer_manifest), "sha256": args.manifest_sha256}))
    root = args.op04_trainer_manifest.resolve().parent
    if (manifest.get("profile") != OP04_PROFILE
            or manifest.get("schema") != "frozen_dspark_op04_resource_package_v1"
            or Path(__file__).resolve() != root / "scripts/run_frozen_dspark_hash_memo_20261011.py"):
        raise ValueError("authenticated original OP04 trainer carrier required")
    inventory = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if inventory != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("exact OP04 trainer carrier closure required")
    for name, digest in manifest["files"].items():
        op04_read({"path": str(root / name), "sha256": digest})
    bindings = json.loads((root / "selection.json").read_text())
    lane = json.loads(op04_read(bindings["metadata"]["lane"]))
    if (bindings["metadata"]["lane"]["sha256"] != OP04_LANE
            or lane.get("candidate") != "dspark_a8"
            or bindings["original_pipeline"] != lane["source"]["src/w1a1_eagle/nine_model_pipeline.py"]
            or bindings["original_pipeline"]["sha256"] != OP04_PIPELINE):
        raise ValueError("original OP04 child lane/pipeline differs")
    validate_op04_policy_memo_selection(manifest, bindings, lane)
    if validate_op04_child_selection(manifest, bindings, lane) is None:
        raise ValueError("OP04 trainer bootstrap not selected")
    original = args.original[1:] if args.original[:1] == ["--"] else args.original
    validate_op04_trainer_argv(original, lane, bindings)
    source = Path(original[1]).parents[1]
    op04_read(lane["source"]["src/w1a1_eagle/continuous_budget.py"])
    op04_read(bindings["original_pipeline"])
    sys.path[:0] = [str(source / "src"), str(source / "scripts")]
    from w1a1_eagle import continuous_budget as budget, nine_model_pipeline as pipeline
    if (Path(budget.__file__).resolve() != Path(lane["source"]["src/w1a1_eagle/continuous_budget.py"]["path"])
            or Path(pipeline.__file__).resolve() != Path(bindings["original_pipeline"]["path"])):
        raise ValueError("wrong original OP04 child modules imported")
    install_policy_memo(budget, pipeline.Files.fingerprint, expected_source_sha256=OP04_POLICY_BUDGET_SHA256,
                        observe=lambda event: print(json.dumps({"trainer_policy_hash_memo": event}), file=sys.stderr, flush=True))
    sys.argv = original[1:]
    runpy.run_path(original[1], run_name="__main__")


def validate_op04_phase_selection(manifest, bindings):
    phase = bindings.get("resume_phase")
    if phase != manifest.get("resume_phase") or phase not in (None, OP04_COMPLETED_PHASE, OP04_SHARD_PHASE):
        raise ValueError("OP04 resume phase selection differs")
    if phase is not None and bindings.get("policy_hash_memo") != OP04_POLICY_PROVENANCE:
        raise ValueError("completed OP04 phase requires selected current-source policy memo")
    if phase == OP04_SHARD_PHASE and bindings.get("trainer_policy_hash_memo") != OP04_CHILD_PROVENANCE:
        raise ValueError("shard OP04 phase requires selected trainer child memo")
    return phase


def validate_op04_policy_memo_selection(manifest, bindings, lane):
    """Check optional provenance/source record without imports or model reads."""
    selected = bindings.get("policy_hash_memo")
    if selected != manifest.get("policy_hash_memo") or selected not in (None, OP04_POLICY_PROVENANCE):
        raise ValueError("OP04 policy hash memo selection/provenance differs")
    if selected is None:
        return
    record = lane.get("source", {}).get("src/w1a1_eagle/continuous_budget.py")
    if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
            or not isinstance(record["path"], str) or not Path(record["path"]).is_absolute()
            or record["sha256"] != OP04_POLICY_BUDGET_SHA256):
        raise ValueError("original OP04 policy budget source required")
    return record


def install_op04_policy_memo(manifest, bindings, lane, pipeline, timed):
    """Explicit new-carrier opt-in; old recovery packets retain their behavior."""
    record = validate_op04_policy_memo_selection(manifest, bindings, lane)
    if record is None:
        return
    op04_read(record)
    from w1a1_eagle import continuous_budget as budget
    if Path(budget.__file__).resolve() != Path(record["path"]):
        raise ValueError("wrong original OP04 policy budget imported")
    return install_policy_memo(
        budget, pipeline.Files.fingerprint, timed_module=timed,
        expected_source_sha256=OP04_POLICY_BUDGET_SHA256,
        observe=lambda event: print(json.dumps({"policy_hash_memo": event}), file=sys.stderr, flush=True),
    )


def op04_read(record):
    path = Path(record["path"])
    if (
        not path.is_absolute()
        or path != path.resolve()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 8 * 1024**2
    ):
        raise ValueError("bounded canonical OP04 metadata/source required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("OP04 source/metadata digest differs: " + str(path))
    return raw


def validate_op04_metadata(bindings, records):
    """Only authenticated small records; never reads checkpoint/model payloads."""
    if bindings.get("resume_phase") == OP04_SHARD_PHASE:
        return validate_op04_shard_boundary(bindings, records)
    lane, state, endpoint, request, admission, ledger = [
        json.loads(records[k])
        for k in ("lane", "state", "endpoint", "request", "admission", "ledger")
    ]
    if (
        hashlib.sha256(records["lane"]).hexdigest() != OP04_LANE
        or lane["candidate"] != "dspark_a8"
        or state["bundle_sha256"] != OP04_LANE
        or (bindings.get("resume_phase") is None
            and state["pending_training"] != bindings["metadata"]["endpoint"])
        or bindings.get("resume_phase") not in (None, OP04_COMPLETED_PHASE)
        or state["admission"] != bindings["metadata"]["admission"]
        or endpoint["timed_evaluation_request"] != bindings["metadata"]["request"]
        or endpoint["bundle_sha256"] != OP04_LANE
        or request["bundle_sha256"] != OP04_LANE
        or request["checkpoint"]["sha256"]
        != "096801ec145a97c99298deed82e255bdc34dd8f526f74acfa06df96857a3feeb"
        or endpoint["checkpoint"]["sha256"] != request["checkpoint"]["sha256"]
        or endpoint["completion_reason"] != "timed_evaluation_boundary"
        or endpoint["committed"] is not True
        or endpoint["status"] != "PASS"
        or request["milestone_index"] != 0
        or request["milestone_seconds"] != 1800
        or request["final_training_complete"] is not False
        or ledger["active_attempt"] is not None
        or ledger["source_sha256"] != OP04_LANE
        or ledger["training_seconds"] != 1800.5557143019978
        or ledger["max_seconds"] != 43200
        or request["elapsed_seconds"] != ledger["training_seconds"]
        or admission["bundle_sha256"] != OP04_LANE
        or admission["candidate"] != "dspark_a8"
        or admission["status"] != "PASS"
        or any(v != "PASS" for v in admission["checks"].values())
        or admission["source_inventory"] != lane["source"]
        or admission["compute_capability"] != [12, 0]
        or admission["artifact_kind"] != "production"
        or admission["schema"] != "nine_model_training_admission_v1"
        or admission["config_sha256"] != lane["config"]["sha256"]
        or endpoint["config_sha256"] != lane["config"]["sha256"]
        or request["config_sha256"] != lane["config"]["sha256"]
        or admission["gpu_uuid"] != lane["gpu_uuid"]
        or endpoint["hardware"]["gpu_uuid"] != lane["gpu_uuid"]
        or lane["source"]["scripts/run_nine_model_lane.py"]["sha256"] != OP04_RUNNER
        or lane["source"]["src/w1a1_eagle/nine_model_pipeline.py"]["sha256"] != OP04_PIPELINE
    ):
        raise ValueError("same committed OP04 pending1800/admission/ledger contract differs")
    if (
        set(admission["checks"])
        != {"backward", "capture_portability", "kernel", "memory", "model", "resource", "source"}
        or state["resource_baseline"]["gpu_uuid"] != lane["gpu_uuid"]
        or not state["resource_baseline"].get("boot_id")
        or bindings["original_runner"] != lane["source"]["scripts/run_nine_model_lane.py"]
        or bindings["original_pipeline"] != lane["source"]["src/w1a1_eagle/nine_model_pipeline.py"]
    ):
        raise ValueError("original OP04 admission/source/baseline contract differs")
    if bindings.get("resume_phase") == OP04_COMPLETED_PHASE:
        validate_op04_completed_boundary(bindings, records, lane, state, endpoint, request)
    return lane, state


OP04_RESUME_QUALITY_STATE = {'profile': 'original5d_resumed_shard_summary_omission_v1', 'runner_sha256': '5d5c390b0f1a6f8e5347b4afb2ff2625033bbc9388fb0f2dda9227b72dc9511a', 'scope': 'all four omitted state summaries only; same retained original evaluation receipt/history', 'state_or_receipt_rewritten': False, 'receipt_report_quality_checks_retained': True}

OP04_CURRENT09_RESUME_QUALITY_STATE = {
    "profile": "original5d_current09_exact_two_omission_nodes_v2",
    "runner_sha256": "5d5c390b0f1a6f8e5347b4afb2ff2625033bbc9388fb0f2dda9227b72dc9511a",
    "predecessor_profile": "exact_current93881_handoff_v1",
    "scope": "exact current09 two omitted nodes; original retained FAILED quality anchor",
    "state_or_receipt_rewritten": False,
    "receipt_report_quality_checks_retained": True,
}

def op04_current09_shard_quality_state(bindings, state, metadata, lane):
    expected = {"evaluation_status": "FAILED", "measurement_status": "MEASUREMENT_COMPLETE",
                "strict_quality_status": "FAILED", "continuation_status": "RESEARCH_CONTINUATION_ALLOWED"}
    predecessor = bindings.get("handoff_predecessor", {})
    if (bindings.get("resume_quality_state") != OP04_CURRENT09_RESUME_QUALITY_STATE
            or predecessor.get("profile") != "exact_current93881_handoff_v1"
            or predecessor.get("boot_id") != "7f6f4496-6b72-494a-8331-8130e942b4f0"
            or predecessor.get("supervisor") != {"pid": 93881, "birth_ticks": 22892480}
            or predecessor.get("controller") != {"pid": 93891, "birth_ticks": 22892507}
            or predecessor.get("minimum_updates_exclusive") != 5598
            or lane["source"]["scripts/run_nine_model_lane.py"]["sha256"] != OP04_RUNNER
            or state.get("status") not in ("training", "restoring_shards", "stopped", "failed")):
        raise ValueError("exact current09 two-level omission selection differs")
    previous = state.get("previous_attempt", {})
    anchor = previous.get("previous_attempt", {}) if isinstance(previous, dict) else {}
    if (not isinstance(previous, dict) or not isinstance(anchor, dict)
            or any(key in node for node in (state, previous) for key in expected)
            or any(node.get("schema") != "nine_model_lane_state_v1"
                   or node.get("bundle_sha256") != OP04_LANE or node.get("candidate") != "dspark_a8"
                   or node.get("evaluation_receipt") != metadata["evaluation_receipt"]
                   or node.get("evaluations") != [metadata["evaluation_receipt"]]
                   for node in (state, previous, anchor))
            or any(node.get("status") != "failed" for node in (previous, anchor))
            or any(anchor.get(key) != value for key, value in expected.items())):
        raise ValueError("exact current09 retained two-level quality/history differs")
    return anchor


def op04_shard_quality_state(bindings, state, metadata, lane):
    expected = {"evaluation_status": "FAILED", "measurement_status": "MEASUREMENT_COMPLETE",
                "strict_quality_status": "FAILED", "continuation_status": "RESEARCH_CONTINUATION_ALLOWED"}
    if all(key in state for key in expected):
        return state
    if bindings.get("resume_quality_state") == OP04_CURRENT09_RESUME_QUALITY_STATE:
        return op04_current09_shard_quality_state(bindings, state, metadata, lane)
    if (bindings.get("resume_quality_state") != OP04_RESUME_QUALITY_STATE
            or bindings.get("handoff_predecessor", {}).get("profile") != "exact_current91602_handoff_v1"
            or any(key in state for key in expected)
            or lane["source"]["scripts/run_nine_model_lane.py"]["sha256"] != OP04_RUNNER
            or state.get("status") not in ("training", "restoring_shards", "stopped", "failed")):
        raise ValueError("original5d resumed shard summary omission contract differs")
    previous = state.get("previous_attempt", {})
    if (not isinstance(previous, dict)
            or previous.get("schema") != "nine_model_lane_state_v1"
            or previous.get("bundle_sha256") != OP04_LANE or previous.get("candidate") != "dspark_a8"
            or previous.get("evaluation_receipt") != metadata["evaluation_receipt"]
            or previous.get("evaluations") != [metadata["evaluation_receipt"]]
            or state.get("evaluation_receipt") != previous["evaluation_receipt"]
            or state.get("evaluations") != previous["evaluations"]
            or any(previous.get(key) != value for key, value in expected.items())):
        raise ValueError("original5d retained evaluation summary/history differs")
    return previous


def validate_op04_shard_boundary(bindings, records):
    """Join a real settled shard checkpoint; original capture/restore gates run."""
    names = ("lane", "state", "endpoint", "request", "admission", "ledger",
             "evaluation_receipt", "evaluation_report", "timed_state", "latest", "training_status")
    lane, state, endpoint, request, admission, ledger, result, report, timed, latest, status = [
        json.loads(records[k]) for k in names]
    metadata = bindings["metadata"]
    quality_state = op04_shard_quality_state(bindings, state, metadata, lane)
    checkpoint = {k: latest[k] for k in ("path", "sha256")}
    counters = endpoint["counters"]
    shard_request = endpoint["shard_request"]
    result_locator = metadata["evaluation_receipt"]
    source_pins = {
        "scripts/run_nine_model_lane.py": OP04_RUNNER,
        "src/w1a1_eagle/nine_model_pipeline.py": OP04_PIPELINE,
        "src/w1a1_eagle/continuous_budget.py": OP04_POLICY_BUDGET_SHA256,
        "scripts/train_nine_model_qat.py": OP04_TRAINER_SHA256,
        "scripts/evaluate_nine_model_timed_checkpoint.py": "37fad07fa309d1be92203fefda303d3b370acb0a8cd8156327a8b12f5a2f8bdf",
        "scripts/export_nine_model_lane_candidate.py": "10ba61f9269256c048ccfac82369a1f49fe569165ffbf80dd2c9548eb3cb9fab",
        "scripts/run_nine_model_lane_endpoint.py": "d7bdfb257bc4d598e9a899914a1f4e879a2769b0656f1cd4768752c43a940fd9",
    }
    # The actual original SIGTERM producer retains pending fields and writes
    # these exact stop fields in its exception/finally path. Actual post-stop
    # bytes must be collected and selected; never replay the pre-stop snapshot.
    stopped = (
        state.get("status") == "stopped"
        and state.get("failure") == {"type": "InterruptedError", "reason": "campaign signal 15"}
        and state.get("checkpoints_deleted") is False
        and "cleanup_failure" not in state
        and state.get("pending_training") in (None, metadata["endpoint"])
    )
    if (
        hashlib.sha256(records["lane"]).hexdigest() != OP04_LANE
        or lane.get("candidate") != "dspark_a8"
        or state.get("schema") != "nine_model_lane_state_v1"
        or state.get("bundle_sha256") != OP04_LANE or state.get("candidate") != "dspark_a8"
        or state.get("campaign_complete") is not False
        or state.get("pending_shard_resume") is not True
        or not ((state.get("status") == "restoring_shards" and state.get("pending_training") == metadata["endpoint"])
                or (state.get("status") == "training" and state.get("pending_training") is None) or stopped)
        or state.get("train_receipt") != metadata["endpoint"]
        or state.get("admission") != metadata["admission"]
        or metadata["admission"]["sha256"] != "8895c1f04c7518eea79e9a1a6081d6997a1742c6c041453f073c3beeefb7bcd8"
        or any(admission.get(k) != v for k, v in dict(schema="nine_model_training_admission_v1", status="PASS",
                artifact_kind="production", bundle_sha256=OP04_LANE, candidate="dspark_a8",
                config_sha256=lane["config"]["sha256"], gpu_uuid=lane["gpu_uuid"], compute_capability=[12, 0]).items())
        or admission.get("checks") != dict.fromkeys(("backward", "capture_portability", "kernel", "memory", "model", "resource", "source"), "PASS")
        or admission.get("source_inventory") != lane["source"]
        or any(endpoint.get(k) != v for k, v in dict(schema="nine_model_stage_receipt_v1", status="PASS", committed=True,
                artifact_kind="production", completion_reason="shard_required", stage="dspark_a8/train",
                bundle_sha256=OP04_LANE, config_sha256=lane["config"]["sha256"]).items())
        or endpoint.get("checkpoint") != checkpoint or latest.get("schema") != "block_qat_checkpoint_v1"
        or latest.get("committed") is not True or latest.get("cursor") != counters
        or latest.get("source_sha256") != hashlib.sha256(json.dumps(
            {**admission["source"], "bundle_sha256": OP04_LANE}, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        or type(counters.get("step")) is not int or counters["step"] <= request["counters"]["step"]
        or counters.get("stage") != "direct" or counters.get("batch_reservation") != shard_request.get("reservation")
        or shard_request["reservation"].get("cursor") != counters.get("data_cursor")
        or shard_request.get("plan_sha256") != counters["data_cursor"].get("dataset_sha256")
        or shard_request.get("plan_sha256") != shard_request["reservation"].get("plan_sha256")
        or shard_request.get("shard_ids") != shard_request["reservation"].get("shard_ids")
        or not 0 < len(shard_request["shard_ids"]) <= 2
        or len(set(shard_request["shard_ids"])) != len(shard_request["shard_ids"])
        or status.get("status") != "shard_required" or status.get("cursor") != counters
        or status.get("checkpoint") != latest or status.get("shard_request") != shard_request
        or ledger.get("schema") != "continuous_training_budget_v1" or ledger.get("source_sha256") != OP04_LANE
        or ledger.get("active_attempt") is not None or ledger.get("max_seconds") != 43200
        or ledger.get("training_seconds") != counters.get("elapsed_seconds")
        or not request["elapsed_seconds"] < ledger["training_seconds"] < 14400
        or endpoint.get("hardware", {}).get("gpu_uuid") != lane["gpu_uuid"]
        or endpoint.get("hardware", {}).get("compute_capability") != [12, 0]
        or state["resource_baseline"].get("gpu_uuid") != lane["gpu_uuid"]
        or not state["resource_baseline"].get("boot_id")
        or bindings["original_runner"] != lane["source"]["scripts/run_nine_model_lane.py"]
        or bindings["original_pipeline"] != lane["source"]["src/w1a1_eagle/nine_model_pipeline.py"]
        or any(lane["source"].get(k, {}).get("sha256") != v for k, v in source_pins.items())
    ):
        raise ValueError("settled OP04 shard checkpoint/admission/ledger contract differs")
    if (
        result_locator["sha256"] != "8ad3fb4c868ac44aee5b43a6c57403341970a505e7af77ac422b8ee0b9e1156c"
        or metadata["evaluation_report"]["sha256"] != "3f06a76517b229dbedeb5855f700f31f7d280c7834c77f43b37fa28027f6a9ea"
        or metadata["request"]["sha256"] != "c8cba647c6642085915169d32e9aac7be917850b0c11ffa9bc93199e3e337c8b"
        or state.get("evaluation_receipt") != result_locator or state.get("evaluations") != [result_locator]
        or quality_state.get("evaluation_status") != result.get("status")
        or result.get("schema") != "nine_model_timed_evaluation_receipt_v1"
        or result.get("research_only") is not True or result.get("bundle_sha256") != OP04_LANE
        or result.get("config_sha256") != lane["config"]["sha256"]
        or result.get("protocol") != request["protocol"]
        or result.get("completed") is not True or result.get("request") != metadata["request"]
        or result.get("checkpoint") != request["checkpoint"] or result.get("evaluation") != metadata["evaluation_report"]
        or any(value.get(k) != expected for value in (quality_state, result, report) for k, expected in
               (("measurement_status", "MEASUREMENT_COMPLETE"), ("strict_quality_status", "FAILED"),
                ("continuation_status", "RESEARCH_CONTINUATION_ALLOWED")))
        or result.get("owned_release", {}).get("owned_process_groups_absent") is not True
        or result.get("owned_release", {}).get("owned_cuda_pids_absent") is not True
        or timed.get("schema") != "nine_model_timed_evaluation_state_v1" or timed.get("pending") is not None
        or timed.get("completed") != [{"request": metadata["request"], "receipt": result_locator}]
        or timed.get("contract") != dict(bundle_sha256=OP04_LANE, candidate="dspark_a8",
                config_sha256=lane["config"]["sha256"], max_seconds=43200,
                milestones_seconds=[1800, 14400, 28800, 43200], protocol=request["protocol"])
    ):
        raise ValueError("completed native evaluation/shard resume contract differs")
    return lane, state


def validate_op04_completed_boundary(bindings, records, lane, state, endpoint, request):
    """Small exact completed-boundary joins; original scientific validators still run."""
    result, report, timed = [json.loads(records[k]) for k in
                             ("evaluation_receipt", "evaluation_report", "timed_state")]
    metadata = bindings["metadata"]
    receipt_locator = metadata["evaluation_receipt"]
    expected_quality = dict(status="FAILED", strict_quality_status="FAILED",
                            measurement_status="MEASUREMENT_COMPLETE",
                            continuation_status="RESEARCH_CONTINUATION_ALLOWED")
    source_pins = {
        "src/w1a1_eagle/continuous_budget.py": OP04_POLICY_BUDGET_SHA256,
        "scripts/evaluate_nine_model_timed_checkpoint.py": "37fad07fa309d1be92203fefda303d3b370acb0a8cd8156327a8b12f5a2f8bdf",
        "scripts/export_nine_model_lane_candidate.py": "10ba61f9269256c048ccfac82369a1f49fe569165ffbf80dd2c9548eb3cb9fab",
        "scripts/run_nine_model_lane_endpoint.py": "d7bdfb257bc4d598e9a899914a1f4e879a2769b0656f1cd4768752c43a940fd9",
    }
    if (
        receipt_locator["sha256"] != "8ad3fb4c868ac44aee5b43a6c57403341970a505e7af77ac422b8ee0b9e1156c"
        or metadata["admission"]["sha256"] != "8895c1f04c7518eea79e9a1a6081d6997a1742c6c041453f073c3beeefb7bcd8"
        or metadata["endpoint"]["sha256"] != "0cfd1f3e4ac7e0175d31d35457813b42e929f13fe96e6a7c5677313eac7356dd"
        or metadata["request"]["sha256"] != "c8cba647c6642085915169d32e9aac7be917850b0c11ffa9bc93199e3e337c8b"
        or metadata["ledger"]["sha256"] != "387de0794ab0191c7625796b065c879386e80d12040cd5576ca75ea57e29341f"
        or metadata["evaluation_report"]["sha256"] != "3f06a76517b229dbedeb5855f700f31f7d280c7834c77f43b37fa28027f6a9ea"
        or state.get("pending_training") is not None
        or state.get("pending_shard_resume") not in (None, False)
        or state.get("status") != "awaiting_evaluation"
        or state.get("evaluation_receipt") != receipt_locator
        or state.get("evaluations") != [receipt_locator]
        or result.get("schema") != "nine_model_timed_evaluation_receipt_v1"
        or result.get("completed") is not True
        or result.get("research_only") is not True
        or any(result.get(k) != v for k, v in expected_quality.items())
        or any(state.get(k) != v for k, v in expected_quality.items() if k != "status")
        or any(report.get(k) != v for k, v in expected_quality.items() if k != "status")
        or result.get("request") != metadata["request"]
        or result.get("training_receipt") != metadata["endpoint"]
        or result.get("checkpoint") != endpoint["checkpoint"]
        or result.get("evaluation") != metadata["evaluation_report"]
        or result.get("bundle_sha256") != OP04_LANE
        or result.get("config_sha256") != lane["config"]["sha256"]
        or result.get("candidate") != lane["candidate"]
        or result.get("protocol") != request["protocol"]
        or result.get("elapsed_seconds") != request["elapsed_seconds"]
        or result.get("milestone_seconds") != 1800
        or result.get("owned_release", {}).get("owned_process_groups_absent") is not True
        or result.get("owned_release", {}).get("owned_cuda_pids_absent") is not True
        or result.get("owned_release", {}).get("other_context_pids") != []
        or timed.get("pending") != metadata["request"]
        or timed.get("completed") != []
        or timed.get("schema") != "nine_model_timed_evaluation_state_v1"
        or timed.get("contract") != {
            "bundle_sha256": OP04_LANE, "candidate": lane["candidate"],
            "config_sha256": lane["config"]["sha256"], "max_seconds": 43200,
            "milestones_seconds": [1800, 14400, 28800, 43200], "protocol": request["protocol"],
        }
        or any(lane["source"].get(k, {}).get("sha256") != v for k, v in source_pins.items())
    ):
        raise ValueError("exact completed OP04 1800 evaluation/resume contract differs")


def derive_op04_main(original_source, operational_controller_source):
    """Alter only release/cleanup and exact admitted pending-first resume handoff."""
    tree = ast.parse(original_source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    controller_tree = ast.parse(operational_controller_source)
    current = next(
        (n for n in controller_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    replacements = {
        n.name: n
        for n in (current.body if current is not None else controller_tree.body)
        if isinstance(n, ast.FunctionDef) and n.name in ("persist_handoff", "release")
    }
    if set(replacements) != {"persist_handoff", "release"}:
        raise ValueError("exact reviewed resource handoff functions missing")

    class Repair(ast.NodeTransformer):
        def __init__(self):
            self.counts = dict(
                release=0, cleanup=0, admission=0, gates=0, selected=0
            )

        def visit_FunctionDef(self, node):
            if node.name == "release":
                self.counts["release"] += 1
                return [replacements["persist_handoff"], replacements["release"]]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            names = [n.id for n in node.targets if isinstance(n, ast.Name)]
            if (
                names == ["admission"]
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "execute"
            ):
                self.counts["admission"] += 1
                return ast.Pass()
            elif (
                names == ["selected"]
                and isinstance(node.value, ast.Subscript)
                and ast.unparse(node.value).startswith("admission['candidate_admissions']")
            ):
                self.counts["selected"] += 1
                node.value = ast.parse('_op04_original_state["admission"]', mode="eval").body
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                call = node.value
                if (
                    call.func.id == "require"
                    and call.args
                    and "admission['gates']" in ast.unparse(call.args[0])
                ):
                    self.counts["gates"] += 1
                    return ast.Pass()
            return self.generic_visit(node)

        def visit_Try(self, node):
            # Only the original final cleanup release must never enter an infinite wait.
            for handler in node.handlers:
                if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException":
                    for statement in handler.body:
                        if isinstance(statement, ast.With):
                            for sub in ast.walk(statement):
                                if isinstance(sub, ast.Try):
                                    for ex in sub.body:
                                        if (
                                            isinstance(ex, ast.Expr)
                                            and isinstance(ex.value, ast.Call)
                                            and isinstance(ex.value.func, ast.Name)
                                            and ex.value.func.id == "release"
                                            and not ex.value.args
                                            and not ex.value.keywords
                                        ):
                                            ex.value.keywords = [
                                                ast.keyword("wait", ast.Constant(False))
                                            ]
                                            self.counts["cleanup"] += 1
            return self.generic_visit(node)

    repair = Repair()
    main = repair.visit(main)
    if repair.counts != dict(
        release=1, cleanup=1, admission=1, gates=1, selected=1
    ):
        raise ValueError(
            "original OP04 controller operational seam differs: " + repr(repair.counts)
        )
    return ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), repair.counts


def derive_op04_timed(source):
    tree = ast.parse(source)
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")

    class Repair(ast.NodeTransformer):
        def __init__(self):
            self.blocks = 0
            self.report = 0

        def visit_For(self, node):
            node = self.generic_visit(node)
            for i, statement in enumerate(node.body):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "release"
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "require_released"
                ):
                    block = node.body[i : i + 5]
                    expected = ast.parse("""release = observer.require_released(runner.process_groups, runner.process_identities)
observed = observer.snapshot()
resource_gate(observed, baseline, plan["resource_policy"])
require(release.get("other_context_pids") == [] and observed.get("dxg_holders") == [], "foreign GPU contexts forbid timed continuation")
atomic_json(stage / "resource-return.json", {"release": release, "resources": observed})""").body
                    if [ast.dump(n, include_attributes=False) for n in block] != [
                        ast.dump(n, include_attributes=False) for n in expected
                    ]:
                        raise ValueError("exact original timed37 resource boundary differs")
                    node.body[i : i + 5] = (
                        ast.parse("""release, observed, resource_wait_wall_seconds = _op04_timed_handoff(observer, runner, baseline, plan["resource_policy"], authorization, stage)
started += resource_wait_wall_seconds
atomic_json(stage / "resource-return.json", {"release": release, "resources": observed})""").body
                    )
                    self.blocks += 1
            return node

        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "report"
                and node.value.func.attr == "update"
            ):
                node.value.keywords.append(
                    ast.keyword(
                        "operational_resource_handoff",
                        ast.Name("_op04_runtime_provenance", ast.Load()),
                    )
                )
                self.report += 1
            return self.generic_visit(node)

    repair = Repair()
    evaluate = repair.visit(evaluate)
    if (repair.blocks, repair.report) != (1, 1):
        raise ValueError("timed37 operational seam count differs")
    return ast.fix_missing_locations(ast.Module(body=[evaluate], type_ignores=[])), {
        "released_stage_wait": repair.blocks,
        "runtime_provenance": repair.report,
    }


def op04_timed_handoff(
    waiter, atomic_json, observer, runner, baseline, policy, authorization, stage
):
    observations = []
    started = time.monotonic()

    def persist(event):
        observations.append(event)
        atomic_json(
            stage / "resource-handoff-observations.json",
            {
                "schema": "op04_released_native_cell_wait_v1",
                "observations": observations,
                "actual_handoff_wall_seconds": time.monotonic() - started,
                "capacity_wait_only_excluded_from_active_evaluation_cap": True,
            },
        )

    proof, observed = waiter(
        observer,
        runner.process_groups,
        runner.process_identities,
        baseline,
        policy,
        authorization=authorization,
        persist=persist,
    )
    elapsed = time.monotonic() - started
    atomic_json(
        stage / "resource-handoff-wait-cost.json",
        {
            "actual_handoff_wall_seconds": elapsed,
            "scope": "released-cell handoff only; request/server startup timings unchanged",
            "capacity_wait_wall_seconds": observations[-1].get("capacity_wait_wall_seconds", 0.0),
            "capacity_wait_only_excluded_from_active_evaluation_cap": True,
        },
    )
    return proof, observed, observations[-1].get("capacity_wait_wall_seconds", 0.0)


def op04_main(argv):
    parser = argparse.ArgumentParser(
        description="Authenticated same-OP04 operational resource handoff; original science and lane remain frozen"
    )
    parser.add_argument("--op04-resource-manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--profile-check-only", action="store_true")
    parser.add_argument("original", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    manifest = json.loads(
        op04_read({"path": str(args.op04_resource_manifest), "sha256": args.manifest_sha256})
    )
    root = args.op04_resource_manifest.resolve().parent
    if (
        manifest.get("profile") != OP04_PROFILE
        or manifest.get("schema") != "frozen_dspark_op04_resource_package_v1"
    ):
        raise ValueError("OP04 resource profile differs")
    inventory = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if inventory != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("exact OP04 operational carrier closure required")
    for name, digest in manifest["files"].items():
        op04_read({"path": str(root / name), "sha256": digest})
    if Path(__file__).resolve() != root / "scripts/run_frozen_dspark_hash_memo_20261011.py":
        raise ValueError("invoke authenticated existing operational proxy")
    bindings = json.loads((root / "selection.json").read_text())
    validate_op04_phase_selection(manifest, bindings)
    records = {k: op04_read(v) for k, v in bindings["metadata"].items()}
    lane, state = validate_op04_metadata(bindings, records)
    validate_op04_policy_memo_selection(manifest, bindings, lane)
    validate_op04_child_selection(manifest, bindings, lane)
    original = op04_read(bindings["original_runner"])
    op04_read(bindings["original_pipeline"])
    helper_source = (root / "resource-helper-source.py").read_bytes()
    controller_source = (root / "resource-controller-source.py").read_bytes()
    main_tree, counts = derive_op04_main(original, controller_source)
    timed_source = op04_read(bindings["original_timed_evaluator"])
    if (
        bindings["original_timed_evaluator"]["sha256"]
        != "37fad07fa309d1be92203fefda303d3b370acb0a8cd8156327a8b12f5a2f8bdf"
    ):
        raise ValueError("original timed37 source required")
    timed_tree, timed_counts = derive_op04_timed(timed_source)
    if args.profile_check_only:
        print(
            json.dumps(
                {
                    "status": "PASS_OPERATIONAL_PROFILE_ONLY",
                    "changes": counts,
                    "timed_changes": timed_counts,
                    "model_loaded": False,
                    "gpu_queried": False,
                    "training_started": False,
                }
            )
        )
        return
    supplied = args.original[1:] if args.original[:1] == ["--"] else args.original
    if supplied != bindings["original_argv"]:
        raise ValueError("exact original same-lane resume argv required")
    if "--resume" not in supplied or "--start" not in supplied:
        raise ValueError("same paid run resume only")
    if (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        != state["resource_baseline"]["boot_id"]
    ):
        raise ValueError("original resource boot differs")
    source = Path(bindings["original_runner"]["path"]).parents[1]
    sys.path[:0] = [str(source / "src"), str(source / "scripts")]
    from w1a1_eagle import nine_model_pipeline as pipeline

    if Path(pipeline.__file__).resolve() != Path(bindings["original_pipeline"]["path"]):
        raise ValueError("wrong original scientific pipeline imported")
    helper = next(
        n
        for n in ast.parse(helper_source).body
        if isinstance(n, ast.FunctionDef) and n.name == "wait_for_resource_return"
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])),
            str(root / "resource-helper-source.py"),
            "exec",
        ),
        pipeline.__dict__,
    )
    import importlib

    timed = importlib.import_module("evaluate_nine_model_timed_checkpoint")
    if Path(timed.__file__).resolve() != Path(bindings["original_timed_evaluator"]["path"]):
        raise ValueError("wrong original timed evaluator imported")
    timed.__dict__["_op04_timed_handoff"] = lambda *a: op04_timed_handoff(
        pipeline.wait_for_resource_return, pipeline.atomic_json, *a
    )
    timed.__dict__["_op04_runtime_provenance"] = {
        "profile": OP04_PROFILE,
        "manifest": {"path": str(args.op04_resource_manifest), "sha256": args.manifest_sha256},
        "original_scientific_source": bindings["original_timed_evaluator"],
        "active_evaluation_cap_seconds": 1200,
        "released_handoff_wait_wall_separate": True,
    }
    exec(compile(timed_tree, bindings["original_timed_evaluator"]["path"], "exec"), timed.__dict__)
    install_op04_policy_memo(manifest, bindings, lane, pipeline, timed)
    install_op04_trainer_bootstrap(manifest, bindings, lane, pipeline,
                                   args.op04_resource_manifest, args.manifest_sha256)
    namespace = runpy.run_path(
        bindings["original_runner"]["path"], run_name="op04_frozen_operational_controller"
    )
    namespace.update(
        wait_for_resource_return=pipeline.wait_for_resource_return, _op04_original_state=state
    )
    exec(compile(main_tree, bindings["original_runner"]["path"], "exec"), namespace)
    sys.argv = supplied
    namespace["main"](supplied[1:])


def main():
    if "--op04-trainer-manifest" in sys.argv[1:]:
        return op04_trainer_main(sys.argv[1:])
    if "--op04-resource-manifest" in sys.argv[1:]:
        return op04_main(sys.argv[1:])
    return historical_main()

if __name__ == '__main__':
    main()
