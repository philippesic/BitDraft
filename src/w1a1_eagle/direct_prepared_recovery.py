"""Preserve an interrupted, unpublished native capture before original recovery.

Completion uses original publication and metadata APIs, never a synthetic PASS.
The unchanged registered preparation reader retains all data/cursor admission.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _identity(value):
    return tuple(getattr(value, key) for key in (
        "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
        "st_size", "st_mtime_ns", "st_ctime_ns"))


def _read(path, *, max_bytes=8 * 1024**2):
    _require(type(max_bytes) is int and max_bytes in (8 * 1024**2, 16 * 1024**2),
             "fixed bounded metadata read limit required")
    path = Path(path)
    before = path.lstat()
    _require(path.resolve() == path and stat.S_ISREG(before.st_mode)
             and before.st_nlink == 1 and before.st_size <= max_bytes,
             "bounded canonical original producer metadata required")
    with path.open("rb") as stream:
        _require(_identity(os.fstat(stream.fileno())) == _identity(before), "producer metadata opened another file")
        raw = stream.read(max_bytes + 1)
        _require(_identity(os.fstat(stream.fileno())) == _identity(before), "producer metadata changed while reading")
    _require(_identity(path.lstat()) == _identity(before), "producer metadata replaced while reading")
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, _identity(before)


def _canonical_directory(path):
    _require(path.is_absolute() and path.resolve() == path,
             "artifact destination must be canonical without symlink ancestors")
    for ancestor in (path, *path.parents):
        item = ancestor.lstat()
        _require(stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode),
                 "artifact destination contains a symlink or non-directory")


def preserve_unpublished_registered_targets(
    plan, registrations, *, run_dir, shard_config, runner, source,
    release, authorization, validate_registration, api,
):
    """Move ONLY proven uncommitted targets aside after actual GPU release.

    Genuine PASS prefixes complete through original publish/metadata consumers.
    Failed/unknown publications refuse, never quarantine. Unpublished partials
    retain the existing preservation path; no data is recaptured or deleted.
    """
    from w1a1_eagle import sustained_block_handoff as original

    authorization()
    original._released(release)
    run = Path(run_dir).resolve()
    actions, observations, completions = [], [], []
    expected_caches = {}
    cache_state_paths = set()
    for root, registration in registrations.items():
        pair = Path(root)
        window = Path(registration["window_root"])
        _require(registration.get("schema") == "prepared_pair_registration_v1"
                 and pair.is_absolute() and pair.resolve() == pair and pair.parent == window
                 and window.parent.name == "prepared-data"
                 and window.parent.parent.parent == run / "direct-runtime"
                 and len(pair.name) == 37 and pair.name.startswith("pair-")
                 and all(c in "0123456789abcdef" for c in pair.name[5:]), "registered pair scope differs")
        validate_registration(registration)
        runtime, _, _ = _read(api.checked(registration["runtime"]))
        operation = Path(runtime["operation"])
        _require(window.parent == operation / "prepared-data"
                 and operation.resolve().parent == run / "direct-runtime", "original runtime operation scope differs")
        initial, _, _ = _read(api.checked(registration["initial_state"]), max_bytes=16 * 1024**2)
        state, pin, fingerprint = _read(pair / "cache-state.json", max_bytes=16 * 1024**2)
        cache_state_paths.add(pin["path"])
        observations.append((pin, fingerprint))
        expected_caches[str(pair)] = (state, pin, fingerprint)
        requested = registration["requested_shards"]
        _require(initial.get("schema") == "block_shard_cache_v1"
                 and initial.get("plan_sha256") == plan.sha256 and initial.get("owner") is None
                 and initial.get("resident") == {} and 0 < len(requested) <= 2
                 and len(set(requested)) == len(requested) and set(requested) <= set(plan.shards)
                 and set(state) == set(initial) and state["schema"] == initial["schema"]
                 and state["plan_sha256"] == plan.sha256
                 and all(state["seals"].get(k) == v for k, v in initial["seals"].items())
                 and set(state["seals"]) - set(initial["seals"]) == set(state["resident"])
                 and set(state["resident"]) <= set(requested), "original registered cache transition differs")
        owner = state["owner"]
        if owner is not None:
            name = owner.get("owner", "")
            _require(owner.get("phase") == "restore" and name.startswith("controller-")
                     and len(name) == 43 and all(c in "0123456789abcdef" for c in name[11:]), "original restore owner differs")
        for shard in requested:
            target = pair / shard
            _require(target.resolve().parent == pair, "original canonical target differs")
            if shard in state["resident"]:
                _require(state["resident"][shard]["directory"] == str(target), "resident target moved")
                metadata = state["seals"][shard].get("original_metadata")
                if metadata is None:
                    completion = _completed_capture(plan, pair, shard, target, operation, state,
                        shard_config, runner, source, run, api)
                    _require(completion is not None, "missing metadata requires genuine PASS publication; preserved intact")
                    completions.append(completion)
                    observations.extend(completion["observations"])
                    continue
                saved, _, _ = _read(api.checked(metadata))
                _require(saved.get("plan_sha256") == plan.sha256 and saved.get("shard") == shard
                         and saved.get("identity") == state["seals"][shard]["identity"]
                         and saved.get("original_directory") == str(target), "original metadata identity differs")
                continue
            if not target.exists() and not target.is_symlink():
                continue
            completion = _completed_capture(plan, pair, shard, target, operation, state,
                shard_config, runner, source, run, api)
            if completion is not None:
                completions.append(completion)
                observations.extend(completion["observations"])
                continue
            _require(owner is not None and shard not in state["seals"], "only a new uncommitted capture may be preserved")
            before = target.lstat()
            _require(stat.S_ISDIR(before.st_mode) and target.resolve() == target, "uncommitted canonical directory required")
            paths = list((run / "attempts" / operation.name / "shard-phases").glob("*/request.json"))
            _require(len(paths) <= 4096, "bounded original request journal exceeded")
            matches = []
            for path in paths:
                request, request_pin, request_stat = _read(path)
                if request.get("directory") != str(target) or request.get("shard") != shard:
                    continue
                stage = path.parent
                _require(not (stage / "publication.json").exists()
                         and not (stage / "publication.json").is_symlink(),
                         "publication present; original completion required, never quarantine")
                _require(request.get("schema") == "block_shard_restore_request_v1"
                         and request.get("plan") == shard_config["plan"]
                         and request.get("plan_sha256") == plan.sha256
                         and request.get("original_identity") is None and request.get("original_metadata") is None
                         and request.get("generation_variant") == plan.value["generation_variant"]
                         and request.get("capture_bytes_bound") == plan.value["shard_bytes"], "original capture request differs")
                producer = shard_config["capture"]["producer"]
                api.checked(producer)
                _require(producer in source.values(), "original capture source not admitted")
                process, process_pin, process_stat = _read(stage / "producer/process.json")
                argv = [str(path) if arg == "{request}" else str(stage / "publication.json")
                        if arg == "{output}" else arg for arg in shard_config["capture"]["argv"]]
                identity = process.get("kernel_identity")
                _require(process.get("argv") == argv and process.get("pid") == process.get("pgid")
                         and identity is not None and identity.get("pid") == process["pid"]
                         and identity in runner.process_identities and process["pid"] in runner.process_groups,
                         "original producer argv/owned birth differs")
                observations.extend([(request_pin, request_stat), (process_pin, process_stat)])
                matches.append({"request": request_pin, "process": process_pin, "producer": producer})
            _require(len(matches) == 1, "one exact owned unpublished capture required")
            # Stat only: no tensor/model reads, no following links out of the target.
            for path in target.rglob("*"):
                item = path.lstat()
                _require(not path.is_symlink() and (stat.S_ISDIR(item.st_mode)
                         or stat.S_ISREG(item.st_mode) and item.st_nlink == 1), "uncommitted target contains foreign links/special files")
            actions.append((target, _identity(before), operation, registration, matches[0]))
    authorization()
    original._released(release)
    for pin, fingerprint in observations:
        _, current, current_stat = _read(Path(pin["path"]),
            max_bytes=16 * 1024**2 if pin["path"] in cache_state_paths else 8 * 1024**2)
        _require(current == pin and current_stat == fingerprint, "original recovery inputs changed before preservation")
    completed = []
    for value in completions:
        result, settled = _finish_publication(plan, value, shard_config, release, authorization, api,
            expected_caches[value["pair"]])
        completed.append(result)
        # Only an exact state produced by our preceding completion can advance
        # this same pair's expectation; never accept a fresh arbitrary owner.
        expected_caches[value["pair"]] = settled
    preserved = []
    for target, fingerprint, operation, registration, producer in actions:
        _require(_identity(target.lstat()) == fingerprint, "uncommitted target changed after release")
        _canonical_directory(operation)
        artifact_parent = operation / "interrupted-artifacts"
        if artifact_parent.exists() or artifact_parent.is_symlink():
            _canonical_directory(artifact_parent)
        else:
            artifact_parent.mkdir()
        _canonical_directory(artifact_parent)
        artifact = artifact_parent / uuid.uuid4().hex
        artifact.mkdir(exist_ok=False)
        _canonical_directory(artifact)
        context = {"schema": "direct_unpublished_capture_preservation_v1", "original_directory": str(target),
                   "preserved_directory": str(artifact / "payload"), "registration": registration,
                   "producer": producer, "directory_identity": list(fingerprint), "plan_sha256": plan.sha256}
        with (artifact / "context.json").open("x") as stream:
            json.dump(context, stream, sort_keys=True, indent=2); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        _canonical_directory(artifact)
        _require(artifact.parent == artifact_parent and artifact_parent.parent == operation
                 and _identity(target.lstat()) == fingerprint,
                 "artifact containment or original directory changed before rename")
        os.rename(target, artifact / "payload")
        preserved.append({"context": api.pin(artifact / "context.json"), "original_directory": str(target),
                          "preserved_directory": str(artifact / "payload")})
    return {"schema": "direct_unpublished_capture_recovery_v1", "preserved": preserved,
            "trainer_seconds_charged": 0, "completed_publications": completed}


def _completed_capture(plan, pair, shard, target, operation, state, config, runner, source, run, api):
    """Read the exact original inline publication guard; no receipt is invented."""
    paths = list((run / "attempts" / operation.name / "shard-phases").glob("*/request.json"))
    _require(len(paths) <= 4096, "bounded original request journal exceeded")
    matches = []
    for path in paths:
        request, request_pin, request_stat = _read(path)
        if request.get("directory") != str(target) or request.get("shard") != shard:
            continue
        output = path.parent / "publication.json"
        if not output.exists() and not output.is_symlink():
            continue
        _require(request.get("schema") == "block_shard_restore_request_v1"
                 and request.get("plan") == config["plan"]
                 and request.get("plan_sha256") == plan.sha256
                 and request.get("original_identity") is None and request.get("original_metadata") is None
                 and request.get("generation_variant") == plan.value["generation_variant"]
                 and request.get("capture_bytes_bound") == plan.value["shard_bytes"],
                 "original completed fresh-capture request differs")
        producer = config["capture"]["producer"]
        api.checked(producer)
        _require(producer in source.values(), "original capture source not admitted")
        process, process_pin, process_stat = _read(path.parent / "producer/process.json")
        argv = [str(path) if arg == "{request}" else str(output) if arg == "{output}" else arg
                for arg in config["capture"]["argv"]]
        birth = process.get("kernel_identity")
        _require(process.get("argv") == argv and process.get("pid") == process.get("pgid")
                 and birth is not None and birth.get("pid") == process["pid"]
                 and birth in runner.process_identities and process["pid"] in runner.process_groups,
                 "original producer argv/owned birth differs")
        receipt, publication_pin, publication_stat = _read(output)
        _require(receipt.get("schema") == "block_shard_publication_v1"
                 and receipt.get("status") == "PASS"
                 and receipt.get("plan_sha256") == plan.sha256
                 and receipt.get("shard") == shard and receipt.get("directory") == str(target)
                 and receipt.get("request") == request_pin,
                 "actual original PASS publication differs")
        _require(not receipt.get("reconstruction_join") and not receipt.get("recovery"),
                 "fresh completion cannot adopt replay/recovery receipt")
        if shard in state["resident"]:
            _require(state["resident"][shard] == {"manifest": receipt["manifest"],
                     "admission": receipt["admission"], "directory": str(target)},
                     "published resident differs from original receipt")
        for locator in (receipt["manifest"], receipt["admission"]):
            api.checked(locator)
        matches.append({"pair": str(pair), "shard": shard, "operation": str(operation),
            "publication": publication_pin, "request": request_pin, "process": process_pin,
            "producer": producer, "observations": [(request_pin, request_stat),
                (process_pin, process_stat), (publication_pin, publication_stat)]})
    _require(len(matches) <= 1, "one exact completed original capture required")
    return matches[0] if matches else None


def _checked_cache(path, expected, loaded=None):
    current = _read(path, max_bytes=16 * 1024**2)
    _require(current == expected and (loaded is None or current[0] == loaded),
             "original completion cache state changed before mutation")
    return current


def _finish_publication(plan, value, config, release, authorization, api, expected):
    """Complete original publication/metadata only while truly released and owned."""
    from w1a1_eagle import sustained_block_handoff as original
    _require(api.file_sha256(Path(api.__file__)) ==
             "a2ad87283999fe1268cd933dce59388c0c0ab544a7a27dcc3797967f305ece4e",
             "original publication consumer source differs")
    authorization(); proof = original._released(release)
    for pin, fingerprint in value["observations"]:
        _, current, current_stat = _read(Path(pin["path"]))
        _require(current == pin and current_stat == fingerprint, "completed publication changed")
    receipt = json.loads(api.checked(value["publication"]).read_text())
    _checked_cache(Path(value["pair"]) / "cache-state.json", expected)
    cache = api.ShardCache(plan, value["pair"], replay_admission=config.get("replay_admission"))
    owner = "controller-" + uuid.uuid4().hex
    active = False
    operation = Path(value["operation"])
    _canonical_directory(operation)
    parent = operation / "publication-completions"
    if not parent.exists() and not parent.is_symlink():
        parent.mkdir()
    _canonical_directory(parent)
    output = parent / uuid.uuid4().hex
    output.mkdir(exist_ok=False)
    _canonical_directory(output)
    try:
        _checked_cache(cache.state_path, expected, cache.state)
        prior = cache.state["owner"]
        if prior is not None:
            cache.release(prior["owner"], released_proof=proof)
        shard = value["shard"]
        if shard in cache.state["resident"]:
            # Re-admit original resident arrays/seals while truly released.
            original.build_catalog(plan, [api.pin(cache.state_path)], output / "resident-validation.json",
                max_bytes=3 * plan.value["shard_bytes"], api=api)
        cache.acquire(owner, "restore"); active = True
        if shard not in cache.state["resident"]:
            cache.publish(shard, receipt["manifest"], receipt["admission"], receipt["directory"])
        root = Path(config["metadata_root"])
        _require(root.is_absolute() and root.resolve() == root, "canonical configured original metadata root required")
        _require(type(config["metadata_max_bytes"]) is int and
                 api.tree_bytes(root) <= config["metadata_max_bytes"], "original metadata quota exceeded")
        destination = root
        if (root / shard).exists():
            # A previous original copy may have stopped before cache._save.
            # Preserve it unchanged; a NEW original copy remains under quota.
            destination = root / ("publication-completion-" + uuid.uuid4().hex)
        metadata = api.preserve_original_metadata(cache, shard, destination,
            max_bytes=config["metadata_max_bytes"])
        _require(api.tree_bytes(root) <= config["metadata_max_bytes"], "original total metadata quota exceeded")
        owned_state = _read(cache.state_path, max_bytes=16 * 1024**2)
        _require(owned_state[0] == cache.state, "original completion cache differs from owned state")
        authorization(); proof = original._released(release)
        _checked_cache(cache.state_path, owned_state, cache.state)
        cache.release(owner, released_proof=proof); active = False
        settled = _read(cache.state_path, max_bytes=16 * 1024**2)
        _require(settled[0] == cache.state, "original completion settled cache differs")
        # The next shard legitimately changes the live cache. Preserve this
        # completed state as immutable evidence before pinning its receipt.
        snapshot = output / "cache-state.json"
        api.atomic_json(snapshot, cache.state)
        record = {"schema": "direct_original_publication_completion_v1", "publication": value["publication"],
            "request": value["request"], "process": value["process"], "producer": value["producer"],
            "cache_state": api.pin(snapshot), "original_cache_path": str(cache.state_path),
            "original_metadata": metadata,
            "trainer_seconds_charged": 0}
        api.atomic_json(output / "receipt.json", record)
        return {"receipt": api.pin(output / "receipt.json"), "shard": shard}, settled
    finally:
        cache.close()
        if active and cache.state["owner"] == {"owner": owner, "phase": "restore"}:
            # A failed release preserves custody. Cleanup cannot mask a primary.
            import sys
            primary = sys.exc_info()[1]
            try:
                owned_state = _read(cache.state_path, max_bytes=16 * 1024**2)
                _require(owned_state[0] == cache.state, "original cleanup cache differs from owned state")
                proof = original._released(release)
                _checked_cache(cache.state_path, owned_state, cache.state)
                cache.release(owner, released_proof=proof)
            except BaseException as cleanup:
                if primary is None:
                    raise
                try:
                    print(json.dumps({"original_publication_cleanup_failure": {
                        "type": type(cleanup).__name__, "reason": str(cleanup),
                        "primary_type": type(primary).__name__, "primary_reason": str(primary)}}),
                        file=sys.stderr, flush=True)
                except BaseException:
                    pass
