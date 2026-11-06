"""Read-only prepared publications for the original rotating block consumer.

Preparation runs only with the student released.  A prepared window retains
original publication paths; crossing a ready boundary only closes/reopens dataset
maps.  It does not capture, replay, evict, save a checkpoint or advance a cursor.
The controller/supervisor integration is deliberately separate from this reader.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

from . import block_shard_lifecycle as lifecycle

SCHEMA = "prepared_block_window_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked(locator, api):
    return api.checked(locator)


def _same_original_seal(left, right):
    """Recovery evidence may grow; original tensors/anchors/metadata cannot change."""
    return left["identity"] == right["identity"] and (
        left.get("original_metadata") is None
        or right.get("original_metadata") is None
        or left["original_metadata"] == right["original_metadata"]
    )


def build_catalog(
    plan,
    cache_locators,
    output,
    *,
    max_bytes,
    publication_shards=None,
    owned_pair_roots=(),
    api=lifecycle,
):
    """Re-admit completed ORIGINAL publications, never relocate their bytes.

    Each input is an externally pinned original cache-state.json. The original
    dataset consumer reuses its completed audit only after exact current-host,
    source, manifest and artifact checks. Native source and original seals join
    separately. This function cannot generate missing data or mutate the cache.
    """
    _require(type(max_bytes) is int and max_bytes > 0, "positive prepared disk bound required")
    if plan.production:
        api.validate_deployed_source(plan)
    records, seals, sources, directories = {}, {}, [], set()
    wanted = None if publication_shards is None else set(publication_shards)
    _require(wanted is None or wanted <= set(plan.shards), "catalog publication selection differs")
    for locator in cache_locators:
        source_path = _checked(locator, api)
        state = json.loads(source_path.read_text())
        _require(
            state.get("schema") == "block_shard_cache_v1"
            and state.get("plan_sha256") == plan.sha256
            and state.get("owner") is None,
            "preparation requires a released original cache of the same logical plan",
        )
        _require(len(state["resident"]) <= 2, "original active cache count exceeds two")
        audit = api.ShardCache.__new__(api.ShardCache)
        audit.plan, audit.root = plan, source_path.parent.resolve()
        audit.dataset_factory, audit.datasets = api.BlockDataset, {}
        audit.state = copy.deepcopy(state)
        _require(
            api.tree_bytes(audit.root) <= 3 * plan.value["shard_bytes"],
            "original cache including staging exceeds admitted physical bound",
        )
        try:
            for shard, record in state["resident"].items():
                directory = audit._directory(record["directory"])
                _require(
                    api.tree_bytes(directory) <= plan.value["shard_bytes"],
                    "actual publication exceeds original shard bound",
                )
                _require(
                    _checked(record["manifest"], api).resolve().is_relative_to(directory)
                    and _checked(record["admission"], api).resolve().is_relative_to(directory),
                    "original manifest/admission outside canonical publication",
                )
                # This is the ORIGINAL completed-admission consumer.  It checks
                # current device/inode/size/mtime/ctime + source + every expected
                # artifact SHA before reusing its completed finite/hash audit.
                dataset = api.ShardCache.dataset(audit, shard)
                _require(
                    set(dataset.chains) == set(plan.shards[shard]),
                    "prepared publication membership differs",
                )
                for key, value in plan.value["geometry"].items():
                    _require(
                        dataset.manifest.get(key) == value, "prepared geometry differs: " + key
                    )
                api.validate_capture_source(plan, shard, dataset)
                identity = {}
                for cid, chain in dataset.chains.items():
                    _require(
                        all(
                            chain[key] == plan.chains[cid][key]
                            for key in ("prompt_id", "prompt_sha256", "domain", "split")
                        ),
                        "prepared original chain metadata differs",
                    )
                    artifacts = {}
                    for key in ("tokens", "features", "logits", "native_receipt"):
                        artifact = chain[key]
                        _require(
                            artifact is not None, "original full teacher/source artifact required"
                        )
                        path = (
                            Path(record["manifest"]["path"]).parent / artifact["path"]
                        ).resolve()
                        _require(
                            path.is_file() and path.is_relative_to(directory),
                            "prepared artifact outside original publication",
                        )
                        artifacts[key] = artifact["sha256"]
                    identity[cid] = {
                        "artifacts": artifacts,
                        "anchors": chain["anchors"],
                        "prompt_length": chain["prompt_length"],
                        "logits_indices": chain.get("logits_indices"),
                    }
                _require(
                    identity == state["seals"][shard]["identity"],
                    "restored teacher/source receipt/tensor bytes changed",
                )
                if wanted is None or shard in wanted:
                    _require(
                        shard not in records or records[shard] == record,
                        "conflicting prepared publication",
                    )
                    records[shard] = copy.deepcopy(record)
                    directories.add(str(directory))
        finally:
            audit.close()
        for shard, seal in state["seals"].items():
            _require(
                shard not in seals or _same_original_seal(seals[shard], seal),
                "conflicting historical seal",
            )
            seals.setdefault(shard, copy.deepcopy(seal))
        sources.append(copy.deepcopy(locator))
    _require(wanted is None or set(records) == wanted, "selected original publication missing")
    total = sum(api.tree_bytes(path) for path in directories)
    _require(total <= max_bytes, "prepared backing storage exceeds explicit disk bound")
    source_roots = {str(Path(row["path"]).resolve().parent) for row in sources}
    owned = [str(Path(path).resolve()) for path in owned_pair_roots]
    _require(
        len(owned) == len(set(owned))
        and set(owned) <= source_roots
        and all(Path(path).name.startswith("pair-") for path in owned),
        "owned pair roots must be explicit original publication cache sources",
    )
    value = {
        "schema": SCHEMA,
        "plan_sha256": plan.sha256,
        "publications": records,
        "seals": seals,
        "source_cache_states": sources,
        "owned_pair_roots": owned,
        "payload_bytes": total,
        "max_window_bytes": max_bytes,
        "producer_source": api.pin(Path(__file__)),
        "original_provider_source": api.pin(Path(inspect.getfile(api.ShardCache))),
        "original_data_source": api.pin(Path(inspect.getfile(api.BlockDataset))),
    }
    # A catalog is immutable evidence, not a second progress database.
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return api.pin(output)


class PreparedCache:
    """At most two active mappings over an immutable admitted backing window."""

    def __init__(self, plan, locator, *, api=lifecycle):
        self.api, self.plan = api, plan
        value = json.loads(_checked(locator, api).read_text())
        _require(
            value.get("schema") == SCHEMA and value.get("plan_sha256") == plan.sha256,
            "prepared window logical identity differs",
        )
        _require(
            set(value["publications"]) <= set(plan.shards)
            and set(value["publications"]) <= set(value["seals"]),
            "prepared membership/seals differ",
        )
        _checked(value["producer_source"], api)
        _require(
            value["producer_source"]["sha256"] == api.file_sha256(Path(__file__)),
            "prepared catalog producer source bytes differ",
        )
        _checked(value["original_provider_source"], api)
        _require(
            Path(value["original_provider_source"]["path"]).resolve()
            == Path(inspect.getfile(api.ShardCache)).resolve(),
            "prepared catalog original source role differs",
        )
        _checked(value["original_data_source"], api)
        _require(
            Path(value["original_data_source"]["path"]).resolve()
            == Path(inspect.getfile(api.BlockDataset)).resolve(),
            "prepared catalog original data consumer differs",
        )
        self.publications = copy.deepcopy(value["publications"])
        self.state = {
            "schema": "block_shard_cache_v1",
            "plan_sha256": plan.sha256,
            "resident": {},
            "seals": copy.deepcopy(value["seals"]),
            "owner": None,
        }
        self.datasets = {}
        self.dataset_factory = api.BlockDataset
        self.payload_bytes = value["payload_bytes"]
        _require(
            0 <= self.payload_bytes <= value["max_window_bytes"], "prepared disk accounting differs"
        )

    def activate(self, shards):
        shards = tuple(dict.fromkeys(shards))
        _require(0 < len(shards) <= 2, "original two-active mapping limit required")
        if any(shard not in self.publications for shard in shards):
            return False
        for shard in tuple(self.datasets):
            if shard not in shards:
                self.datasets.pop(shard).close()
        self.state["resident"] = {s: self.publications[s] for s in shards}
        return True

    def dataset(self, shard):
        # Actual original consumer: expected manifest SHA, completed admission,
        # current artifact fingerprints and allow_synthetic=False are retained.
        return self.api.ShardCache.dataset(self, shard)

    def close(self):
        self.api.ShardCache.close(self)

    def disk_bytes(self):
        return self.payload_bytes


def _reserve_prepared(self, cursor, count=2, *, require_teacher=True, reservation=None):
    original = self.cache.api.RotatingBlockProvider.reserve_batch
    try:
        return original(
            self, cursor, count, require_teacher=require_teacher, reservation=reservation
        )
    except self.cache.api.ShardRequired as error:
        if not self.cache.activate(error.shard_ids):
            raise
        return original(
            self, cursor, count, require_teacher=require_teacher, reservation=error.reservation
        )


class PreparedBlockProvider(lifecycle.RotatingBlockProvider):
    reserve_batch = _reserve_prepared


def open_prepared_provider(data, prepared_window, *, api=lifecycle):
    """Use the caller's admitted ORIGINAL consumer, including its source globals."""
    _require(
        data.get("provider") == "rotating_block_v1", "original rotating data provider required"
    )
    plan = api.FrozenShardPlan.load({key: data[key] for key in ("path", "sha256")})
    api.validate_deployed_source(plan)
    return provider_from_catalog(plan, prepared_window, api=api)


def provider_from_catalog(plan, prepared_window, *, api=lifecycle):
    cls = type(
        "PreparedOriginalBlockProvider",
        (api.RotatingBlockProvider,),
        {"reserve_batch": _reserve_prepared},
    )
    return cls(plan, PreparedCache(plan, prepared_window, api=api))


def install_child_provider(api, data, prepared_window):
    """Operational hook for a source-authenticated original child namespace.

    The runtime must authenticate ``api`` before calling and record both source
    roles. Only the exact selected data configuration is intercepted. Returns an
    undo callback; an unrelated provider remains the original consumer.
    """
    original, selected = api.open_provider, copy.deepcopy(data)

    def open_selected(actual):
        if actual != selected:
            return original(actual)
        return open_prepared_provider(actual, prepared_window, api=api)

    api.open_provider = open_selected

    def restore():
        _require(api.open_provider is open_selected, "prepared provider hook replaced externally")
        api.open_provider = original

    return restore


def _released(release):
    proof = release()
    _require(
        isinstance(proof, dict)
        and proof.get("owned_process_groups_absent") is True
        and proof.get("owned_cuda_pids_absent") is True,
        "actual student/teacher process and CUDA release required for preparation",
    )
    return proof


def future_shards(plan, cursor_payload, seals, *, api=lifecycle):
    """Original order plus actual progress; unknown lengths are never consumed."""
    cache = type("HistoryOnly", (), {"state": {"seals": seals}})()
    provider = api.RotatingBlockProvider(plan, cache)
    cursor = provider.restore_cursor(copy.deepcopy(cursor_payload))
    result = []
    progress = cursor.progress.get(str(cursor.epoch), {})
    for cid in plan.order(cursor.split, cursor.seed, cursor.epoch):
        shard = plan.chain_shards[cid]
        seal = seals.get(shard)
        length = None if seal is None else len(seal["identity"][cid]["anchors"])
        if length is not None and progress.get(cid, 0) == length:
            continue
        if shard not in result:
            result.append(shard)
    # The next epoch is not speculatively materialized.  The original epoch-tail
    # reservation remains the authoritative boundary if this window reaches it.
    return result


def prepare_window(
    plan,
    cursor_payload,
    shard_config,
    history_cache_locators,
    window_root,
    *,
    run_dir,
    attempt,
    runner,
    release,
    authorization,
    source,
    max_window_bytes,
    reserve_free_bytes,
    merge_seals,
    register_cache=None,
    available_bytes=None,
    owned_pair_roots=(),
    target_window_bytes=None,
    required_shards=(),
    batch_reservation=None,
    api=lifecycle,
):
    """Automatically capture a bounded future window through ORIGINAL consumers.

    Must run under the exclusive controller/campaign lock with student absent.
    ``merge_seals`` is a serialized controller callback: it accepts an authentic
    released cache-state locator, merges ONLY original seals into canonical
    history, and returns fresh canonical cache-state locators.  This function
    never modifies a paid cursor or relocates any historical replay directory.
    ``available_bytes(path)`` must measure the destination's PHYSICAL backing
    filesystem; default statvfs is suitable for /mnt/c, not WSL virtual free.
    """
    import uuid

    _require(callable(merge_seals), "serialized canonical seal-history merge callback required")
    _require(callable(register_cache), "durable owned-pair registration callback required")
    _require(
        type(max_window_bytes) is int
        and max_window_bytes > 0
        and type(reserve_free_bytes) is int
        and reserve_free_bytes >= 0,
        "explicit physical disk and reserve bounds required",
    )
    if target_window_bytes is None:
        target_window_bytes = max_window_bytes
    _require(type(target_window_bytes) is int and 0 < target_window_bytes <= max_window_bytes,
             "positive preparation target within hard disk bound required")
    _require(isinstance(required_shards, (list, tuple))
             and len(set(required_shards)) == len(required_shards)
             and set(required_shards) <= set(plan.shards),
             "exact original required shard inventory differs")
    authorization()
    _released(release)
    root = Path(window_root).resolve()
    _require(not root.exists(), "fresh prepared window root required; preserve previous attempt")
    root.mkdir(parents=True, exist_ok=False)
    if available_bytes is None:
        _require(
            root == Path("/mnt/c") or root.is_relative_to(Path("/mnt/c")),
            "default capacity guard requires C physical filesystem, not WSL virtual free",
        )

        def available_bytes(path):
            info = os.statvfs(path)
            return info.f_bavail * info.f_frsize

    history_locators = list(history_cache_locators)
    historical, current_records, cache_paths = {}, {}, {}
    for locator in history_locators:
        path = _checked(locator, api)
        state = json.loads(path.read_text())
        _require(
            state.get("schema") == "block_shard_cache_v1"
            and state.get("plan_sha256") == plan.sha256
            and state.get("owner") is None,
            "window history requires released original cache",
        )
        cache_paths[str(path.resolve().parent)] = path
        for shard, seal in state["seals"].items():
            _require(
                shard not in historical or _same_original_seal(historical[shard], seal),
                "canonical historical seals conflict",
            )
            historical.setdefault(shard, copy.deepcopy(seal))
        current_records.update(copy.deepcopy(state["resident"]))
    unchanged_cursor = copy.deepcopy(cursor_payload)
    ordered = future_shards(plan, unchanged_cursor, historical, api=api)
    selected, phases, prepared_states, used = [], [], [], 0
    owned = set(str(Path(path).resolve()) for path in owned_pair_roots)
    carried_states = {
        str(Path(locator["path"]).resolve().parent): copy.deepcopy(locator)
        for locator in history_locators
        if str(Path(locator["path"]).resolve().parent) in owned
    }
    # Retain ready original data only when it is actually upcoming in this epoch.
    for shard in ordered:
        if shard in current_records:
            selected.append(shard)
            used += api.tree_bytes(current_records[shard]["directory"])
    _require(used <= max_window_bytes, "ready publications already exceed prepared disk bound")
    pending = [s for s in ordered if s not in selected]
    stop_reason = "epoch_lookahead_complete"
    while pending:
        # Keep every ready publication, but require no byte-volume floor before
        # TRAIN. Check only between fully published, released and merged pairs;
        # the original exact next-batch reservation remains the stopping gate.
        if set(required_shards) <= set(selected):
            # A metadata-kind ShardRequired may describe only the first part of
            # a batch. Re-derive it with the ORIGINAL consumer as seals arrive;
            # empty residency forces its reservation boundary before array IO.
            view = type("ReservationHistory", (), {"state": {
                "seals": historical, "resident": {},
            }})()
            provider = api.RotatingBlockProvider(plan, view)
            restored = provider.restore_cursor(copy.deepcopy(unchanged_cursor))
            try:
                provider.reserve_batch(restored, 2, reservation=batch_reservation)
            except api.ShardRequired as required:
                if (required.reservation.get("kind") == "batch"
                        and set(required.shard_ids) <= set(selected)):
                    stop_reason = "bounded_training_window_ready"
                    break
            else:
                raise RuntimeError("original missing-resident reservation boundary changed")
        authorization()
        _released(release)
        pair = pending[:2]
        # Historical reconstruction must use its ORIGINAL canonical root.  Keep
        # such a restore separate from first captures, so no source path aliases
        # or replay relocation exceptions are introduced.
        old = pair[0] in historical
        if old:
            pair = pair[:1]
        elif len(pair) == 2 and pair[1] in historical:
            pair = pair[:1]
        bound = len(pair) * plan.value["shard_bytes"]
        free = available_bytes(root)
        _require(type(free) is int and free >= 0, "physical available-byte observation required")
        if used + bound > max_window_bytes or free - bound < reserve_free_bytes:
            stop_reason = "physical_disk_window_bound"
            break
        if old:
            metadata_locator = historical[pair[0]].get("original_metadata")
            _require(
                metadata_locator is not None, "historical replay lacks original canonical metadata"
            )
            metadata = json.loads(_checked(metadata_locator, api).read_text())
            canonical = Path(metadata["original_directory"]).resolve()
            _require(
                canonical.name == pair[0] and str(canonical.parent) in cache_paths,
                "historical restore must use its admitted original canonical cache",
            )
            cache = api.ShardCache(
                plan, canonical.parent, replay_admission=shard_config.get("replay_admission")
            )
            # Original reconstruction may evict other original resident data.
            # Refuse that destructive collision with this window's retained data.
            excess = max(0, len(set(cache.state["resident"]) | set(pair)) - 2)
            evictions = [s for s in cache.state["resident"] if s not in pair][:excess]
            if any(s in selected for s in evictions):
                stop_reason = "canonical_replay_would_evict_prepared_publication"
                break
        else:
            cache_root = root / ("pair-" + uuid.uuid4().hex)
            cache = api.ShardCache(
                plan, cache_root, replay_admission=shard_config.get("replay_admission")
            )
            cache.state["seals"] = copy.deepcopy(historical)
            cache._save()
            authorization()
            _released(release)
            register_cache(api.pin(cache.state_path), list(pair))
        config = copy.deepcopy(shard_config)
        config["cache_root"] = str(cache.root)
        reservation = {
            "kind": "prepared_lookahead",
            "plan_sha256": plan.sha256,
            "cursor": copy.deepcopy(unchanged_cursor),
            "shard_ids": pair,
        }
        request = {"plan_sha256": plan.sha256, "shard_ids": pair, "reservation": reservation}
        pair_phases = api.restore_requested_shards(
            config,
            request,
            cache=cache,
            run_dir=run_dir,
            attempt=attempt,
            runner=runner,
            release=lambda: _released(release),
            authorization=authorization,
            source=source,
        )
        _require(cache.state["owner"] is None, "original preparation did not release ownership")
        _require(cursor_payload == unchanged_cursor, "preparation changed paid data cursor")
        actual = sum(api.tree_bytes(cache.state["resident"][s]["directory"]) for s in pair)
        used += actual
        _require(used <= max_window_bytes, "actual prepared data exceeded admitted window")
        _require(
            available_bytes(root) >= reserve_free_bytes,
            "physical free reserve consumed during preparation",
        )
        state_locator = api.pin(cache.state_path)
        if str(cache.root) in carried_states:
            carried_states[str(cache.root)] = state_locator
        # Native source/data gates have passed before canonical history is extended.
        fresh_history = merge_seals(state_locator)
        _require(
            isinstance(fresh_history, list) and fresh_history,
            "seal merge must return fresh original canonical state locators",
        )
        for locator in fresh_history:
            state = json.loads(_checked(locator, api).read_text())
            _require(
                state.get("plan_sha256") == plan.sha256 and state.get("owner") is None,
                "canonical history merge returned active or different plan",
            )
            for shard, seal in historical.items():
                _require(
                    state["seals"].get(shard) == seal,
                    "canonical merge lost or changed original historical seal",
                )
            for shard in pair:
                _require(
                    state["seals"].get(shard) == cache.state["seals"][shard],
                    "canonical merge did not retain actual prepared seal",
                )
        history_locators = fresh_history
        historical.update(copy.deepcopy(cache.state["seals"]))
        if not old:
            prepared_states.append(state_locator)
            owned.add(str(cache.root))
        phases.extend(pair_phases)
        selected.extend(pair)
        del pending[: len(pair)]
        cache.close()
    _require(selected, "disk/replay boundary leaves no prepared original publication")
    # Canonical history may have been extended by the trusted callback, so use its
    # returned fresh pins; no unobserved mutable pointer is silently re-pinned.
    catalog = build_catalog(
        plan,
        history_locators + list(carried_states.values()) + prepared_states,
        root / "catalog.json",
        max_bytes=max_window_bytes,
        publication_shards=selected,
        owned_pair_roots=sorted(owned),
        api=api,
    )
    result = {
        "schema": "prepared_block_window_receipt_v1",
        "catalog": catalog,
        "plan_sha256": plan.sha256,
        "cursor": unchanged_cursor,
        "preparation_only": True,
        "trainer_seconds_charged": 0,
        "selected_shards": selected,
        "phase_costs": phases,
        "stop_reason": stop_reason,
        "payload_bytes": used,
        "max_window_bytes": max_window_bytes,
        "reserve_free_bytes": reserve_free_bytes,
        "target_window_bytes": target_window_bytes,
        "required_shards": list(required_shards),
    }
    api.atomic_json(root / "preparation-receipt.json", result)
    return result



_ARCHIVE_PROFILE = "exact_catalog_cache_state_inventory_v1"
_ARCHIVE_MEMBER_BYTES = 16 * 1024**2
_ARCHIVE_CHUNK_BYTES = 1024**2


def _archive_stat(value):
    return tuple(getattr(value, key) for key in (
        "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_gid",
        "st_size", "st_mtime_ns", "st_ctime_ns"))


def _cache_state_archive_inventory(locators):
    """Finite exact catalog inventory; match the normal16MiB metadata cap."""
    members, identities, seen = [], {}, set()
    for locator in locators:
        _require(isinstance(locator, dict) and set(locator) == {"path", "sha256"},
                 "exact original cache-state locator required")
        path = Path(locator["path"])
        info = path.lstat()
        _require(path.is_absolute() and path.resolve() == path
                 and path.name == "cache-state.json" and stat.S_ISREG(info.st_mode)
                 and info.st_nlink == 1 and info.st_size <= _ARCHIVE_MEMBER_BYTES
                 and str(path) not in seen, "bounded original cache-state metadata required")
        seen.add(str(path)); identities[str(path)] = _archive_stat(info)
        members.append({"original": copy.deepcopy(locator), "bytes": info.st_size})
    return {"profile": _ARCHIVE_PROFILE, "members": members,
            "total_bytes": sum(row["bytes"] for row in members)}, identities


def _copy_cache_state_archive(locator, snapshot, member, identity):
    """Stream exact metadata bytes; no tensor/model or cache ownership changes."""
    path, snapshot = Path(locator["path"]), Path(snapshot)
    _require(_archive_stat(path.lstat()) == identity, "cache-state changed before archive")
    _require(snapshot.parent.resolve() == snapshot.parent, "canonical archive directory required")
    digest, count = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source, snapshot.open("xb") as output:
        _require(_archive_stat(os.fstat(source.fileno())) == identity,
                 "cache-state changed while opening archive")
        for block in iter(lambda: source.read(_ARCHIVE_CHUNK_BYTES), b""):
            count += len(block)
            _require(count <= member["bytes"] and count <= _ARCHIVE_MEMBER_BYTES,
                     "cache-state archive exceeded exact inventory")
            digest.update(block); output.write(block)
        _require(count == member["bytes"] and digest.hexdigest() == locator["sha256"]
                 and _archive_stat(os.fstat(source.fileno())) == identity
                 and _archive_stat(path.lstat()) == identity,
                 "cache-state archive source identity or bytes changed")
        output.flush(); os.fsync(output.fileno())
    return {"path": str(snapshot), "sha256": digest.hexdigest()}


def _read_cache_state_archive(record):
    """Bound before IO and authenticate the exact immutable JSON read."""
    path = Path(record["path"])
    before = path.lstat()
    _require(path.is_absolute() and path.resolve() == path and stat.S_ISREG(before.st_mode)
             and before.st_nlink == 1 and before.st_size <= _ARCHIVE_MEMBER_BYTES,
             "bounded original cache-state archive required")
    identity = _archive_stat(before)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        _require(_archive_stat(os.fstat(stream.fileno())) == identity,
                 "cache-state archive changed on open")
        raw = stream.read(_ARCHIVE_MEMBER_BYTES + 1)
        _require(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == record["sha256"]
                 and _archive_stat(os.fstat(stream.fileno())) == identity
                 and _archive_stat(path.lstat()) == identity,
                 "cache-state archive identity or SHA256 differs")
    return json.loads(raw)

def retire_consumed_window(
    plan,
    cursor_payload,
    catalog_locator,
    *,
    release,
    authorization,
    merge_seals,
    replay_admission=None,
    reservation=None,
    api=lifecycle,
):
    """Retire ONLY fully consumed owned pair publications through original evict.

    Save exact old cache-state bytes before mutation. All native/replay recovery
    gates run in the original consumer; canonical history acknowledges the new
    recovery proof BEFORE deletion. Original metadata/seals and empty roots stay.
    """
    import uuid

    authorization()
    _released(release)
    _require(callable(merge_seals), "serialized recovery-history merge callback required")
    catalog_path = _checked(catalog_locator, api).resolve()
    catalog = json.loads(catalog_path.read_text())
    # Reuse source binding checks without opening a dataset or accepting stale pins.
    reader = PreparedCache(plan, catalog_locator, api=api)
    before_cursor = copy.deepcopy(cursor_payload)
    upcoming = future_shards(plan, cursor_payload, reader.state["seals"], api=api)
    protected = set(upcoming[:2])
    if reservation is not None:
        _require(
            reservation.get("plan_sha256") == plan.sha256
            and reservation.get("cursor") == cursor_payload
            and isinstance(reservation.get("shard_ids"), list)
            and set(reservation["shard_ids"]) <= set(plan.shards),
            "retirement pending reservation differs from exact paid cursor",
        )
        protected.update(reservation["shard_ids"])
    root = catalog_path.parent
    output = root / ("retirement-" + uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    archives, retired, kept, histories, remaining, owned_states = [], [], {}, [], [], []
    owned = set(catalog.get("owned_pair_roots", []))
    _require(
        owned
        <= {str(Path(row["path"]).resolve().parent) for row in catalog["source_cache_states"]},
        "retirement ownership not joined to catalog source cache states",
    )
    histories = [
        copy.deepcopy(row)
        for row in catalog["source_cache_states"]
        if str(Path(row["path"]).resolve().parent) not in owned
    ]
    archive_budget, archive_identities = _cache_state_archive_inventory(catalog["source_cache_states"])
    context = {
        "schema": "prepared_retirement_context_v1",
        "cache_state_archive_budget": archive_budget,
        "catalog": catalog_locator,
        "cursor": before_cursor,
        "reservation": copy.deepcopy(reservation),
        "owned_pair_roots": sorted(owned),
    }
    api.atomic_json(output / "context.json", context)
    progress_record = {
        "schema": "prepared_retirement_progress_v1",
        "catalog": catalog_locator,
        "cache_owners": {},
        "deleting": {},
        "canonical_history": histories,
    }
    api.atomic_json(output / "progress.json", progress_record)
    bytes_before = api.tree_bytes(root)
    payload_removed = 0
    progress = cursor_payload["progress"].get(str(cursor_payload["epoch"]), {})
    for locator, member in zip(catalog["source_cache_states"], archive_budget["members"]):
        snapshot = output / ("cache-state-" + str(len(archives)) + ".json")
        archived = _copy_cache_state_archive(locator, snapshot, member,
                                             archive_identities[locator["path"]])
        _require(api.pin(snapshot) == archived, "retirement state archive differs")
        archives.append({"original": locator, "snapshot": archived})
    for locator in catalog["source_cache_states"]:
        _require(_archive_stat(Path(locator["path"]).lstat()) == archive_identities[locator["path"]],
                 "cache-state inventory changed before archive commit")
    api.atomic_json(output / "source-state-archives.json", archives)
    for locator in catalog["source_cache_states"]:
        state_path = Path(locator["path"]).resolve()
        # Original canonical cache is never part of this cleanup. Only explicit
        # prepared pair roots owned by this very immutable catalog are eligible.
        if str(state_path.parent) not in owned:
            continue
        _require(
            state_path.parent.name.startswith("pair-"), "retirement pair root identity differs"
        )
        state_path = _checked(locator, api)
        state = json.loads(state_path.read_text())
        _require(
            state.get("plan_sha256") == plan.sha256 and state.get("owner") is None,
            "retirement requires same-plan released original pair cache",
        )
        candidates = []
        for shard in state["resident"]:
            if shard not in catalog["publications"]:
                kept[shard] = "not-selected-by-this-catalog"
                continue
            _require(
                state["seals"][shard]["identity"] == reader.state["seals"][shard]["identity"],
                "retirement original seal changed",
            )
            chains = plan.shards[shard]
            complete = all(
                plan.chains[cid]["split"] == cursor_payload["split"]
                and progress.get(cid, 0) == len(state["seals"][shard]["identity"][cid]["anchors"])
                for cid in chains
            )
            if shard in protected or not complete:
                kept[shard] = "next-reservation-or-incomplete-original-anchors"
            else:
                candidates.append(shard)
        if not candidates:
            owned_states.append(copy.deepcopy(locator))
            if state["resident"]:
                remaining.append(copy.deepcopy(locator))
            continue
        cache = api.ShardCache(plan, state_path.parent, replay_admission=replay_admission)
        owner = "prepared-retirement-" + uuid.uuid4().hex
        authorization()
        _released(release)
        progress_record["cache_owners"][str(state_path.resolve())] = owner
        api.atomic_json(output / "progress.json", progress_record)
        cache.acquire(owner, "restore")
        try:
            for shard in candidates:
                if cache.state["seals"][shard].get("recovery") is None:
                    _require(
                        replay_admission is not None,
                        "retirement requires admitted original recovery",
                    )
                    api.enable_replay_recovery(cache, shard)
                else:
                    cache.set_recovery(shard, cache.state["seals"][shard]["recovery"])
            cache.release(owner, released_proof=_released(release))
            recovery_state = api.pin(cache.state_path)
            fresh_history = merge_seals(recovery_state)
            _require(
                isinstance(fresh_history, list) and fresh_history,
                "retirement recovery proof must persist to canonical history",
            )
            for history_locator in fresh_history:
                history = json.loads(_checked(history_locator, api).read_text())
                _require(
                    history.get("plan_sha256") == plan.sha256 and history.get("owner") is None,
                    "retirement canonical history not released or same plan",
                )
                for shard in candidates:
                    _require(
                        history["seals"].get(shard) == cache.state["seals"][shard]
                        and history["seals"][shard].get("recovery") is not None,
                        "canonical history did not preserve admitted recovery proof",
                    )
            histories = fresh_history
            progress_record["canonical_history"] = fresh_history
            api.atomic_json(output / "progress.json", progress_record)
            cache.acquire(owner, "restore")
            for shard in candidates:
                authorization()
                _released(release)
                directory = cache._directory(cache.state["resident"][shard]["directory"])
                size = api.tree_bytes(directory)
                progress_record["deleting"][shard] = {
                    "state_path": str(cache.state_path.resolve()),
                    "directory": str(directory),
                    "recovery": cache.state["seals"][shard]["recovery"],
                    "payload_bytes": size,
                }
                api.atomic_json(output / "progress.json", progress_record)
                cache.evict(shard)
                payload_removed += size
                retired.append(shard)
            _require(cursor_payload == before_cursor, "retirement changed paid data cursor")
        finally:
            cache.close()
            if cache.state["owner"] == {"owner": owner, "phase": "restore"}:
                cache.release(owner, released_proof=_released(release))
        owned_states.append(api.pin(cache.state_path))
        if cache.state["resident"]:
            remaining.append(api.pin(cache.state_path))
    result = {
        "schema": "prepared_block_window_retirement_v1",
        "original_catalog": catalog_locator,
        "cursor": before_cursor,
        "trainer_seconds_charged": 0,
        "retired_shards": retired,
        "protected_shards": kept,
        "source_state_archives": archives,
        "payload_bytes_removed": payload_removed,
        "window_tree_bytes_before": bytes_before,
        "window_tree_bytes_after": api.tree_bytes(root),
        "canonical_history": histories,
        "remaining_cache_states": remaining,
        "owned_pair_roots": sorted(owned),
        "owned_cache_states": owned_states,
    }
    api.atomic_json(output / "receipt.json", result)
    return result


def reconcile_retirement(
    plan,
    cursor_payload,
    catalog_locator,
    *,
    release,
    authorization,
    merge_seals,
    replay_admission=None,
    reservation=None,
    api=lifecycle,
):
    """Reconcile ONLY a recorded interrupted original retirement, never re-pin blindly."""
    import uuid

    authorization()
    _released(release)
    catalog_path = _checked(catalog_locator, api).resolve()
    catalog = json.loads(catalog_path.read_text())
    reader = PreparedCache(plan, catalog_locator, api=api)
    current = {}
    changed = False
    for locator in catalog["source_cache_states"]:
        path = Path(locator["path"]).resolve()
        value = json.loads(path.read_text())
        current[str(path)] = (path, value)
        changed |= api.file_sha256(path) != locator["sha256"]
    if not changed:
        return {
            "schema": "prepared_retirement_reconciliation_v1",
            "changed": False,
            "catalog": catalog_locator,
            "cursor": copy.deepcopy(cursor_payload),
        }
    # A complete context+archive is mandatory; an unfinished observer snapshot
    # cannot authorize any current-state mutation or new source pin.
    accepted = None
    for directory in sorted(catalog_path.parent.glob("retirement-*")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        if not all(
            (directory / name).is_file()
            for name in ("context.json", "progress.json", "source-state-archives.json")
        ):
            continue
        context = json.loads((directory / "context.json").read_text())
        if context.get("catalog") != catalog_locator:
            continue
        _require(
            context.get("schema") == "prepared_retirement_context_v1"
            and context.get("cursor") == cursor_payload
            and context.get("reservation") == reservation
            and context.get("owned_pair_roots") == sorted(catalog["owned_pair_roots"]),
            "interrupted retirement cursor/reservation/ownership context differs",
        )
        archives = json.loads((directory / "source-state-archives.json").read_text())
        _require(
            [row["original"] for row in archives] == catalog["source_cache_states"],
            "retirement archive inventory differs from original catalog",
        )
        budget = context.get("cache_state_archive_budget")
        if budget is not None:
            _require(isinstance(budget, dict) and set(budget) == {"profile", "members", "total_bytes"}
                     and budget["profile"] == _ARCHIVE_PROFILE
                     and isinstance(budget["members"], list)
                     and [row.get("original") for row in budget["members"]] == catalog["source_cache_states"]
                     and all(set(row) == {"original", "bytes"} and type(row["bytes"]) is int
                             and 0 <= row["bytes"] <= _ARCHIVE_MEMBER_BYTES for row in budget["members"])
                     and type(budget["total_bytes"]) is int
                     and budget["total_bytes"] == sum(row["bytes"] for row in budget["members"]),
                     "exact original metadata archive inventory differs")
        baseline = {}
        total = 0
        for index, row in enumerate(archives):
            snapshot = Path(row["snapshot"]["path"])
            _require(
                snapshot.parent == directory.resolve()
                and snapshot.name == "cache-state-" + str(index) + ".json"
                and row["snapshot"]["sha256"] == row["original"]["sha256"],
                "retirement archive does not preserve exact original source bytes",
            )
            size = snapshot.stat().st_size
            total += size
            _require(size <= _ARCHIVE_MEMBER_BYTES and
                     (total <= 32 * 1024**2 if budget is None else
                      size == budget["members"][index]["bytes"] and total <= budget["total_bytes"]),
                     "retirement metadata archive bound exceeded")
            baseline[str(Path(row["original"]["path"]).resolve())] = _read_cache_state_archive(row["snapshot"])
        _require(budget is None or total == budget["total_bytes"],
                 "retirement archive total differs from exact inventory")
        progress_record = json.loads((directory / "progress.json").read_text())
        _require(
            progress_record.get("schema") == "prepared_retirement_progress_v1"
            and progress_record.get("catalog") == catalog_locator,
            "interrupted retirement progress identity differs",
        )
        accepted = directory, baseline, progress_record
        break
    _require(accepted is not None, "changed window has no complete authentic retirement context")
    directory, baseline, progress_record = accepted
    owned = set(catalog["owned_pair_roots"])
    protected = set(future_shards(plan, cursor_payload, reader.state["seals"], api=api)[:2])
    if reservation is not None:
        _require(
            reservation.get("plan_sha256") == plan.sha256
            and reservation.get("cursor") == cursor_payload,
            "retirement recovery pending reservation differs",
        )
        protected.update(reservation["shard_ids"])
    progress = cursor_payload["progress"].get(str(cursor_payload["epoch"]), {})

    def eligible(shard):
        return (
            shard in catalog["publications"]
            and shard not in protected
            and all(
                plan.chains[cid]["split"] == cursor_payload["split"]
                and progress.get(cid, 0)
                == len(reader.state["seals"][shard]["identity"][cid]["anchors"])
                for cid in plan.shards[shard]
            )
        )

    removed, interrupted = [], []
    # Validate EVERY state transition before changing any source or accepting a pin.
    for name, (path, value) in current.items():
        old = baseline[name]
        _require(
            set(value) == set(old)
            and value.get("schema") == old.get("schema")
            and value.get("plan_sha256") == plan.sha256
            and set(value["seals"]) == set(old["seals"]),
            "retirement recovery cache schema/plan/seal inventory changed",
        )
        allowed_owner = {"owner": progress_record["cache_owners"].get(name), "phase": "restore"}
        _require(
            value.get("owner") is None
            or (
                str(path.parent) in owned
                and allowed_owner["owner"] is not None
                and value["owner"] == allowed_owner
            ),
            "retirement recovery cache owner differs from recorded original owner",
        )
        _require(
            set(value["resident"]) <= set(old["resident"])
            and all(
                value["resident"][shard] == old["resident"][shard] for shard in value["resident"]
            ),
            "retirement recovery added or changed a physical publication",
        )
        gone = set(old["resident"]) - set(value["resident"])
        _require(
            not gone or str(path.parent) in owned,
            "retirement recovery changed unowned canonical publications",
        )
        for shard, seal in value["seals"].items():
            previous = old["seals"][shard]
            _require(
                seal["identity"] == previous["identity"]
                and seal.get("original_metadata") == previous.get("original_metadata")
                and set(seal) == set(previous),
                "retirement recovery immutable original seal changed",
            )
            proof_changed = seal.get("recovery") != previous.get("recovery")
            if proof_changed:
                _require(
                    previous.get("recovery") is None
                    and seal.get("recovery") is not None
                    and eligible(shard),
                    "retirement recovery changed unrelated proof",
                )
            if (
                proof_changed
                or shard in gone
                or progress_record["deleting"].get(shard, {}).get("state_path") == name
            ):
                _require(
                    eligible(shard) and seal.get("recovery") is not None,
                    "retirement recovery lacks consumed original recovery proof",
                )
                audit = api.ShardCache.__new__(api.ShardCache)
                audit.plan, audit.root, audit.state = plan, path.parent, copy.deepcopy(value)
                audit.replay_admission, audit._save = replay_admission, lambda: None
                api.ShardCache.set_recovery(audit, shard, seal["recovery"])
        for shard in gone:
            intent = progress_record["deleting"].get(shard)
            _require(
                intent is not None
                and intent["state_path"] == name
                and intent["recovery"] == value["seals"][shard]["recovery"]
                and not Path(old["resident"][shard]["directory"]).exists(),
                "retirement recovery removed unrecorded or surviving physical bytes",
            )
            removed.append(shard)
        for shard, intent in progress_record["deleting"].items():
            if intent["state_path"] == name and shard in value["resident"]:
                _require(
                    eligible(shard)
                    and intent["directory"] == value["resident"][shard]["directory"]
                    and intent["recovery"] == value["seals"][shard]["recovery"],
                    "interrupted original eviction intent differs",
                )
                interrupted.append((name, shard))
    # Source-validated proof additions are acknowledged by canonical history before
    # replaying any unfinished ORIGINAL eviction. No paid cursor changes occur.
    histories = []
    for name, (path, value) in current.items():
        if str(path.parent) not in owned:
            continue
        cache = api.ShardCache(plan, path.parent, replay_admission=replay_admission)
        if cache.state["owner"] is not None:
            cache.release(cache.state["owner"]["owner"], released_proof=_released(release))
        fresh = merge_seals(api.pin(cache.state_path))
        _require(
            isinstance(fresh, list) and fresh,
            "retirement recovery canonical acknowledgment missing",
        )
        for locator in fresh:
            history = json.loads(_checked(locator, api).read_text())
            _require(
                history.get("plan_sha256") == plan.sha256 and history.get("owner") is None,
                "retirement recovery canonical history identity differs",
            )
            for shard in removed + [item[1] for item in interrupted]:
                _require(
                    history["seals"][shard]["identity"] == reader.state["seals"][shard]["identity"]
                    and history["seals"][shard].get("recovery") is not None,
                    "retirement recovery proof not persisted canonically",
                )
        histories = fresh
        pending = [shard for state_name, shard in interrupted if state_name == name]
        if pending:
            owner = progress_record["cache_owners"][name]
            cache.acquire(owner, "restore")
            try:
                for shard in pending:
                    authorization()
                    _released(release)
                    target = Path(cache.state["resident"][shard]["directory"])
                    if not target.exists():
                        # Original evict removed the directory before its final
                        # atomic state save. Recreate ONLY an empty deletion
                        # target so the same original gated evict can commit.
                        _require(
                            target.parent.resolve() == cache.root,
                            "interrupted eviction canonical directory differs",
                        )
                        target.mkdir(exist_ok=False)
                    cache.evict(shard)
                    removed.append(shard)
            finally:
                cache.close()
                cache.release(owner, released_proof=_released(release))
    paths = [api.pin(path) for path, _ in current.values()]
    surviving = set().union(
        *(set(json.loads(_checked(row, api).read_text())["resident"]) for row in paths)
    )
    output = catalog_path.parent / ("retirement-recovery-" + uuid.uuid4().hex)
    output.mkdir(exist_ok=False)
    fresh_catalog = build_catalog(
        plan,
        paths,
        output / "catalog.json",
        max_bytes=catalog["max_window_bytes"],
        publication_shards=set(catalog["publications"]) & surviving,
        owned_pair_roots=sorted(owned),
        api=api,
    )
    result = {
        "schema": "prepared_retirement_reconciliation_v1",
        "changed": True,
        "catalog": fresh_catalog,
        "original_catalog": catalog_locator,
        "context": api.pin(directory / "context.json"),
        "cursor": copy.deepcopy(cursor_payload),
        "removed_shards": sorted(set(removed)),
        "canonical_history": histories,
    }
    api.atomic_json(output / "receipt.json", result)
    return result


def recover_registered_preparation(
    plan,
    registrations,
    output,
    *,
    run_dir,
    cursor_payload,
    release,
    authorization,
    merge_seals,
    validate_registration,
    max_bytes,
    api=lifecycle,
):
    """Re-admit only registered original publications after unfinished PREP.

    Registration is durable before a teacher starts. No directory discovery or
    missing publication is inferred; an incomplete declared directory fails shut.
    """
    authorization()
    _released(release)
    _require(callable(validate_registration), "registered producer/runtime validator required")
    directory = Path(run_dir).resolve()
    states = []
    for root, registration in registrations.items():
        path = Path(root)
        window = Path(registration["window_root"])
        _require(
            registration.get("schema") == "prepared_pair_registration_v1"
            and path.is_absolute()
            and path.resolve() == path
            and path.parent == window
            and window.parent.name == "prepared-data"
            and window.parent.parent.parent == directory / "direct-runtime"
            and path.name.startswith("pair-")
            and len(path.name) == 37
            and all(char in "0123456789abcdef" for char in path.name[5:]),
            "registered prepared pair is outside its exact run/window scope",
        )
        validate_registration(registration)
        initial = json.loads(_checked(registration["initial_state"], api).read_text())
        requested = registration["requested_shards"]
        _require(
            initial.get("schema") == "block_shard_cache_v1"
            and initial.get("plan_sha256") == plan.sha256
            and initial.get("owner") is None
            and initial.get("resident") == {}
            and isinstance(requested, list)
            and 0 < len(requested) <= 2
            and len(set(requested)) == len(requested)
            and set(requested) <= set(plan.shards),
            "registered empty original cache or requested shard identity differs",
        )
        state_path = path / "cache-state.json"
        _require(
            state_path.is_file() and not state_path.is_symlink(),
            "registered original cache state is missing",
        )
        state = json.loads(state_path.read_text())
        _require(
            set(state) == set(initial)
            and state.get("schema") == initial["schema"]
            and state.get("plan_sha256") == plan.sha256
            and all(state["seals"].get(shard) == seal for shard, seal in initial["seals"].items())
            and set(state["seals"]) - set(initial["seals"]) == set(state["resident"])
            and set(state["resident"]) <= set(requested),
            "registered original cache seal/publication transition differs",
        )
        owner = state.get("owner")
        if owner is not None:
            name = owner.get("owner", "")
            _require(
                owner.get("phase") == "restore"
                and name.startswith("controller-")
                and len(name) == 43
                and all(c in "0123456789abcdef" for c in name[11:]),
                "registered original restore owner differs",
            )
        for shard in requested:
            target = path / shard
            _require(
                target.resolve().parent == path, "registered shard canonical directory differs"
            )
            if shard not in state["resident"]:
                _require(not target.exists(), "incomplete registered original publication retained")
            else:
                _require(
                    state["resident"][shard]["directory"] == str(target),
                    "registered publication moved from its original canonical directory",
                )
                if plan.production:
                    original = state["seals"][shard].get("original_metadata")
                    _require(
                        original is not None,
                        "incomplete registered original source metadata retained",
                    )
                    metadata = json.loads(_checked(original, api).read_text())
                    _require(
                        metadata.get("plan_sha256") == plan.sha256
                        and metadata.get("shard") == shard
                        and metadata.get("identity") == state["seals"][shard]["identity"]
                        and metadata.get("original_directory") == str(target),
                        "registered original metadata seal/source differs",
                    )
        states.append((state_path, state))
    # All registration/source/immutable-state checks precede ownership changes.
    locators = []
    for path, state in states:
        cache = api.ShardCache(plan, path.parent)
        if state["owner"] is not None:
            cache.release(state["owner"]["owner"], released_proof=_released(release))
        locators.append(api.pin(cache.state_path))
        cache.close()
    catalog = build_catalog(
        plan, locators, output, max_bytes=max_bytes, owned_pair_roots=list(registrations), api=api
    )
    admitted = json.loads(_checked(catalog, api).read_text())
    future_shards(plan, cursor_payload, admitted["seals"], api=api)
    histories = []
    for locator in locators:
        state = json.loads(_checked(locator, api).read_text())
        fresh = merge_seals(locator)
        _require(
            isinstance(fresh, list) and fresh,
            "recovered publication canonical acknowledgment missing",
        )
        for record in fresh:
            history = json.loads(_checked(record, api).read_text())
            _require(
                history.get("owner") is None
                and history.get("plan_sha256") == plan.sha256
                and all(
                    shard in history["seals"] and _same_original_seal(history["seals"][shard], seal)
                    for shard, seal in state["seals"].items()
                ),
                "recovered original publication was not retained canonically",
            )
        histories = fresh
    result = {
        "schema": "registered_preparation_recovery_v1",
        "catalog": catalog,
        "cache_states": locators,
        "owned_pair_roots": list(registrations),
        "canonical_history": histories,
        "cursor": copy.deepcopy(cursor_payload),
        "trainer_seconds_charged": 0,
    }
    api.atomic_json(Path(output).parent / "recovery.json", result)
    return result


def migrate_catalog(plan, locator, output, *, expected_producer_source, api=lifecycle):
    """Re-admit old publications with current ORIGINAL validators, never old code.

    The exclusive runtime must bind expected_producer_source to its authenticated
    predecessor runtime. Old bytes must still exist at that immutable source pin.
    Stale cache-state pins require retirement reconciliation, not blind migration.
    """
    old = json.loads(_checked(locator, api).read_text())
    _require(
        old.get("schema") == SCHEMA and old.get("plan_sha256") == plan.sha256,
        "catalog migration logical identity differs",
    )
    _require(
        old.get("producer_source") == expected_producer_source,
        "catalog migration predecessor producer identity differs",
    )
    _checked(expected_producer_source, api)
    for field, cls in (
        ("original_provider_source", api.ShardCache),
        ("original_data_source", api.BlockDataset),
    ):
        _checked(old[field], api)
        _require(
            Path(old[field]["path"]).resolve() == Path(inspect.getfile(cls)).resolve(),
            "catalog migration original scientific source role differs",
        )
    fresh = build_catalog(
        plan,
        old["source_cache_states"],
        output,
        max_bytes=old["max_window_bytes"],
        publication_shards=old["publications"],
        owned_pair_roots=old["owned_pair_roots"],
        api=api,
    )
    rebuilt = json.loads(_checked(fresh, api).read_text())
    _require(
        all(
            rebuilt[field] == old[field]
            for field in ("publications", "seals", "owned_pair_roots", "payload_bytes")
        ),
        "catalog migration changed original publication or ownership identity",
    )
    result = {
        "schema": "prepared_catalog_migration_v1",
        "original_catalog": locator,
        "catalog": fresh,
        "previous_producer": expected_producer_source,
        "current_producer": rebuilt["producer_source"],
        "original_scientific_admission_rechecked": True,
    }
    api.atomic_json(Path(output).parent / "migration.json", result)
    return result


def main(argv=None):
    from .direct_training_runtime import main as runtime_main

    return runtime_main(argv)
