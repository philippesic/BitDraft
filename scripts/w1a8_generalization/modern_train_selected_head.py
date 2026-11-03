"""Explicit512 FIT-selected head-row capacity contrast from resident newdata3840.

Both sign-learning arms use the same48-group expanded FIT CE population and
original356-token replay bank. The false arm freezes external head masters;
the true arm learns512 reachable-label-selected rows. Both start at3840 and
retain the original balanced1920 ancestry and19200-update horizon.
The parent scale offsets, Adam moments, RNG and cumulative accounting migrate
unchanged. Selected head masters are external; false retains scale-only Adam.
"""

from __future__ import annotations

# Project path bootstrap precedes optional Torch/project imports.
# ruff: noqa: E402
import argparse
import copy
import io
import json
import math
import os
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path("/home/philip/binary-eagle-decoding")
SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / "src"), str(PROJECT / "third_party/llama.cpp/gguf-py")]
import modern_resident_rows as resident_helpers
import modern_selected_head as head_helpers
import modern_train_expanded_ce as frozen_ce_helpers
import modern_train_expanded_ce_resident as resident_trainer
import modern_train_native_replay as native_helpers
import modern_train_protected as protected_helpers
import torch
from modern_actor import (
    SCALE_STATE_SCHEMA,
    _ancestry,
    build_modern_actor,
    file_pin,
)
from modern_actual_data import ActualDataset
from modern_actual_union import Audit, UnionActualDataset, build_union
from modern_resident_rows import ResidentRows
from modern_selected_head import SelectedHead, absolute_map, select_head_rows
from modern_selected_sign import _check_frozen
from modern_train_protected import (
    ProtectedReplaySchedule,
    cpu,
    equal,
    forward_semantics,
    normalized_replay_loss,
    require,
    validate_original_reference,
)
from modern_train_protected import (
    write as original_write,
)

from w1a1_eagle.native_step import ExternalNativeF32Embedding
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH

V1_OBJECTIVE = (
    "Original CE sum/fixed4 from2 ordinary fit windows plus retention*mean "
    "protected-token margin deficit on1 independent group-balanced fit replay; "
    "no retention on ordinary CE windows."
)
WARM_CURSOR = 1920
MARGIN_FLOOR = 0.1
SCHEMA = "w1a8_selected_head_native_replay_checkpoint_v1"
ARTIFACTS = Path("/mnt/c/Users/philip/binary-eagle-decoding-artifacts")
STORAGE_FLOOR = 2 * 1024**3 + 64 * 1024**2
OLD_MANIFEST_SHA256 = "ccc5fb9e89c86daece91e9b29327a285a71c3f35362e9efd60c40b4a92d8f5aa"
REFERENCE_SHA256 = "f8a7809fb1b05c9adfb7eb1ed811d35c45b42e481d6b75da1f4395930a53a643"
NATIVE_TRAINER_SHA256 = "1e47569cfb609da4003b30a78f96821f95faeb23fc429ebb3b01903a13b4245a"
WARM_CHECKPOINT_SHA256 = "7842ede528af87bb9716e4c0738c6fd5a061dc625266fa61d0d6f2ac42abc5da"
WARM_CHECKPOINT = (
    ARTIFACTS / "w1a8-modern-protected-20261103-01/all-replay01/checkpoint-00001920.pt"
)
FROZEN_CE_SHA256 = "76420c95fb0e5cd0550c90bd9b22319bb04dd4548cfd241c4b5d0adf74fd2315"
RESIDENT_HELPER_SHA256 = "18ae284e46bc04a29e917217d39a77d6b1540003728c9895589ccbf781ee0130"
PROOF_SCHEMA = "w1a8_resident_union_parity_v1"
PROOF_STATUS = "REAL_UNION_RESIDENT_ROWS_BYTE_AND_WINDOW_PARITY_PASS"
RESIDENT_TRAINER_SHA256 = "ab410b5fd5196eef40257c87255ecdff0f8b2fdf83e8a9cd8facf5000e6e9263"
HEAD_HELPER_SHA256 = "34b24bd17cf45834b48fb7fa1044644d65a1f6632e843ed8bc7c22d24a9e5739"
HEAD_START = 3840
PARENT_CHECKPOINT = ARTIFACTS / "w1a8-expanded-ce-resident-new24-3840-01/checkpoint-00003840.pt"
PARENT_SHA256 = "f4a8d2d31b1016816827d38884f61bbbf881cf029a1a44cc48f94af71e838475"
PARENT_BYTES = 1196657
PARENT_GGUF = ARTIFACTS / "w1a8-expanded-ce-resident-new24-3840-01/model.gguf"
PARENT_GGUF_SHA256 = "0b37d4247bb406b01fba7bae402b269477e62c3e984ac08da25f34c8612319fa"
PARENT_GGUF_BYTES = 40802176


def reachable_fit_records(windows):
    """Select capacity from exactly the CE-reachable FIT labels, never DEV."""
    result = []
    for w in windows:
        require(w["role"] == "fit", "head row selection accepts FIT windows only")
        require(
            len(w["labels"]) == len(w["native_supervision_reachable_mask"]),
            "head selector reachability depth differs",
        )
        require(
            all(type(value) is bool for value in w["native_supervision_reachable_mask"]),
            "head selector reachability mask must be Boolean",
        )
        labels = [
            label
            for label, reachable in zip(
                w["labels"], w["native_supervision_reachable_mask"], strict=True
            )
            if reachable
        ]
        if labels:
            result.append(
                {
                    "role": "fit",
                    "group_id": w["group_id"],
                    "labels": labels,
                    "source_split": "TRAIN",
                }
            )
    require(result, "no reachable FIT labels for head selection")
    return result


def scale_scope(actor):
    paths = list(CANDIDATE_D_BASE_TO_PATH.values())
    require(
        not actor.scale_only and set(actor.linears) == set(paths) and len(paths) == 9,
        "head trainer requires explicit non-scale-only nine-projection actor",
    )
    return paths, tuple(actor.linears[p].scale_offset for p in paths)


def load_parent_scales(actor, state):
    """Validate the original scale schema on the new explicit non-scale-only actor."""
    paths, _ = scale_scope(actor)
    require(
        set(state) == {"schema", "ancestry", "scale_offsets"}
        and state["schema"] == SCALE_STATE_SCHEMA
        and state["ancestry"] == _ancestry(actor),
        "parent scale-state ancestry/schema differs; learned-O ancestry is forbidden",
    )
    require(
        set(state["scale_offsets"]) == set(CANDIDATE_D_BASE_TO_PATH),
        "parent nine-scale mask differs",
    )
    _check_frozen(actor, include_selected_initial=True)
    pending = []
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        value, destination = state["scale_offsets"][base], actor.linears[path].scale_offset
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.shape == destination.shape
            and value.is_contiguous()
            and not value.requires_grad
            and torch.isfinite(value).all().item(),
            "invalid parent scale tensor: " + base,
        )
        pending.append((destination, value.to(destination.device).clone()))
    with torch.no_grad():
        for destination, value in pending:
            destination.copy_(value)
    require(len(paths) == len(pending) == 9, "parent scale transfer incomplete")


def make_optimizer(selected):
    """Keep original one-group Adam literally intact in the disabled control."""
    selected.check()
    paths, scales = scale_scope(selected.actor)
    groups = [{"params": scales, "lr": 1e-5}]
    if selected.learn_signs:
        groups.append({"params": (selected.latent,), "lr": 1e-3})
    optimizer = torch.optim.Adam(groups, lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
    clipped = scales + ((selected.latent,) if selected.learn_signs else ())
    require(id(selected.latent) not in {id(p) for p in scales}, "head master aliases a scale")
    return optimizer, paths, clipped


def migrate_parent_optimizer(saved, optimizer, old_paths, new_paths, learn_signs):
    migrated = resident_trainer.named_optimizer_state(saved, old_paths, new_paths)
    if learn_signs:
        fresh = optimizer.state_dict()
        require(len(fresh["param_groups"]) == 2 and not fresh["state"], "fresh head Adam required")
        extra = copy.deepcopy(fresh["param_groups"][1])
        extra["params"] = [max(migrated["param_groups"][0]["params"]) + 1]
        migrated["param_groups"].append(extra)
    else:
        require(len(optimizer.param_groups) == 1, "disabled head must have scale-only Adam")
    optimizer.load_state_dict(migrated)


def head_origin(parent, parent_pin):
    return {
        "checkpoint": copy.deepcopy(parent_pin),
        "parent_contract": copy.deepcopy(parent["contract"]),
        "global_start_cursor": HEAD_START,
        "initial_costs": dict(parent["costs"]),
        "initial_exposure": dict(parent["exposure"]),
        "initial_ce_stage_exposure": copy.deepcopy(parent["ce_stage_exposure"]),
        "initial_trainer_seconds": parent["trainer_seconds"],
    }


def validate_parent(parent, expected, pin):
    require(
        pin == {"path": str(PARENT_CHECKPOINT), "sha256": PARENT_SHA256, "bytes": PARENT_BYTES},
        "exact resident newdata3840 parent required",
    )
    require(
        parent.get("cursor") == HEAD_START and parent.get("schema") == resident_trainer.SCHEMA,
        "parent requires resident3840 scale-only checkpoint",
    )
    require(
        expected["trainer"] == file_pin(resident_trainer.__file__)
        and expected["trainer"]["sha256"] == RESIDENT_TRAINER_SHA256,
        "parent resident trainer source differs",
    )
    resident_trainer.validate_resume(parent, expected)
    require(
        expected["expanded_ce"] is True
        and expected["optimizer_scope"] == "all"
        and expected["horizon"] == 19200
        and expected["seed"] == 37
        and expected["native_fit_bank_tokens"] == 356
        and expected["batch_windows"] == 2
        and expected["retention"] == 10.0
        and len(expected["optimized_scale_paths"]) == 9,
        "parent data/objective/optimizer contract differs",
    )
    require(
        len(parent["optimizer"]["param_groups"]) == 1
        and parent["optimizer"]["param_groups"][0]["lr"] == 1e-5,
        "parent requires original scale-only Adam1e-5",
    )


def sign_metrics(selected):
    initial = selected.head.latent_sign.detach().index_select(0, selected.rows)
    current = selected.latent.detach()
    return {
        "selected_rows": len(selected.rows),
        "changed_latents": int(current.ne(initial).sum()),
        "hard_sign_flips": int(current.ge(0).ne(initial.ge(0)).sum()),
        "max_abs_displacement": float((current - initial).abs().max()),
    }


def gradient_metrics(selected):
    value = selected.latent.grad
    require(
        selected.learn_signs and value is not None and value.shape == selected.latent.shape,
        "actual selected-head gradient required",
    )
    finite = bool(torch.isfinite(value).all())
    require(finite, "nonfinite selected-head gradient")
    return {
        "shape": list(value.shape),
        "finite": finite,
        "nonzero_elements": int(value.ne(0).sum()),
        "nonzero_rows": int(value.ne(0).any(-1).sum()),
        "norm": float(value.norm()),
        "max_abs": float(value.abs().max()),
    }


class CadencedGuard:
    """Cheap read-only checkpoints; every write requires fresh projected capacity.

    Local signal flags are checked on every invocation. Polls occur at the next
    guard checkpoint after their deadline; individual blocking operations retain
    their original behavior. Recovery-only writes may finish a paid checkpoint
    after a stop, while still enforcing fresh storage and exact byte reserves.
    """

    def __init__(
        self,
        project,
        output,
        *,
        control_path=None,
        c_filesystem=Path("/mnt/c"),
        stop_poll_seconds=0.25,
        disk_poll_seconds=1.0,
        clock=time.monotonic,
    ):
        require(
            0 < stop_poll_seconds <= 0.25 and 0 < disk_poll_seconds <= 1.0,
            "guard cadence exceeds declared bounds",
        )
        self.project, self.output = Path(project), Path(output)
        self.c_filesystem = Path(c_filesystem)
        self.control_path = (
            control_path or Path.home() / ".config/binary-eagle-decoding/gpu-control.json"
        )
        self.stop_poll_seconds, self.disk_poll_seconds = stop_poll_seconds, disk_poll_seconds
        self.clock = clock
        self.stop_requested = False
        self._external_stop = False
        self._last_stop = self._last_disk = -math.inf

    def request_stop(self, signum=None, frame=None):
        self.stop_requested = True

    def _raise_stop(self):
        if self.stop_requested or self._external_stop:
            raise RuntimeError("GPU stop requested")

    def _poll_stop(self, now, *, force=False):
        if force or now - self._last_stop >= self.stop_poll_seconds:
            c = json.loads(Path(self.control_path).read_text())["rtx5080"]
            self._external_stop = bool(
                c.get("pause_requested")
                or c.get("stop_all_requested")
                or (self.output / "STOP").exists()
            )
            self._last_stop = now

    def _poll_disk(self, now, *, force=False, write_bytes=0):
        if force or now - self._last_disk >= self.disk_poll_seconds:
            for filesystem, reserve in ((self.project, 0), (self.c_filesystem, write_bytes)):
                stat = os.statvfs(filesystem)
                if stat.f_bavail * stat.f_frsize < STORAGE_FLOOR + reserve:
                    raise RuntimeError("storage floor/write reserve reached: " + str(filesystem))
            self._last_disk = now

    def __call__(self):
        self._raise_stop()
        now = self.clock()
        self._poll_stop(now)
        self._raise_stop()
        self._poll_disk(now)
        self._raise_stop()

    def before_write(self, write_bytes, *, recovery=False):
        require(type(write_bytes) is int and write_bytes >= 0, "invalid output write reserve")
        if not recovery:
            self._raise_stop()
        now = self.clock()
        try:
            self._poll_stop(now, force=True)
        except (OSError, ValueError, KeyError):
            if not recovery:
                raise
            self._external_stop = True
        if not recovery:
            self._raise_stop()
        self._poll_disk(now, force=True, write_bytes=write_bytes)
        if not recovery:
            self._raise_stop()


def guarded_write(path, value, guard, *, recovery=False):
    """Reserve exact UTF-8 JSON temporary bytes before the original atomic writer."""
    size = len((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    guard.before_write(size, recovery=recovery)
    original_write(path, value)


def guarded_checkpoint(path, state, guard, *, recovery=False):
    """Serialize before reserving exact temporary bytes; verify before publication."""
    path = Path(path)
    payload = io.BytesIO()
    torch.save(state, payload)
    encoded = payload.getvalue()
    guard.before_write(len(encoded), recovery=recovery)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    require(equal(state, loaded), "checkpoint roundtrip differs")
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_resident_proof(proof_path, manifest_path, manifest):
    require(
        Path(proof_path).suffix == ".json" and 0 < Path(proof_path).stat().st_size <= 1024**2,
        "resident proof must be bounded JSON metadata",
    )
    pin = file_pin(proof_path)
    require(0 < pin["bytes"] <= 1024**2, "resident proof metadata size invalid")
    proof = json.loads(Path(proof_path).read_text())
    require(file_pin(proof_path) == pin, "resident proof changed while reading")
    require(
        proof.get("schema") == PROOF_SCHEMA and proof.get("status") == PROOF_STATUS,
        "real-union resident byte-parity admission required",
    )
    for key, path in (
        ("manifest", manifest_path),
        ("helper", resident_helpers.__file__),
        ("original_reader", Path(__file__).with_name("modern_actual_data.py")),
    ):
        require(proof.get(key) == file_pin(path), "resident proof source pin differs: " + key)
    for key in (
        "all_matrices_byte_equal",
        "all_sampled_materializations_byte_equal",
        "all_embeddings_byte_equal",
    ):
        require(proof.get(key) is True, "resident parity proof failed: " + key)
    descriptors = {
        (str(Path(ref["path"]).resolve()), ref["sha256"], ref["bytes"], tuple(ref["shape"]))
        for key in ("raw_files", "embedding_files")
        for ref in manifest[key]
    }
    require(
        type(proof.get("unique_descriptors")) is int
        and proof["unique_descriptors"] == len(descriptors) > 0,
        "resident proof matrix denominator differs",
    )
    require(
        type(proof.get("matrix_bytes_compared")) is int
        and proof["matrix_bytes_compared"] == sum(ref[2] for ref in descriptors) > 0,
        "resident proof matrix byte denominator differs",
    )
    require(
        type(proof.get("sampled_windows")) is int
        and proof["sampled_windows"] >= len(manifest["groups"]) > 0,
        "resident proof must sample at least every group",
    )
    require(
        type(proof.get("embedding_rows")) is int
        and proof["embedding_rows"] == len(manifest["required_embedding_ids"]) > 0,
        "resident proof embedding denominator differs",
    )
    return pin, proof


def bind_resident(data, config):
    require(isinstance(data, UnionActualDataset), "resident binding requires explicit union reader")
    require(not isinstance(data._rows, ResidentRows), "union already has a resident reader")
    cache = ResidentRows(**config)
    data._rows = cache
    return cache


def load_embeddings(data, device, guard):
    embeddings = {}
    for token in data.manifest["required_embedding_ids"]:
        guard()
        embeddings[token] = data.embedding_lookup(token, device=device)
    return embeddings


def resident_contract(config, proof_pin, proof):
    return {
        "config": copy.deepcopy(config),
        "proof": proof_pin,
        "proof_report": proof,
        "source": file_pin(resident_helpers.__file__),
        "frozen_ce_trainer": file_pin(frozen_ce_helpers.__file__),
        "runtime_accounting": "current process only; cold cache on resume is legitimate",
        "row_sha_checks": "unchanged original materializer; all row checks retained",
        "guard": {
            "stop_poll_seconds": 0.25,
            "read_only_disk_poll_seconds": 1.0,
            "storage_floor_bytes": STORAGE_FLOOR,
            "force_write_reserve": True,
        },
    }


def capture_protocol(union, audit):
    """Pin the new capture's actual HTTP requests and its frozen target references."""
    member = union["members"][1]["metadata"]
    root = Path(member["capture"]["path"]).parent
    freeze_path = str(root / "freeze.json")
    sources = {ref["path"]: ref for ref in member["source_pins"]}
    require(freeze_path in sources, "new capture freeze missing source pin")
    freeze_ref = sources[freeze_path]
    freeze = audit.read(freeze_ref)
    inputs = {ref["path"]: ref for ref in freeze["input_pins"]}
    result = [freeze_ref]
    new_groups = [
        entries[0]["group"]
        for entries in union["group_provenance"].values()
        if entries[0]["member"] == 1
    ]
    for group in new_groups:
        request_ref = file_pin(root / group["id"] / "request.json")
        reference_ref = file_pin(root / group["id"] / "target-reference.json")
        request = audit.read(request_ref)
        reference = audit.read(reference_ref)
        require(
            all(
                request.get(key) == value
                for key, value in {
                    "n_predict": 80,
                    "temperature": 0,
                    "seed": 1234,
                    "cache_prompt": False,
                    "return_tokens": True,
                    "stream": False,
                }.items()
            ),
            "new actual request differs from frozen80-token protocol",
        )
        require(
            inputs.get(reference["request"]["path"]) == reference["request"]
            and inputs.get(reference["response"]["path"]) == reference["response"],
            "new target-only request/response lacks frozen input pin",
        )
        require(
            audit.read(reference["request"]) == request,
            "new actual request differs from frozen target request",
        )
        result.extend([request_ref, reference_ref, reference["request"], reference["response"]])
    require(len(new_groups) >= 24, "new capture lacks24 request protocol receipts")
    return result


def controlled_views(original, union, original_pin):
    """Require an old-first audited union, preserving every original window.

    Storage indices may be interned by the union; map those indices explicitly
    rather than mistaking an array index for a semantic feature identity.
    build_union must separately authenticate all source bytes before this call.
    """
    members = union["members"]
    require(len(members) == 2, "exact old plus one independent new capture required")
    require(members[0]["manifest"] == original_pin, "original manifest pin/order changed")
    old_groups = {g["id"] for g in original["groups"]}
    old_components = {g["group_id"] for g in original["groups"]}
    require(len(old_groups) == len(original["groups"]), "duplicate original groups")
    new_groups = [g for g in union["groups"] if g["id"] not in old_groups]
    require(
        len(new_groups) >= 24
        and Counter(g["domain"] for g in new_groups)
        == {domain: len(new_groups) // 3 for domain in ("code", "reasoning", "prose")}
        and all(g["role"] == "fit" and g["group_id"] not in old_components for g in new_groups)
        and len({g["group_id"] for g in new_groups}) == len(new_groups),
        "new capture needs independent all-FIT groups, balanced at least8 per domain",
    )
    for group in union["groups"]:
        if group["id"] in old_groups:
            require(
                group == next(g for g in original["groups"] if g["id"] == group["id"]),
                "original group changed",
            )
    require(
        members[1]["metadata"]["capture"] != original["capture"]
        and members[1]["metadata"]["capture"]["path"] != original["capture"]["path"],
        "new groups require a genuinely independent actual capture",
    )
    new_ids = {g["id"] for g in new_groups}
    require(
        all(
            g["id"] in new_ids
            for g in (
                p["group"]
                for entries in union["group_provenance"].values()
                for p in entries
                if p["member"] == 1
            )
        )
        and set(members[1]["window_ids"]).isdisjoint(members[0]["window_ids"]),
        "new member reuses original groups or windows",
    )
    windows = {w["id"]: w for w in union["windows"]}
    require(len(windows) == len(union["windows"]), "duplicate union windows")
    old = []
    maps = members[0]["index_maps"]["raw_rows"]
    for window in original["windows"]:
        expected = copy.deepcopy(window)
        expected["prefix_raw_rows"] = [maps[i] for i in window["prefix_raw_rows"]]
        expected["seed_raw_row"] = maps[window["seed_raw_row"]]
        require(windows.get(window["id"]) == expected, "original FIT/DEV window changed")
        old.append(windows[window["id"]])
    old_ids = {w["id"] for w in old}
    extra = [w for w in union["windows"] if w["id"] not in old_ids]
    require(
        extra
        and all(w["role"] == "fit" and w["group"] in new_ids for w in extra)
        and {w["group"] for w in extra} == new_ids
        and {w["id"] for w in extra} == set(members[1]["window_ids"]),
        "every new window must be independent FIT; no extra DEV allowed",
    )
    require(not union["missing_embedding_ids"], "union has missing native F32 embeddings")
    require(
        set(original["required_embedding_ids"]) <= set(union["required_embedding_ids"]),
        "old embedding token coverage changed",
    )
    fit = [w for w in old if w["role"] == "fit"]
    dev = [w for w in old if w["role"] == "development"]
    return fit, dev, extra


def ce_order(windows):
    order = list(range(len(windows)))
    random.Random(37).shuffle(order)
    return order


def ce_at(windows, order, cursor):
    return [windows[order[(2 * cursor + offset) % len(order)]] for offset in range(2)]


def named_optimizer_state(saved_state, old_paths, new_paths):
    """Join Adam moments by scale path; never assume incidental integer IDs.

    Identity-order migration returns a deep copy without altering any Adam field.
    """
    require(
        len(old_paths) == len(new_paths) == 9
        and len(set(old_paths)) == 9
        and set(old_paths) == set(new_paths),
        "nine named optimizer scales differ",
    )
    result = copy.deepcopy(saved_state)
    require(len(result["param_groups"]) == 1, "one Adam parameter group required")
    group = result["param_groups"][0]
    params = group["params"]
    require(len(params) == len(set(params)) == 9, "Adam parameter IDs differ")
    require(set(result["state"]) == set(params), "Adam moments missing or extra")
    by_name = dict(zip(old_paths, params, strict=True))
    # load_state_dict maps this ordered group to the live parameters. Keep each
    # original state key attached to its original named scale.
    group["params"] = [by_name[path] for path in new_paths]
    return result


def empty_ce_exposure():
    return {
        kind: {"windows": 0, "supported_labels": 0, "groups": {}, "window_ids": {}}
        for kind in ("old", "new")
    }


def count_ce(exposure, window, supported, old_ids):
    row = exposure["old" if window["id"] in old_ids else "new"]
    row["windows"] += 1
    row["supported_labels"] += supported
    for key, value in (("groups", window["group_id"]), ("window_ids", window["id"])):
        row[key][value] = row[key].get(value, 0) + 1


def head_totals(cursor, costs, exposure, seconds, ce_stage_exposure, origin):
    result = stage_totals(cursor, costs, exposure, seconds, origin)
    deltas = {}
    for kind in ("old", "new"):
        before, after = origin["initial_ce_stage_exposure"][kind], ce_stage_exposure[kind]
        delta = {key: after[key] - before[key] for key in ("windows", "supported_labels")}
        for key in ("groups", "window_ids"):
            counts = {
                name: after[key].get(name, 0) - before[key].get(name, 0)
                for name in sorted(set(after[key]) | set(before[key]))
            }
            require(
                all(value >= 0 for value in counts.values()), "head CE exposure moved backwards"
            )
            delta[key] = {name: value for name, value in counts.items() if value}
        require(
            all(delta[key] >= 0 for key in ("windows", "supported_labels")),
            "head CE exposure moved backwards",
        )
        deltas[kind] = delta
    result["ce_stage_exposure"] = deltas
    return result


def validate_head_optimizer(state, cursor, learn_signs):
    groups = state["param_groups"]
    require(
        len(groups) == (2 if learn_signs else 1)
        and len(groups[0]["params"]) == 9
        and groups[0]["lr"] == 1e-5
        and "name" not in groups[0],
        "head Adam scale group differs",
    )
    scale_ids = groups[0]["params"]
    require(scale_ids == list(range(9)), "canonical nine Adam path IDs differ")
    for group in groups:
        require(
            tuple(group["betas"]) == (0.9, 0.999)
            and group["eps"] == 1e-8
            and group.get("weight_decay", 0) == 0
            and not group.get("amsgrad", False)
            and not group.get("maximize", False)
            and not group.get("capturable", False)
            and not group.get("differentiable", False)
            and not group.get("decoupled_weight_decay", False)
            and group.get("foreach") is None
            and group.get("fused") is None,
            "head Adam algorithm/defaults differ",
        )
    require(
        len(set(scale_ids)) == 9 and set(scale_ids) <= set(state["state"]),
        "nine inherited Adam states required",
    )
    if learn_signs:
        require(
            len(groups[1]["params"]) == 1 and groups[1]["lr"] == 1e-3,
            "head Adam sign group differs",
        )
        head_id = groups[1]["params"][0]
        require(head_id == 9, "canonical selected-head Adam ID differs")
        if cursor == HEAD_START:
            require(head_id not in state["state"], "initial head Adam must have empty moments")
        else:
            require(
                head_id in state["state"]
                and float(state["state"][head_id]["step"]) == cursor - HEAD_START,
                "head Adam step differs from new-stage cursor",
            )
            for key in ("exp_avg", "exp_avg_sq"):
                require(
                    torch.isfinite(state["state"][head_id][key]).all().item(),
                    "nonfinite head Adam moments",
                )
        allowed = set(scale_ids) | ({head_id} if cursor > HEAD_START else set())
    else:
        allowed = set(scale_ids)
    require(set(state["state"]) == allowed, "unexpected Adam state outside selected scope")


def validate_continuation(
    saved, origin, ce_exposure, ce_windows, order, reverse_ids, replay_schedule, native_refs
):
    validate_head_optimizer(saved["optimizer"], saved["cursor"], saved["contract"]["learn_signs"])
    shapes = [
        saved["selected_state"]["scale_offsets"][base].shape for base in CANDIDATE_D_BASE_TO_PATH
    ]
    shapes.append(saved["selected_state"]["selected_latent"].shape)
    for key, row in saved["optimizer"]["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            value = row[name]
            require(
                value.dtype == torch.float32
                and value.shape == shapes[key]
                and value.is_contiguous()
                and torch.isfinite(value).all().item(),
                "selected-head Adam moment geometry/dtype differs",
            )
    # Declared validation view: the immutable original counters still start at1920,
    # while the new head Adam owns a separate3840-based step counter. Never mutate
    # the checkpoint or reinterpret its saved contract as a legacy checkpoint.
    scale_ids = saved["optimizer"]["param_groups"][0]["params"]
    scale_state = {
        "param_groups": saved["optimizer"]["param_groups"][:1],
        "state": {key: saved["optimizer"]["state"][key] for key in scale_ids},
    }
    counter_view = {**saved, "schema": resident_trainer.SCHEMA, "optimizer": scale_state}
    resident_trainer.validate_continuation(
        counter_view,
        origin,
        ce_exposure,
        ce_windows,
        order,
        reverse_ids,
        replay_schedule,
        native_refs,
    )
    expected = head_totals(
        saved["cursor"],
        saved["costs"],
        saved["exposure"],
        saved["trainer_seconds"],
        ce_exposure,
        saved["contract"]["head_start"],
    )
    require(saved.get("head_stage") == expected, "head-stage accounting differs")


def validate_resume(saved, contract):
    require(saved.get("schema") == SCHEMA, "resume requires expanded-CE checkpoint schema")
    require(saved["contract"] == contract, "resume contract differs")
    for key in (
        "selected_state",
        "head_stage",
        "optimizer",
        "rng",
        "cuda_rng",
        "python_rng",
        "ce_stage_exposure",
        "resident_runtime",
    ):
        require(key in saved, "resume missing state: " + key)
    runtime = saved["resident_runtime"]
    require(
        isinstance(runtime, dict) and runtime.get("schema") == resident_helpers.SCHEMA,
        "resume resident runtime schema differs",
    )
    require(
        runtime.get("source_ancestry")
        == {
            "resident_reader": contract["resident"]["source"],
            "original_reader": contract["resident"]["proof_report"]["original_reader"],
        },
        "resume resident runtime source differs",
    )
    for key, value in contract["resident"]["config"].items():
        require(runtime.get(key) == value, "resume resident runtime configuration differs: " + key)
    require(
        type(runtime.get("cached_bytes")) is int
        and 0 <= runtime["cached_bytes"] <= runtime["max_cache_bytes"],
        "resume resident runtime byte accounting differs",
    )
    require(
        all(
            type(runtime.get(key)) is int and runtime[key] >= 0
            for key in ("hits", "misses", "fallbacks", "cached_files", "backing_mmap_files")
        )
        and runtime["fallbacks"] <= runtime["misses"]
        and runtime["backing_mmap_files"] <= 4,
        "resume resident runtime counters differ",
    )


def warm_origin(saved, source_pin):
    return {
        "checkpoint": source_pin,
        "original_contract": saved["contract"],
        "global_start_cursor": WARM_CURSOR,
        "initial_costs": dict(saved["costs"]),
        "initial_exposure": dict(saved["exposure"]),
        "initial_trainer_seconds": saved["trainer_seconds"],
        "original_checkpoint_status": saved["status"],
    }


def native_fit_references(fit_windows, original_refs, supported_target_ids):
    """Expand FIT protections from native emission ancestry, never from DEV labels."""
    result = {}
    for window in fit_windows:
        require(window["role"] == "fit", "native replay accepts fit windows only")
        require(window["id"] not in result, "duplicate native fit window")
        original = original_refs[window["id"]]
        require(
            original["role"] == "fit"
            and original["labels"] == window["labels"]
            and original["native_prefix"] == window["native_prefix"],
            "native reference ancestry differs",
        )
        depth = window["max_depth"]
        require(
            len(window["labels"])
            == len(original["margins"])
            == len(original["supported"])
            == depth,
            "native reference depth differs",
        )
        protections = window["native_protections"]
        require(
            [p["proposal_depth"] for p in protections] == list(range(window["native_prefix"]))
            and window["native_prefix"] <= depth,
            "native emitted mask is not a bounded prefix",
        )
        margins = list(original["margins"])
        for p in protections:
            d, label = p["proposal_depth"], p["token"]
            require(
                label == window["labels"][d]
                and p["response_index"] == window["response_label_start"] + d,
                "native protection label does not join verified response",
            )
            require(
                original["supported"][d] is True and label in supported_target_ids,
                "native protected label lacks draft vocabulary support",
            )
            margin = original["margins"][d]
            require(
                isinstance(margin, (int, float)) and math.isfinite(margin),
                "native protected reference margin must be finite",
            )
            margins[d] = max(MARGIN_FLOOR, min(margin, 2.0))
        result[window["id"]] = {
            **original,
            "protected": [d < window["native_prefix"] for d in range(depth)],
            "margins": margins,
        }
    return result


def native_schedule(fit, references):
    schedule = ProtectedReplaySchedule(fit, references, seed=37)
    empty = schedule.contract.pop("groups_without_actor_matched_protections")
    schedule.contract.update(
        {
            "eligibility": (
                "ALL native-emitted FIT successes, including original GPU-mismatched tokens"
            ),
            "groups_without_native_protections": empty,
            "margin_target": "max(0.1,min(original_GPU_reference_margin,2.0))",
            "selection_cursor": "global training cursor; stage starts at1920",
        }
    )
    return schedule


def validate_v1_warm_start(saved, expected, old_schedule, helper_pin):
    """Only the exact all-scale lambda10 balanced1920 checkpoint may migrate."""
    require(
        saved.get("cursor") == WARM_CURSOR and saved.get("status") in ("RECOVERY", "COMPLETE"),
        "warm-start requires original balanced1920 checkpoint",
    )
    old = saved["contract"]
    require(
        all(old["trainer"][k] == helper_pin[k] for k in ("sha256", "bytes")),
        "pinned original protected trainer helper differs",
    )
    for key in (
        "manifest",
        "actor",
        "view_source",
        "optimizer",
        "retention",
        "seed",
        "horizon",
        "batch_windows",
        "margin_cap",
        "optimizer_scope",
        "replay_windows_per_update",
        "context",
        "runtime",
        "precision",
        "reference",
        "reference_freeze",
        "original_trainer",
        "numeric_forward_semantics_sha256",
        "original_reference_objective",
        "ce_schedule",
        "optimized_scale_paths",
        "native_evaluation_masks",
        "readout",
    ):
        require(old.get(key) == expected.get(key), "warm-start contract differs: " + key)
    require(
        old.get("objective") == V1_OBJECTIVE and old.get("protected_schedule") == old_schedule,
        "warm-start requires original actor-intersection bank/objective",
    )
    require(
        old["optimizer_scope"] == "all" and old["retention"] == 10.0 and old["horizon"] == 19200,
        "warm-start requires all-scale lambda10 original horizon",
    )
    groups = saved["optimizer"]["param_groups"]
    require(
        len(groups) == 1
        and len(groups[0]["params"]) == 9
        and groups[0]["lr"] == 1e-5
        and tuple(groups[0]["betas"]) == (0.9, 0.999)
        and groups[0]["eps"] == 1e-8,
        "warm-start optimizer state differs",
    )
    require(
        saved["exposure"]["ce_windows"] == 2 * WARM_CURSOR
        and saved["exposure"]["replay_windows"] == WARM_CURSOR,
        "warm-start exposure/cursor differs",
    )
    require(
        all(
            isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
            for v in saved["costs"].values()
        )
        and math.isfinite(saved["trainer_seconds"]),
        "warm-start physical costs invalid",
    )
    for key in ("scales", "rng", "cuda_rng", "python_rng"):
        require(key in saved, "warm-start missing continuation state: " + key)


def original_validation_packet(saved, origin):
    """Validate ancestry without rewinding the actual resumed optimizer/RNG state."""
    return {
        **saved,
        "contract": origin["original_contract"],
        "cursor": origin["global_start_cursor"],
        "exposure": origin["initial_exposure"],
        "costs": origin["initial_costs"],
        "trainer_seconds": origin["initial_trainer_seconds"],
        "status": origin["original_checkpoint_status"],
    }


def stage_totals(cursor, costs, exposure, trainer_seconds, origin):
    """Keep original cumulative work and explicit migration-stage increments."""
    require(cursor >= origin["global_start_cursor"], "stage cursor precedes warm start")
    incremental_costs = {k: costs[k] - origin["initial_costs"][k] for k in costs}
    incremental_exposure = {k: exposure[k] - origin["initial_exposure"][k] for k in exposure}
    seconds = trainer_seconds - origin["initial_trainer_seconds"]
    require(
        seconds >= 0
        and all(v >= 0 for v in incremental_costs.values())
        and all(v >= 0 for v in incremental_exposure.values()),
        "stage accounting moved backwards",
    )
    return {
        "global_start_cursor": origin["global_start_cursor"],
        "stage_updates": cursor - origin["global_start_cursor"],
        "stage_trainer_seconds": seconds,
        "stage_costs": incremental_costs,
        "stage_exposure": incremental_exposure,
    }


def main(*, qualification_runner=None):
    parser = argparse.ArgumentParser(
        description=(
            "FIT-only selected-head zero-export/gradient/compact-resume qualification"
            if qualification_runner is not None
            else __doc__
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="exact original manifest")
    parser.add_argument("--union-manifest", type=Path, required=True)
    parser.add_argument("--resident-proof", type=Path, required=True)
    parser.add_argument("--resident-max-cache-bytes", type=int, default=1024**3)
    parser.add_argument("--resident-max-file-bytes", type=int)
    parser.add_argument("--resident-available-reserve-bytes", type=int, default=2 * 1024**3)
    parser.add_argument(
        "--resident-capacity-mode", choices=["fallback", "error"], default="fallback"
    )
    parser.add_argument("--expanded-ce", choices=["true"], default="true")
    parser.add_argument("--learn-signs", choices=["false", "true"], required=True)
    parser.add_argument(
        "--extra-fit-readout",
        action="store_true",
        help="compatibility option; all old/new FIT and DEV readouts are always emitted",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["train"], default="train")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--original-trainer", type=Path, default=Path(__file__).with_name("modern_train.py")
    )
    parser.add_argument("--scope", choices=["all"], default="all")
    parser.add_argument(
        "--protected-trainer",
        type=Path,
        default=Path(__file__).with_name("modern_train_protected.py"),
    )
    parser.add_argument(
        "--final-updates",
        "--steps",
        dest="steps",
        type=int,
        default=3842 if qualification_runner is not None else 5760,
    )
    parser.add_argument("--retention", type=float, choices=[10.0], default=10.0)
    continuation = parser.add_mutually_exclusive_group(required=True)
    continuation.add_argument("--parent", type=Path)
    continuation.add_argument("--resume", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if qualification_runner is not None:
        require(
            args.steps == HEAD_START + 2, "real qualification requires exactly two unique updates"
        )
    if not args.allow_cuda or not 0 < args.steps <= 19200 or args.retention < 0:
        raise ValueError("explicit sole GPU owner / known horizon required")
    args.out = args.out.resolve()
    require(
        args.out.is_relative_to(ARTIFACTS) and args.out != ARTIFACTS,
        "compact outputs must go directly to a unique C: artifact directory",
    )
    require(
        file_pin(args.manifest)["sha256"] == OLD_MANIFEST_SHA256,
        "exact frozen original manifest required",
    )
    require(
        file_pin(args.reference)["sha256"] == REFERENCE_SHA256, "exact frozen reference02 required"
    )
    require(
        file_pin(native_helpers.__file__)["sha256"] == NATIVE_TRAINER_SHA256,
        "original all-native trainer source changed",
    )
    require(
        file_pin(frozen_ce_helpers.__file__)["sha256"] == FROZEN_CE_SHA256,
        "frozen controlled-CE trainer source changed",
    )
    require(
        file_pin(resident_helpers.__file__)["sha256"] == RESIDENT_HELPER_SHA256,
        "admitted resident helper source changed",
    )
    require(
        file_pin(resident_trainer.__file__)["sha256"] == RESIDENT_TRAINER_SHA256
        and file_pin(head_helpers.__file__)["sha256"] == HEAD_HELPER_SHA256,
        "frozen resident/head helper source changed",
    )
    guard = CadencedGuard(PROJECT, args.out)
    signal.signal(signal.SIGTERM, guard.request_stop)
    signal.signal(signal.SIGINT, guard.request_stop)
    guard.before_write(64 * 1024)
    args.out.mkdir(parents=True, exist_ok=False)

    def write_output(path, value, *, recovery=False):
        guarded_write(path, value, guard, recovery=recovery)

    guard()
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5080":
        raise ValueError("RTX5080 required")
    device = torch.device("cuda:0")
    original_data = ActualDataset(args.manifest)
    data = UnionActualDataset(args.union_manifest)
    audit = Audit(guard)
    audited_union = build_union(
        [m["manifest"]["path"] for m in data.manifest["members"]], audit=audit
    )
    require(data.manifest == audited_union, "union differs from freshly audited source members")
    fit, dev, extra_fit = controlled_views(
        original_data.manifest, data.manifest, file_pin(args.manifest)
    )
    protocol_pins = capture_protocol(data.manifest, audit)
    proof_pin, proof = validate_resident_proof(
        args.resident_proof, args.union_manifest, data.manifest
    )
    cache_config = {
        "max_cache_bytes": args.resident_max_cache_bytes,
        "max_file_bytes": (
            args.resident_max_cache_bytes
            if args.resident_max_file_bytes is None
            else args.resident_max_file_bytes
        ),
        "available_reserve_bytes": args.resident_available_reserve_bytes,
        "capacity_mode": args.resident_capacity_mode,
    }
    cache = bind_resident(data, cache_config)
    if data.manifest["missing_embedding_ids"]:
        raise ValueError("native F32 embeddings missing")
    embeddings = load_embeddings(data, device, guard)
    lookup = ExternalNativeF32Embedding(
        lookup=embeddings.__getitem__,
        vocab_size=248320,
        hidden_size=2560,
        target_identity="frozen native Qwen3.5-4B Q4_K_M actual parent",
        target_sha256=data.manifest["target"]["sha256"],
        capture_sha256=file_pin(args.union_manifest)["sha256"],
    )
    actor = build_modern_actor(
        PROJECT / "runs/w1a8-modern-packed-init-20261103-01/model.gguf",
        PROJECT / "runs/spec-shortlist-20261030/models/hf-eagle-4B/config.source.json",
        embedding_lookup=lookup,
        device=device,
        scale_only=False,
    )
    absolute = actor.drafter.d2t + torch.arange(50000, device=device)
    reverse = torch.full((248320,), -1, device=device, dtype=torch.long)
    reverse[absolute] = torch.arange(50000, device=device)
    contract = {
        "manifest": file_pin(args.manifest),
        "actor": actor.contract,
        "trainer": file_pin(resident_trainer.__file__),
        "view_source": file_pin(Path(__file__).with_name("modern_actual_data.py")),
        "optimizer": {"name": "Adam", "lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8},
        "retention": args.retention,
        "seed": 37,
        "horizon": 19200,
        "batch_windows": 2,
        "margin_cap": 2.0,
        "objective": (
            "Original CE sum/fixed4 from2 ordinary fit windows plus retention*mean "
            "protected-token margin deficit on1 independent group-balanced fit replay; "
            "no retention on ordinary CE windows."
        ),
        "optimizer_scope": args.scope,
        "replay_windows_per_update": 1,
        "context": (
            "detached current-actor KV rebuilt from exact observed rawfeature lineage; "
            "own FC/prenorm in goldprefix unroll"
        ),
        "runtime": "original immutable packed-runtime03; bridgefix overlay deliberately separate",
        "precision": "F32Torch/A8, F16KV, TF32disabled",
    }

    def forward(window):
        batch = data.materialize(window, device)
        adapter = actor.adapter
        cache = adapter.build_context_cache(
            batch.prefix_tokens, batch.prefix_rawfeatures, chunk_size=64
        )
        state = adapter.encode_feature(batch.seed_rawfeature)
        token = window["seed_token"]
        values = []
        for d in range(window["max_depth"]):
            step = adapter.decode_step(token, state, window["anchor"] + d, cache)
            cache = step.cache
            values.append(step.logits)
            state = step.pre_norm
            token = window["labels"][d]
        return torch.stack(values)

    def inspect(window, values):
        labels = torch.tensor(window["labels"], device=device)
        classes = reverse[labels]
        proposals = absolute[values.argmax(-1)]
        correct = proposals.eq(labels)
        prefix = int(correct.long().cumprod(0).sum())
        margins = []
        for d, cls in enumerate(classes.tolist()):
            if cls < 0:
                margins.append(None)
                continue
            other = values[d].scatter(0, classes[d : d + 1], -torch.inf).max()
            margins.append(float((values[d, cls] - other).detach()))
        return {
            "id": window["id"],
            "role": window["role"],
            "native_prefix": window["native_prefix"],
            "actor_prefix": prefix,
            "proposals": proposals.detach().cpu().tolist(),
            "labels": window["labels"],
            "supported": classes.ge(0).cpu().tolist(),
            "margins": margins,
            "protected": [d < min(window["native_prefix"], prefix) for d in range(len(classes))],
        }

    def readout(name, windows):
        begin = time.perf_counter()
        rows = []
        with torch.no_grad(), selected.round():
            for index, w in enumerate(windows):
                guard()
                rows.append(inspect(w, forward(w)))
                if index % 128 == 0:
                    print(
                        json.dumps({"stage": name, "windows": index, "total": len(windows)}),
                        flush=True,
                    )
        value = {"rows": rows, "wall_seconds": time.perf_counter() - begin}
        write_output(args.out / (name + ".json"), value)
        return value

    require(
        len(fit) == 1414 and len(dev) == 724, "full actual parent36 fit/development views required"
    )
    native_masks = {
        role: sum(len(w["native_protections"]) for w in fit + dev if w["role"] == role)
        for role in ("fit", "development")
    }
    require(
        native_masks == {"fit": 356, "development": 193}, "all549 native success masks must remain"
    )
    reference = json.loads(args.reference.read_text())
    reference_contract = json.loads(args.reference.with_name("freeze.json").read_text())
    refs = validate_original_reference(
        reference, reference_contract, contract, fit + dev, args.original_trainer, __file__
    )
    helper_pin = file_pin(protected_helpers.__file__)
    requested_helper = file_pin(args.protected_trainer)
    require(
        all(helper_pin[k] == requested_helper[k] for k in ("sha256", "bytes")),
        "loaded protected helper differs from explicitly pinned original source",
    )
    require(
        forward_semantics(Path(protected_helpers.__file__).read_text())
        == forward_semantics(Path(__file__).read_text()),
        "protected forward semantics changed",
    )
    old_schedule = ProtectedReplaySchedule(fit, refs, seed=37).contract
    native_refs = native_fit_references(fit, refs, set(absolute.detach().cpu().tolist()))
    require(
        sum(sum(row["protected"]) for row in native_refs.values()) == 356,
        "all356 native FIT successes must enter the expanded bank",
    )
    schedule = native_schedule(fit, native_refs)
    require(
        len({w["group_id"] for w in fit}) == len(schedule.groups) == 24,
        "all24 FIT groups must remain in the expanded native bank",
    )
    order = ce_order(fit)
    paths, selected_parameters = scale_scope(actor)
    contract.update(
        {
            "reference": file_pin(args.reference),
            "reference_freeze": file_pin(args.reference.with_name("freeze.json")),
            "original_trainer": file_pin(args.original_trainer),
            "numeric_forward_semantics_sha256": forward_semantics(Path(__file__).read_text()),
            "original_reference_objective": reference_contract["objective"],
            "protected_schedule": schedule.contract,
            "ce_schedule": {
                "seed": 37,
                "order": [fit[i]["id"] for i in order],
                "selection": "(2*cursor+offset)%N; offset=0,1",
            },
            "optimized_scale_paths": paths,
            "native_evaluation_masks": native_masks,
            "readout": "all1414 fit and all724 development windows; dev never sampled for training",
        }
    )
    cursor = 0
    train_seconds = 0.0
    costs = {
        "ce_forward_seconds": 0.0,
        "replay_forward_seconds": 0.0,
        "shared_backward_optimizer_seconds": 0.0,
    }
    exposure = {
        "ce_windows": 0,
        "ce_supported_labels": 0,
        "replay_windows": 0,
        "replay_protected_tokens": 0,
    }
    source_checkpoint = args.resume or args.parent
    source_pin = file_pin(source_checkpoint)
    warm_pin = file_pin(WARM_CHECKPOINT)
    require(warm_pin["sha256"] == WARM_CHECKPOINT_SHA256, "balanced1920 checkpoint changed")
    saved = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    require(
        file_pin(source_checkpoint) == source_pin, "continuation checkpoint changed while reading"
    )
    parent_pin = file_pin(PARENT_CHECKPOINT)
    require(
        parent_pin["sha256"] == PARENT_SHA256 and parent_pin["bytes"] == PARENT_BYTES,
        "fixed resident newdata3840 parent changed",
    )
    if args.parent:
        require(source_pin == parent_pin, "explicit fixed parent checkpoint required")
        parent = saved
    else:
        require(saved.get("schema") == SCHEMA, "resume requires selected-head checkpoint")
        parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
        require(file_pin(PARENT_CHECKPOINT) == parent_pin, "parent changed while reading")
    origin = parent["contract"]["warm_start"]
    require(warm_pin == origin["checkpoint"], "original balanced1920 ancestry changed")
    original_packet = torch.load(WARM_CHECKPOINT, map_location="cpu", weights_only=False)
    require(file_pin(WARM_CHECKPOINT) == warm_pin, "original warm checkpoint changed")
    validate_v1_warm_start(original_packet, contract, old_schedule, helper_pin)
    require(origin == warm_origin(original_packet, warm_pin), "original warm accounting changed")
    contract.update(
        {
            "objective": (
                "Original CE2windows/fixed4 plus lambda10 mean margin-deficit replay "
                "over ALL356 native FIT successes; "
                "target=max(0.1,min(original_GPU_margin,2)); declared v1 bank/objective migration"
            ),
            "protected_schedule": schedule.contract,
            "warm_start": origin,
            "protected_helper_source": helper_pin,
            "explicit_original_protected_source": requested_helper,
            "native_fit_bank_tokens": 356,
            "native_margin_floor": MARGIN_FLOOR,
            "native_reference_intersection_extras": 356
            - sum(sum(refs[w["id"]]["protected"]) for w in fit),
            "stage_kind": "declared_objective_bank_migration_from_balanced1920_v1",
        }
    )
    # The original contract above is used only to validate the immutable v1
    # ancestry. Ordinary CE expansion is declared in this separate schema.
    original_ce_schedule = copy.deepcopy(contract["ce_schedule"])
    ce_fit = fit + extra_fit if args.expanded_ce == "true" else fit
    order = ce_order(ce_fit)
    contract.update(
        {
            "schema": resident_trainer.SCHEMA,
            "objective": (
                "CE sum/fixed4 from2 ordinary FIT windows; expanded_ce controls old-only "
                "versus union FIT population. Lambda10 mean native margin-deficit replay "
                "uses immutable original356 FIT bank, original reference02, floor0.1 cap2."
            ),
            "stage_kind": "controlled_ordinary_ce_expansion_with_original_native_replay",
            "expanded_ce": args.expanded_ce == "true",
            "union_manifest": file_pin(args.union_manifest),
            "union_source": file_pin(Path(__file__).with_name("modern_actual_union.py")),
            "native_trainer_source": file_pin(native_helpers.__file__),
            "original_manifest": file_pin(args.manifest),
            "original_ce_schedule": original_ce_schedule,
            "ce_schedule": {
                "seed": 37,
                "order": [ce_fit[i]["id"] for i in order],
                "selection": "(2*cursor+offset)%N; offset=0,1",
            },
            "replay_bank": "original356 FIT successes /24 original groups /reference02 only",
            "new_fit_groups": sorted({w["group_id"] for w in extra_fit}),
            "extra_fit_windows": len(extra_fit),
            "extra_fit_native_tokens": sum(len(w["native_protections"]) for w in extra_fit),
            "new_capture_protocol_pins": protocol_pins,
            "readout": "original1414 FIT and724 DEV; extraFIT separately optional",
            "extra_fit_readout": parent["contract"]["extra_fit_readout"],
            "checkpoint_schema": resident_trainer.SCHEMA,
            "optimizer_parameter_names": paths,
        }
    )
    # Declare the resident source/proof/config only after unchanged original
    # reference and warm-state ancestry have passed their original validators.
    contract["resident"] = resident_contract(cache_config, proof_pin, proof)
    validate_parent(parent, contract, parent_pin)
    supported_ids = set(absolute.detach().cpu().tolist())
    resident_trainer.validate_continuation(
        parent,
        origin,
        parent["ce_stage_exposure"],
        ce_fit,
        order,
        supported_ids,
        schedule,
        native_refs,
    )
    parent_contract = copy.deepcopy(contract)
    load_parent_scales(actor, parent["scales"])
    selection = select_head_rows(
        reachable_fit_records(ce_fit),
        absolute_map(actor).cpu(),
        source_pins=[file_pin(args.union_manifest)],
        count=512,
    )
    selected = SelectedHead(
        actor, selection, base_state_pin=parent_pin, learn_signs=args.learn_signs == "true"
    )
    require(
        torch.equal(selected.latent.detach().abs(), torch.full_like(selected.latent, 0.5)),
        "selected head must start from exact initializer plus/minus0.5",
    )
    optimizer, paths, selected_parameters = make_optimizer(selected)
    head_start = head_origin(parent, parent_pin)
    contract = {
        **parent_contract,
        "schema": SCHEMA,
        "checkpoint_schema": SCHEMA,
        "trainer": file_pin(__file__),
        "stage_kind": "declared512_reachable_FIT_head_sign_capacity_contrast_from_newdata3840",
        "learn_signs": selected.learn_signs,
        "head_start": head_start,
        "selected_head_source": file_pin(head_helpers.__file__),
        "head_selection": selection,
        "selected_head_contract": copy.deepcopy(selected._source_contract),
        "parent_resident_source": file_pin(resident_trainer.__file__),
        "head_optimizer": {
            "scale_lr": 1e-5,
            "sign_lr": 1e-3 if selected.learn_signs else None,
            "parameter_order": "nine original scale paths then optional external selected rows",
        },
        "readout": "all original1414 FIT/all724 DEV/all newFIT; DEV never sampled for training",
        "extra_fit_readout": True,
    }
    if args.resume:
        require(
            saved["contract"].get("head_start") == head_start,
            "selected-head parent ancestry differs",
        )
        validate_resume(saved, contract)
        selected.load_state(saved["selected_state"])
        optimizer.load_state_dict(saved["optimizer"])
        cursor, train_seconds = saved["cursor"], saved["trainer_seconds"]
        costs, exposure = dict(saved["costs"]), dict(saved["exposure"])
        ce_stage_exposure = copy.deepcopy(saved["ce_stage_exposure"])
        validate_continuation(
            saved, origin, ce_stage_exposure, ce_fit, order, supported_ids, schedule, native_refs
        )
    else:
        migrate_parent_optimizer(
            parent["optimizer"],
            optimizer,
            parent_contract["optimized_scale_paths"],
            paths,
            selected.learn_signs,
        )
        saved = parent
        cursor, train_seconds = parent["cursor"], parent["trainer_seconds"]
        costs, exposure = dict(parent["costs"]), dict(parent["exposure"])
        ce_stage_exposure = copy.deepcopy(parent["ce_stage_exposure"])
        validate_head_optimizer(optimizer.state_dict(), cursor, selected.learn_signs)
    old_fit_ids = {w["id"] for w in fit}
    torch.set_rng_state(saved["rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    require(
        HEAD_START <= cursor < args.steps <= 19200,
        "head continuation must extend cursor within unchanged19200 horizon",
    )
    write_output(args.out / "freeze.json", contract)
    write_output(args.out / "selection.json", selection)
    write_output(
        args.out / "continuation.json",
        {
            "mode": "resume" if args.resume else "parent_migration",
            "checkpoint": source_pin,
            "parent": parent_pin,
            "loaded_global_cursor": cursor,
            "requested_global_cursor": args.steps,
        },
    )

    def snapshot(status="RECOVERY"):
        return {
            "schema": SCHEMA,
            "ce_stage_exposure": copy.deepcopy(ce_stage_exposure),
            "resident_runtime": cache.report(),
            "contract": contract,
            "selected_state": selected.state(),
            "optimizer": cpu(optimizer.state_dict()),
            "cursor": cursor,
            "trainer_seconds": train_seconds,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
            "costs": dict(costs),
            "exposure": dict(exposure),
            "status": status,
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
            "head_stage": head_totals(
                cursor, costs, exposure, train_seconds, ce_stage_exposure, head_start
            ),
        }

    def save(status):
        path = args.out / f"checkpoint-{cursor:08d}.pt"
        if path.exists():
            return path
        state = snapshot(status)
        validate_continuation(
            state, origin, ce_stage_exposure, ce_fit, order, supported_ids, schedule, native_refs
        )
        guarded_checkpoint(path, state, guard, recovery=status == "INTERRUPTED")
        # Complete the checkpoint publication receipt even if a stop arrives
        # after its bytes were already committed. This performs no GPU work.
        write_output(args.out / "latest.json", file_pin(path), recovery=True)
        return path

    def update(*, capture_gradient=False):
        nonlocal cursor, train_seconds
        guard()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = 0.0
        ce_value = 0.0
        supported = 0
        ce_counts = []
        with selected.round():
            for w in ce_at(ce_fit, order, cursor):
                window_supported = 0
                values = forward(w)
                classes = reverse[torch.tensor(w["labels"], device=device)]
                ce = values.sum() * 0.0
                for d, cls in enumerate(classes.tolist()):
                    if cls >= 0 and w["native_supervision_reachable_mask"][d]:
                        ce = ce + torch.nn.functional.cross_entropy(
                            values[d : d + 1], classes[d : d + 1]
                        )
                        supported += 1
                        window_supported += 1
                loss = loss + ce / 4.0
                ce_value += float(ce.detach()) / 4.0
                ce_counts.append((w, window_supported))
            torch.cuda.synchronize()
            ce_seconds = time.perf_counter() - t
            replay_begin = time.perf_counter()
            replay_group, replay_window = schedule.at(cursor)
            replay_values = forward(replay_window)
            replay_classes = reverse[torch.tensor(replay_window["labels"], device=device)]
            replay_loss, replay_tokens = normalized_replay_loss(
                replay_values, replay_classes, native_refs[replay_window["id"]], cap=2.0
            )
            loss = loss + args.retention * replay_loss
            torch.cuda.synchronize()
            replay_seconds = time.perf_counter() - replay_begin
            backward_begin = time.perf_counter()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            loss.backward()
        gradients = gradient_metrics(selected) if capture_gradient else None
        norm = torch.nn.utils.clip_grad_norm_(selected_parameters, 1.0)
        if not bool(torch.isfinite(norm)):
            raise ValueError("nonfinite gradient")
        optimizer.step()
        selected.clamp_()
        for path in paths:
            actor.linears[path].project_scales_()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        train_seconds += elapsed
        costs["ce_forward_seconds"] += ce_seconds
        costs["replay_forward_seconds"] += replay_seconds
        costs["shared_backward_optimizer_seconds"] += time.perf_counter() - backward_begin
        exposure["ce_windows"] += 2
        exposure["ce_supported_labels"] += supported
        exposure["replay_windows"] += 1
        exposure["replay_protected_tokens"] += replay_tokens
        for w, window_supported in ce_counts:
            count_ce(ce_stage_exposure, w, window_supported, old_fit_ids)
        cursor += 1
        row = {
            "update": cursor,
            "update_seconds": elapsed,
            "CE": ce_value,
            "retention_mean": float(replay_loss.detach()),
            "replay_group": replay_group,
            "replay_window": replay_window["id"],
            "replay_protected_tokens": replay_tokens,
            "ce_forward_seconds": ce_seconds,
            "replay_forward_seconds": replay_seconds,
            "loss": float(loss.detach()),
            "supported_labels": supported,
            "gradient_norm": float(norm),
            "trainer_seconds": train_seconds,
            **stage_totals(cursor, costs, exposure, train_seconds, origin),
        }
        if gradients is not None:
            row["head_gradient"] = gradients
        row["head_stage"] = head_totals(
            cursor, costs, exposure, train_seconds, ce_stage_exposure, head_start
        )
        return row

    def restore(state):
        nonlocal cursor, train_seconds, costs, exposure, ce_stage_exposure
        validate_resume(state, contract)
        validate_continuation(
            state,
            origin,
            state["ce_stage_exposure"],
            ce_fit,
            order,
            supported_ids,
            schedule,
            native_refs,
        )
        selected.load_state(state["selected_state"])
        optimizer.load_state_dict(state["optimizer"])
        cursor, train_seconds = state["cursor"], state["trainer_seconds"]
        costs, exposure = dict(state["costs"]), dict(state["exposure"])
        ce_stage_exposure = copy.deepcopy(state["ce_stage_exposure"])
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])

    if qualification_runner is not None:
        qualification_runner(
            {
                "args": args,
                "guard": guard,
                "write": write_output,
                "actor": actor,
                "selected": selected,
                "optimizer": optimizer,
                "cache": cache,
                "contract": contract,
                "parent_pin": parent_pin,
                "ce_fit": ce_fit,
                "replay_windows": [
                    w
                    for w in fit
                    if native_refs[w["id"]]["protected"] and any(native_refs[w["id"]]["protected"])
                ],
                "snapshot": snapshot,
                "restore": restore,
                "save": save,
                "update": update,
                "forward": forward,
                "inspect": inspect,
                "started": started,
            }
        )
        return

    guard.before_write(0)
    receipts = (args.out / "updates.jsonl").open("x")
    try:
        while cursor < args.steps:
            row = update()
            encoded_row = json.dumps(row) + "\n"
            guard.before_write(len(encoded_row.encode("utf-8")))
            receipts.write(encoded_row)
            receipts.flush()
            if cursor % 48 == 0:
                print(json.dumps(row), flush=True)
            if cursor % 480 == 0:
                save("RECOVERY")
        checkpoint = (
            save("COMPLETE")
            if cursor % 480
            else Path(json.loads((args.out / "latest.json").read_text())["path"])
        )
    except BaseException:
        save("INTERRUPTED")
        raise
    finally:
        receipts.close()
    readout("after-development", dev)
    readout("after-fit", fit)
    readout("after-extra-fit", extra_fit)
    selected.check()
    guard.before_write(actor.initializer["bytes"])
    exported = selected.export(args.out / "model.gguf")
    write_output(args.out / "export.json", exported)
    write_output(
        args.out / "result.json",
        {
            "status": "TRAINED_EXPORTED_NOT_NATIVE_QUALIFIED",
            "updates": cursor,
            "trainer_seconds": train_seconds,
            "wall_seconds": time.perf_counter() - started,
            "checkpoint": file_pin(checkpoint),
            "model": exported["output"],
            "reference": file_pin(args.reference),
            "retention": args.retention,
            "optimizer_scope": args.scope,
            "optimized_scale_paths": paths,
            "selected_head_metrics": sign_metrics(selected),
            "learn_signs": selected.learn_signs,
            "costs": costs,
            "exposure": exposure,
            "resident_runtime": cache.report(),
            "resident_proof": proof_pin,
            "ce_stage_exposure": ce_stage_exposure,
            "expanded_ce": args.expanded_ce == "true",
            "native_evaluation_masks": native_masks,
            "warm_start": origin["checkpoint"],
            "stage": stage_totals(cursor, costs, exposure, train_seconds, origin),
            "head_stage": head_totals(
                cursor, costs, exposure, train_seconds, ce_stage_exposure, head_start
            ),
            "scope": (
                "selected-head capacity contrast with immutable union CE/original replay; "
                "no untouched generalization/quality/throughput claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
