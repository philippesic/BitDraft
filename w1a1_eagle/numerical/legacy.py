import hashlib
import struct
from typing import Protocol, cast

import numpy as np
import torch
import zstandard
from torch import Tensor

from w1a1_eagle.checkpoints.store import StateRef, Store
from w1a1_eagle.contracts import Record, integer, number, record, sequence, string
from w1a1_eagle.data import Artifact
from w1a1_eagle.numerical.worker import TensorWorker
from w1a1_eagle.records import canonical, identity
from w1a1_eagle.training import Progress


class Decoder(Protocol):
    def decompress(self, data: bytes, size: int) -> bytes: ...


class Compression(Protocol):
    def ZstdDecompressor(self) -> Decoder: ...


def materialize(source: Artifact, receipt: Record, store: Store) -> Artifact:
    source.verify()
    if (
        receipt.get("format") != "dflash.byte_shuffle_zstd.stream.v1"
        or receipt.get("archive_sha256") != source.sha256
    ):
        raise ValueError("unsupported or unauthenticated legacy checkpoint receipt")
    expected = string(receipt["raw_sha256"])
    raw_bytes = integer(receipt["raw_bytes"])
    decoder = cast(Compression, zstandard).ZstdDecompressor()
    digest = hashlib.sha256()
    count = 0

    def write(output: object) -> None:
        nonlocal count
        from typing import BinaryIO

        target = cast(BinaryIO, output)
        with source.path.open("rb") as stream:
            if stream.read(11) != b"DFLASHBSZ1\n":
                raise ValueError("legacy checkpoint magic mismatch")
            while header := stream.read(16):
                if len(header) != 16:
                    raise ValueError("torn legacy checkpoint header")
                size, encoded = cast(tuple[int, int], struct.unpack("<QQ", header))
                if not 1 <= size <= 8 * 1024 * 1024 or encoded > 16 * 1024 * 1024:
                    raise ValueError("legacy chunk exceeds bound")
                coded = stream.read(encoded)
                if len(coded) != encoded:
                    raise ValueError("torn legacy checkpoint payload")
                block = decoder.decompress(coded, size)
                if len(block) != size:
                    raise ValueError("legacy decompressed chunk size mismatch")
                n = size // 4 * 4
                restored = (
                    np.frombuffer(block[:n], dtype=np.uint8).reshape(4, -1).T.copy().tobytes()
                    + block[n:]
                )
                digest.update(restored)
                count += len(restored)
                if count > raw_bytes:
                    raise ValueError("legacy logical size exceeds receipt")
                target.write(restored)
        if count != raw_bytes or digest.hexdigest() != expected:
            raise ValueError("legacy raw digest mismatch")

    state = store.commit_with(write, "authenticated-legacy-torch-zip", 0, raw_bytes)
    return state.artifact


def canonical_name(name: str) -> str:
    if name == "output_norm_enc":
        return "encoder_norm"
    name = name.replace("layers.", "blocks.")
    for old, new in (
        ("attn_q_norm", "q_norm"),
        ("attn_k_norm", "k_norm"),
        ("ffn_gate", "gate"),
        ("ffn_up", "up"),
        ("ffn_down", "down"),
    ):
        name = name.replace("." + old + ".", "." + new + ".")
        if name.endswith("." + old):
            name = name[: -len(old)] + new
    return name


def group_name(name: str) -> str:
    return {
        "latent_sign": "sign",
        "scale": "scale",
        "midpoint": "midpoint",
        "log_gain": "gain",
        "log_clip": "clip",
    }.get(name.rsplit(".", 1)[-1], "norm")


def import_state(
    raw: Artifact, worker: TensorWorker, destination: Store, max_bytes: int
) -> StateRef:
    raw.verify()
    state = record(
        cast(object, torch.load(raw.path, map_location="cpu", weights_only=False, mmap=True))
    )
    if state.get("version") != 2 or state.get("resume_capable") is not True:
        raise ValueError("legacy import requires a complete version-2 resumable own-run state")
    config = record(state["config"])
    if identity(config) != state["config_hash"]:
        raise ValueError("legacy configuration digest mismatch")
    if worker.model.origin is None or record(config["source"])["sha256"] != worker.model.origin[0]:
        raise ValueError("legacy initializer differs from new worker")
    spec = record(state["component_spec"])
    if (
        spec.get("head") != "Q4_K"
        or spec.get("embedding") != "Q4_K"
        or spec.get("tied") is not True
    ):
        raise ValueError("legacy frozen head contract differs")
    progress = record(state["progress"])
    updates, cursor = integer(progress["updates"]), integer(progress["cursor"])
    if updates < 0 or cursor < 0:
        raise ValueError("legacy progress is negative")
    parameters = record(state["parameters"])
    expected = dict(worker.model.named_parameters())
    mapped: dict[str, Tensor] = {}
    names: dict[str, str] = {}
    for name, value in parameters.items():
        target = canonical_name(name)
        if target in mapped or target not in expected or not torch.is_tensor(value):
            raise ValueError("legacy parameter coverage differs: " + name)
        if (
            value.shape != expected[target].shape
            or value.dtype != expected[target].dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("legacy parameter geometry/finiteness differs: " + name)
        mapped[target] = value
        names[name] = target
    if set(mapped) != set(expected):
        raise ValueError("legacy parameter set does not match qualified HE4 graph")
    old_optimizer = record(state["optimizer"])
    old_groups = sequence(old_optimizer["param_groups"])
    new_optimizer = cast(Record, worker.optimizer.state_dict())
    new_groups = sequence(new_optimizer["param_groups"])
    old_states_raw = old_optimizer["state"]
    if not isinstance(old_states_raw, dict):
        raise ValueError("legacy optimizer states are malformed")
    old_states = cast(dict[object, object], old_states_raw)
    converted: dict[int, object] = {}
    if len(old_groups) != len(new_groups):
        raise ValueError("legacy optimizer group count differs")
    for old_value, new_value in zip(old_groups, new_groups, strict=True):
        old, new = record(old_value), record(new_value)
        for field in (
            "name",
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        ):
            if old.get(field) != new.get(field):
                raise ValueError("legacy optimizer policy differs: " + field)
        old_names = [name for name in parameters if group_name(name) == old["name"]]
        new_names = [name for name in expected if group_name(name) == new["name"]]
        old_ids = sequence(old["params"])
        new_ids = sequence(new["params"])
        if len(old_ids) != len(old_names) or len(new_ids) != len(new_names):
            raise ValueError("optimizer name/id coverage differs")
        by_name = {
            names[name]: old_states[integer(key)]
            for name, key in zip(old_names, old_ids, strict=True)
        }
        for name, key in zip(new_names, new_ids, strict=True):
            fields = record(by_name[name])
            if set(fields) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("legacy Adam field coverage differs")
            step = fields["step"]
            if not torch.is_tensor(step) or float(step) != updates:
                raise ValueError("legacy Adam step does not match completed updates")
            for field in ("exp_avg", "exp_avg_sq"):
                tensor = fields[field]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.shape != expected[name].shape
                    or tensor.dtype != expected[name].dtype
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError("legacy optimizer moment differs: " + name + "/" + field)
            converted[integer(key)] = fields
    new_optimizer["state"] = converted
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(mapped[name])
    worker.optimizer.load_state_dict(new_optimizer)
    worker.updates = updates
    worker.optimizer_updates = updates
    worker.cursor = cursor
    worker.progress = Progress(
        updates=updates,
        cursor=cursor,
        attempts=updates,
        wall_seconds=number(progress["training_seconds"]),
    )
    rng = state["rng"]
    if not isinstance(rng, tuple) or len(rng) != 4:
        raise ValueError("legacy RNG tuple differs")
    saved_rng = cast(tuple[object, object, object, object], rng)
    if not torch.is_tensor(saved_rng[0]):
        raise ValueError("legacy Torch RNG missing")
    torch.set_rng_state(saved_rng[0])
    cuda = saved_rng[3]
    if worker.model.vocabulary.device.type == "cuda":
        if not isinstance(cuda, list) or any(
            not torch.is_tensor(v) for v in cast(list[object], cuda)
        ):
            raise ValueError("legacy CUDA RNG missing")
        torch.cuda.set_rng_state_all(cast(list[Tensor], cuda))
    worker.lineage = (raw.sha256,)
    worker.ready = True
    reference = worker.checkpoint(destination, max_bytes)
    destination.atomic(
        destination.root / "legacy-import.json",
        canonical(
            {
                "legacy_raw_sha256": raw.sha256,
                "legacy_config": state["config_hash"],
                "updates": updates,
                "cursor": cursor,
                "progress": progress,
                "state": reference.artifact.sha256,
                "scope": "mapped parameters/Adam and Torch CUDA RNG; original raw archive retains complete Python/NumPy RNG and historical accounting",
                "historical_attempt_counts": "completed updates known; failed/discarded counts not supplied by legacy format",
            }
        ),
    )
    return reference
