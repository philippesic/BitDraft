import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.contracts import Record, integer, record, sequence
from w1a1_eagle.data import CaptureSpec as CaptureSpec
from w1a1_eagle.data import Supervision, supervision

HalfArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float16]]"
FloatArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
IntArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.int32]]"

TAPS = (2, 6, 10, 14, 18, 22, 26, 30)


@dataclass(frozen=True)
class Capture:
    tokens: tuple[int, ...]
    features: Tensor
    windows: tuple[Supervision, ...]

    @property
    def bytes(self) -> int:
        return self.features.numel() * self.features.element_size()


def load_capture(spec: CaptureSpec, byte_budget: int, hidden: int = 2560) -> Capture:
    if type(byte_budget) is not int or byte_budget < 1 or hidden < 1:
        raise ValueError("explicit capture byte budget required")
    if spec.metadata.size > min(16 * 1024 * 1024, byte_budget // 32):
        raise ValueError("metadata exceeds capture budget")
    spec.metadata.verify()
    metadata = record(cast(object, json.loads(spec.metadata.path.read_text())))
    if (
        metadata.get("taps") != list(TAPS)
        or metadata.get("feature_semantics") != "zero_based_pre_block_input_all_tokens"
    ):
        raise ValueError("capture tap or feature semantics mismatch")
    tokens = tuple(integer(v) for v in sequence(metadata["input_tokens"]))
    positions = tuple(integer(v) for v in sequence(metadata["logit_positions"]))
    labels = tuple(integer(v) for v in sequence(metadata["greedy_ids"]))
    prompt_tokens = integer(metadata["prompt_tokens"])
    eog = tuple(integer(v) for v in sequence(metadata["eog_ids"]))
    if any(value in eog for value in labels[:-1]):
        raise ValueError("post-EOG label exposure")
    selected = supervision(
        spec.identity,
        tokens,
        prompt_tokens,
        positions,
        labels,
        labels[-1] if labels and labels[-1] in eog else None,
        str(metadata.get("window_policy", "full15")),
    )
    feature_dtype = metadata.get("feature_dtype", "float32")
    if feature_dtype not in ("float32", "float16"):
        raise ValueError("unsupported feature storage dtype")
    suffix = "f16" if feature_dtype == "float16" else "f32"
    storage_size = 2 if feature_dtype == "float16" else 4
    n = len(tokens)
    expected_size = n * hidden * len(TAPS) * 4
    if expected_size * 3 + spec.metadata.size * 32 > byte_budget:
        raise ValueError("capture materialization exceeds byte budget")
    files = {ref.path.name: ref for ref in spec.files}
    if len(files) != len(spec.files):
        raise ValueError("duplicate capture filenames")
    for tap in TAPS:
        ref = files.get(f"layer-{tap}.{suffix}")
        if ref is None or ref.size != n * hidden * storage_size:
            raise ValueError("capture feature geometry mismatch")
    for ref in spec.files:
        ref.verify()
    for name, expected in (
        ("inputs.npy", tokens),
        ("label_positions.npy", positions),
        ("labels.npy", labels),
    ):
        if name in files:
            array = cast(IntArray, np.load(files[name].path, mmap_mode="r", allow_pickle=False))
            if (
                array.dtype != np.dtype("int32")
                or (array.ndim != 1 or array.size != len(expected))
                or tuple(integer(v) for v in sequence(cast(object, array.tolist()))) != expected
            ):
                raise ValueError("capture array metadata disagreement")
    arrays: list[FloatArray] = []
    for tap in TAPS:
        path = files[f"layer-{tap}.{suffix}"].path
        if storage_size == 2:
            mapped = cast(HalfArray, np.memmap(path, dtype="<f2", mode="r", shape=(n, hidden)))
            arrays.append(mapped.astype(np.float32))
        else:
            arrays.append(
                cast(FloatArray, np.memmap(path, dtype="<f4", mode="r", shape=(n, hidden)))
            )
    features = torch.from_numpy(cast(FloatArray, np.concatenate(arrays, axis=1)))
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite teacher features")
    return Capture(tokens, features, selected)


def validate_capture(spec: CaptureSpec, byte_budget: int) -> Record:
    torch.set_num_threads(2)
    checked = load_capture(spec, byte_budget)
    return {
        "status": "passed",
        "checks": "product loader: authenticated arrays, geometry, finite values, tokens, labels and EOG",
        "supervision_windows": len(checked.windows),
        "materialized_bytes": checked.bytes,
    }


class CaptureCache:
    def __init__(self, byte_budget: int, hidden: int = 2560) -> None:
        if type(byte_budget) is not int or byte_budget < 1:
            raise ValueError("positive cache budget required")
        self.byte_budget = byte_budget
        self.hidden = hidden
        self.entries: OrderedDict[str, Capture] = OrderedDict()

    def get(self, spec: CaptureSpec) -> Capture:
        key = (
            spec.identity
            + ":"
            + spec.metadata.sha256
            + ":"
            + ":".join(ref.sha256 for ref in spec.files)
        )
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        self.entries.clear()
        capture = load_capture(spec, self.byte_budget, self.hidden)
        self.entries[key] = capture
        return capture
