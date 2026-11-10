import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.contracts import integer, record, sequence
from w1a1_eagle.data import Artifact, Window, windows

FloatArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
IntArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.int32]]"

TAPS = (2, 6, 10, 14, 18, 22, 26, 30)


@dataclass(frozen=True)
class CaptureSpec:
    identity: str
    metadata: Artifact
    files: tuple[Artifact, ...]


@dataclass(frozen=True)
class Capture:
    tokens: tuple[int, ...]
    features: Tensor
    windows: tuple[Window, ...]

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
    selected = windows(
        spec.identity,
        tokens,
        prompt_tokens,
        positions,
        labels,
        labels[-1] if labels and labels[-1] in eog else None,
    )
    n = len(tokens)
    expected_size = n * hidden * len(TAPS) * 4
    if expected_size * 3 + spec.metadata.size * 32 > byte_budget:
        raise ValueError("capture materialization exceeds byte budget")
    files = {ref.path.name: ref for ref in spec.files}
    if len(files) != len(spec.files):
        raise ValueError("duplicate capture filenames")
    for tap in TAPS:
        ref = files.get(f"layer-{tap}.f32")
        if ref is None or ref.size != n * hidden * 4:
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
    arrays = [
        cast(
            FloatArray,
            np.memmap(files[f"layer-{tap}.f32"].path, dtype="<f4", mode="r", shape=(n, hidden)),
        )
        for tap in TAPS
    ]
    features = torch.from_numpy(cast(FloatArray, np.concatenate(arrays, axis=1)))
    if not bool(torch.isfinite(features).all()):
        raise ValueError("nonfinite teacher features")
    return Capture(tokens, features, selected)


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
