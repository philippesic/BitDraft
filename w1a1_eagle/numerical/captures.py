import json
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor

from w1a1_eagle.contracts import Record, integer, record, sequence
from w1a1_eagle.data import CaptureSpec as CaptureSpec
from w1a1_eagle.data import Supervision, capture_supervision

HalfArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float16]]"
FloatArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.float32]]"
IntArray: TypeAlias = "np.ndarray[tuple[int, ...], np.dtype[np.int32]]"

TAPS = (2, 6, 10, 14, 18, 22, 26, 30)
READERS = 4


@dataclass(frozen=True)
class Capture:
    tokens: tuple[int, ...]
    features: Tensor
    windows: tuple[Supervision, ...]

    @property
    def bytes(self) -> int:
        return self.features.numel() * self.features.element_size()


def finite(values: "HalfArray | FloatArray") -> bool:
    if values.dtype == np.float16:
        half = values.view(np.uint16)
        return not bool(((half >= 0x7C00) & (half < 0x8000) | (half >= 0xFC00)).any())
    single = values.view(np.uint32)
    return not bool(((single >= 0x7F800000) & (single < 0x80000000) | (single >= 0xFF800000)).any())


def load_capture(
    spec: CaptureSpec,
    byte_budget: int,
    hidden: int = 2560,
    readers: ThreadPoolExecutor | None = None,
) -> Capture:
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
    selected = capture_supervision(spec.identity, metadata)
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
    feature_names = {f"layer-{tap}.{suffix}" for tap in TAPS}
    for ref in spec.files:
        if ref.path.name not in feature_names:
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
    shape = (len(TAPS), n, hidden)
    taps: HalfArray | FloatArray = (
        np.empty(shape, dtype=np.float16)
        if storage_size == 2
        else np.empty(shape, dtype=np.float32)
    )
    raw = taps.data.cast("B")
    size = n * hidden * storage_size
    parts = [
        (files[f"layer-{tap}.{suffix}"], raw[index * size : (index + 1) * size])
        for index, tap in enumerate(TAPS)
    ]
    del raw
    if readers is None:
        for ref, part in parts:
            ref.read_verified(part)
    else:
        reads = [readers.submit(ref.read_verified, part) for ref, part in parts]
        wait(reads)
        for read in reads:
            read.result()
    del parts
    if not finite(taps):
        raise ValueError("nonfinite teacher features")
    stored = torch.from_numpy(taps)
    features = torch.empty((n, len(TAPS), hidden), dtype=stored.dtype)
    features.copy_(stored.permute(1, 0, 2))
    del stored, taps
    features = features.view(n, len(TAPS) * hidden)
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
        self.loader: ThreadPoolExecutor | None = None
        self.readers: ThreadPoolExecutor | None = None
        self.pending: tuple[str, Future[Capture]] | None = None

    @staticmethod
    def key(spec: CaptureSpec) -> str:
        return (
            spec.identity
            + ":"
            + spec.metadata.sha256
            + ":"
            + ":".join(ref.sha256 for ref in spec.files)
        )

    def clear(self) -> None:
        self.discard()
        self.entries.clear()

    def discard(self) -> None:
        if self.pending is not None:
            _, future = self.pending
            self.pending = None
            future.cancel()
            wait((future,))

    def load(self, spec: CaptureSpec) -> Capture:
        if self.readers is None:
            self.readers = ThreadPoolExecutor(READERS, "capture-read")
        return load_capture(spec, self.byte_budget, self.hidden, self.readers)

    def get(self, spec: CaptureSpec) -> Capture:
        key = self.key(spec)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        future = None
        if self.pending is not None and self.pending[0] == key:
            future = self.pending[1]
            self.pending = None
        self.clear()
        capture = None
        if future is not None and future.exception() is None:
            capture = future.result()
        del future
        if capture is None:
            capture = self.load(spec)
        self.entries[key] = capture
        return capture

    def prefetch(self, spec: CaptureSpec) -> None:
        key = self.key(spec)
        if key in self.entries or (self.pending is not None and self.pending[0] == key):
            return
        self.discard()
        if self.loader is None:
            self.loader = ThreadPoolExecutor(1, "capture-prefetch")
        self.pending = key, self.loader.submit(self.load, spec)
