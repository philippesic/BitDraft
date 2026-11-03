"""Optional instance-local immutable RAM snapshots of authenticated native F32 arrays.

No trainer integration or global replacement is performed. Explicit usage:
``cache = ResidentRows(...); dataset._rows = cache``. Cache misses delegate all
file hashing and mmap validation to the unchanged ``modern_actual_data._Rows``.
Hits intentionally use the authenticated snapshot without reopening/restatting a
source. Window materialization must retain its existing per-row SHA checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import modern_actual_data
import numpy as np
from modern_actual_data import _Rows

SCHEMA = "w1a8_resident_native_f32_rows_v1"
DEFAULT_CACHE_BYTES = 1024**3
DEFAULT_AVAILABLE_RESERVE = 2 * 1024**3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_pin(path):
    path = Path(path).resolve()
    value = path.read_bytes()
    return {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def linux_available_bytes():
    """Read actual Linux MemAvailable; fail closed if the kernel omits it."""
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "MemAvailable:":
                _require(len(fields) == 3 and fields[2] == "kB", "invalid Linux MemAvailable")
                value = int(fields[1]) * 1024
                _require(value >= 0, "negative Linux MemAvailable")
                return value
    raise RuntimeError("Linux MemAvailable is unavailable; cannot authorize RAM copy")


class ResidentRows:
    """Bounded, non-evicting snapshots above a private original four-mmap LRU.

    ``capacity_mode='fallback'`` delegates uncached requests to the original
    reader when file/cache/RAM limits are reached. ``'error'`` raises MemoryError
    instead. Neither mode evicts an existing RAM snapshot or changes bytes.
    The instance is synchronous, like the original reader; share neither one
    between concurrent materializers without external serialization.
    """

    def __init__(
        self,
        *,
        max_cache_bytes=DEFAULT_CACHE_BYTES,
        max_file_bytes=None,
        available_reserve_bytes=DEFAULT_AVAILABLE_RESERVE,
        capacity_mode="fallback",
    ):
        max_file_bytes = max_cache_bytes if max_file_bytes is None else max_file_bytes
        _require(type(max_cache_bytes) is int and max_cache_bytes >= 0, "invalid cache byte limit")
        _require(
            type(max_file_bytes) is int and 0 <= max_file_bytes <= max_cache_bytes,
            "file byte limit must not exceed cache byte limit",
        )
        _require(
            type(available_reserve_bytes) is int and available_reserve_bytes >= 0,
            "invalid available RAM reserve",
        )
        _require(capacity_mode in ("fallback", "error"), "unknown capacity mode")
        self.max_cache_bytes = max_cache_bytes
        self.max_file_bytes = max_file_bytes
        self.available_reserve_bytes = available_reserve_bytes
        self.capacity_mode = capacity_mode
        self._backing = _Rows(limit=4)
        self._cache = {}
        self._identities = {}
        self._resolved_paths = {}
        self._bytes = self._hits = self._misses = self._fallbacks = 0
        self._fallback_reasons = Counter()
        self._copy_refusals = Counter()
        self._sources = {
            "resident_reader": _source_pin(__file__),
            "original_reader": _source_pin(modern_actual_data.__file__),
        }

    def _descriptor(self, spec):
        _require(isinstance(spec, dict), "array descriptor must be a dictionary")
        raw_path = os.fspath(spec["path"])
        _require(isinstance(raw_path, str), "array path must be text")
        # Restrict the API to captured native arrays before any payload access.
        _require(Path(raw_path).suffix == ".f32", "only native .f32 array sources are allowed")
        path = self._resolved_paths.get(raw_path)
        if path is None:
            path = str(Path(raw_path).resolve())
            _require(Path(path).suffix == ".f32", "resolved source must be a native .f32 array")
            self._resolved_paths[raw_path] = path
            self._resolved_paths[path] = path
        shape = spec["shape"]
        _require(
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(type(n) is int and n > 0 for n in shape),
            "invalid native F32 matrix shape",
        )
        size, digest = spec["bytes"], spec["sha256"]
        _require(type(size) is int and size == shape[0] * shape[1] * 4, "wrong source F32 geometry")
        _require(isinstance(digest, str) and SHA256.fullmatch(digest), "invalid source SHA256")
        identity = (digest, size, tuple(shape))
        _require(
            path not in self._identities or self._identities[path] == identity,
            "conflicting resident array source pin",
        )
        return path, identity

    def _capacity_reason(self, size):
        if size > self.max_file_bytes:
            return "file_limit"
        if size > self.max_cache_bytes - self._bytes:
            return "cache_limit"
        try:
            available = linux_available_bytes()
        except (OSError, ValueError, RuntimeError):
            return "available_unreadable"
        if available - size < self.available_reserve_bytes:
            return "available_limit"
        return None

    def _uncached(self, spec, path, identity, reason):
        self._copy_refusals[reason] += 1
        if self.capacity_mode == "error":
            raise MemoryError("resident F32 cache refused copy: " + reason)
        result = self._backing.matrix(spec)
        self._identities[path] = identity
        self._fallbacks += 1
        self._fallback_reasons[reason] += 1
        return result

    def matrix(self, spec):
        """Return exact little-endian F32 values with the original reader's shape."""
        path, identity = self._descriptor(spec)
        if path in self._cache:
            self._hits += 1
            return self._cache[path]
        self._misses += 1
        size = identity[1]
        reason = self._capacity_reason(size)
        if reason:
            return self._uncached(spec, path, identity, reason)
        source = self._backing.matrix(spec)
        self._identities[path] = identity
        _require(
            source.dtype == np.dtype("<f4")
            and source.shape == identity[2]
            and source.flags.c_contiguous
            and source.nbytes == size,
            "original reader returned incompatible F32 matrix",
        )
        # Authentication/page reads can change MemAvailable. Check again directly
        # before the only payload allocation, not merely before first mmap/hash.
        reason = self._capacity_reason(size)
        if reason:
            self._copy_refusals[reason] += 1
            if self.capacity_mode == "error":
                raise MemoryError("resident F32 cache refused copy: " + reason)
            self._fallbacks += 1
            self._fallback_reasons[reason] += 1
            return source
        try:
            immutable = source.tobytes(order="C")
        except MemoryError:
            self._copy_refusals["allocation"] += 1
            if self.capacity_mode == "error":
                raise
            self._fallbacks += 1
            self._fallback_reasons["allocation"] += 1
            return source
        # bytes owns the allocation. No writable base or mmap alias survives in
        # the cached object; setflags(write=True) therefore also fails closed.
        result = np.frombuffer(immutable, dtype="<f4").reshape(identity[2])
        _require(
            not result.flags.writeable and result.flags.c_contiguous,
            "resident F32 snapshot must be immutable and contiguous",
        )
        self._cache[path] = result
        self._bytes += size
        return result

    def report(self, *, include_descriptors=False):
        """Return JSON-safe accounting and authenticated snapshot/source ancestry."""
        descriptors = [
            {
                "path": path,
                "sha256": identity[0],
                "bytes": identity[1],
                "shape": list(identity[2]),
                "resident": path in self._cache,
            }
            for path, identity in sorted(self._identities.items())
        ]
        encoded = json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode()
        value = {
            "schema": SCHEMA,
            "source_ancestry": {key: dict(ref) for key, ref in self._sources.items()},
            "max_cache_bytes": self.max_cache_bytes,
            "max_file_bytes": self.max_file_bytes,
            "available_reserve_bytes": self.available_reserve_bytes,
            "capacity_mode": self.capacity_mode,
            "cached_bytes": self._bytes,
            "cached_files": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "fallbacks": self._fallbacks,
            "fallback_reasons": dict(self._fallback_reasons),
            "copy_refusals": dict(self._copy_refusals),
            "backing_mmap_limit": self._backing.limit,
            "backing_mmap_files": len(self._backing.maps),
            "authenticated_descriptors": len(descriptors),
            "descriptors_sha256": hashlib.sha256(encoded).hexdigest(),
            "hit_semantics": "immutable first-authenticated snapshot; no source stat/reopen",
            "row_hash_checks": "unchanged caller materializer responsibility",
        }
        if include_descriptors:
            value["descriptors"] = descriptors
        return value
