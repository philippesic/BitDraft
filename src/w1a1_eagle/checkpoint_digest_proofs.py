"""Ephemeral full-hash proofs for one checkpoint publication transaction.

Never persist these proofs or carry them between save calls. A receipt digest
alone cannot create a proof: the first match reads the entire regular file.
"""

import hashlib
import os
import stat
from pathlib import Path


def regular_file_identity(path):
    """Strict no-follow identity; refuse linked or nonregular payloads."""
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError("checkpoint proof requires a regular file with one link")
    return _stat_identity(value)


def _stat_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _full_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


class CheckpointDigestProofs:
    """Save-local cache authenticated by actual bytes and strict file identity.

    Context binds a freshly validated receipt and its own file identity. The
    post-rename hash precedes sidecar creation, so its initially unbound proof
    may bind once to that new sidecar. Later context changes require a full hash.
    Independent callers and forked processes cannot reuse this authority.
    """

    def __init__(self):
        self._pid = os.getpid()
        self._proofs = {}

    def matches(self, path, expected_sha256, *, context=None):
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
        ):
            return False
        if os.getpid() != self._pid:
            self._proofs.clear()
            self._pid = os.getpid()
        path = Path(path).absolute()
        before = regular_file_identity(path)
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _stat_identity(opened) != before
            ):
                self._proofs.pop(path, None)
                return False
            prior = self._proofs.get(path)
            reusable = (
                prior is not None
                and prior[0] == before
                and prior[1] == expected_sha256
                and (prior[2] is None or prior[2] == context)
            )
            actual = prior[1] if reusable else _full_digest(stream)
            if (
                _stat_identity(os.fstat(stream.fileno())) != before
                or regular_file_identity(path) != before
                or actual != expected_sha256
            ):
                self._proofs.pop(path, None)
                return False
        self._proofs[path] = (before, actual, context)
        return True
