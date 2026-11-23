"""A synthetic store laid out like a `PortableStore` but written by another tool, for the
adoption tests (RFC 0003 section 8.2). No real data.

``<root>/store@6/<d[:2]>/<d>/<f>/`` holds ``result.json`` (the key record whose digest is ``d``
and the outputs), ``context.json`` (the canonical JSON of the context, whose sha256 is ``f``)
and one ``<sha256>.npy`` per array. Containers and side directories are ``0o2770``, entry
directories are sealed ``0o2550`` and files are ``0o440``: the group-written example of RFC
0003 section 9. ``tmp/`` is the tool's staging directory, ``trash/`` its quarantine and
``provenance/`` a store-level record directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import typing as ty

import attrs
import numpy as np

from pydra.utils.digest import jcs

VERSION = "store@6"
STAGING = "tmp"
QUARANTINE = "trash"
SIDECAR = "provenance"
FILE_MODE = 0o440
ENTRY_MODE = 0o2750
SEALED_MODE = 0o2550
CONTAINER_MODE = 0o2770
SIDE_MODE = 0o2770


@attrs.frozen
class Entry:
    """One entry the fixture wrote: its address and what it holds."""

    version: str
    digest: str
    fingerprint: str
    key: dict
    context: dict
    score: float
    vec: ty.Any

    @property
    def shard(self) -> str:
        return self.digest[:2]

    def path(self, root: os.PathLike, version: str | None = None) -> str:
        return os.path.join(
            os.fspath(root),
            version or self.version,
            self.shard,
            self.digest,
            self.fingerprint,
        )


def key_digest(key: ty.Mapping) -> str:
    return hashlib.sha256(jcs(dict(key))).hexdigest()


def npy_bytes(array: ty.Any) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array(
        buf, np.ascontiguousarray(array), version=(1, 0), allow_pickle=False
    )
    return buf.getvalue()


def _dir(path: str, mode: int) -> None:
    os.makedirs(path, exist_ok=True)
    os.chmod(path, mode)


def _file(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
    os.chmod(path, FILE_MODE)


def prepare_root(root: os.PathLike) -> None:
    """Give ``root`` the user's own group, so a set-group-ID bit on the directories below it
    is kept (BSD drops it for a group the user is not in), and make the side directories.
    """
    root = os.fspath(root)
    os.makedirs(root, exist_ok=True)
    os.chown(root, -1, os.getgid())
    for name in (STAGING, QUARANTINE, SIDECAR):
        _dir(os.path.join(root, name), SIDE_MODE)


def write_entry(
    root: os.PathLike,
    i: int,
    *,
    version: str = VERSION,
    bad_key: bool = False,
    bad_context: bool = False,
) -> Entry:
    """Write entry ``i``. ``bad_key`` records a key that does not digest to the entry's
    directory; ``bad_context`` a context that does not give its fingerprint directory.
    """
    root = os.fspath(root)
    key = {"scheme": "toy-key@6", "task": "features", "inputs": {"wav": f"sub-{i:02d}"}}
    context = {"python": "3.12", "tool": "6.0", "host_class": "cpu"}
    digest = key_digest(key)
    raw_context = jcs(context)
    fingerprint = hashlib.sha256(raw_context).hexdigest()
    vec = np.arange(4, dtype="<f8") * (i + 1)
    blob = npy_bytes(vec)
    sha = hashlib.sha256(blob).hexdigest()
    score = 0.5 + i
    recorded_key = dict(key, inputs={"wav": "someone-else"}) if bad_key else key
    if bad_context:
        raw_context = jcs(dict(context, tool="5.9"))
    result = {"key": recorded_key, "outputs": {"score": score, "vec": {"npy": sha}}}
    _dir(os.path.join(root, version), CONTAINER_MODE)
    shard = os.path.join(root, version, digest[:2])
    _dir(shard, CONTAINER_MODE)
    _dir(os.path.join(shard, digest), CONTAINER_MODE)
    entry = os.path.join(shard, digest, fingerprint)
    _dir(entry, ENTRY_MODE)
    _file(os.path.join(entry, "result.json"), json.dumps(result).encode())
    _file(os.path.join(entry, "context.json"), raw_context)
    _file(os.path.join(entry, f"{sha}.npy"), blob)
    os.chmod(entry, SEALED_MODE)
    return Entry(version, digest, fingerprint, key, context, score, vec)


def build_tree(root: os.PathLike, n: int = 3, **kw: ty.Any) -> list[Entry]:
    """Prepare ``root`` and write ``n`` good entries."""
    prepare_root(root)
    return [write_entry(root, i, **kw) for i in range(n)]
