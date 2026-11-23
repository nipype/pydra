"""Job payloads of a worker pool (RFC 0005 section 5.2).

A ticket names each job's payload, never a pickle: either a file ``<pool root>/payloads/<sha256>``,
read only after the ownership walk of RFC 0003 section 6.4 passes for that directory and the digest
matches, or JSON carried in the ticket by a codec whose jobs are JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import typing as ty
from pathlib import Path

import attrs

from pydra.engine.portable.trust import EntryNotTrusted, check_private
from pydra.utils.digest import jcs
from pydra.utils.storagedir import IS_WINDOWS, StorageDir, StoreUnavailable
from pydra.workers.pool.storage import FILE_MODE, POLICY, PRIVATE_PICKLE, _create

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

__all__ = [
    "JobCodec",
    "PayloadMismatch",
    "PAYLOADS",
    "PrivatePickle",
    "read_payload",
    "write_payload",
]

PAYLOADS = "payloads"
"""The directory of a pool's payload files, under its root, each named by its sha256."""
JSON_MEDIA = "application/json"


class PayloadMismatch(ValueError):
    """A payload that is not the one its ticket names: another digest, or another codec."""


@ty.runtime_checkable
class JobCodec(ty.Protocol):
    """How a pool carries a job from the coordinator to a worker.

    ``name`` is recorded in each payload. A codec whose ``encode`` gives ``application/json`` has its
    jobs carried in the ticket; any other is written to the job's work directory.
    """

    name: str

    def encode(self, job: "Job") -> tuple[bytes, str]:
        """The job's bytes and their media type."""

    def decode(self, data: bytes, store: ty.Any) -> "Job":
        """The job `encode` gave ``data`` for, attached to ``store``."""


@attrs.frozen
class PrivatePickle:
    """The default codec: the job cloudpickled to ``_job.pklz`` in its ``0o700`` work directory, the
    trust ``_job.pklz`` has under the Slurm worker, made explicit and checked."""

    name: ty.ClassVar[str] = PRIVATE_PICKLE
    media_type: ty.ClassVar[str] = "application/vnd.pydra.job+cloudpickle"

    def encode(self, job: "Job") -> tuple[bytes, str]:
        import cloudpickle

        return cloudpickle.dumps(job), self.media_type

    def decode(self, data: bytes, store: ty.Any) -> "Job":
        import cloudpickle

        job = cloudpickle.loads(data)
        if store is not None:
            job.store = store
        return job


def _make_private_dir(path: Path) -> None:
    """Make ``path`` ``0o700`` if it is absent; an existing one is left as it is."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        return
    if not IS_WINDOWS:
        os.chmod(path, 0o700)


def write_payload(codec: JobCodec, job: "Job", root: os.PathLike) -> dict:
    """Encode ``job`` and return its payload for a ticket.

    A JSON encoding is returned inline as ``{"codec", "sha256", "json"}``, the digest taken of its
    JCS form. Any other is written once to ``<root>/payloads/<sha256>`` (``0o600``; the directory
    made ``0o700`` if absent) under the pool's ``root``, and ``{"codec", "path", "sha256"}`` is
    returned; it is read by every attempt of the job and removed only by
    `~pydra.workers.pool.storage.PoolRoot.collect` once no unsettled ticket names it.

    Raises
    ------
    PayloadMismatch
        The payload file holds other bytes than its name digests.
    """
    data, media = codec.encode(job)
    if media == JSON_MEDIA:
        value = json.loads(data)
        canonical = jcs(value)
        return {
            "codec": codec.name,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "json": value,
        }
    digest = hashlib.sha256(data).hexdigest()
    where = Path(root).absolute() / PAYLOADS
    _make_private_dir(where)
    with StorageDir.open_root(where, POLICY, trust="trusted") as handle:
        if _create(handle, digest, data) == "exists":
            raise PayloadMismatch(f"{where / digest} holds other bytes")
        handle.set_mode(FILE_MODE, digest)
    return {"codec": codec.name, "path": str(where / digest), "sha256": digest}


def read_payload(codec: JobCodec, payload: dict, store: ty.Any) -> "Job":
    """The job a ticket's ``payload`` names, decoded by ``codec`` and attached to ``store``.

    A file payload is read only when its work directory and the file pass the ownership walk
    (owned by this user and writable by no one else; on Windows, under the user's local application
    data) and its sha256 is the payload's.

    Raises
    ------
    EntryNotTrusted
        The walk failed.
    PayloadMismatch
        The codec or the digest is not the payload's.
    """
    if payload.get("codec") != codec.name:
        raise ValueError(
            f"the payload's codec is {payload.get('codec')!r}, not {codec.name!r}"
        )
    if "json" in payload:
        data = jcs(payload["json"])
    else:
        path = Path(payload["path"])
        try:
            handle = StorageDir.open_root(path.parent, POLICY, trust="private")
        except StoreUnavailable as err:
            raise EntryNotTrusted(str(err)) from err
        with handle:
            check_private(handle, [path.name])
            with handle.open_read(path.name) as f:
                data = f.read()
    if hashlib.sha256(data).hexdigest() != payload.get("sha256"):
        raise PayloadMismatch(
            f"the payload of {payload.get('path', 'a ticket')} is not its digest's"
        )
    return codec.decode(data, store)
