"""What a portable store asks of identity: the key, the context and the fingerprint
(RFC 0003 section 4)."""

from __future__ import annotations

import hashlib
import typing as ty

import attrs

from pydra.engine.context import RECORDED_FIELDS, job_context
from pydra.utils.digest import jcs
from pydra.utils.identity import IDENTITY_SCHEME, record_checksum

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

__all__ = [
    "Context",
    "KeyRecord",
    "StoreIdentity",
    "PydraIdentityV1",
    "PydraIdentityV2",
]

Context = ty.Mapping[str, ty.Any]
KeyRecord = ty.Optional[ty.Mapping[str, ty.Any]]


@ty.runtime_checkable
class StoreIdentity(ty.Protocol):
    """The identity a `PortableStore` is configured with. It must be picklable, because
    ``context(job, "execute")`` runs in worker processes.

    Attributes
    ----------
    name : str
        The layout header's ``identity.name``.
    fingerprint_name : str
        The header's ``identity.fingerprint``.
    recorded_only : frozenset of str
        The header's ``identity.recorded_only``: context fields kept in ``context.json``
        that do not split slots.
    """

    name: str
    fingerprint_name: str
    recorded_only: frozenset[str]

    def key(self, job: "Job") -> KeyRecord:
        """The record the job's checksum is the digest of, or None."""
        ...

    def check_key(self, checksum: str, key: KeyRecord) -> bool:
        """Whether ``key`` digests to ``checksum``."""
        ...

    def context(self, job: "Job", phase: ty.Literal["lookup", "execute"]) -> Context:
        """The context of ``job`` for the lookup in the submitting or locking process, or for
        publishing in the process that ran it."""
        ...

    def fingerprint(self, context: Context) -> str:
        """The slot name of ``context``: 64 lower-case hex characters."""
        ...


def _fields(value: ty.Iterable[str]) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"recorded_only is a collection of field names, got {value!r}")
    return frozenset(value) | RECORDED_FIELDS


POOL_FIELD = "pool"
"""The section a pool worker adds to the execute context (RFC 0005 section 8)"""


def _v2_fields(value: ty.Iterable[str]) -> frozenset[str]:
    return _fields(value) | {POOL_FIELD}


@attrs.frozen
class PydraIdentityV1:
    """``pydra-identity@1``: pydra's checksum is not the digest of a stored record, the context
    is `pydra.engine.context.job_context` built once in the submitting process and carried with
    the job, so both phases return it, and the fingerprint
    (``pydra-context@1``) is the full sha256 of the RFC 8785 canonical JSON of the context
    without the ``recorded_only`` fields. ``recorded_only`` always holds
    ``environment_repository`` and ``environment_facts``, whatever else is given."""

    recorded_only: frozenset[str] = attrs.field(factory=frozenset, converter=_fields)
    name: ty.ClassVar[str] = "pydra-identity@1"
    fingerprint_name: ty.ClassVar[str] = "pydra-context@1"

    @recorded_only.validator
    def _strings(self, _attribute: ty.Any, value: frozenset[str]) -> None:
        for field in value:
            if not isinstance(field, str):
                raise TypeError(f"recorded_only holds field names, got {field!r}")

    def key(self, job: "Job") -> KeyRecord:
        return None

    def check_key(self, checksum: str, key: KeyRecord) -> bool:
        return True

    def context(self, job: "Job", phase: ty.Literal["lookup", "execute"]) -> Context:
        if phase == "execute":
            return (
                job.context
            )  # the one built in the submitting process and carried (section 4)
        job.checksum  # collects the library modules the task's code reaches
        return job_context(
            job.task, job.environment, job.resolved if job.resolves_reference else None
        )

    def fingerprint(self, context: Context) -> str:
        slotted = {k: v for k, v in context.items() if k not in self.recorded_only}
        return hashlib.sha256(jcs(slotted)).hexdigest()


@attrs.frozen
class PydraIdentityV2(PydraIdentityV1):
    """``pydra-identity@2`` (RFC 0004 section 3.1): the checksum is the digest of the job's
    identity record, which is the key. The context and the fingerprint are
    `PydraIdentityV1`'s. ``recorded_only`` always holds ``pool``, the section a pool
    worker adds to the execute context (RFC 0005 section 8), beside
    ``environment_repository`` and ``environment_facts``, whatever else is given."""

    recorded_only: frozenset[str] = attrs.field(
        factory=frozenset,
        converter=_v2_fields,
        validator=PydraIdentityV1._strings,
    )
    name: ty.ClassVar[str] = IDENTITY_SCHEME

    def key(self, job: "Job") -> KeyRecord:
        """The job's record as carried with it from the submitting process, never rebuilt
        from the task (a worker may be unable to read the function's source)."""
        return job.record

    def check_key(self, checksum: str, key: KeyRecord) -> bool:
        """Whether ``key`` is a ``pydra-identity@2`` record that digests to ``checksum``,
        or a supplied record whose ``supplied`` is ``checksum``."""
        if not isinstance(key, ty.Mapping) or key.get("scheme") != IDENTITY_SCHEME:
            return False
        try:
            return record_checksum(key) == checksum
        except (KeyError, TypeError, ValueError):
            return False
