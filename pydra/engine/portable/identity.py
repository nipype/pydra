"""What a portable store asks of identity: the key, the context and the fingerprint
(RFC 0003 section 4)."""

from __future__ import annotations

import hashlib
import typing as ty

import attrs

from pydra.engine.absent import absence_suffix
from pydra.engine.context import RECORDED_FIELDS, job_context
from pydra.utils.digest import jcs
from pydra.utils.identity import IDENTITY_SCHEME, record_checksum

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

__all__ = [
    "Context",
    "KeyRecord",
    "StoreIdentity",
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
    scheme: int | None
    """The ``pydra-identity`` version whose checksums this identity looks up; None for
    another scheme."""

    def key(self, job: "Job") -> KeyRecord:
        """The record the job's checksum is the digest of, or None."""
        ...

    def supplied_key(self, checksum: str) -> KeyRecord:
        """The key record of a checksum supplied from outside (a migrated slot), or None
        for an identity that records no keys."""
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
    ``environment_repository`` and ``environment_facts``, whatever else is given. It is the
    reader of roots written before ``pydra-identity@2`` (`migrate`) and the base of
    `PydraIdentityV2`; no store writes it by default."""

    recorded_only: frozenset[str] = attrs.field(factory=frozenset, converter=_fields)
    name: ty.ClassVar[str] = "pydra-identity@1"
    fingerprint_name: ty.ClassVar[str] = "pydra-context@1"
    scheme: ty.ClassVar[int | None] = 1

    def supplied_key(self, checksum: str) -> KeyRecord:
        return None

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
    scheme: ty.ClassVar[int | None] = 2

    def supplied_key(self, checksum: str) -> KeyRecord:
        return {"scheme": IDENTITY_SCHEME, "supplied": checksum}

    def key(self, job: "Job") -> KeyRecord:
        """The job's record as carried with it from the submitting process, never rebuilt
        from the task (a worker may be unable to read the function's source). A job whose
        checksum carries an absence suffix (RFC 0002 section 2.3(a)) adds its ``absent``
        triples (`absent_form`)."""
        triples = job.absent
        if not triples:
            return job.record
        return {**job.record, ABSENT_KEY: absent_form(triples)}

    def check_key(self, checksum: str, key: KeyRecord) -> bool:
        """Whether ``key`` is a ``pydra-identity@2`` record that digests to ``checksum``,
        or a supplied record whose ``supplied`` is ``checksum``. A record with ``absent``
        triples digests to the digest of the record without them plus their
        `~pydra.engine.absent.absence_suffix`; ``checksum`` itself is never parsed."""
        if not isinstance(key, ty.Mapping) or key.get("scheme") != IDENTITY_SCHEME:
            return False
        base = {k: v for k, v in key.items() if k != ABSENT_KEY}
        try:
            suffix = ""
            if ABSENT_KEY in key:
                triples = absent_triples(key[ABSENT_KEY])
                if not triples or absent_form(triples) != key[ABSENT_KEY]:
                    return False  # an empty or unsorted list is never written
                suffix = absence_suffix(triples)
            return record_checksum(base) + suffix == checksum
        except (KeyError, TypeError, ValueError):
            return False


ABSENT_KEY = "absent"
"""The field of a ``pydra-identity@2`` key holding a job's absence triples"""


def absent_form(triples: ty.Iterable[tuple[str, int, str]]) -> list[list[ty.Any]]:
    """The JSON form of absence triples, the one `~pydra.engine.absent.absence_suffix`
    digests: ``[field, {"$int": "<index>"}, cause]`` each, sorted"""
    return [[f, {"$int": str(int(i))}, c] for f, i, c in sorted(triples)]


def absent_triples(form: ty.Any) -> list[tuple[str, int, str]]:
    """The ``(field, index, cause)`` triples of a key's ``absent`` (see `absent_form`).

    Raises
    ------
    ValueError
        ``form`` is not a list of ``[str, {"$int": str}, str]``.
    """
    if not isinstance(form, list):
        raise ValueError(f"absent is a list of triples, got {form!r}")
    triples = []
    for item in form:
        if not (
            isinstance(item, list)
            and len(item) == 3
            and isinstance(item[0], str)
            and isinstance(item[2], str)
            and isinstance(item[1], dict)
            and set(item[1]) == {"$int"}
            and isinstance(item[1]["$int"], str)
        ):
            raise ValueError(
                f"an absence triple is [field, {{'$int': index}}, cause], got {item!r}"
            )
        triples.append((item[0], int(item[1]["$int"]), item[2]))
    return triples
