"""Typed absent values (RFC 0002 section 2.3(a)).

A task may return :class:`Absent` for any output. A reader's input field declares what it
does with one through its ``absent`` attribute: ``"propagate"`` (the element completes
absent without being looked up, shared, dispatched or committed), ``"accept"`` (the value
is passed as resolved) or ``"drop"`` (a combined value is passed without its absent
elements). Left unset, a field that reads a single value propagates and one that reads a
combined value accepts. A field whose whole value is an ``Absent`` propagates under
``"drop"``, as there are no elements to drop; its triple has the element index -1
(``WHOLE``).

Outside the runner's resolution of an element, an ``Absent`` passes the type coercion of
an output field and of an input field whose ``absent`` is ``"accept"`` or ``"drop"``.

A combined read of a node that is absent as a whole (a splitter input resolved to an
``Absent``, or it inherits such a node's state) is an empty list; any other read of it is
its ``Absent``.

A task whose checksum may not see an accepted or dropped ``Absent`` (a supplied checksum,
or a field the checksum skips) has :func:`absence_suffix` of :func:`absence_triples`
appended to its job checksum.
"""

import contextlib
import contextvars
import hashlib
import typing as ty

import attrs

from pydra.utils.digest import jcs

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

__all__ = [
    "Absent",
    "AbsentMode",
    "absence_identity",
    "absence_suffix",
    "absence_triples",
]

AbsentMode = ty.Literal["propagate", "accept", "drop"]

ABSENT_MODES = ("propagate", "accept", "drop")

# The element index of the triple of an ``Absent`` that is a field's whole value
WHOLE = -1


@attrs.frozen
class Absent:
    """A value that is not there, and why

    Attributes
    ----------
    cause : str
        Free text naming why; a project defines its own vocabulary. It enters the hash
    source : str | None
        The node and element that produced it; provenance only, never hashed
    """

    cause: str
    source: str | None = None

    def __bytes_repr__(self, cache) -> ty.Iterator[bytes]:
        yield b"pydra.Absent(cause="
        yield self.cause.encode()
        yield b")"


def absence_triples(job: "Job") -> list[tuple[str, int, str]]:
    """The ``(field, element index, cause)`` of every ``Absent`` the checksum of a job may
    not see, sorted

    These are the absent values a field accepted or dropped (a field's whole value has
    index -1, an element of a combined value its position before dropping), restricted
    to the fields the checksum does not hash: every field of a task with a supplied
    checksum (``__checksum__`` or ``checksum=``), else the fields with
    ``container_path``. A task whose checksum pydra computes from hashed values sees an
    ``Absent`` through its ``cause``, so it has none.

    Parameters
    ----------
    job : Job
        A job formed by the graph runner
    """
    from pydra.utils.general import get_fields

    accepted = getattr(job, "_accepted_absent", None) or ()
    if not accepted:
        return []
    task = job.task
    if hasattr(task, "__checksum__"):
        unseen = None
    else:
        unseen = {
            f.name for f in get_fields(task) if getattr(f, "container_path", False)
        }
    return sorted(t for t in accepted if unseen is None or t[0] in unseen)


_RESOLVING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pydra_absent_resolving", default=False
)


@contextlib.contextmanager
def resolving() -> ty.Iterator[None]:
    """While the runner resolves an element's inputs, an ``Absent`` passes the coercion
    of every input field, whose ``absent`` mode is applied right after"""
    token = _RESOLVING.set(True)
    try:
        yield
    finally:
        _RESOLVING.reset(token)


def is_resolving() -> bool:
    """Whether the runner is resolving an element's inputs (see :func:`resolving`)"""
    return _RESOLVING.get()


def absence_identity(
    task_type: str, producer: str | None, triple: tuple[str, int, str]
) -> str:
    """The checksum of a job that completes absent, made without the task's checksum,
    so no ``checksum=`` hook ever sees an ``Absent``: ``<task type>-absent-`` and the hex
    SHA-256 of the canonical JSON of ``[producer, [field, {"$int": index}, cause]]``

    Parameters
    ----------
    task_type : str
        The task type, as it begins the task's checksum
    producer : str, optional
        The unsuffixed checksum of the job the ``Absent`` was read from, None if it was
        not read from an upstream job
    triple : tuple[str, int, str]
        ``(field, element index, cause)`` of the propagated ``Absent``
    """
    field, index, cause = triple
    form = [producer, [field, {"$int": str(int(index))}, cause]]
    return f"{task_type}-absent-" + hashlib.sha256(jcs(form)).hexdigest()


def absence_suffix(triples: ty.Sequence[tuple[str, int, str]]) -> str:
    """``-absent-`` and the hex SHA-256 of the sorted triples, ``""`` for none

    The digest is taken over the RFC 8785 canonical JSON of
    ``[[field, {"$int": "<index>"}, cause], ...]``, the encoding RFC 0004's key record
    carries the triples in, so a store can recompute the suffix from the record. The
    suffix holds no path separator, so the checksum stays one path component.

    Parameters
    ----------
    triples : Sequence[tuple[str, int, str]]
        ``(field, element index, cause)`` of each ``Absent``
    """
    if not triples:
        return ""
    form = [[f, {"$int": str(int(i))}, c] for f, i, c in sorted(triples)]
    return "-absent-" + hashlib.sha256(jcs(form)).hexdigest()
