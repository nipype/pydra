"""Typed absent values (RFC 0002 section 2.3(a)).

A task may return :class:`Absent` for any output. A reader's input field declares what it
does with one through its ``absent`` attribute: ``"propagate"`` (the default: the element
completes absent without being looked up, shared, dispatched or committed), ``"accept"``
(the value is passed as resolved), ``"drop"`` (a combined value is passed without its
absent elements) or ``"fill"`` (each is replaced with the field's ``fill``). A single value
that is an ``Absent`` propagates under ``"drop"``, as there are no elements to drop. An
``Absent`` left inside the value of a field that does not accept it raises
:class:`AbsentNotAccepted`. The triple of an ``Absent`` that is a field's whole value has
the element index -1 (``WHOLE``).

Outside the runner's resolution of an element, an ``Absent`` passes the type coercion of
an output field and of an input field whose ``absent`` is ``"accept"`` or ``"drop"``.

Every read of a node that is absent as a whole (a splitter input resolved to an
``Absent``, or its state inherits such a node's) is that ``Absent``. A combining reader then
applies its mode: ``"accept"`` passes the ``Absent``, ``"drop"`` an empty list, ``"fill"``
the fill, and ``"propagate"`` completes the reader absent.

A task whose checksum may not see an accepted or dropped ``Absent`` (a supplied checksum,
or a field the checksum skips), or that filled one, has :func:`absence_suffix` of
:func:`absence_triples` appended to its job checksum.
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
    "AbsentNotAccepted",
    "absence_identity",
    "absence_suffix",
    "absence_triples",
    "task_definition",
]

AbsentMode = ty.Literal["propagate", "accept", "drop", "fill"]

ABSENT_MODES = ("propagate", "accept", "drop", "fill")

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


class AbsentNotAccepted(TypeError):
    """An ``Absent`` would reach a task through a field that does not declare
    ``absent="accept"``."""


def contains_absent(value: ty.Any) -> bool:
    """Whether ``value`` is an ``Absent`` or holds one in a list, tuple, set or dict."""
    if isinstance(value, Absent):
        return True
    if isinstance(value, dict):
        return any(contains_absent(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_absent(v) for v in value)
    return False


def absence_triples(job: "Job") -> list[tuple[str, int, str]]:
    """The ``(field, element index, cause)`` of every ``Absent`` the checksum of a job may
    not see, sorted

    These are the absent values a field filled, accepted or dropped (a field's whole value
    has index -1, an element of a combined value its position before dropping), restricted
    to the fields the checksum does not hash: every field of a task with a supplied checksum
    (``__checksum__`` or ``checksum=``), else the fields with ``container_path``. An
    ``Absent`` a field holds as a constant counts too, at index -1 or its position in a list.

    Parameters
    ----------
    job : Job
        A job formed by the graph runner
    """
    from pydra.utils.general import get_fields

    if job._propagated_absent is not None:
        return []  # its identity is the absence record, never suffixed
    task = job.task
    fields = get_fields(task)
    if hasattr(task, "__checksum__"):
        unseen = {f.name for f in fields}
    else:
        unseen = {f.name for f in fields if getattr(f, "container_path", False)}
    filled = {f.name for f in fields if f.absent == "fill"}
    accepted = set(job._accepted_absent)
    if not unseen:
        return sorted(t for t in accepted if t[0] in filled)
    recorded = {t[0] for t in accepted}
    for name in unseen - recorded:
        value = getattr(task, name, None)
        if isinstance(value, Absent):
            accepted.add((name, WHOLE, value.cause))
        elif isinstance(value, (list, tuple)):
            accepted.update(
                (name, i, v.cause) for i, v in enumerate(value) if isinstance(v, Absent)
            )
    return sorted(t for t in accepted if t[0] in unseen | filled)


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


def task_definition(task) -> tuple[str, str]:
    """The task type and the hash of the task's definition without its inputs: its
    executor (as the checksum hashes it, by code or declared version) and its outputs
    class, as a normal checksum hashes them

    Parameters
    ----------
    task : Task
        The task
    """
    from pydra.utils.hash import Cache, hash_function

    cache = Cache(code_hash=task._code_hash)
    executor = task._executor_name
    parts = {"Outputs": hash_function(task.Outputs, cache=cache)}
    if executor is not None:
        parts[executor] = hash_function(task._hashed_value(executor), cache=cache)
    return task._task_type(), hash_function(sorted(parts.items()))


def absence_identity(
    definition: tuple[str, str], producer: str | None, triple: tuple[str, int, str]
) -> str:
    """The checksum of a job that completes absent, made without the task's checksum,
    so no ``checksum=`` hook ever sees an ``Absent``: ``<task type>-absent-`` and the hex
    SHA-256 of the canonical JSON of
    ``[definition hash, producer, [field, {"$int": index}, cause]]``. Different tasks
    reading the same ``Absent`` differ; the same task reading it at several elements
    has one identity, as its results are equal.

    Parameters
    ----------
    definition : tuple[str, str]
        The task type and the hash of the task's definition (:func:`task_definition`)
    producer : str, optional
        The unsuffixed checksum of the job the ``Absent`` was read from, None if it was
        not read from an upstream job
    triple : tuple[str, int, str]
        ``(field, element index, cause)`` of the propagated ``Absent``
    """
    task_type, definition_hash = definition
    field, index, cause = triple
    form = [definition_hash, producer, [field, {"$int": str(int(index))}, cause]]
    return f"{task_type}-absent-" + hashlib.sha256(jcs(form)).hexdigest()


def absence_suffix(triples: ty.Sequence[tuple[str, int, str]]) -> str:
    """``-absent-`` and the hex SHA-256 of the sorted triples, ``""`` for none

    The digest is taken over the RFC 8785 canonical JSON of
    ``[[field, {"$int": "<index>"}, cause], ...]``, the encoding of RFC 0004's key record.
    The suffix holds no path separator.

    Parameters
    ----------
    triples : Sequence[tuple[str, int, str]]
        ``(field, element index, cause)`` of each ``Absent``
    """
    if not triples:
        return ""
    form = [[f, {"$int": str(int(i))}, c] for f, i, c in sorted(triples)]
    return "-absent-" + hashlib.sha256(jcs(form)).hexdigest()
