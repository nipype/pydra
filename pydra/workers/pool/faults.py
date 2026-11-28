"""Fault records, the classifier and the fault budgets of a pool (RFC 0005 section 6).

A fault is typed where it happens by a classifier, ``classify(exc, phase) -> FaultRecord``, and each
kind has one effect: a content fault is the job's own (retried up to ``content_attempts``); a
resource fault moves the job up its ladder (a batched call is bisected first,
`pydra.workers.pool.bisect`); an infrastructure fault requeues the job, and holds its residency once
the residency has failed ``infrastructure_replacements`` times in the coordinator's epoch; a crash
(`WorkerLost`) is retried on a fresh process until the job, alone, has lost ``crashes_charged``.

A fault's class survives every wrapper: a record attached to an exception (`attach`), or to any
exception it was raised from or during, is the fault, whatever classifier is in use (`fault_of`).
"""

from __future__ import annotations

import typing as ty

import attrs

__all__ = [
    "Classifier",
    "FaultBudgets",
    "FaultRecord",
    "Phase",
    "WorkerLost",
    "attach",
    "attached",
    "classify",
    "fault_of",
]

Phase = ty.Literal["load", "call", "pool"]
Kind = ty.Literal["content", "resource", "infrastructure", "policy"]

# Exceptions that are out of memory, by qualified name, so nothing is imported to match them.
OUT_OF_MEMORY = frozenset(
    (
        "builtins.MemoryError",
        "torch.OutOfMemoryError",
        "torch.cuda.OutOfMemoryError",
        "jax.errors.ResourceExhaustedError",
    )
)

_RECORD = "__pydra_fault__"
_MESSAGE = 2000


class WorkerLost(RuntimeError):
    """The process that ran a call died or broke its protocol: a crash (RFC 0005 section 6).

    A process-owning provider's client raises it (or raises from it) when its process ends, or
    answers out of protocol, during a request.
    """


@attrs.frozen
class FaultRecord:
    """A typed fault.

    Attributes
    ----------
    kind : {"content", "resource", "infrastructure", "policy"}
    type : str
        The exception's class name, or a pool name (``ResourceExhausted``, ``JobCrashedWorker``,
        ``Unplaceable``).
    message : str
    phase : {"load", "call", "pool"}
    reason : str, optional
        What made it: ``"oom"`` or ``"timeout"`` (a resource fault), ``"crash"`` (a lost process).
    gap : bool
        An unclassified exception from the pool's machinery.
    """

    kind: Kind
    type: str
    message: str = ""
    phase: Phase = "call"
    reason: str | None = None
    gap: bool = False

    @property
    def crash(self) -> bool:
        """Whether the fault is a lost process (`WorkerLost`)."""
        return self.reason == "crash"

    def to_json(self) -> dict:
        return attrs.asdict(self)

    @classmethod
    def from_json(cls, record: dict) -> "FaultRecord":
        return cls(**record)


Classifier = ty.Callable[[BaseException, Phase], FaultRecord]


def _chain(exc: BaseException) -> ty.Iterator[BaseException]:
    """``exc``, then what it was raised from (``__cause__``) or during (``__context__``)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None


def _qualnames(exc: BaseException) -> set[str]:
    return {f"{t.__module__}.{t.__qualname__}" for t in type(exc).__mro__}


def _message(exc: BaseException, cause: BaseException) -> str:
    text = str(exc)
    if cause is not exc:
        text = f"{text} (from {type(cause).__name__}: {cause})"
    return text[:_MESSAGE]


def attach(exc: BaseException, record: FaultRecord) -> BaseException:
    """Attach ``record`` to ``exc`` and return ``exc``: whatever wraps or re-raises it, its fault is
    ``record`` (`fault_of`)."""
    setattr(exc, _RECORD, record)
    return exc


def attached(exc: BaseException) -> FaultRecord | None:
    """The record attached to ``exc``, or to the nearest exception of its chain; None when none is."""
    for link in _chain(exc):
        record = getattr(link, _RECORD, None)
        if isinstance(record, FaultRecord):
            return record
    return None


def classify(exc: BaseException, phase: Phase) -> FaultRecord:
    """The default classifier.

    Out of memory (`OUT_OF_MEMORY`, by qualified name, anywhere in the exception's chain) is a
    resource fault at any phase; a `WorkerLost` in the chain is a crash (an infrastructure fault
    with reason ``"crash"``). Otherwise an exception from a call (the task's own code) is content,
    and one from a load or the pool's machinery is an infrastructure fault flagged ``gap``.
    """
    name = type(exc).__name__
    links = list(_chain(exc))
    for link in links:
        if _qualnames(link) & OUT_OF_MEMORY:
            return FaultRecord(
                "resource", name, _message(exc, link), phase, reason="oom"
            )
    for link in links:
        if isinstance(link, WorkerLost):
            return FaultRecord(
                "infrastructure", name, _message(exc, link), phase, reason="crash"
            )
    if phase == "call":
        return FaultRecord("content", name, str(exc)[:_MESSAGE], phase)
    return FaultRecord("infrastructure", name, str(exc)[:_MESSAGE], phase, gap=True)


def fault_of(
    exc: BaseException, phase: Phase, classifier: Classifier = classify
) -> FaultRecord:
    """The fault of ``exc`` at ``phase``: the record attached to it or its chain (`attached`) when
    there is one, else ``classifier``'s. A load is never a job's: a content record at a load is the
    residency's, an infrastructure fault."""
    record = attached(exc)
    if record is None:
        record = classifier(exc, phase)
    if phase == "load" and record.kind == "content":
        record = attrs.evolve(record, kind="infrastructure", phase="load")
    return record


@attrs.frozen
class FaultBudgets:
    """How often a job, a residency or a worker class may fail before its fault is final (RFC 0005
    section 6). Every default is pydra's.

    Attributes
    ----------
    content_attempts : int
        Runs of a job whose task raises, each retry under a ``restart`` generation (1: no retry).
    resource_moves : int
        Moves of a job up its ticket's ladder after an out-of-memory call or a timeout, counted from
        its ``ladder`` generations at the ``call`` phase.
    infrastructure_replacements : int
        Infrastructure faults of one residency in a coordinator epoch after which the residency is
        held on its queue and its jobs are ``NotRun(held=True)``.
    crashes_charged : int
        Crashes charged to a job (alone in its call, or in its lost worker's in-flight call) after
        which it is ``failed`` (``JobCrashedWorker``).
    start_failures : int
        Consecutive failed starts after which a worker class is broken and its jobs are held.
    listing_failures : int
        Consecutive polls whose listing of the substrate's workers failed after which every class
        with work is broken and its jobs are held.
    """

    content_attempts: int = 1
    resource_moves: int = 2
    infrastructure_replacements: int = 3
    crashes_charged: int = 2
    start_failures: int = 7
    listing_failures: int = 7

    def __attrs_post_init__(self) -> None:
        low = [
            name
            for name in (
                "content_attempts",
                "infrastructure_replacements",
                "crashes_charged",
                "start_failures",
                "listing_failures",
            )
            if getattr(self, name) < 1
        ]
        if low:
            raise ValueError(f"{', '.join(low)} must be at least 1")
        if self.resource_moves < 0:
            raise ValueError("resource_moves is not negative")

    def to_json(self) -> dict:
        return attrs.asdict(self)

    @classmethod
    def from_json(cls, record: dict) -> "FaultBudgets":
        """The budgets `to_json` wrote; a missing field takes its default.

        Raises
        ------
        ValueError
            An unknown field.
        """
        unknown = set(record) - {f.name for f in attrs.fields(cls)}
        if unknown:
            raise ValueError(f"unknown fault budgets: {sorted(unknown)}")
        return cls(**record)
