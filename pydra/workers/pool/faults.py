"""Fault records and budgets of a pool (RFC 0005 section 6).

The worker types each fault where it happens through a classifier, ``classify(exc, phase)``,
records it in the outcome it writes, and applies the budgets below.
"""

from __future__ import annotations

import typing as ty

import attrs

__all__ = [
    "FaultBudgets",
    "FaultRecord",
    "Phase",
    "Classifier",
    "default_classifier",
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
        What made a resource fault: ``"oom"`` or ``"timeout"``.
    gap : bool
        An unclassified exception from the pool's machinery.
    """

    kind: Kind
    type: str
    message: str = ""
    phase: Phase = "call"
    reason: str | None = None
    gap: bool = False

    def to_json(self) -> dict:
        return attrs.asdict(self)

    @classmethod
    def from_json(cls, record: dict) -> "FaultRecord":
        return cls(**record)


Classifier = ty.Callable[[BaseException, Phase], FaultRecord]


def _qualnames(exc: BaseException) -> set[str]:
    return {f"{t.__module__}.{t.__qualname__}" for t in type(exc).__mro__}


def default_classifier(exc: BaseException, phase: Phase) -> FaultRecord:
    """Out of memory (`OUT_OF_MEMORY`) is a resource fault anywhere; otherwise an exception from a
    call (the task's own code) is content, and one from a load or the pool's machinery is an
    infrastructure fault flagged ``gap``."""
    name = type(exc).__name__
    message = str(exc)[:2000]
    if _qualnames(exc) & OUT_OF_MEMORY:
        return FaultRecord("resource", name, message, phase, reason="oom")
    if phase == "call":
        return FaultRecord("content", name, message, phase)
    return FaultRecord("infrastructure", name, message, phase, gap=True)


@attrs.frozen
class FaultBudgets:
    """How often a job may fail before its fault is final.

    Attributes
    ----------
    content_attempts : int
        Runs of a job whose task raises, each retry under a ``restart`` generation (1: no retry).
    crashes_charged : int
        Charged crashes after which a job is ``failed`` (``JobCrashedWorker``).
    resource_moves : int
        Moves up the ticket's ladder after an out-of-memory call or a timeout.
    infrastructure_replacements : int
        Infrastructure faults of a job (each a requeue) after which it is ``failed``
        (``InfrastructureFault``).
    """

    content_attempts: int = 1
    crashes_charged: int = 2
    resource_moves: int = 2
    infrastructure_replacements: int = 3

    def __attrs_post_init__(self) -> None:
        if (
            self.content_attempts < 1
            or self.crashes_charged < 1
            or self.infrastructure_replacements < 1
        ):
            raise ValueError(
                "content_attempts, crashes_charged and infrastructure_replacements are at "
                "least 1"
            )
        if self.resource_moves < 0:
            raise ValueError("resource_moves is not negative")
