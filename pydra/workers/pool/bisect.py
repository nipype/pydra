"""One bisection engine for local and pool execution (RFC 0005 sections 4.4, 6).

`bisect` runs a list of jobs as one call and settles each. A call that raises a resource fault
or a crash as a whole reloads the residency (``reload``) and is halved until the failing job is
alone; a half that completed keeps its results. A lone job's resource fault is returned for the
caller to move up the ladder; a lone job's crash is run again until it has been lost
``crashes_charged`` times in a row (``JobCrashedWorker``, a content fault). A crash whose jobs all
completed afterwards is reported (``survived``) and charged to no job. Any other raise fails each
job of the call with that fault; a non-`Exception` propagates, and a load fault is raised out
unchanged. An
exception element of a batched return is that job's fault and never a trigger: content, or the
record attached to it (a job's own run that raised, typed by the caller's classifier).
"""

from __future__ import annotations

import typing as ty

import attrs

from pydra.workers.pool.faults import Classifier, FaultRecord, attached, fault_of

__all__ = ["Abandon", "Settled", "bisect"]

T = ty.TypeVar("T")


class Abandon(Exception):
    """Raised by a call that ended without a verdict (its claim lost, its worker stopping, its job
    superseded): `bisect` re-raises it, and nothing is split or retried."""


@attrs.frozen
class Settled:
    """What became of one job of a bisected call.

    Attributes
    ----------
    value : object
        The job's result, when ``fault`` is None.
    fault : FaultRecord, optional
        Its fault: content (its own exception element, a non-trigger raise of its call, or
        ``JobCrashedWorker``), resource (alone and still out of memory), or whatever else its
        whole call raised.
    error : BaseException, optional
        The exception behind ``fault``, if one was raised.
    charged : bool
        ``JobCrashedWorker``: the job alone was lost ``crashes_charged`` times.
    """

    value: ty.Any = None
    fault: FaultRecord | None = None
    error: BaseException | None = None
    charged: bool = False


def bisect(
    items: ty.Sequence[T],
    call: ty.Callable[[list[T]], ty.Sequence[ty.Any]],
    *,
    classify: Classifier,
    reload: ty.Callable[[FaultRecord], None] | None = None,
    lost: ty.Callable[[T], int] | None = None,
    crashes_charged: int = 2,
    survived: ty.Callable[[FaultRecord], None] | None = None,
) -> list[Settled]:
    """Run ``items`` as one ``call`` and settle each, bisecting on the resource and crash triggers.

    Parameters
    ----------
    items : sequence
        The jobs of the call.
    call : callable
        ``call(jobs) -> [output or exception, ...]``, one element per job, in order; it raises when
        the call fails as a whole, and raises `Abandon` when it ended without a verdict.
    classify : callable
        ``classify(exc, "call") -> FaultRecord`` for an exception the whole call raised, when no
        record is attached to it (`fault_of`).
    reload : callable, optional
        Called with the fault after each trigger, before anything runs again.
    lost : callable, optional
        The crashes of a job recorded elsewhere (its charged generations); the job is charged when
        that count or the engine's own count of its lone losses in a row reaches
        ``crashes_charged``, so the engine terminates whatever the hook returns.
    crashes_charged : int
        Crashes of a job alone after which it is ``JobCrashedWorker``.
    survived : callable, optional
        Called with each crash whose jobs all completed in the end.

    Returns
    -------
    list of Settled
        One per item, in order.

    Raises
    ------
    Abandon
        The call ended without a verdict.
    BaseException
        A fault at a load (its record's phase is ``"load"``), as the call raised it.
    """
    settled: list[Settled | None] = [None] * len(items)
    counts: dict[int, int] = {}
    crashes: list[tuple[FaultRecord, list[int]]] = []

    def count(index: int) -> int:
        counts[index] = counts.get(index, 0) + 1
        recorded = lost(items[index]) if lost is not None else 0
        return max(recorded, counts[index])

    def run(indices: list[int]) -> None:
        try:
            outputs = list(call([items[i] for i in indices]))
        except Abandon:
            raise
        except Exception as exc:
            fault = fault_of(exc, "call", classify)
            if fault.phase == "load":
                raise
            if fault.kind != "resource" and not fault.crash:
                for i in indices:
                    settled[i] = Settled(fault=fault, error=exc)
                return
            if fault.crash:
                crashes.append((fault, list(indices)))
            if reload is not None:
                reload(fault)
            if len(indices) > 1:
                half = (len(indices) + 1) // 2
                run(indices[:half])
                run(indices[half:])
                return
            [i] = indices
            if fault.kind == "resource":
                settled[i] = Settled(fault=fault, error=exc)
                return
            losses = count(i)
            if losses >= crashes_charged:
                settled[i] = Settled(
                    fault=FaultRecord(
                        "content",
                        "JobCrashedWorker",
                        f"the job's call crashed {losses} times (the budget is "
                        f"{crashes_charged}); the last: {fault.type}: {fault.message}",
                        "call",
                    ),
                    error=exc,
                    charged=True,
                )
                return
            run(indices)
            return
        if len(outputs) != len(indices):
            mismatch = ValueError(
                f"a batched call of {len(indices)} jobs returned {len(outputs)} elements"
            )
            fault = FaultRecord("content", "ValueError", str(mismatch), "call")
            for i in indices:
                settled[i] = Settled(fault=fault, error=mismatch)
            return
        for i, output in zip(indices, outputs):
            if isinstance(output, BaseException):
                record = attached(output) if isinstance(output, Exception) else None
                settled[i] = Settled(
                    fault=record
                    or FaultRecord(
                        "content", type(output).__name__, str(output)[:2000], "call"
                    ),
                    error=output,
                )
            else:
                settled[i] = Settled(value=output)

    if items:
        run(list(range(len(items))))
    if survived is not None:
        for fault, indices in crashes:
            if all(
                settled[i] is not None and settled[i].fault is None for i in indices
            ):
                survived(fault)
    return ty.cast(list[Settled], settled)
