"""The settings every process of a pool reads (RFC 0005 sections 4.3, 5.3)."""

from __future__ import annotations

import attrs

from pydra.engine.lease import LeaseSettings

__all__ = ["PoolSettings"]


def _fraction(_inst, attribute, value) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{attribute.name} is in [0, 1], not {value!r}")


def _positive(_inst, attribute, value) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{attribute.name} must be positive, not {value!r}")


@attrs.frozen
class PoolSettings:
    """How a pool's processes poll, wait and bound calls.

    Attributes
    ----------
    lease : LeaseSettings
        The pool's lease settings; a worker whose four staleness fields differ from ``pool.json``'s
        refuses to start.
    poll_s : float
        Seconds between claim attempts while nothing is claimable.
    idle_polls : int
        Polls with nothing claimable after which a worker exits.
    orphan_after_s : float
        Seconds without a live coordinator lease after which a worker a substrate started exits.
    call_timeout : float, optional
        A call's time bound for a job that declares no ``ComputeResources.time``; None: no deadline.
    timeout_grace : float
        Seconds added to a call's deadline, and the time a worker has to stop an expired call before
        its beater kills it.
    drain_s : float
        Seconds a worker given SIGTERM waits for the call in flight before releasing its claims.
    coordinator_wait : float, optional
        Seconds a new coordinator waits for another live coordinator of the pool before it raises
        ``PoolBusy``; None waits without bound.
    max_queued_jobs : int, optional
        Jobs written as tickets per queue and not yet settled; the rest wait in the submitting
        process. None: 4 x ``quantum_jobs`` x ``max_workers`` of the queue's class.
    target_wall_s : float
        Seconds the waiting jobs of a queue should take with the workers scaling starts for them,
        when a per-job time is measured.
    unmeasured_workers : int
        Workers wanted per queue when no per-job time is measured.
    recheck_fraction : float
        The fraction of settled jobs run again alone on a fresh residency and compared with the
        committed entry (RFC 0005 section 4.4), sampled by checksum; 0: none. The re-check runs in
        the coordinator's process, on a thread of its own: that process needs the task's
        providers, its dependencies and, for a GPU task, a device (a re-check it cannot run is
        recorded as not run). A worker-side re-check is a follow-up.
    start_deadline_s : float, optional
        Seconds from the first failure of a streak after which a worker class with waiting jobs is
        broken, ahead of the ``start_failures`` budget, and its jobs are ``NotRun(held=True)`` with
        the latest error. A streak is the class's failed starts in a row (with no worker of the
        class live) or a queue's failed ticket writes in a row; it ends when a worker is ready, a
        write succeeds, or the class has no waiting jobs. The break is judged only at a failed
        attempt, never ahead of a due retry. None: only the fault budgets break a class.
    start_deadline_failures : int
        Failed attempts a streak needs, as well as ``start_deadline_s``, before it breaks the class.
    max_warm_sessions : int
        Warm environments (a started container, Apptainer instance or Lmod resolution, one per
        environment and resolved reference) a worker keeps open; opening one more closes the least
        recently used idle one. 4 by default (unmeasured): an idle container holds no device memory
        and little host memory, so the bound is on processes, mounts and scratch directories.
    """

    lease: LeaseSettings = attrs.field(factory=LeaseSettings)
    poll_s: float = attrs.field(default=5.0, validator=_positive)
    idle_polls: int = attrs.field(default=12, validator=_positive)
    orphan_after_s: float = attrs.field(default=600.0, validator=_positive)
    call_timeout: float | None = attrs.field(default=None, validator=_positive)
    timeout_grace: float = attrs.field(default=60.0, validator=_positive)
    drain_s: float = attrs.field(default=20.0, validator=_positive)
    coordinator_wait: float | None = attrs.field(default=None, validator=_positive)
    max_queued_jobs: int | None = attrs.field(default=None, validator=_positive)
    target_wall_s: float = attrs.field(default=600.0, validator=_positive)
    unmeasured_workers: int = attrs.field(default=1, validator=_positive)
    recheck_fraction: float = attrs.field(default=0.0, validator=_fraction)
    start_deadline_s: float | None = attrs.field(default=120.0, validator=_positive)
    start_deadline_failures: int = attrs.field(default=3, validator=_positive)
    max_warm_sessions: int = attrs.field(default=4, validator=_positive)

    def to_json(self) -> dict:
        record = attrs.asdict(self, recurse=False)
        record["lease"] = self.lease.to_json()
        return record

    @classmethod
    def from_json(cls, record: dict) -> "PoolSettings":
        """The settings `to_json` wrote; a missing field takes its default.

        Raises
        ------
        ValueError
            An unknown field.
        """
        known = {f.name for f in attrs.fields(cls)}
        unknown = set(record) - known
        if unknown:
            raise ValueError(f"unknown pool settings: {sorted(unknown)}")
        fields = dict(record)
        if "lease" in fields:
            fields["lease"] = LeaseSettings.from_json(fields["lease"])
        return cls(**fields)
