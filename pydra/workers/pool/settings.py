"""The settings every process of a pool reads (RFC 0005 sections 4.3, 5.3)."""

from __future__ import annotations

import attrs

from pydra.engine.lease import LeaseSettings

__all__ = ["PoolSettings"]


def _positive(_inst, attribute, value) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{attribute.name} must be positive, not {value!r}")


@attrs.frozen
class PoolSettings:
    """How a pool's processes poll, wait and bound calls. Each default and its derivation is
    in RFC 0005 section 5.3.

    Attributes
    ----------
    lease : LeaseSettings
        The pool's lease settings; a worker whose staleness fields differ refuses to start.
    poll_s : float
        Seconds between claim attempts while nothing is claimable.
    idle_polls : int
        Empty polls after which a worker exits.
    orphan_after_s : float
        Seconds without a live coordinator after which a substrate's worker exits.
    call_timeout : float, optional
        A call's bound for a job that declares no ``time``; None: no deadline.
    timeout_grace : float
        Seconds added to a call's deadline before its beater kills it.
    drain_s : float
        Seconds a worker given SIGTERM waits for its call before releasing its claims.
    coordinator_wait : float, optional
        Seconds a coordinator waits for another live one before ``PoolBusy``; None: no bound.
    max_queued_jobs : int, optional
        Tickets' jobs per queue not yet settled; None: 4 x ``quantum_jobs`` x ``max_workers``.
    target_wall_s : float
        Seconds a queue's waiting jobs should take with the workers scaling starts.
    unmeasured_workers : int
        Workers wanted per queue while no per-job time is measured.
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
