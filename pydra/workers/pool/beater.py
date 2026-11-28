"""The pool's use of the lease beater (RFC 0005 section 5.3).

A worker beats its worker and claim leases from a `pydra.engine.lease.LeaseBeater` subprocess
through a `BeaterClient`. The pool's beater runs `pool_expire` when a call's deadline passes and
kills the worker if it does not acknowledge the expiry within ``timeout_grace``.
"""

from __future__ import annotations

import logging
import typing as ty
from pathlib import Path

from pydra.engine.lease import BeaterClient, LeaseBeater, LeaseSettings

__all__ = ["EXPIRE", "pool_beater", "pool_expire", "expiry_payload"]

logger = logging.getLogger("pydra.pool")

EXPIRE = "pydra.workers.pool.beater:pool_expire"


def expiry_payload(
    root: Path, settings: LeaseSettings, queue: str, ticket: str, jobs: list[dict]
) -> dict:
    """What `pool_expire` needs to write a call's ``expired`` outcomes, as JSON."""
    return {
        "root": str(root),
        "lease": settings.to_json(),
        "queue": queue,
        "ticket": ticket,
        "jobs": jobs,
    }


def pool_expire(token_dir: Path, payload: dict) -> None:
    """Write an ``expired`` outcome under its attempt for each job of the call whose deadline passed;
    the first writer wins against a late ``done`` (RFC 0005 sections 5.1, 5.3). Runs in the beater.
    """
    from pydra.workers.pool.storage import PoolRoot

    settings = LeaseSettings.from_json(payload["lease"])
    with PoolRoot.open(payload["root"], settings, create=False, probe=False) as pool:
        for job in payload["jobs"]:
            pool.outcome(
                payload["queue"],
                payload["ticket"],
                (job["checksum"], job["fingerprint"]),
                job["attempt"],
                {"outcome": "expired", "reason": "timeout"},
            )
    logger.warning(
        "call of %d job(s) under %s passed its deadline: expired",
        len(payload["jobs"]),
        token_dir,
    )


def pool_beater(
    timeout_grace: float,
    *,
    on_lost: ty.Callable[[Path, str], None] | None = None,
    on_expired: ty.Callable[[Path], None] | None = None,
    on_dead: ty.Callable[[str], None] | None = None,
) -> BeaterClient:
    """Start the worker's beater: `pool_expire` on a deadline, the worker killed ``timeout_grace``
    seconds after an expiry it did not acknowledge."""
    return BeaterClient(
        LeaseBeater(on_expire=EXPIRE, kill_after=timeout_grace),
        on_lost=on_lost,
        on_expired=on_expired,
        on_dead=on_dead,
    )
