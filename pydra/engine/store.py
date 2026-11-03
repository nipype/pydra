"""Result stores: where jobs run and where their results are kept."""

import asyncio
import hashlib
import json
import os
import typing as ty
from pathlib import Path
import attrs
from filelock import SoftFileLock, Timeout
from pydra.engine.result import Result, load_result, save
from pydra.utils.general import ensure_list

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

Context = ty.Mapping[str, ty.Any]
"""The execution context that produced a result, recorded beside it."""

REUSE_POLICIES = ("exact", "any")


@attrs.define
class Lease:
    """The right to run the job identified by ``checksum`` in ``cache_dir``.

    The lease is held while it is entered, either with ``with`` or ``async with``,
    and released on exit.
    """

    checksum: str
    cache_dir: Path
    lockfile: Path
    _lock: SoftFileLock | None = attrs.field(default=None, init=False, repr=False)

    def __enter__(self) -> "Lease":
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        self._lock = SoftFileLock(self.lockfile)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()

    async def __aenter__(self) -> "Lease":
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        lock = SoftFileLock(self.lockfile)
        timeout = 0.1
        while True:
            try:
                lock.acquire(timeout=0)
                break
            except Timeout:
                await asyncio.sleep(timeout)
                if timeout <= 2:
                    timeout *= 2
        self._lock = lock
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


def context_key(context: Context) -> str:
    """The name of the slot holding results recorded in ``context``."""
    return hashlib.sha256(
        json.dumps(dict(context), sort_keys=True).encode()
    ).hexdigest()[:16]


@ty.runtime_checkable
class ResultStore(ty.Protocol):
    """Interface between jobs and the storage of their results.

    A job is identified by its checksum and the context it runs in. Each pair has
    its own slot, so results recorded in different contexts never overwrite each other.
    """

    def cache_dir(self, checksum: str, context: Context) -> Path:
        """The directory the job identified by ``checksum`` runs in under ``context``."""

    def lookup(self, checksum: str, context: Context) -> Result | None:
        """Return the stored result for ``checksum``, or None if there is none.

        The slot of ``context`` is searched in every location first. If the store's
        reuse policy allows it, a result recorded in another context is returned
        otherwise, marked with ``Result.cross_context``.
        """

    def lockfile(self, checksum: str, context: Context) -> Path:
        """The lock file guarding the job identified by ``checksum`` under ``context``."""

    def acquire(self, checksum: str, context: Context) -> Lease:
        """Return the lease guarding the job identified by ``checksum`` under ``context``."""

    def commit(self, lease: Lease, result: Result, job: "Job | None" = None) -> None:
        """Store ``result``, and ``job`` if given, as those of the job held under ``lease``.

        Both are written together, so a reader never sees a result without its job.
        """


class PickleStore:
    """Results pickled into ``<cache_root>/<checksum>/<context key>/_result.pklz``.

    Parameters
    ----------
    cache_root : os.PathLike
        The directory jobs run in and results are written to.
    readonly_caches : list[os.PathLike], optional
        Further directories searched for results, in order, after ``cache_root``.
    reuse : str, optional
        Which stored results ``lookup`` returns given the current context: ``"exact"``
        (default) only those recorded in an equal context, ``"any"`` all of them,
        those from a different context marked with ``Result.cross_context``.
    """

    def __init__(
        self,
        cache_root: os.PathLike,
        readonly_caches: list[os.PathLike] | None = None,
        reuse: str = "exact",
    ):
        if reuse not in REUSE_POLICIES:
            raise ValueError(f"reuse must be one of {REUSE_POLICIES}, not {reuse!r}")
        self.cache_root = Path(cache_root)
        self.readonly_caches = [Path(p) for p in ensure_list(readonly_caches)]
        self.reuse = reuse

    def cache_dir(self, checksum: str, context: Context) -> Path:
        return self.cache_root / checksum / context_key(context)

    def lookup(self, checksum: str, context: Context) -> Result | None:
        locations = [self.cache_root] + self.readonly_caches
        slot = f"{checksum}/{context_key(context)}"
        for location in locations:
            result = load_result(slot, [location])
            if result is not None:
                return result
        if self.reuse == "any":
            for location in locations:
                directory = location / checksum
                if not directory.is_dir():
                    continue
                for other in sorted(p.name for p in directory.iterdir() if p.is_dir()):
                    result = load_result(f"{checksum}/{other}", [location])
                    if result is not None:
                        result.cross_context = True
                        return result
        return None

    def lockfile(self, checksum: str, context: Context) -> Path:
        return self.cache_root / f"{checksum}-{context_key(context)}.lock"

    def acquire(self, checksum: str, context: Context) -> Lease:
        return Lease(
            checksum,
            self.cache_dir(checksum, context),
            self.lockfile(checksum, context),
        )

    def commit(self, lease: Lease, result: Result, job: "Job | None" = None) -> None:
        save(lease.cache_dir, result=result, job=job)
