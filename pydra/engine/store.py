"""Result stores: where jobs run and where their results are kept."""

import asyncio
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


@ty.runtime_checkable
class ResultStore(ty.Protocol):
    """Interface between jobs and the storage of their results."""

    def cache_dir(self, checksum: str) -> Path:
        """The directory the job identified by ``checksum`` runs in."""

    def lookup(self, checksum: str, context: Context | None = None) -> Result | None:
        """Return the stored result for ``checksum``, or None if there is none."""

    def lockfile(self, checksum: str) -> Path:
        """The lock file guarding the job identified by ``checksum``."""

    def acquire(self, checksum: str) -> Lease:
        """Return the lease guarding the job identified by ``checksum``."""

    def commit(self, lease: Lease, result: Result, job: "Job | None" = None) -> None:
        """Store ``result``, and ``job`` if given, as those of the job held under ``lease``.

        Both are written together, so a reader never sees a result without its job.
        """


class PickleStore:
    """Results pickled into ``<cache_root>/<checksum>/_result.pklz``.

    Parameters
    ----------
    cache_root : os.PathLike
        The directory jobs run in and results are written to.
    readonly_caches : list[os.PathLike], optional
        Further directories searched for results, in order, after ``cache_root``.
    """

    def __init__(
        self,
        cache_root: os.PathLike,
        readonly_caches: list[os.PathLike] | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.readonly_caches = [Path(p) for p in ensure_list(readonly_caches)]

    def cache_dir(self, checksum: str) -> Path:
        return self.cache_root / checksum

    def lookup(self, checksum: str, context: Context | None = None) -> Result | None:
        return load_result(checksum, [self.cache_root] + self.readonly_caches)

    def lockfile(self, checksum: str) -> Path:
        return self.cache_dir(checksum).with_suffix(".lock")

    def acquire(self, checksum: str) -> Lease:
        return Lease(checksum, self.cache_dir(checksum), self.lockfile(checksum))

    def commit(self, lease: Lease, result: Result, job: "Job | None" = None) -> None:
        save(lease.cache_dir, result=result, job=job)
