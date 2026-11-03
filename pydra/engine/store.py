"""Result stores: where jobs run and where their results are kept."""

import asyncio
import errno
import hashlib
import json
import os
import shutil
import sys
import time
import typing as ty
from pathlib import Path
import attrs
from filelock import SoftFileLock, Timeout
from pydra.engine.result import Result, load_result, save
from pydra.utils.general import ensure_list

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job

Context = ty.Mapping[str, ty.Any]
"""The execution context that produced a result, recorded beside it."""

REMOVE_ATTEMPTS = 5
"""How many times ``remove_from_cache`` tries to remove a path held open by another process."""

REMOVE_BACKOFF = 0.1
"""Seconds ``remove_from_cache`` waits after its first failed attempt, doubled after each further one."""

_BUSY_ERRNOS = frozenset({errno.EBUSY, errno.ENOTEMPTY})
_BUSY_WINERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION


class CacheBusyError(OSError):
    """A path in a cache could not be removed because another process holds a file in it open.

    ``filename`` is the path being removed and ``filename2`` the file that was held open.
    """


def is_busy(error: OSError) -> bool:
    """Whether ``error`` reports a file held open by another process."""
    return (
        getattr(error, "winerror", None) in _BUSY_WINERRORS
        or error.errno in _BUSY_ERRNOS
    )


def remove_from_cache(path: Path) -> None:
    """Remove the file or directory tree at ``path``.

    A removal that fails because another process holds a file open is tried
    ``REMOVE_ATTEMPTS`` times with exponential backoff, then raises ``CacheBusyError``.
    Any other error is raised as it is.
    """
    path = Path(path)
    failed: list[str] = []

    def record(func, failed_path, error):
        failed.append(failed_path)
        raise error

    if sys.version_info >= (3, 12):
        handler = {"onexc": record}
    else:
        handler = {"onerror": lambda f, p, exc_info: record(f, p, exc_info[1])}
    delay = REMOVE_BACKOFF
    for attempt in range(1, REMOVE_ATTEMPTS + 1):
        failed.clear()
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, **handler)
            else:
                path.unlink()
            return
        except OSError as error:
            if not is_busy(error):
                raise
            if attempt == REMOVE_ATTEMPTS:
                held = failed[0] if failed else path
                raise CacheBusyError(
                    error.errno,
                    f"could not remove {path} after {REMOVE_ATTEMPTS} attempts because "
                    f"{held} is held open by another process (on Windows typically a "
                    "virus scanner, a search indexer or a process whose working directory "
                    "is inside it; on NFS a process on any client)",
                    str(path),
                    None,
                    str(held),
                ) from error
        time.sleep(delay)
        delay *= 2


@ty.runtime_checkable
class ReusePolicy(ty.Protocol):
    """Decides whether a result recorded in one execution context is reused in another."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        """Whether the result of ``task`` recorded in ``stored`` is reused in ``current``."""


class ExactReuse:
    """Reuses only results recorded in an equal context."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        return dict(stored) == dict(current)


class AnyReuse:
    """Reuses results recorded in any context."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        return True


REUSE_POLICIES = {"exact": ExactReuse, "any": AnyReuse}


def reuse_policy(reuse: "str | ReusePolicy") -> ReusePolicy:
    """The policy named by ``reuse``, or ``reuse`` itself if it is a policy."""
    if isinstance(reuse, ReusePolicy):
        return reuse
    if isinstance(reuse, str) and reuse in REUSE_POLICIES:
        return REUSE_POLICIES[reuse]()
    raise ValueError(
        f"reuse must be one of {tuple(REUSE_POLICIES)} or a ReusePolicy, not {reuse!r}"
    )


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

    def lookup(
        self, checksum: str, context: Context, task: "Task | None"
    ) -> Result | None:
        """Return the stored result for ``checksum``, or None if there is none.

        The slot of ``context`` is searched in every location first. Otherwise a
        result recorded in another context is returned if the store's reuse policy
        accepts its context for ``task``, marked with ``Result.cross_context`` and
        carrying the context it was recorded in.
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
    reuse : str | ReusePolicy, optional
        Which stored results ``lookup`` returns given the current context: ``"exact"``
        (default) only those recorded in an equal context, ``"any"`` all of them, a
        ``ReusePolicy`` those whose context it accepts. Those from a different context
        are marked with ``Result.cross_context``.
    """

    def __init__(
        self,
        cache_root: os.PathLike,
        readonly_caches: list[os.PathLike] | None = None,
        reuse: "str | ReusePolicy" = "exact",
    ):
        self.cache_root = Path(cache_root)
        self.readonly_caches = [Path(p) for p in ensure_list(readonly_caches)]
        self.reuse = reuse_policy(reuse)

    def cache_dir(self, checksum: str, context: Context) -> Path:
        return self.cache_root / checksum / context_key(context)

    def lookup(
        self, checksum: str, context: Context, task: "Task | None"
    ) -> Result | None:
        locations = [self.cache_root] + self.readonly_caches
        slot = f"{checksum}/{context_key(context)}"
        for location in locations:
            result = load_result(slot, [location])
            if result is not None:
                return result
        if type(self.reuse) is ExactReuse:
            return None
        for location in locations:
            directory = location / checksum
            if not directory.is_dir():
                continue
            for other in sorted(p.name for p in directory.iterdir() if p.is_dir()):
                result = load_result(f"{checksum}/{other}", [location])
                if result is not None and self.reuse.accept(
                    result.context, context, task
                ):
                    result.cross_context = True
                    return result
        return None

    def slots(self, checksum: str | None = None) -> ty.Iterator[Path]:
        """The directories of the jobs run under ``cache_root``, of one job if its
        ``checksum`` is given. Each holds the job's ``_result.pklz``, ``_error.pklz``
        and outputs; its parent is named by the job's checksum."""
        checksums = (
            [checksum]
            if checksum is not None
            else sorted(p.name for p in self.cache_root.iterdir() if p.is_dir())
        )
        for name in checksums:
            directory = self.cache_root / name
            if directory.is_dir():
                yield from sorted(p for p in directory.iterdir() if p.is_dir())

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
