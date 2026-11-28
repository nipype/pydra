"""Result stores: where jobs run and where their results are kept."""

import errno
import hashlib
import json
import logging
import os
import shutil
import sys
import typing as ty
import warnings
from pathlib import Path
from pydra.engine.context import RECORD
from pydra.engine.lease import DirectoryLease, Lease, LeaseLost, LeaseSettings
from pydra.utils.storage import CLOCK_PREFIX, PROBE_PREFIX
from pydra.engine.result import Result, load_result, save
from pydra.utils.general import CACHE_FILE_MODE, ensure_list, atomic_write
from pydra.utils.retry import Backoff

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job

Context = ty.Mapping[str, ty.Any]
"""The execution context that produced a result, recorded beside it (see
`pydra.engine.context.job_context`)."""

CONTEXT_FILE = "_context.json"
"""The file in a slot holding the context its result was recorded in, read by
``PickleStore.lookup`` before the result is loaded."""

LEASE_SUFFIX = ".lock"
"""The suffix of a job's lease in ``PickleStore.cache_root``."""

SUCCESS_DIR = ".pydra-succeeded"
"""The directory in ``PickleStore.cache_root`` holding the jobs' success records: one
empty file per slot, named ``<checksum>-<context key>``, whose presence says that the
slot holds a result that did not error (see `PickleStore.success_record`)."""

logger = logging.getLogger("pydra")

_record_exists = os.path.exists

REMOVE_RETRY = Backoff(first=0.1, cap=0.8, attempts=5)
"""How ``remove_from_cache`` retries a removal blocked by held or pending files: five tries
over 1.5 s, the time an antivirus scan or an NFS silly-rename usually holds a file."""

_BUSY_ERRNOS = frozenset({errno.EBUSY})
_BUSY_WINERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
_NOT_EMPTY_ERRNOS = frozenset({errno.ENOTEMPTY})
_NOT_EMPTY_WINERRORS = frozenset({145})  # ERROR_DIR_NOT_EMPTY
_PENDING_WINERRORS = frozenset({303})  # ERROR_DELETE_PENDING
_SILLY_RENAME_PREFIX = ".nfs"


class CacheBusyError(OSError):
    """A path in a cache could not be removed because files in it stayed held open or
    pending deletion.

    ``filename`` is the path being removed, ``filename2`` the first such file and
    ``held`` all of them.
    """


class CacheWriterError(OSError):
    """Entries appeared in a cache directory while it was being removed.

    ``filename`` is the directory being removed, ``filename2`` the first such entry and
    ``leftovers`` all of them.
    """


def _cache_error(cls, error, message, path, files, attribute):
    winerror = getattr(error, "winerror", None)
    raised = cls(error.errno, message, str(path), winerror, str(files[0]))
    if winerror is not None and getattr(raised, "winerror", None) is None:
        raised.winerror = winerror
    setattr(raised, attribute, [str(f) for f in files])
    return raised


def is_busy(error: OSError) -> bool:
    """Whether ``error`` reports a file held open by another process."""
    return (
        getattr(error, "winerror", None) in _BUSY_WINERRORS
        or error.errno in _BUSY_ERRNOS
    )


def _is_not_empty(error: OSError) -> bool:
    return (
        getattr(error, "winerror", None) in _NOT_EMPTY_WINERRORS
        or error.errno in _NOT_EMPTY_ERRNOS
    )


def _is_pending(entry: Path) -> bool:
    """Whether ``entry`` is an NFS silly-rename, a file Windows has marked for deletion,
    or already gone; any other error reading it is raised."""
    if entry.name.startswith(_SILLY_RENAME_PREFIX):
        return True
    try:
        os.stat(entry)
    except FileNotFoundError:
        return True
    except OSError as error:
        if getattr(error, "winerror", None) in _PENDING_WINERRORS:
            return True
        raise
    return False


def _leftovers(directory: Path) -> list[Path]:
    """The entries other than directories beneath ``directory``."""
    found = []
    for root, dirs, files in os.walk(directory):
        found.extend(
            Path(root, d) for d in dirs if os.path.islink(os.path.join(root, d))
        )
        found.extend(Path(root, f) for f in files)
    return sorted(found)


def _remove_directories(path: Path) -> None:
    """Remove the empty directory tree at ``path``, deepest first."""
    for root, dirs, _ in os.walk(path, topdown=False):
        for d in dirs:
            if not os.path.islink(os.path.join(root, d)):
                os.rmdir(os.path.join(root, d))
    os.rmdir(path)


def remove_from_cache(path: Path, retry: Backoff | None = None) -> None:
    """Remove the file or directory tree at ``path``; a path already gone counts as removed.

    A removal blocked by files held open (sharing violation, ``EBUSY``) or pending deletion
    (NFS ``.nfs*`` silly-renames, Windows delete-pending files) is tried as ``retry`` says
    (default `REMOVE_RETRY`), then raises ``CacheBusyError``; once only pending files are left,
    later attempts remove only directories. A directory not empty for any other reason raises
    ``CacheWriterError`` at once. Any other error is raised as it is.
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
    directories_only = False
    tries = (retry or REMOVE_RETRY).tries()
    for attempt in tries:
        failed.clear()
        try:
            if directories_only:
                _remove_directories(path)
            elif path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, **handler)
            else:
                path.unlink()
            return
        except FileNotFoundError:
            if not os.path.lexists(path):
                return
            if tries.last:
                raise
            continue
        except OSError as error:
            blocked = Path(failed[0] if failed else error.filename or path)
            if _is_not_empty(error):
                held = _leftovers(blocked if blocked.is_dir() else path)
                written = [entry for entry in held if not _is_pending(entry)]
                if written:
                    raise _cache_error(
                        CacheWriterError,
                        error,
                        f"entries appeared in {path} while it was being removed: "
                        + ", ".join(map(str, written)),
                        path,
                        written,
                        "leftovers",
                    ) from error
                held = held or [blocked]
            elif is_busy(error):
                held = [blocked]
            else:
                raise
            directories_only = all(_is_pending(entry) for entry in held)
            if tries.last:
                raise _cache_error(
                    CacheBusyError,
                    error,
                    f"could not remove {path} after {attempt + 1} attempts: "
                    + ", ".join(map(str, held))
                    + " held open or pending deletion",
                    path,
                    held,
                    "held",
                ) from error


@ty.runtime_checkable
class ReusePolicy(ty.Protocol):
    """Decides whether a result recorded in one execution context is reused in another."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        """Whether the result of ``task`` recorded in ``stored`` is reused in ``current``."""


def _known(context: Context) -> bool:
    """Whether every part of a context is known: its layout is this pydra's and the
    identities of the packages its task reaches are recorded"""
    packages = context.get("packages")
    return (
        "unknown" not in context
        and context.get("record") == RECORD
        and isinstance(packages, ty.Mapping)
        and None not in packages.values()
        and all(context.get(k) is not None for k in ConsistentReuse.EQUAL)
        and isinstance(context.get("python"), str)
    )


def _minor(version: str) -> list[str]:
    return version.split(".")[:2]


class ExactReuse:
    """Reuses only results recorded in an equal context, every part of which is known.
    A result recorded in an equal context shares the current one's slot, so it is
    found without consulting the policy."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        return _known(stored) and dict(stored) == dict(current)


class ConsistentReuse:
    """Reuses results recorded under the same Python major.minor and implementation,
    the same platform and environment, and the same versions of the distributions the
    task reaches (``packages``), all of them known. The full Python version and the
    other installed distributions (``installed``) may differ."""

    EQUAL = ("record", "implementation", "platform", "environment", "packages")

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        return (
            _known(stored)
            and _known(current)
            and all(stored[k] == current[k] for k in self.EQUAL)
            and _minor(stored["python"]) == _minor(current["python"])
        )


class AnyReuse:
    """Reuses results recorded in any context."""

    def accept(self, stored: Context, current: Context, task: "Task | None") -> bool:
        return True


REUSE_POLICIES = {
    "exact": ExactReuse,
    "consistent": ConsistentReuse,
    "any": AnyReuse,
}


class DeclaredVersionWarning(UserWarning):
    """A stored result of a task with a declared version was computed from other code
    than the task's under the same version (see `check_declared_code`)."""


def _moved(stored: Context, code: ty.Mapping[str, str]) -> dict[str, str]:
    """The declared versions whose code digest in ``stored`` differs from ``code``'s,
    with the stored digest"""
    recorded = stored.get("code")
    if not isinstance(recorded, ty.Mapping):
        return {}
    return {k: v for k, v in recorded.items() if k in code and code[k] != v}


def check_declared_code(
    store: "ResultStore", checksum: str, context: Context, task: "Task | None"
) -> list[tuple[Path, Context]] | None:
    """Warn if a result stored for ``checksum`` was computed from other code than
    ``context`` records for a declared version it reaches (a ``code`` digest, see
    `pydra.engine.context.job_context`, differs), saying whether ``lookup`` would
    reuse it. Nothing is checked once a result is stored in ``context`` itself.

    Returns the stored contexts it read, to be passed to ``lookup`` as ``stored``, or
    None if it read none."""
    code = context.get("code")
    if not code or (store.cache_dir(checksum, context) / CONTEXT_FILE).exists():
        return None
    stored = list(store.stored_contexts(checksum))
    moved: dict[str, set[str]] = {}
    for _, other in stored:
        for identity, digest in _moved(other, code).items():
            moved.setdefault(identity, set()).add(digest)
    if not moved:
        return stored
    reuse = getattr(store, "reuse", None)
    first = next(
        (
            other
            for _, other in stored
            if reuse is not None
            and type(reuse) is not ExactReuse
            and reuse.accept(other, context, task)
        ),
        None,
    )
    reused = first is not None and bool(_moved(first, code))
    name = type(task).__name__ if task is not None else "A task"
    changes = "; ".join(
        f"{qualname} declares version={version!r}, but its code has changed (code "
        f"digest {', '.join(sorted(moved[identity]))} there, {code[identity]} now)"
        for identity in sorted(moved)
        for qualname, _, version in [identity.partition("==")]
    )
    warnings.warn(
        f"{name} has a stored result computed from other code under the same "
        f"declared versions: {changes}. That result is "
        f"{'reused' if reused else 'not reused'}: bump version= or remove it",
        DeclaredVersionWarning,
        stacklevel=2,
    )
    return stored


def reuse_policy(reuse: "str | ReusePolicy") -> ReusePolicy:
    """The policy named by ``reuse``, or ``reuse`` itself if it is a policy."""
    if isinstance(reuse, ReusePolicy):
        return reuse
    if isinstance(reuse, str) and reuse in REUSE_POLICIES:
        return REUSE_POLICIES[reuse]()
    raise ValueError(
        f"reuse must be one of {tuple(REUSE_POLICIES)} or a ReusePolicy, not {reuse!r}"
    )


def _is_store_internal(name: str) -> bool:
    """Whether ``name`` in a cache root is a lease, a success record or a probe, not a
    job's directory."""
    return (
        name.endswith(LEASE_SUFFIX)
        or name == SUCCESS_DIR
        or name.startswith((PROBE_PREFIX, CLOCK_PREFIX))
    )


def lease_still_held(lease: Lease) -> bool:
    """Whether ``lease`` has not been taken over: asked of the storage if the lease
    can `~pydra.engine.lease.DirectoryLease.prove` it, otherwise read from its
    ``lost`` attribute if it has one."""
    prove = getattr(lease, "prove", None)
    if callable(prove):
        return bool(prove())
    return not getattr(lease, "lost", None)


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
        self,
        checksum: str,
        context: Context,
        task: "Task | None",
        stored: "list[tuple[Path, Context]] | None" = None,
    ) -> Result | None:
        """Return the stored result for ``checksum``, or None if there is none.

        The slot of ``context`` is searched in every location first. Otherwise a
        result recorded in another context is returned if the store's reuse policy
        accepts its context for ``task``, marked with ``Result.cross_context`` and
        carrying the context it was recorded in and, as ``Result.reused_in``,
        ``context``. ``stored``, the `stored_contexts` of ``checksum`` already read
        (by `check_declared_code`), is searched instead of reading them again.
        """

    def stored_contexts(self, checksum: str) -> ty.Iterator[tuple[Path, Context]]:
        """The slot and recorded context of each result stored for ``checksum``, in
        every location, in the order ``lookup`` considers them."""

    def lockfile(self, checksum: str, context: Context) -> Path:
        """The path of the lease guarding the job identified by ``checksum`` under
        ``context``. It exists while the lease is held."""

    def acquire(self, checksum: str, context: Context) -> Lease:
        """Return the lease guarding the job identified by ``checksum`` under
        ``context``, not yet entered. A store whose storage cannot carry its lease
        raises `~pydra.utils.storage.StorageUnsupported` when the lease is entered."""

    def commit(self, lease: Lease, result: Result, job: "Job | None" = None) -> None:
        """Store ``result``, and ``job`` if given, as those of the job held under ``lease``.

        Both are written together, so a reader never sees a result without its job.
        A store raises `~pydra.engine.lease.LeaseLost`, writing nothing, if ``lease``
        was taken over by another process while it was held.
        """

    def read_succeeded(
        self, checksum: str, context: Context, task: "Task | None"
    ) -> Result | None:
        """The result of ``checksum`` under ``context``, read without the lease from a success record;
        None when there is none (by default a store keeps none), and None for a result that is
        errored or from another context, since the runner settles any result it returns as stored.
        """
        return None


class PickleStore(ResultStore):
    """Results pickled into ``<cache_root>/<checksum>/<context key>/_result.pklz``.

    Looking up a result unpickles its files, so ``cache_root`` and every entry of
    ``readonly_caches`` trust everyone who can write to them with code execution as the
    reading user. Only point the store at caches whose writers you trust. Files are
    written atomically with mode ``0o644`` adjusted by the umask.

    A slot whose job succeeded carries a success record in `SUCCESS_DIR` (see
    `success_record`), so the submitter reads its result without the lease and does
    not dispatch the job.

    Parameters
    ----------
    cache_root : os.PathLike
        The directory jobs run in and results are written to.
    readonly_caches : list[os.PathLike], optional
        Further directories searched for results, in order, after ``cache_root``.
    reuse : str | ReusePolicy, optional
        Which stored results ``lookup`` returns given the current context (see
        `REUSE_POLICIES`): ``"consistent"`` (default) those recorded under the same
        Python minor version, platform, environment and versions of the distributions
        the task reaches, ``"exact"`` only those recorded in an equal context,
        ``"any"`` all of them, a ``ReusePolicy`` those whose context it accepts. Those
        from a different context are marked with ``Result.cross_context`` and record
        the current one as ``Result.reused_in``.
    lease_settings : LeaseSettings, optional
        How the `~pydra.engine.lease.DirectoryLease` guarding each job beats, ages and
        waits: when a lease left by a dead process is taken over (``stale_after``) and
        how long a wait for a live one lasts (``timeout``). The defaults suit a cache
        on NFS; see `~pydra.engine.lease.LeaseSettings`.
    """

    def __init__(
        self,
        cache_root: os.PathLike,
        readonly_caches: list[os.PathLike] | None = None,
        reuse: "str | ReusePolicy" = "consistent",
        lease_settings: LeaseSettings | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.readonly_caches = [Path(p) for p in ensure_list(readonly_caches)]
        self.reuse = reuse_policy(reuse)
        self.lease_settings = (
            lease_settings if lease_settings is not None else LeaseSettings()
        )

    def cache_dir(self, checksum: str, context: Context) -> Path:
        return self.cache_root / checksum / context_key(context)

    def lookup(
        self,
        checksum: str,
        context: Context,
        task: "Task | None",
        stored: "list[tuple[Path, Context]] | None" = None,
    ) -> Result | None:
        locations = [self.cache_root] + self.readonly_caches
        slot = f"{checksum}/{context_key(context)}"
        for location in locations:
            result = load_result(slot, [location])
            if result is not None:
                return result
        if type(self.reuse) is ExactReuse:
            return None
        for slot, other in self.stored_contexts(checksum) if stored is None else stored:
            if not self.reuse.accept(other, context, task):
                continue
            result = load_result(f"{checksum}/{slot.name}", [slot.parent.parent])
            if result is not None:
                result.cross_context = True
                result.reused_in = dict(context)
                return result
        return None

    def stored_contexts(self, checksum: str) -> ty.Iterator[tuple[Path, Context]]:
        locations = [self.cache_root] + self.readonly_caches
        for location in locations:
            directory = location / checksum
            if not directory.is_dir():
                continue
            for slot in sorted(p for p in directory.iterdir() if p.is_dir()):
                try:
                    yield slot, json.loads((slot / CONTEXT_FILE).read_text())
                except (OSError, ValueError):
                    continue  # still being written, or not written by this pydra

    def slots(self, checksum: str | None = None) -> ty.Iterator[Path]:
        """The directories of the jobs run under ``cache_root``, of one job if its
        ``checksum`` is given. Each holds the job's ``_result.pklz``, ``_error.pklz``
        and outputs; its parent is named by the job's checksum."""
        checksums = (
            [checksum]
            if checksum is not None
            else sorted(
                p.name
                for p in self.cache_root.iterdir()
                if p.is_dir() and not _is_store_internal(p.name)
            )
        )
        for name in checksums:
            directory = self.cache_root / name
            if directory.is_dir():
                yield from sorted(p for p in directory.iterdir() if p.is_dir())

    def lockfile(self, checksum: str, context: Context) -> Path:
        return self.cache_root / f"{checksum}-{context_key(context)}{LEASE_SUFFIX}"

    def success_record(self, checksum: str, context: Context) -> Path:
        """The path of the record saying that the slot of ``checksum`` under
        ``context`` in ``cache_root`` holds a result that did not error.

        `commit` writes it as its last step, only for such a result and only while
        the lease is still held; `withdraw_success` removes it under the lease before
        the slot is rewritten. A slot without one (errored, running, in a read-only
        cache, written before records existed, or whose record could not be
        written) is read under its lease."""
        return self.cache_root / SUCCESS_DIR / f"{checksum}-{context_key(context)}"

    def read_succeeded(
        self, checksum: str, context: Context, task: "Task | None"
    ) -> Result | None:
        """The result of ``checksum`` under ``context``, read without the lease, if
        its slot in ``cache_root`` has a success record (one existence check); None
        if it has none, or if `lookup` returns no result, an errored one or one from
        another context, in which case the job is to be run under its lease.

        The result is read through `lookup`, the call a job makes on a hit under its
        lease, so a store that overrides it to refuse, verify or record reuse sees
        this read too."""
        if not _record_exists(self.success_record(checksum, context)):
            return None
        try:
            result = self.lookup(checksum, context, task)
        except Exception as e:  # removed, rewritten or unreadable: run it leased
            logger.debug("Reading the result of %s without its lease: %s", checksum, e)
            return None
        if result is None or result.errored or result.cross_context:
            return None
        return result

    def withdraw_success(self, lease: DirectoryLease) -> None:
        """Remove the success record of the job held under ``lease``, before its slot
        is rewritten; called with the lease held."""
        self._success_record_of(lease).unlink(missing_ok=True)

    @staticmethod
    def _success_record_of(lease: DirectoryLease) -> Path:
        name = lease.lockfile.name[: -len(LEASE_SUFFIX)]
        return lease.lockfile.parent / SUCCESS_DIR / name

    def acquire(self, checksum: str, context: Context) -> DirectoryLease:
        return DirectoryLease(
            checksum,
            self.cache_dir(checksum, context),
            self.lockfile(checksum, context),
            settings=self.lease_settings,
        )

    def commit(
        self, lease: DirectoryLease, result: Result, job: "Job | None" = None
    ) -> None:
        if not lease_still_held(lease):
            raise LeaseLost(
                f"The lease {lease.lockfile} was taken over by another process while "
                f"held, so this process does not write its result to {lease.cache_dir}",
                lease.lockfile,
            )
        lease.cache_dir.mkdir(parents=True, exist_ok=True)
        with atomic_write(lease.cache_dir / CONTEXT_FILE, "w") as f:
            json.dump(result.context, f, sort_keys=True)
        save(lease.cache_dir, result=result, job=job)
        if result.errored:
            return
        if not lease_still_held(lease):
            raise LeaseLost(
                f"The lease {lease.lockfile} was taken over by another process while "
                f"its result was written to {lease.cache_dir}, so this process does not "
                "record it as a success",
                lease.lockfile,
            )
        record = self._success_record_of(lease)
        try:
            record.parent.mkdir(exist_ok=True)
            os.close(os.open(record, os.O_WRONLY | os.O_CREAT, CACHE_FILE_MODE))
        except OSError as e:
            logger.warning(
                "Could not write the success record %s (%s); the job's result is "
                "stored and is read under its lease when next reused",
                record,
                e,
            )
