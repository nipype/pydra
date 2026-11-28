"""Leases: the right to run a job, held across processes and hosts sharing a cache.

`Lease` is the interface a `pydra.engine.store.ResultStore` returns from ``acquire``.
`DirectoryLease` is the default, made for a cache on shared storage such as NFS.
"""

import asyncio
import errno
import getpass
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import typing as ty
import uuid
from pathlib import Path
import attrs
from pydra.utils.general import atomic_write
from pydra.utils.storage import require_lease_storage, storage_now

logger = logging.getLogger("pydra")

STALE_AFTER = 300.0
"""Default seconds, on the storage clock, after a lease's last heartbeat at which it is
stale and another process takes it over (see `LeaseSettings`)."""

RENEW_FRACTION = 0.1
"""Default heartbeat interval as a fraction of ``stale_after``."""

ATTRIBUTE_DELAY = 60.0
"""Default seconds a client may see a directory's mtime late: the NFS default
``acdirmax``."""

TIMEOUT = 600.0
"""Default seconds a wait for a lease lasts before `LeaseTimeout`."""

POLL_MIN = 0.05
"""Seconds before the second attempt to take a held lease."""

POLL_MAX = 2.0
"""Most seconds between two attempts to take a held lease."""

MISSED_BEATS = 1
"""Heartbeats a live holder may miss (a slow or hung write) without being taken over."""

RELEASE_PATIENCE = 5.0
"""Most seconds a release retries renaming the holder's directory aside while the
storage refuses it; at most ``MISSED_BEATS * renew_interval`` for given settings."""

HOLDER_FILE = "holder.json"
"""The holder record in a lease's token directory."""

_TOKEN = re.compile(r"^[0-9a-f]{32}$")
_ESTALE = getattr(errno, "ESTALE", None)
IS_WINDOWS = sys.platform == "win32"


def _delete_pending(err: OSError) -> bool:
    """Whether ``err`` may be Windows refusing a mkdir on, in or a listing of a
    directory pending deletion: one removed while another handle to it is open."""
    return IS_WINDOWS and isinstance(err, PermissionError)


class LeaseTimeout(TimeoutError):
    """A wait for a lease reached its bound while another process held the lease.

    Attributes
    ----------
    lockfile : Path
        The lease waited for.
    holder : dict | None
        The holder's record (``owner``, ``host``, ``pid``, ``started``), None if
        none could be read.
    age : float | None
        Seconds on the storage clock since the holder's last heartbeat.
    """

    def __init__(
        self,
        message: str,
        lockfile: Path,
        holder: dict | None,
        age: float | None,
    ):
        super().__init__(message)
        self.lockfile = lockfile
        self.holder = holder
        self.age = age


class LeaseUnavailable(LeaseTimeout):
    """A stale lease could not be taken over because the storage refused it (for
    example the lease directory belongs to another user and is not writable). The
    message names the lease, its holder and the refusal; ``error`` is the refusal."""

    def __init__(
        self,
        message: str,
        lockfile: Path,
        holder: dict | None,
        age: float | None,
        error: OSError,
    ):
        super().__init__(message, lockfile, holder, age)
        self.error = error


class LeaseRefused(PermissionError):
    """The storage kept refusing to make or list the lease directory for longer than
    `release_patience`: a permission problem, not a lease another process holds.
    ``error`` is the refusal."""

    def __init__(self, message: str, lockfile: Path, error: OSError):
        super().__init__(error.errno, message, str(lockfile))
        self.lockfile = lockfile
        self.error = error


class LeaseLost(RuntimeError):
    """A lease was taken over by another process while it was held, so what it
    guarded is not written by its former holder.

    Attributes
    ----------
    lockfile : Path
        The lease that was lost.
    """

    def __init__(self, message: str, lockfile: Path):
        super().__init__(message)
        self.lockfile = lockfile


@ty.runtime_checkable
class Lease(ty.Protocol):
    """The right to run one job, held while it is entered with ``with`` or
    ``async with`` and released on exit."""

    def __enter__(self) -> "Lease":
        """Take the lease, waiting for it if another process holds it."""

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release the lease."""

    async def __aenter__(self) -> "Lease":
        """Take the lease without blocking the event loop while waiting."""

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        """Release the lease."""

    def try_take(self) -> float | None:
        """Make one attempt to take the lease without waiting for it.

        Returns
        -------
        float or None
            None if the lease is held now (release it with ``__exit__``), otherwise the
            seconds to wait before the next call. The first call of a wait starts its
            bounds; a later call continues the same wait, and raises `LeaseTimeout`
            (leaving the lease untaken) once a bound has passed.
        """


@attrs.frozen
class LeaseSettings:
    """How a `DirectoryLease` beats, ages and waits (defaults derived in RFC 0003 section 5.3).

    A holder renews its lease every ``renew_fraction * stale_after`` seconds; another process
    takes it over once the holder's newest mark is older than ``stale_after`` on the storage
    clock. A waiter behind a holder that keeps renewing waits as long as the holder runs:
    ``timeout`` counts only time with no renewal seen and no takeover possible (a stale lease
    the storage refuses to let this process remove, `LeaseUnavailable`, or an undeletable lock
    file of an earlier pydra), and restarts whenever the newest mark advances. ``max_wait``, if
    set, caps the whole wait.

    Construction raises ``ValueError`` unless a live holder is never taken over,
    ``(1 + missed_beats) * renew_fraction * stale_after + attribute_delay < stale_after``, and a
    dead holder is taken over before a waiter gives up, ``timeout >= stale_after +
    attribute_delay``.

    Parameters
    ----------
    stale_after : float
        Seconds on the storage clock after the last heartbeat at which a lease is stale.
    timeout : float
        Seconds without a renewal of a lease that cannot be taken over after which a
        wait ends in `LeaseTimeout`.
    renew_fraction : float
        Heartbeat interval as a fraction of ``stale_after``.
    attribute_delay : float
        Seconds a client may see a directory's mtime late on this storage.
    max_wait : float | None
        Seconds after which any wait ends in `LeaseTimeout`, even behind a holder that
        keeps renewing; None (default) for no cap.
    poll_min, poll_max : float
        Seconds before the second attempt to take a held lease, and most seconds between two.
    missed_beats : int
        Heartbeats a live holder may miss (a slow or hung write) without being taken over.
    release_patience : float
        Most seconds a release retries a refused rename (see `release_patience`).
    """

    stale_after: float = STALE_AFTER
    timeout: float = TIMEOUT
    renew_fraction: float = RENEW_FRACTION
    attribute_delay: float = ATTRIBUTE_DELAY
    max_wait: float | None = None
    poll_min: float = attrs.field(default=POLL_MIN, eq=False)
    poll_max: float = attrs.field(default=POLL_MAX, eq=False)
    missed_beats: int = attrs.field(default=MISSED_BEATS, eq=False)
    release_patience: float = attrs.field(default=RELEASE_PATIENCE, eq=False)

    def __attrs_post_init__(self):
        if self.max_wait is not None and self.max_wait < 0:
            raise ValueError(f"max_wait must not be negative (got {self.max_wait})")
        if self.stale_after <= 0 or self.timeout < 0 or self.attribute_delay < 0:
            raise ValueError(
                "stale_after must be positive and timeout and attribute_delay not "
                f"negative (got {self.stale_after}, {self.timeout}, "
                f"{self.attribute_delay})"
            )
        if not 0 < self.renew_fraction < 1:
            raise ValueError(
                f"renew_fraction must be between 0 and 1 (got {self.renew_fraction})"
            )
        if not 0 < self.poll_min <= self.poll_max or self.missed_beats < 0:
            raise ValueError(
                f"need 0 < poll_min <= poll_max and missed_beats >= 0 (got {self.poll_min}, "
                f"{self.poll_max}, {self.missed_beats})"
            )
        shown = (
            1 + self.missed_beats
        ) * self.renew_interval + self.attribute_delay  # oldest mark of a live holder
        if shown >= self.stale_after:
            raise ValueError(
                f"a live holder could be taken over: (1 + {self.missed_beats}) * "
                f"{self.renew_interval:g} + {self.attribute_delay:g} = {shown:g} is not "
                f"less than stale_after {self.stale_after:g}"
            )
        if self.timeout < self.stale_after + self.attribute_delay:
            raise ValueError(
                f"timeout {self.timeout:g} would give up before a dead holder is taken "
                f"over: it must be at least stale_after + attribute_delay = "
                f"{self.stale_after + self.attribute_delay:g}"
            )

    @property
    def renew_interval(self) -> float:
        """Seconds between two heartbeats."""
        return self.renew_fraction * self.stale_after

    @property
    def visibility_timeout(self) -> float:
        """Seconds a reader waits for another client's write to show: ``attribute_delay``
        and one poll, so never less than the delay the lease assumes."""
        return self.attribute_delay + self.poll_max


def release_patience(settings: LeaseSettings) -> float:
    """Most seconds a refusal explained by another process's open handle is retried: a
    release renaming its directory aside, and on Windows a mkdir or listing of a lease
    directory pending deletion."""
    return min(
        settings.release_patience, settings.missed_beats * settings.renew_interval
    )


def _owner() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no user name in some containers
        return str(os.getuid()) if hasattr(os, "getuid") else "unknown"


def _describe(holder: dict | None, age: float | None) -> str:
    if not holder:
        who = "an unknown holder"
    else:
        who = f"pid {holder.get('pid', '?')} on host {holder.get('host', '?')}" + (
            f" (owner {holder['owner']})" if holder.get("owner") else ""
        )
    if age is not None:
        who += f", last renewed {age:.1f}s ago on the storage clock"
    return who


class _Heartbeat(threading.Thread):
    """Renews a lease by making ``beat-<n+1>`` and removing ``beat-<n>`` in the holder's
    token directory, which moves the directory's mtime on the storage. A beat that
    finds the token directory gone marks the lease lost."""

    def __init__(self, lease: "DirectoryLease", token_dir: Path):
        super().__init__(name=f"pydra-lease-{token_dir.parent.name}", daemon=True)
        self.lease = lease
        self.token_dir = token_dir
        self.stopped = threading.Event()

    def run(self):
        settings = self.lease.settings
        interval = settings.renew_interval
        beat = 0
        last_ok = due = time.monotonic()
        while True:
            due += interval
            if self.stopped.wait(max(0.0, due - time.monotonic())):
                return
            try:
                os.mkdir(self.token_dir / f"beat-{beat + 1}")
            except OSError as err:
                if err.errno == errno.ENOENT or err.errno == _ESTALE:
                    self.lease._lose("its holder directory was removed")
                    return
                if time.monotonic() - last_ok > settings.stale_after:
                    self.lease._lose(f"it could not be renewed for too long ({err})")
                    return
                logger.debug(
                    "Could not renew the lease %s (%s); trying again in %.1fs",
                    self.token_dir.parent,
                    err,
                    interval,
                )
                continue
            try:
                os.rmdir(self.token_dir / f"beat-{beat}")
            except OSError:
                pass
            beat += 1
            last_ok = time.monotonic()


@attrs.define
class DirectoryLease:
    """A lease taken by ``mkdir`` and kept alive by a heartbeat.

    The lease is the directory ``lockfile``. Of the processes making it, one succeeds
    and holds the lease; it records itself in ``<lockfile>/<token>/holder.json``
    (owner, host, pid and start time) under a random token. A heartbeat thread renews
    the lease every ``settings.renew_interval`` by making and removing an entry in the
    token directory. Another process takes the lease over once the newest mtime in it
    is older than ``settings.stale_after``, measured on the storage server's clock (the
    mtime of a file just created beside the lease), never this host's. Takeover renames
    the holder's token directory, so of several processes taking one lease over exactly
    one succeeds, and a holder that was taken over finds out at its next heartbeat or
    when it calls `prove`. A waiter waits while the holder keeps renewing; it gives up
    in `LeaseTimeout`, which names the holder, only after ``settings.timeout`` without
    a renewal in which the lease could not be taken over, or after
    ``settings.max_wait`` if that is set.

    A regular file at ``lockfile`` is a lock written by an earlier pydra
    (``filelock.SoftFileLock``). It is judged by its own mtime and removed once older
    than ``stale_after``, whether or not the process that wrote it still runs (see
    `LeaseSettings`).

    A holder taken over while it still ran (a process stopped for longer than
    ``stale_after``) must not write what the lease guards: `prove` tells it whether it
    still holds the lease, and `pydra.engine.store.PickleStore.commit` refuses with
    `LeaseLost` when it does not.

    Parameters
    ----------
    checksum : str
        The checksum of the job the lease is for.
    cache_dir : Path
        The directory the job runs in.
    lockfile : Path
        The lease directory.
    settings : LeaseSettings
        How the lease beats, ages and waits.
    clock : callable, optional
        Returns the current time on the storage; by default `storage_now` of the
        lease's parent directory.
    monotonic : callable, optional
        Returns seconds on a local monotonic clock, for the wait's bounds.
    sleep : callable, optional
        Sleeps between attempts of a synchronous wait.
    """

    checksum: str
    cache_dir: Path
    lockfile: Path
    settings: LeaseSettings = attrs.field(factory=LeaseSettings)
    clock: ty.Callable[[], float] | None = attrs.field(default=None, repr=False)
    monotonic: ty.Callable[[], float] = attrs.field(default=time.monotonic, repr=False)
    sleep: ty.Callable[[float], None] = attrs.field(default=time.sleep, repr=False)
    broke: list[dict] = attrs.field(factory=list, init=False, repr=False)
    lost: str | None = attrs.field(default=None, init=False, repr=False)
    _token: str | None = attrs.field(default=None, init=False, repr=False)
    _heartbeat: _Heartbeat | None = attrs.field(default=None, init=False, repr=False)
    _last_seen: tuple[dict | None, float | None] = attrs.field(
        default=(None, None), init=False, repr=False
    )
    _newest_seen: float | None = attrs.field(default=None, init=False, repr=False)
    _refused: OSError | None = attrs.field(default=None, init=False, repr=False)
    _pending: OSError | None = attrs.field(default=None, init=False, repr=False)
    _pending_since: float | None = attrs.field(default=None, init=False, repr=False)
    _unreadable_since: float | None = attrs.field(default=None, init=False, repr=False)
    _deadline: float = attrs.field(default=0.0, init=False, repr=False)
    _cap: float | None = attrs.field(default=None, init=False, repr=False)
    _delay: float = attrs.field(default=POLL_MIN, init=False, repr=False)
    _waiting: bool = attrs.field(default=False, init=False, repr=False)

    @property
    def held(self) -> bool:
        """Whether this lease is entered and has not been found taken over."""
        return self._token is not None and self.lost is None

    def holder(self) -> dict | None:
        """The record of the process holding the lease, None if none can be read."""
        try:
            return self._read_holder()[0]
        except PermissionError as err:
            if not _delete_pending(err):
                raise
            return None

    def prove(self) -> bool:
        """Whether this process still holds the lease, asked of the storage: a
        directory is made and removed in the holder's token directory, which fails
        once the lease was taken over. A lease found taken over is marked lost."""
        if self._token is None or self.lost is not None:
            return False
        proof = self.lockfile / self._token / f"proof-{uuid.uuid4().hex}"
        try:
            os.mkdir(proof)
        except OSError as err:
            if err.errno == errno.ENOENT or err.errno == _ESTALE:
                self._lose("its holder directory was removed")
                return False
            return True  # not shown lost; the heartbeat judges a lasting failure
        try:
            os.rmdir(proof)
        except OSError:
            pass
        return True

    # Taking

    def _now(self) -> float:
        if self.clock is not None:
            return self.clock()
        return storage_now(self.lockfile.parent)

    def _prepare(self) -> None:
        self.lockfile.parent.mkdir(parents=True, exist_ok=True)
        require_lease_storage(self.lockfile.parent)
        self.broke, self.lost, self._last_seen = [], None, (None, None)
        self._newest_seen, self._refused = None, None
        self._delay = self.settings.poll_min
        self._unreadable_since = None
        self._pending, self._pending_since = None, None
        now = self.monotonic()
        self._deadline = now + self.settings.timeout
        max_wait = self.settings.max_wait
        self._cap = now + max_wait if max_wait is not None else None

    def _try_take(self) -> bool:
        """Make the lease directory and record this process in it; whether it is held."""
        try:
            os.mkdir(self.lockfile)
        except FileExistsError:
            return False
        except FileNotFoundError:
            self.lockfile.parent.mkdir(parents=True, exist_ok=True)
            return False
        except PermissionError as err:
            if not _delete_pending(err):
                raise
            self._refused = self._pending = err
            return False
        token = uuid.uuid4().hex
        token_dir = self.lockfile / token
        try:
            os.mkdir(token_dir)
        except OSError as err:
            # the lease was removed as empty by a process taking it over or
            # releasing; APFS answers EINVAL instead of ENOENT when the rmdir races
            # this mkdir, Windows a PermissionError
            if _delete_pending(err):
                self._refused = self._pending = err
                return False
            if err.errno in (errno.ENOENT, errno.EINVAL, _ESTALE):
                return False
            raise
        try:
            others = [e for e in os.listdir(self.lockfile) if e != token]
        except FileNotFoundError:
            others = ["gone"]
        if others:
            # the directory was removed and made again by another taker between our
            # two mkdirs: whoever sees another entry withdraws
            shutil.rmtree(token_dir, ignore_errors=True)
            try:
                os.rmdir(self.lockfile)
            except OSError:
                pass
            return False
        record = {
            "owner": _owner(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": os.stat(token_dir).st_mtime,
            "token": token,
        }
        with atomic_write(token_dir / HOLDER_FILE, "w") as f:
            json.dump(record, f)
        self._token = token
        self._heartbeat = _Heartbeat(self, token_dir)
        self._heartbeat.start()
        return True

    # Judging and taking over

    def _read_holder(self) -> tuple[dict | None, list[str]]:
        """The holder's record and the entries of the lease directory."""
        try:
            entries = os.listdir(self.lockfile)
        except NotADirectoryError:
            return self._read_legacy(), []
        except FileNotFoundError:
            return None, []
        for entry in sorted(entries):
            name = entry[len("gone-") :] if entry.startswith("gone-") else entry
            if _TOKEN.match(name):
                try:
                    record = json.loads(
                        (self.lockfile / entry / HOLDER_FILE).read_text()
                    )
                except (OSError, ValueError):
                    continue
                if isinstance(record, dict):
                    return record, entries
        return None, entries

    def _read_legacy(self) -> dict | None:
        try:
            lines = self.lockfile.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            return None
        record = {}
        if lines and lines[0].strip().isdigit():
            record["pid"] = int(lines[0].strip())
        if len(lines) > 1 and lines[1].strip():
            record["host"] = lines[1].strip()
        return record or None

    def _newest_mtime(self, entries: list[str]) -> tuple[float | None, OSError | None]:
        """The newest mtime of the lease, its entries and theirs (the holder record
        and heartbeats), so a renewal shows even where a directory's own mtime is
        reported late, and the error of a mark that exists but could not be read.
        The mtime is None if the lease is gone.

        A mark that is gone is skipped. The children of an entry that cannot be
        listed are not marks; one that is listed but cannot be read is reported."""
        mtimes, unreadable = [], None
        paths = [self.lockfile]
        for entry in entries:
            paths.append(self.lockfile / entry)
            try:
                paths.extend(self.lockfile / entry / e for e in os.listdir(paths[-1]))
            except OSError:
                pass
        for path in paths:
            try:
                mtimes.append(os.lstat(path).st_mtime)
            except OSError as err:
                if err.errno not in (errno.ENOENT, errno.ENOTDIR, _ESTALE):
                    unreadable = err
        try:  # read again after listing, so a takeover renamed meanwhile shows
            mtimes.append(os.lstat(self.lockfile).st_mtime)
        except FileNotFoundError:
            return None, unreadable
        except OSError as err:
            unreadable = err
        return (max(mtimes) if mtimes else None), unreadable

    def _judge(self) -> bool:
        """Take over the lease if it is stale; whether it was removed."""
        try:
            st = os.lstat(self.lockfile)
        except FileNotFoundError:
            return True
        except PermissionError as err:
            return self._held_pending(err)
        if not os.path.isdir(self.lockfile) or os.path.islink(self.lockfile):
            return self._judge_legacy(st)
        try:
            holder, entries = self._read_holder()
        except PermissionError as err:
            return self._held_pending(err)
        if entries and all(e.startswith("gone-") for e in entries):
            return self._clear_remains(entries)
        now = self._now()  # before the marks: a stall after this only shrinks the age
        newest, unreadable = self._newest_mtime(entries)
        if newest is None:
            return unreadable is None
        self._saw(newest)
        age = now - newest
        self._last_seen = (holder, age)
        if not self._judgeable(unreadable, now):
            return False
        if age <= self.settings.stale_after:
            return False
        tokens = [e for e in entries if _TOKEN.match(e)]
        if tokens:
            claimed = tokens[0]
        elif entries:
            claimed = sorted(entries)[0]  # a takeover that died before it finished
        else:
            try:
                os.rmdir(self.lockfile)  # a taker that died between its two mkdirs
            except PermissionError as err:
                self._refused = err
                return False
            except OSError:
                return False
            logger.debug("Removed the empty stale lease %s", self.lockfile)
            return True
        gone = self.lockfile / f"gone-{uuid.uuid4().hex}"
        try:
            had_contents = bool(os.listdir(self.lockfile / claimed))
        except OSError:
            had_contents = False
        try:
            os.rename(self.lockfile / claimed, gone)
        except OSError as err:
            if err.errno in (errno.ENOENT, errno.EINVAL, _ESTALE):
                return False  # another process took it over first
            if err.errno in (errno.EACCES, errno.EPERM):
                self._refused = err
                return False
            raise
        marker = f"claimed-{claimed}"
        if not self._claim(gone, marker, had_contents):
            return False
        self._remove(gone, keep_last=marker)
        if tokens:
            self._broke(holder, age)
        else:
            logger.debug("Removed the remains of a takeover in %s", self.lockfile)
        return True

    def _judgeable(self, unreadable: OSError | None, now: float) -> bool:
        """Whether the lease's age can be judged by its readable marks.

        A mark that cannot be read may be the holder's newest. While one is, the lease
        is judged only once ``stale_after`` has passed on the storage clock since a
        mark was first seen unreadable: by then a fresh mark that stayed unreadable
        would be stale too, and the takeover's rename decides whether it is held."""
        if unreadable is None:
            self._unreadable_since = None
            return True
        if self._unreadable_since is None:
            self._unreadable_since = now
            logger.debug("A mark in %s cannot be read: %s", self.lockfile, unreadable)
        return now - self._unreadable_since >= self.settings.stale_after

    def _clear_remains(self, entries: list[str]) -> bool:
        """Remove a lease holding only ``gone-`` entries, whatever their age; whether it
        was removed. They are what a release or a takeover renamed aside and could not
        remove yet (on Windows, while another process had a file in them open), never
        a holder: a holder's directory is its bare token."""
        for entry in entries:
            shutil.rmtree(self.lockfile / entry, ignore_errors=True)
        try:
            os.rmdir(self.lockfile)
        except FileNotFoundError:
            return True
        except OSError:
            return False  # not emptied yet, or a new holder made its token meanwhile
        logger.debug("Removed the released remains in %s", self.lockfile)
        return True

    def _judge_legacy(self, st: os.stat_result) -> bool:
        self._saw(st.st_mtime)
        age = self._now() - st.st_mtime
        holder = self._read_legacy()
        self._last_seen = (holder, age)
        if age <= self.settings.stale_after:
            return False
        try:
            os.unlink(self.lockfile)  # fails on a directory, so never on a new lease
        except FileNotFoundError:
            return True
        except IsADirectoryError:
            return False
        except PermissionError as err:
            self._refused = err
            return False
        self._broke(holder, age)
        return True

    def _broke(self, holder: dict | None, age: float) -> None:
        """Record that a holder's lease was taken over."""
        self.broke.append({"holder": holder, "age": age})
        logger.warning(
            "Took over the lease %s from %s (stale after %.1fs)",
            self.lockfile,
            _describe(holder, age),
            self.settings.stale_after,
        )

    def _claim(self, gone: Path, marker: str, had_contents: bool) -> bool:
        """Whether this process claimed the entry it renamed to ``gone``.

        A rename does not settle this alone: on Windows a rename opens its source and
        renames the open handle, so two processes renaming one entry can both
        succeed, the second moving the directory the first just moved. Each then
        makes ``marker`` in the directory it moved, which is one directory whatever
        its name now, and only one ``mkdir`` succeeds. A process whose directory was
        moved on before its ``mkdir`` finds nothing at ``gone``.

        The claimant removes the directory's contents before its marker (see
        `_remove`). A ``mkdir`` that lands after the marker is gone finds the
        directory emptied, though it ``had_contents`` before the rename, and loses
        too; it removes what it moved. Only a directory that was already empty can
        be claimed twice, which counts no takeover.

        Raises
        ------
        OSError
            The marker could not be made for another reason than losing (for
            example the storage is full).
        """
        try:
            os.mkdir(gone / marker)
        except OSError as err:
            lost = err.errno in (
                errno.ENOENT,
                errno.EEXIST,
                _ESTALE,
            ) or _delete_pending(err)
            if not lost:
                raise
            logger.debug("Lost the claim of %s: %s", gone, err)
            return False
        if had_contents:
            try:
                rest = [e for e in os.listdir(gone) if e != marker]
            except OSError:
                rest = None  # moved on by a later rename: this mkdir still won
            if rest == []:
                logger.debug("Lost the claim of %s: being removed", gone)
                self._remove(gone)
                return False
        return True

    def _remove(self, claimed: Path, keep_last: str | None = None) -> None:
        """Remove a claimed entry, keeping its claim marker ``keep_last`` until its
        other contents are gone, then every ``gone-`` entry beside it (claimed or
        released remains, or a claim in flight that has lost), then the lease
        directory, if nothing else is in it."""
        try:
            contents = [e for e in os.listdir(claimed) if e != keep_last]
        except OSError:
            contents = []
        for entry in contents:
            path = claimed / entry
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        for last in ([claimed / keep_last] if keep_last else []) + [claimed]:
            try:
                os.rmdir(last)
            except OSError:
                pass
        shutil.rmtree(claimed, ignore_errors=True)  # anything made meanwhile
        try:
            remains = [e for e in os.listdir(self.lockfile) if e.startswith("gone-")]
        except OSError:
            remains = []
        for entry in remains:
            shutil.rmtree(self.lockfile / entry, ignore_errors=True)
        try:
            os.rmdir(self.lockfile)
        except OSError:
            pass

    def _saw(self, newest: float) -> None:
        """Note the newest mark in the lease; restart the wait's bound if it advanced,
        since the holder has then been shown alive."""
        if self._newest_seen is not None and newest > self._newest_seen:
            self._deadline = self.monotonic() + self.settings.timeout
            self._refused = None
        if self._newest_seen is None or newest > self._newest_seen:
            self._newest_seen = newest

    def _timeout(self, capped: bool) -> LeaseTimeout:
        holder, age = self._last_seen
        if holder is None:
            holder = self.holder()
        held_by = f"the lease {self.lockfile}, held by {_describe(holder, age)}"
        if self._refused is not None and not capped:
            err = self._refused
            return LeaseUnavailable(
                f"Cannot take over {held_by}: it is stale, but removing it was refused "
                f"({err.strerror or err}). Its owner or an administrator can remove "
                f"{self.lockfile}; a cache shared between users needs group-writable "
                "directories",
                lockfile=self.lockfile,
                holder=holder,
                age=age,
                error=err,
            )
        if capped:
            why = f"max_wait {self.settings.max_wait:.1f}s passed"
            if self._pending is not None:
                why += (
                    "; the storage was refusing to make or list it "
                    f"({self._pending.strerror or self._pending})"
                )
        else:
            why = (
                f"no renewal was seen for {self.settings.timeout:.1f}s and it could not "
                "be taken over"
            )
        return LeaseTimeout(
            f"Gave up waiting for {held_by}: {why}; a lease is taken over once not "
            f"renewed for {self.settings.stale_after:.1f}s",
            lockfile=self.lockfile,
            holder=holder,
            age=age,
        )

    def _attempt(self) -> bool:
        self._pending = None
        if self._try_take():
            return True
        return self._judge() and self._try_take()

    def _step(self) -> float | None:
        """Try to take the lease once: None if it is held now, otherwise the seconds to
        wait before the next try. Raises `LeaseTimeout` when a bound has passed."""
        if self._attempt():
            return None
        now = self.monotonic()
        self._check_pending(now)
        if self._cap is not None and now >= self._cap:
            raise self._timeout(capped=True) from self._pending
        if now >= self._deadline:
            raise self._timeout(capped=False)
        bound = self._deadline if self._cap is None else min(self._deadline, self._cap)
        delay = min(self._delay, bound - now)
        self._delay = min(self._delay * 2, self.settings.poll_max)
        return delay

    def _held_pending(self, err: PermissionError) -> bool:
        """Count a refusal of the lease directory as held on Windows (False); raise it
        elsewhere. A directory pending deletion keeps its name until its last handle
        closes."""
        if not _delete_pending(err):
            raise err
        self._pending = err
        return False

    def _check_pending(self, now: float) -> None:
        """Raise `LeaseRefused` once refusals that a directory pending deletion explains
        have lasted `release_patience`."""
        if self._pending is None:
            self._pending_since = None
            return
        if self._pending_since is None:
            self._pending_since = now
        patience = release_patience(self.settings)
        if now - self._pending_since < patience:
            return
        err = self._pending
        raise LeaseRefused(
            f"Cannot take the lease {self.lockfile}: the storage refused to make or list "
            f"it for {patience:.1f}s ({err.strerror or err}). Check that this user can "
            f"write {self.lockfile.parent}",
            lockfile=self.lockfile,
            error=err,
        ) from err

    # Releasing

    def _lose(self, why: str) -> None:
        self.lost = why
        logger.warning(
            "The lease %s was taken over while held (%s); the job it guards may run "
            "twice",
            self.lockfile,
            why,
        )

    def _release(self) -> None:
        for delay in self._releasing():
            self.sleep(delay)

    async def _arelease(self) -> None:
        for delay in self._releasing():
            await asyncio.sleep(delay)

    def _releasing(self) -> ty.Iterator[float]:
        """Release the lease, yielding the seconds to wait before each retry.

        The holder's directory is renamed aside, then removed; only a directory that is gone shows
        the lease taken over. A refused rename (on Windows, while another process has the directory
        or a file in it open) is retried for `release_patience` seconds, within which no waiter sees
        the stopped holder stale; a rename still refused then leaves the lease to be taken over once
        stale."""
        self._waiting = False
        heartbeat, self._heartbeat = self._heartbeat, None
        token, self._token = self._token, None
        if heartbeat is not None:
            heartbeat.stopped.set()
            heartbeat.join(self.settings.renew_interval)
        if token is None:
            return
        token_dir, gone = self.lockfile / token, self.lockfile / f"gone-{token}"
        patience = release_patience(self.settings)
        give_up = self.monotonic() + patience
        delay = min(self.settings.poll_min / 5, patience / 4)
        while True:
            try:
                os.rename(token_dir, gone)
                break
            except OSError as err:
                if err.errno in (errno.ENOENT, _ESTALE):
                    if self.lost is None:
                        self._lose("its holder directory was gone at release")
                    return
                if self.monotonic() >= give_up:
                    logger.warning(
                        "The lease %s could not be released (%s); it is taken over "
                        "once not renewed for %.1fs",
                        self.lockfile,
                        err,
                        self.settings.stale_after,
                    )
                    return
                logger.debug("Retrying the release of %s: %s", self.lockfile, err)
            yield delay
            delay = min(delay * 2, self.settings.poll_max / 4)
        self._remove(gone)

    # Protocol

    def try_take(self) -> float | None:
        """One pass of the wait of ``__enter__``: None once the lease is held, otherwise
        the seconds before the next attempt. The first call of a wait runs
        ``_prepare``, which sets its bounds; later calls keep them. `LeaseTimeout` ends
        the wait."""
        if not self._waiting:
            self._prepare()
            self._waiting = True
        try:
            delay = self._step()
        except BaseException:
            self._waiting = False
            self._release()
            raise
        if delay is None:
            self._waiting = False
        return delay

    def __enter__(self) -> "DirectoryLease":
        self._prepare()
        try:
            while (delay := self._step()) is not None:
                self.sleep(delay)
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._release()

    async def __aenter__(self) -> "DirectoryLease":
        self._prepare()
        try:
            while (delay := self._step()) is not None:
                await asyncio.sleep(delay)
        except BaseException:
            await self._arelease()
            raise
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._arelease()
