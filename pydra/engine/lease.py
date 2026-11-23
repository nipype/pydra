"""Leases: the right to run a job, held across processes and hosts sharing a cache.

`Lease` is the interface a `pydra.engine.store.ResultStore` returns from ``acquire``.
`DirectoryLease` is the default, made for a cache on shared storage such as NFS.
"""

import asyncio
import errno
import getpass
import importlib
import json
import logging
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import typing as ty
import uuid
import weakref
from pathlib import Path
import attrs
from pydra.utils.general import atomic_write
from pydra.utils.storage import require_lease_storage, storage_now
from pydra.utils.storagedir import IS_WINDOWS

logger = logging.getLogger("pydra")

STALE_AFTER = 300.0
"""Default seconds, on the storage clock, after a lease's last heartbeat at which it is
stale and another process takes it over (see `LeaseSettings`)."""

MAX_HONOURED_STALE_AFTER = 4 * STALE_AFTER
"""Default most seconds of a holder's recorded ``stale_after`` a taker honours (see
`LeaseSettings`), or the taker's own ``stale_after`` if that is larger."""

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
_BEATS = ("thread", "external")
_ESTALE = getattr(errno, "ESTALE", None)


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

    # the checksum of the job the lease is for
    checksum: str
    # the directory a job holding the lease writes its files in, None if it has none
    cache_dir: "Path | None"
    # whether holding it excludes other takers; False: a lookup made before taking it is
    # still current when it is held
    exclusive: bool

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

    A holder renews its lease every ``renew_fraction * stale_after`` seconds and records its
    ``stale_after``. Another process takes the lease over once the holder's newest mark is
    older, on the storage clock, than the larger of its own ``stale_after`` and the holder's,
    at most ``max_honoured_stale_after`` (`holder_stale_after`). A waiter behind a holder that
    keeps renewing waits as long as the holder runs: ``timeout`` counts only time with no
    renewal seen and no takeover possible (a stale lease the storage refuses to let this
    process remove, `LeaseUnavailable`, or an undeletable lock file of an earlier pydra), and
    restarts whenever the newest mark advances. ``max_wait``, if set, caps the whole wait.

    Construction raises ``ValueError`` unless a live holder is never taken over,
    ``(1 + missed_beats) * renew_fraction * stale_after + attribute_delay < stale_after``, and a
    dead holder is taken over before a waiter gives up, ``timeout >= stale_after +
    attribute_delay``.

    Two settings are equal when their four staleness fields are (``stale_after``,
    ``renew_fraction``, ``attribute_delay``, ``timeout``); ``max_wait``,
    ``max_honoured_stale_after`` and ``holder`` may differ between processes sharing leases.

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
    max_honoured_stale_after : float
        Most seconds of a holder's recorded ``stale_after`` this process honours, and
        what a holder record that exists but cannot be read counts as. Default
        `MAX_HONOURED_STALE_AFTER`, or ``stale_after`` if larger; it must be at least
        ``stale_after``. A waiter's ``timeout`` stretches by at most
        ``max_honoured_stale_after - stale_after``.
    poll_min, poll_max : float
        Seconds before the second attempt to take a held lease, and most seconds between two.
    missed_beats : int
        Heartbeats a live holder may miss (a slow or hung write) without being taken over.
    release_patience : float
        Most seconds a release retries a refused rename (see `release_patience`).
    holder : {"named", "anonymous"}
        What the holder record names. ``"named"`` (default) writes the owner, host and pid;
        ``"anonymous"`` writes only the token, the start time and ``stale_after``, and
        messages say "another process", so nothing in the lease directory names a user,
        a host or a job.
    """

    stale_after: float = STALE_AFTER
    timeout: float = TIMEOUT
    renew_fraction: float = RENEW_FRACTION
    attribute_delay: float = ATTRIBUTE_DELAY
    max_wait: float | None = attrs.field(default=None, eq=False)
    holder: ty.Literal["named", "anonymous"] = attrs.field(default="named", eq=False)
    max_honoured_stale_after: float = attrs.field(
        default=attrs.Factory(
            lambda self: max(MAX_HONOURED_STALE_AFTER, self.stale_after),
            takes_self=True,
        ),
        eq=False,
    )
    poll_min: float = attrs.field(default=POLL_MIN, eq=False)
    poll_max: float = attrs.field(default=POLL_MAX, eq=False)
    missed_beats: int = attrs.field(default=MISSED_BEATS, eq=False)
    release_patience: float = attrs.field(default=RELEASE_PATIENCE, eq=False)

    def __attrs_post_init__(self):
        if self.holder not in ("named", "anonymous"):
            raise ValueError(
                f"holder must be 'named' or 'anonymous' (got {self.holder!r})"
            )
        if not self.stale_after <= self.max_honoured_stale_after < float("inf"):
            raise ValueError(
                "max_honoured_stale_after must be finite and at least stale_after (got "
                f"{self.max_honoured_stale_after} and {self.stale_after})"
            )
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

    def to_json(self) -> dict:
        """The four staleness fields as a JSON object (``max_wait`` left out)."""
        return {name: getattr(self, name) for name in _SETTINGS_FIELDS}

    @classmethod
    def from_json(cls, record: dict) -> "LeaseSettings":
        """The settings `to_json` wrote; a record with other keys is refused.

        Raises
        ------
        ValueError
            A field is missing or unknown, or the values break an inequality.
        """
        keys = set(record)
        if keys != set(_SETTINGS_FIELDS):
            missing = sorted(set(_SETTINGS_FIELDS) - keys)
            unknown = sorted(keys - set(_SETTINGS_FIELDS))
            raise ValueError(
                f"lease settings need exactly {', '.join(_SETTINGS_FIELDS)} "
                f"(missing {missing}, unknown {unknown})"
            )
        return cls(**{name: record[name] for name in _SETTINGS_FIELDS})


_SETTINGS_FIELDS = ("stale_after", "renew_fraction", "attribute_delay", "timeout")


def read_holder_record(
    entries: ty.Iterable[str], read: ty.Callable[[str], "str | bytes"]
) -> tuple[dict | None, bool]:
    """The first holder record among ``entries`` (the names in a lease directory; a
    ``gone-`` prefix is allowed), each read by ``read(entry)``, or None; and whether a
    record exists that could not be read. ``read`` raises ``FileNotFoundError`` or
    ``NotADirectoryError`` for an entry without one."""
    unreadable = False
    for entry in sorted(entries):
        name = entry[len("gone-") :] if entry.startswith("gone-") else entry
        if not _TOKEN.match(name):
            continue
        try:
            record = json.loads(read(entry))
        except (FileNotFoundError, NotADirectoryError, ValueError):
            continue
        except OSError:
            unreadable = True
            continue
        if isinstance(record, dict):
            return record, unreadable
    return None, unreadable


def release_patience(settings: LeaseSettings) -> float:
    """Most seconds a refusal explained by another process's open handle is retried: a
    release renaming its directory aside, and on Windows a mkdir or listing of a lease
    directory pending deletion."""
    return min(
        settings.release_patience, settings.missed_beats * settings.renew_interval
    )


def holder_stale_after(
    record: dict | None, settings: LeaseSettings, *, unreadable: bool = False
) -> float:
    """The ``stale_after`` to judge a holder by: the larger of ``settings.stale_after``
    and the one its record carries, at most ``settings.max_honoured_stale_after``.

    A record without a valid one (an older pydra wrote it) counts as the default
    `STALE_AFTER`. With no record, the cap if a record exists but could not be read
    (``unreadable``, another user's on a shared root), otherwise the taker's own."""
    own, cap = settings.stale_after, settings.max_honoured_stale_after
    if not isinstance(record, dict):
        return cap if unreadable else own
    recorded = record.get("stale_after", STALE_AFTER)
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or not 0 < recorded < float("inf")
    ):
        recorded = STALE_AFTER
    return min(max(own, float(recorded)), cap)


def _owner() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no user name in some containers
        return str(os.getuid()) if hasattr(os, "getuid") else "unknown"


def _describe(holder: dict | None, age: float | None, anonymous: bool = False) -> str:
    if anonymous:
        who = "another process"
    elif not holder:
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

    The lease is the directory ``lockfile``: the process whose ``mkdir`` succeeds holds it
    and records itself in ``<lockfile>/<token>/holder.json`` under a random token, and a
    heartbeat renews it every ``settings.renew_interval``. Another process takes it over
    once its newest mtime, on the storage server's clock, is older than `holder_stale_after`;
    of several takers exactly one succeeds. A waiter waits while the holder renews and gives
    up in `LeaseTimeout` (naming the holder) after ``settings.timeout`` without a renewal
    in which the lease could not be taken over, or after ``settings.max_wait``. A regular
    file at ``lockfile`` (an earlier pydra's ``filelock.SoftFileLock``) is removed once older
    than ``stale_after``.

    A holder taken over while it still ran must not write what the lease guards: `prove`
    tells it whether it still holds the lease, and `pydra.engine.store.PickleStore.commit`
    refuses with `LeaseLost` when it does not.

    A lease that guards no job, such as a worker pool's claim, worker or coordinator
    lease, is made by `DirectoryLease.at` and has no checksum.

    Parameters
    ----------
    checksum : str | None
        The checksum of the job the lease is for; None for a named lease.
    cache_dir : Path | None
        The directory the job runs in; None for a named lease.
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
    beat : "thread" or "external"
        Who renews the lease once taken: a heartbeat thread in this process
        (default), or another process, a `LeaseBeater` reached through
        `BeaterClient`, given `token_dir`; with ``"external"`` no thread is started.
    refused_since : float, optional
        When, on ``monotonic``, an earlier lease object of the same ``lockfile`` began
        seeing the refusals `LeaseRefused` bounds (its `refusing_since`), so a caller
        that makes a lease per attempt still reaches the bound.
    """

    exclusive: ty.ClassVar[bool] = True

    checksum: str | None
    cache_dir: Path | None
    lockfile: Path
    settings: LeaseSettings = attrs.field(factory=LeaseSettings)
    clock: ty.Callable[[], float] | None = attrs.field(default=None, repr=False)
    monotonic: ty.Callable[[], float] = attrs.field(default=time.monotonic, repr=False)
    sleep: ty.Callable[[float], None] = attrs.field(default=time.sleep, repr=False)
    beat: ty.Literal["thread", "external"] = attrs.field(
        default="thread", kw_only=True, validator=attrs.validators.in_(_BEATS)
    )
    refused_since: float | None = attrs.field(default=None, kw_only=True, repr=False)
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
    _restarted: float = attrs.field(default=0.0, init=False, repr=False)
    _record_unreadable: bool = attrs.field(default=False, init=False, repr=False)
    _cap: float | None = attrs.field(default=None, init=False, repr=False)
    _delay: float = attrs.field(default=POLL_MIN, init=False, repr=False)
    _waiting: bool = attrs.field(default=False, init=False, repr=False)

    @classmethod
    def at(
        cls,
        lockfile: Path,
        settings: LeaseSettings,
        *,
        beat: ty.Literal["thread", "external"] = "thread",
        clock: ty.Callable[[], float] | None = None,
        monotonic: ty.Callable[[], float] = time.monotonic,
        sleep: ty.Callable[[float], None] = time.sleep,
        refused_since: float | None = None,
    ) -> "DirectoryLease":
        """A named lease: the lease directory ``lockfile``, guarding no job.

        Parameters
        ----------
        lockfile : Path
            The lease directory.
        settings : LeaseSettings
            How the lease beats, ages and waits.
        beat : "thread" or "external"
            See `DirectoryLease`.
        clock, monotonic, sleep : callable, optional
            See `DirectoryLease`.
        refused_since : float, optional
            See `DirectoryLease`.
        """
        return cls(
            None,
            None,
            Path(lockfile),
            settings=settings,
            clock=clock,
            monotonic=monotonic,
            sleep=sleep,
            beat=beat,
            refused_since=refused_since,
        )

    @property
    def refusing_since(self) -> float | None:
        """When, on ``monotonic``, the refusals `LeaseRefused` bounds began, None while the
        latest attempt saw none."""
        return self._pending_since

    @property
    def held(self) -> bool:
        """Whether this lease is entered and has not been found taken over."""
        return self._token is not None and self.lost is None

    @property
    def token_dir(self) -> Path | None:
        """The holder's token directory while this lease is entered, else None; the
        directory a `LeaseBeater` beats for a lease taken with ``beat="external"``."""
        if self._token is None:
            return None
        return self.lockfile / self._token

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
        self._pending, self._pending_since = None, self.refused_since
        now = self.monotonic()
        self._restarted = now
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
            "started": os.stat(token_dir).st_mtime,
            "token": token,
            "stale_after": self.settings.stale_after,
        }
        if self.settings.holder == "named":
            record = {
                "owner": _owner(),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                **record,
            }
        with atomic_write(token_dir / HOLDER_FILE, "w") as f:
            json.dump(record, f)
        self._token = token
        if self.beat == "thread":
            self._heartbeat = _Heartbeat(self, token_dir)
            self._heartbeat.start()
        return True

    # Judging and taking over

    def _read_holder(self) -> tuple[dict | None, list[str]]:
        """The holder's record and the entries of the lease directory. Sets
        ``_record_unreadable`` if a record exists that could not be read."""
        self._record_unreadable = False
        try:
            entries = os.listdir(self.lockfile)
        except NotADirectoryError:
            return self._read_legacy(), []
        except FileNotFoundError:
            return None, []
        record, self._record_unreadable = read_holder_record(
            entries, lambda entry: (self.lockfile / entry / HOLDER_FILE).read_text()
        )
        return record, entries

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
        stale_after = holder_stale_after(
            holder, self.settings, unreadable=self._record_unreadable
        )
        if stale_after > self.settings.stale_after:
            # wait for a dead holder with longer bounds as long as for one with ours, up
            # to max_honoured_stale_after
            self._deadline = max(
                self._deadline,
                self._restarted
                + self.settings.timeout
                + stale_after
                - self.settings.stale_after,
            )
        if age <= stale_after:
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
            _describe(holder, age, self.settings.holder == "anonymous"),
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
            self._restarted = self.monotonic()
            self._deadline = self._restarted + self.settings.timeout
            self._refused = None
        if self._newest_seen is None or newest > self._newest_seen:
            self._newest_seen = newest

    def _timeout(self, capped: bool) -> LeaseTimeout:
        holder, age = self._last_seen
        if holder is None:
            holder = self.holder()
        held_by = (
            f"the lease {self.lockfile}, held by "
            f"{_describe(holder, age, self.settings.holder == 'anonymous')}"
        )
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


# Beating from another process


class BeaterDead(RuntimeError):
    """The `LeaseBeater` behind a `BeaterClient` is gone: its pipe reached end of
    file, or it did not acknowledge a request in time. Its holder takes no new lease
    under it."""


def _reference(fn: "ty.Callable | str") -> str:
    """The ``module:qualname`` an importable callable is found by."""
    if isinstance(fn, str):
        module, _, qualname = fn.partition(":")
    else:
        module = getattr(fn, "__module__", None) or ""
        qualname = getattr(fn, "__qualname__", None) or ""
    if not module or not qualname or module == "__main__" or "<" in qualname:
        raise ValueError(
            f"on_expire must be importable, a module-level callable or its "
            f"'module:qualname' (got {fn!r})"
        )
    return f"{module}:{qualname}"


def _resolve(reference: str) -> ty.Callable:
    module, _, qualname = reference.partition(":")
    obj: ty.Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


@attrs.define
class _Beaten:
    """One token directory a `LeaseBeater` holds."""

    path: Path
    interval: float
    stale_after: float
    n: int
    due: float
    last_ok: float
    beating: bool = True
    at: float | None = None
    payload: ty.Any = None
    kill_at: float | None = None


@attrs.frozen
class LeaseBeater:
    """A loop, run in a process of its own, that beats the token directories of
    leases another process (the holder) took with ``beat="external"``.

    Each directory is beaten by `_Heartbeat`'s rule at its lease's renewal interval:
    ``beat-<n+1>`` is made and ``beat-<n>`` removed. A directory found gone
    (``ENOENT``/``ESTALE``), or failing for longer than ``stale_after``, is lost: it
    is no longer beaten and the holder is sent ``lost``. A directory given a deadline
    stops being beaten when the deadline passes (``time.time()`` on this host);
    ``on_expire(token_dir, payload)`` then runs here, and the holder is sent
    ``expired``. The loop ends when the holder's pipe closes. Start one with
    `BeaterClient`.

    Parameters
    ----------
    on_expire : callable or str, optional
        Called as ``on_expire(token_dir, payload)`` in the beater process when a
        deadline passes; a module-level callable or its ``"module:qualname"``,
        imported by the beater.
    kill_after : float, optional
        Seconds after ``expired`` within which the holder must send
        ``ack_expired``; if it does not, the beater kills the holder's process
        (``SIGKILL``) and exits. None (default) never kills.
    """

    on_expire: "ty.Callable | str | None" = attrs.field(default=None)
    kill_after: float | None = None

    @on_expire.validator
    def _check_on_expire(self, attribute, value) -> None:
        if value is not None:
            _reference(value)

    def config(self, holder_pid: int) -> dict:
        """The JSON the beater process is started with."""
        return {
            "on_expire": None if self.on_expire is None else _reference(self.on_expire),
            "kill_after": self.kill_after,
            "holder_pid": holder_pid,
        }

    @staticmethod
    def run(config: dict, commands: ty.BinaryIO, replies: ty.BinaryIO) -> None:
        """The loop: read requests from ``commands``, write acknowledgements and
        events to ``replies``, beat in between; return at end of file."""
        _BeaterLoop(config, replies).run(commands)


PARENT_POLL = 1.0
"""Most seconds a `LeaseBeater` runs without checking that its holder lives; it must stay
far below any ``renew_interval`` (30 s by default), so a dead holder's lease stops being
beaten long before a waiter judges it."""


class _Holder:
    """The beater's view of its holder process: whether it lives, and killing it,
    never a process that reused its pid. On POSIX the holder is the beater's parent,
    and its death reparents the beater; on Windows a handle to it is kept open."""

    def __init__(self, pid: int):
        self.pid = pid
        self.handle = None
        self.watched = True
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel32.OpenProcess.restype = wintypes.HANDLE
            self.kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            self.kernel32.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel32.WaitForSingleObject.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
            ]
            self.kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            synchronize, terminate = 0x00100000, 0x0001
            self.handle = self.kernel32.OpenProcess(synchronize | terminate, False, pid)
            self.watched = bool(self.handle)
        elif os.getppid() != pid:
            self.watched = False  # started through an intermediary: end of file only
        if not self.watched:
            logger.warning("The lease beater cannot watch its holder, pid %s", pid)

    def alive(self) -> bool:
        if not self.watched:
            return True
        if self.handle is not None:
            return self.kernel32.WaitForSingleObject(self.handle, 0) != 0
        return os.getppid() == self.pid

    def kill(self) -> None:
        if not self.alive():
            return
        if self.handle is not None:
            self.kernel32.TerminateProcess(self.handle, 1)
            return
        try:
            os.kill(self.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass


class _BeaterLoop:
    def __init__(self, config: dict, replies: ty.BinaryIO):
        self.replies = replies
        self.holder_pid = config["holder_pid"]
        self.kill_after = config.get("kill_after")
        reference = config.get("on_expire")
        self.on_expire = _resolve(reference) if reference else None
        self.dirs: dict[str, _Beaten] = {}
        self.holder = _Holder(self.holder_pid)

    def send(self, message: dict) -> None:
        self.replies.write(json.dumps(message).encode() + b"\n")
        self.replies.flush()

    def run(self, commands: ty.BinaryIO) -> None:
        inbox: "queue.Queue[dict | None]" = queue.Queue()

        def read() -> None:
            try:
                for line in commands:
                    if line.strip():
                        inbox.put(json.loads(line))
            finally:
                inbox.put(None)

        threading.Thread(target=read, name="pydra-beater-reader", daemon=True).start()
        self.send({"ready": os.getpid()})
        while True:
            try:
                message = inbox.get(timeout=min(self.wait(), PARENT_POLL))
            except queue.Empty:
                message = {}
            if message is None:
                return
            if not self.holder.alive():
                logger.warning("The lease holder, pid %s, is gone", self.holder_pid)
                return
            if message:
                self.handle(message)
            if self.service():
                return

    def wait(self) -> float:
        now, wall = time.monotonic(), time.time()
        wakes = []
        for entry in self.dirs.values():
            if entry.beating:
                wakes.append(entry.due - now)
                if entry.at is not None:
                    wakes.append(entry.at - wall)
            if entry.kill_at is not None:
                wakes.append(entry.kill_at - now)
        return max(0.0, min(wakes)) if wakes else 3600.0

    def handle(self, message: dict) -> None:
        op, key = message.get("op"), message.get("dir")
        reply: dict = {"ack": message.get("id")}
        try:
            if op == "hold":
                self.hold(key, LeaseSettings.from_json(message["settings"]))
            elif op == "release":
                self.dirs.pop(key, None)
            elif op == "deadline":
                entry = self.dirs[key]
                entry.at, entry.payload = message.get("at"), message.get("payload")
            elif op == "ack_expired":
                if key in self.dirs:
                    self.dirs[key].kill_at = None
            else:
                raise ValueError(f"unknown request {op!r}")
        except Exception as err:
            reply["error"] = f"{type(err).__name__}: {err}"
        self.send(reply)

    def hold(self, key: str, settings: LeaseSettings) -> None:
        path = Path(key)
        try:  # continue the numbering of any beat already in the directory
            n = max(
                (int(e[5:]) for e in os.listdir(path) if re.match(r"^beat-\d+$", e)),
                default=0,
            )
        except OSError:
            n = 0
        now = time.monotonic()
        self.dirs[key] = _Beaten(
            path, settings.renew_interval, settings.stale_after, n, now, now
        )

    def service(self) -> bool:
        """Run what is due; whether the loop ends (it killed the holder)."""
        now, wall = time.monotonic(), time.time()
        for key, entry in list(self.dirs.items()):
            if entry.beating and entry.at is not None and wall >= entry.at:
                self.expire(key, entry)
            elif entry.beating and now >= entry.due:
                self.beat(key, entry)
            if entry.kill_at is not None and time.monotonic() >= entry.kill_at:
                self.kill()
                return True
        return False

    def beat(self, key: str, entry: _Beaten) -> None:
        entry.due += entry.interval
        now = time.monotonic()
        if entry.due < now:  # fell behind (a slow write): beat on from now
            entry.due = now + entry.interval
        try:
            os.mkdir(entry.path / f"beat-{entry.n + 1}")
        except FileExistsError:
            entry.n += 1
            return
        except OSError as err:
            if err.errno in (errno.ENOENT, _ESTALE):
                self.lose(key, entry, "its holder directory was removed")
            elif now - entry.last_ok > entry.stale_after:
                self.lose(key, entry, f"it could not be renewed for too long ({err})")
            return
        try:
            os.rmdir(entry.path / f"beat-{entry.n}")
        except OSError:
            pass
        entry.n += 1
        entry.last_ok = time.monotonic()

    def lose(self, key: str, entry: _Beaten, reason: str) -> None:
        entry.beating = False
        self.send({"event": "lost", "dir": key, "reason": reason})

    def expire(self, key: str, entry: _Beaten) -> None:
        entry.beating = False
        if self.on_expire is not None:
            try:
                self.on_expire(entry.path, entry.payload)
            except Exception:
                logger.exception("on_expire failed for %s", entry.path)
        if self.kill_after is not None:
            entry.kill_at = time.monotonic() + self.kill_after
        self.send({"event": "expired", "dir": key})

    def kill(self) -> None:
        logger.warning(
            "Killing pid %s: it did not stop an expired call within %ss",
            self.holder_pid,
            self.kill_after,
        )
        self.holder.kill()


def _beater_main() -> None:
    """The beater process: ``python -c`` runs this with the config as argument.
    Requests arrive on stdin; replies leave on the original stdout, which is then
    pointed at stderr so that nothing else writes to the pipe."""
    for name in ("SIGINT", "SIGTERM"):  # it ends when its pipe closes or its
        if hasattr(signal, name):  # holder dies, so it outlives a group's SIGTERM
            signal.signal(getattr(signal, name), signal.SIG_IGN)
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    config = json.loads(sys.argv[1])
    logging.basicConfig(level=logging.WARNING)
    try:
        loop = _BeaterLoop(config, replies)
    except Exception as err:
        replies.write(json.dumps({"error": f"{type(err).__name__}: {err}"}).encode())
        replies.write(b"\n")
        replies.flush()
        raise SystemExit(1)
    try:
        loop.run(sys.stdin.buffer)
    except BrokenPipeError:
        pass


_CLIENTS: "weakref.WeakSet[BeaterClient]" = weakref.WeakSet()


def _after_fork_in_child() -> None:
    """In a forked child, point every `BeaterClient`'s pipe descriptors at
    ``os.devnull``: the child is not the holder, and a copy of the holder's end
    kept open would keep its beater from seeing end of file."""
    clients = list(_CLIENTS)
    if not clients:
        return
    null = os.open(os.devnull, os.O_RDWR)
    try:
        for client in clients:
            client._dying = True
            client.dead = "the holder forked; a child does not hold its leases"
            for stream in (client.process.stdin, client.process.stdout):
                try:
                    os.dup2(null, stream.fileno())
                except (OSError, ValueError):
                    pass
    finally:
        os.close(null)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class BeaterClient:
    """The holder's end of a pipe to a `LeaseBeater`, which it starts as a child process and stops
    when closed (or when its holder exits, which closes the pipe).

    The holder takes a lease with ``beat="external"`` and calls `hold` with it, running nothing
    under the lease before `hold` returns, and calls `release` before releasing the lease. The
    beater's events arrive on a reader thread: ``lost`` marks the lease lost (its next
    `DirectoryLease.prove` answers False), ``expired`` reports a passed deadline. The callbacks run
    in order on a thread of their own, so a callback may make requests.

    The beater ends when the pipe closes or when it finds its holder dead (within `PARENT_POLL`
    seconds), even while a forked child of the holder lives. It ignores ``SIGINT`` and ``SIGTERM``.
    A request not acknowledged within the lease's ``renew_interval``, or the pipe reaching end of
    file, means the beater is dead: the request and every later one raise `BeaterDead`, and the
    holder should take no new lease and exit.

    Parameters
    ----------
    beater : LeaseBeater, optional
        What the beater runs; by default no callback and no kill.
    startup_timeout : float
        Seconds to wait for the beater process to start (default 60).
    on_lost : callable, optional
        Called as ``on_lost(token_dir, reason)`` after the lease was marked lost.
    on_expired : callable, optional
        Called as ``on_expired(token_dir)`` after ``expired`` arrived.
    on_dead : callable, optional
        Called as ``on_dead(reason)`` once the beater is found dead.
    """

    def __init__(
        self,
        beater: LeaseBeater | None = None,
        *,
        startup_timeout: float = 60.0,
        on_lost: ty.Callable[[Path, str], None] | None = None,
        on_expired: ty.Callable[[Path], None] | None = None,
        on_dead: ty.Callable[[str], None] | None = None,
    ):
        self.beater = beater if beater is not None else LeaseBeater()
        self.on_lost, self.on_expired, self.on_dead = on_lost, on_expired, on_dead
        self.lost: dict[Path, str] = {}
        self.expired: set[Path] = set()
        self.dead: str | None = None
        self._dying = False
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._acks: dict[int, dict] = {}
        self._next = 0
        self._settings: dict[Path, LeaseSettings] = {}
        self._leases: dict[Path, DirectoryLease] = {}
        self._ready: dict | None = None
        self._calls: "queue.Queue[tuple | None]" = queue.Queue()
        self._caller = threading.Thread(
            target=self._call, name="pydra-beater-callbacks", daemon=True
        )
        self._caller.start()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-P",
                "-c",
                "from pydra.engine.lease import _beater_main; _beater_main()",
                json.dumps(self.beater.config(os.getpid())),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._reader = threading.Thread(
            target=self._read, name="pydra-beater-client", daemon=True
        )
        self._reader.start()
        _CLIENTS.add(self)
        with self._changed:
            self._changed.wait_for(
                lambda: self._ready is not None or self.dead is not None,
                startup_timeout,
            )
            ready, dead = self._ready, self.dead
        if ready is None or "error" in ready:
            why = (ready or {}).get("error") or dead or "it did not start in time"
            why = f"the beater did not start: {why}"
            self._die(why)
            self.close()
            raise BeaterDead(why)

    @property
    def alive(self) -> bool:
        """Whether the beater is still answering."""
        return self.dead is None

    # Requests

    def hold(self, lease: DirectoryLease) -> None:
        """Have the beater beat ``lease.token_dir`` at its settings' renewal
        interval; returns once it acknowledged. ``lease`` is marked lost when the
        beater reports its directory lost.

        Raises
        ------
        ValueError
            ``lease`` is not held, or was taken with ``beat="thread"`` (its own
            heartbeat beats it).
        """
        if lease.beat != "external":
            raise ValueError(
                f"{lease.lockfile} was taken with beat={lease.beat!r}; only a lease "
                'taken with beat="external" is beaten by a LeaseBeater'
            )
        token_dir, settings = lease.token_dir, lease.settings
        if token_dir is None:
            raise ValueError(f"{lease.lockfile} is not held")
        with self._lock:
            self._settings[token_dir] = settings
            self._leases[token_dir] = lease
            self.expired.discard(token_dir)
            self.lost.pop(token_dir, None)
        self._request(
            {"op": "hold", "dir": str(token_dir), "settings": settings.to_json()},
            settings.renew_interval,
        )

    def release(self, token_dir: Path) -> None:
        """Have the beater stop beating ``token_dir``; returns once it acknowledged,
        after which the holder releases the lease."""
        token_dir = Path(token_dir)
        self._request(
            {"op": "release", "dir": str(token_dir)}, self._interval(token_dir)
        )
        with self._lock:
            self._settings.pop(token_dir, None)
            self._leases.pop(token_dir, None)

    def deadline(
        self, token_dir: Path, at: float | None, payload: ty.Any = None
    ) -> None:
        """Set (or, with ``at=None``, clear) the time, as ``time.time()`` on this
        host, at which the beater stops beating ``token_dir``, runs
        ``on_expire(token_dir, payload)`` and sends ``expired``. ``payload`` is
        JSON."""
        token_dir = Path(token_dir)
        self._request(
            {"op": "deadline", "dir": str(token_dir), "at": at, "payload": payload},
            self._interval(token_dir),
        )

    def ack_expired(self, token_dir: Path) -> None:
        """Tell the beater the expired call under ``token_dir`` was stopped, so that
        a `LeaseBeater` with ``kill_after`` does not kill this process."""
        token_dir = Path(token_dir)
        self._request(
            {"op": "ack_expired", "dir": str(token_dir)}, self._interval(token_dir)
        )

    # Waiting

    def wait_expired(self, token_dir: Path, timeout: float | None = None) -> bool:
        """Whether ``expired`` arrived for ``token_dir`` within ``timeout`` seconds."""
        token_dir = Path(token_dir)
        with self._changed:
            return self._changed.wait_for(lambda: token_dir in self.expired, timeout)

    def wait_dead(self, timeout: float | None = None) -> bool:
        """Whether the beater was found dead within ``timeout`` seconds."""
        with self._changed:
            return self._changed.wait_for(lambda: self.dead is not None, timeout)

    def close(self) -> None:
        """Close the pipe, which ends the beater, and wait for it to exit."""
        with self._changed:
            self._dying = True
            if self.dead is None:
                self.dead = "it was closed"
            self._changed.notify_all()
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._reader.join(10)
        self.process.stdout.close()
        self._calls.put(None)
        if threading.current_thread() is not self._caller:
            self._caller.join(10)

    def __enter__(self) -> "BeaterClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # Internals

    def _interval(self, token_dir: Path) -> float:
        with self._lock:
            settings = self._settings.get(token_dir)
        if settings is None:
            raise ValueError(f"{token_dir} is not held by this beater")
        return settings.renew_interval

    def _request(self, message: dict, timeout: float) -> None:
        with self._changed:
            if self.dead is not None:
                raise BeaterDead(self.dead)
            self._next += 1
            ident = message["id"] = self._next
        try:
            self.process.stdin.write(json.dumps(message).encode() + b"\n")
            self.process.stdin.flush()
        except OSError as err:
            self._die(f"its pipe is closed ({err})")
            raise BeaterDead(self.dead) from err
        with self._changed:
            answered = self._changed.wait_for(
                lambda: ident in self._acks or self.dead is not None, timeout
            )
            reply = self._acks.pop(ident, None)
        if reply is None:
            if not answered:
                self._die(f"it did not acknowledge {message['op']} within {timeout:g}s")
            raise BeaterDead(self.dead)
        if "error" in reply:
            raise ValueError(f"the beater refused {message['op']}: {reply['error']}")

    def _die(self, why: str) -> None:
        """Record the beater dead, stop it, wake every waiter and queue
        ``on_dead``."""
        with self._lock:
            if self._dying:
                return
            self._dying = True
        logger.warning("The lease beater (pid %s) is dead: %s", self.process.pid, why)
        if self.process.poll() is None:
            self.process.kill()
        with self._changed:
            self.dead = why
            self._changed.notify_all()
        if self.on_dead is not None:
            self._calls.put((self.on_dead, why))

    def _call(self) -> None:
        """Run the queued callbacks in order until `close`."""
        while (item := self._calls.get()) is not None:
            fn, *args = item
            try:
                fn(*args)
            except Exception:
                logger.exception("A lease beater callback failed")

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                self._dispatch(message)
        except (OSError, ValueError):
            pass
        self._die("its pipe reached end of file")

    def _dispatch(self, message: dict) -> None:
        if "ready" in message or ("error" in message and "ack" not in message):
            with self._changed:
                self._ready = message
                self._changed.notify_all()
            return
        if "ack" in message:
            with self._changed:
                self._acks[message["ack"]] = message
                self._changed.notify_all()
            return
        token_dir = Path(message.get("dir", ""))
        if message.get("event") == "lost":
            reason = message.get("reason", "lost")
            with self._changed:
                self.lost[token_dir] = reason
                lease = self._leases.get(token_dir)
                self._changed.notify_all()
            if lease is not None and lease.lost is None:
                lease._lose(reason)
            if self.on_lost is not None:
                self._calls.put((self.on_lost, token_dir, reason))
        elif message.get("event") == "expired":
            with self._changed:
                self.expired.add(token_dir)
                self._changed.notify_all()
            if self.on_expired is not None:
                self._calls.put((self.on_expired, token_dir))
