"""The worker-pool protocol over shared storage (RFC 0005 sections 5.1-5.3 and 5.5).

A pool is a directory on storage that every worker and the submitting process see::

    <root>/                                   0o700, owned by the pool's user
      pool.json                               "pool@1": pool id, the four LeaseSettings, codec,
                                              owner uid, segment size, probe results
      coordinator/<epoch>/                    one per coordinator, taken by mkdir; lease/, ended
      queues/<label>/                         label = sha256 of the queue name
        name                                  the queue name, written once
        tickets/<digest>.json                 a ticket as JCS; digest = sha256 of those bytes
        segments/<seq>/slot-<n>               the index: a regular file holding a ticket digest
        segments/<seq>/jobs-<n>-<count>       the job count of slot n's ticket
        segments/<seq>/res-<n>                slot n's ticket has a final outcome for every job
        segments/<seq>/retired                every slot of the segment is taken and resolved
        slotted/<digest>                      the ticket's first slot, "<seq>/<n>"
        claims/<digest>.lease/                the claim's DirectoryLease
        losses/<digest>/<n>                   a takeover of the ticket
        outcome/<digest>/<job>-<attempt>.json one per job and attempt; the first writer wins
        settled/<digest>-<count>              the ticket's outcome count reached <count>
        held-<epoch>-<label>                  a hold of a residency on this queue
      follow/<job>/<g>                        generation g of a job: the ticket that holds it
      follow/<job>/<g>.started                attempt <pool id>.<g> was started, by which worker
      payloads/<sha256>                       a job's payload, read by every attempt of the job

``<job>`` is ``<checksum>-<fingerprint>`` when that is a plain lower-case name of at most 200
characters, else ``h-<sha256>`` of the two.

Every file is created once and never modified: it is staged under a private name, flushed, linked
to its name (renamed, on Windows), and its directory flushed. A create that meets ``EEXIST`` reads
the file back and counts it as its own when the bytes, which carry a fresh uuid, are its own (a
lost reply to a retransmitted request). ``ESTALE`` on a read is read as absent. No symlink is made
and no host-bound lock is taken: every exclusion is a `DirectoryLease`.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import re
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.engine.lease import DirectoryLease, LeaseSettings, LeaseTimeout
from pydra.engine.portable import store as portable_store
from pydra.engine.portable.layout import FINGERPRINT
from pydra.utils.digest import jcs
from pydra.utils.retry import Backoff
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
)

__all__ = [
    "Claim",
    "CorruptTicket",
    "JobHeld",
    "PoolOwnerMismatch",
    "PoolRoot",
    "PoolSettingsMismatch",
    "PoolStorageUnsupported",
    "ListingBehind",
    "Ticket",
    "TicketJob",
    "job_name",
    "queue_label",
]


STOP_MARK = "stop.json"
"""The worker record a coordinator writes to stop a worker, which drains on it as on SIGTERM."""

EXIT_MARK = "exit.json"
"""The worker record a worker writes as it exits, read where its process is not visible."""

POOL_FORMAT = "pool@1"
TICKET_FORMAT = "pool-ticket@1"
OUTCOME_FORMAT = "pool-outcome@1"
HEADER = "pool.json"
PRIVATE_PICKLE = "private-pickle@1"
SEGMENT_ITEMS = 1000
"""Slots per segment, recorded in ``pool.json``; every process of a pool uses the recorded value.
A listing of one segment stays a few tens of kilobytes, so per-claim work does not grow with
the queue (RFC 0005 section 5)."""

OUTCOME_LISTING = Backoff(first=0.05, cap=1.0)
"""How `outcomes` lists again while fewer records show than the settled mark names."""

OUTCOMES = ("done", "failed", "held", "withdrawn", "expired")
FINAL = frozenset(("done", "failed", "held", "withdrawn"))
"""Outcomes after which a job leaves its ticket; ``expired`` waits for a taker."""
WITHDRAW_REASONS = ("cancel", "replaced", "split", "refused")
CAUSES = ("put", "replaced", "split", "takeover", "restart", "ladder")
PHASES = ("call", "load", None)

FILE_MODE = 0o600
DIR_MODE = 0o700
POLICY = StorePolicy(
    file_mode=FILE_MODE,
    entry_mode=DIR_MODE,
    container_mode=DIR_MODE,
    side_mode=DIR_MODE,
    staging_mode=DIR_MODE,
)
_TRUST: ty.Literal["private", "trusted"] = "trusted" if IS_WINDOWS else "private"

_ESTALE = getattr(errno, "ESTALE", None)
_QUEUE_NAME = re.compile(r"^[^#\s/\\]+(#\S+)?$")
_PLAIN_JOB = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,199}$")
_SEQ = re.compile(r"^[0-9]{12}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_SLOT = re.compile(r"^slot-([0-9]+)$")
_JOBS = re.compile(r"^jobs-([0-9]+)-([1-9][0-9]*)$")
_RES = re.compile(r"^res-([0-9]+)$")
_GEN = re.compile(r"^(0|[1-9][0-9]*)$")
_EPOCH = re.compile(r"^[1-9][0-9]*$")
_HELD = re.compile(r"^held-([1-9][0-9]*)-([0-9a-f]{64})$")
_SETTLED = re.compile(r"^([0-9a-f]{64})-([1-9][0-9]*)$")
_STAGED = ".tmp-"
_PAYLOAD_KEYS = frozenset(("codec", "path", "sha256", "json"))

JobKey = tuple[str, str]
"""A job's ``(checksum, fingerprint)``."""


class PoolSettingsMismatch(ValueError):
    """A process's `LeaseSettings` differ from the ones recorded in ``pool.json``.

    Attributes
    ----------
    fields : list of str
        The differing fields among ``stale_after``, ``renew_fraction``, ``attribute_delay`` and
        ``timeout``.
    """

    def __init__(self, message: str, fields: list[str]):
        super().__init__(message)
        self.fields = fields


class PoolOwnerMismatch(PermissionError):
    """The pool belongs to another user: a pool is single-user (RFC 0005 section 5.2)."""


class PoolStorageUnsupported(StoreUnavailable):
    """The storage lacks a primitive the protocol needs, or shows directory changes later than
    ``attribute_delay``; the message names what `probe` found."""


class ListingBehind(LookupError):
    """A listing that still shows fewer records than a settled mark counts, at the bound of its wait.

    Attributes
    ----------
    found : dict
        The records listed.
    expected : int
    """

    def __init__(self, message: str, found: dict, expected: int):
        super().__init__(message)
        self.found = found
        self.expected = expected


class CorruptTicket(ValueError):
    """A ticket file that is not a ticket, or not the ticket its name digests."""


class JobHeld(RuntimeError):
    """A put of a job that a live ticket other than the one being put already holds.

    Attributes
    ----------
    job : JobKey
    ticket : str
        The digest of the ticket that holds it.
    """

    def __init__(self, job: JobKey, ticket: str):
        super().__init__(f"job {job[0]} ({job[1]}) is held by ticket {ticket}")
        self.job = job
        self.ticket = ticket


# -- names -----------------------------------------------------------------------------------


def queue_label(name: str) -> str:
    """The directory name of queue ``name``: the sha256 of the name.

    Raises
    ------
    ValueError
        ``name`` is not a queue name: a class, with ``#<residency digest>`` when it has one.
    """
    if not isinstance(name, str) or not _QUEUE_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a queue name: a worker class, with #<residency digest> for a "
            "residency"
        )
    return hashlib.sha256(name.encode()).hexdigest()


def job_name(checksum: str, fingerprint: str) -> str:
    """The name of a job's follow directory and the prefix of its outcome records."""
    raw = f"{checksum}-{fingerprint}"
    if _PLAIN_JOB.fullmatch(raw):
        return raw
    return "h-" + hashlib.sha256(f"{checksum}\0{fingerprint}".encode()).hexdigest()


def _key(job: "TicketJob | JobKey") -> JobKey:
    if isinstance(job, TicketJob):
        return (job.checksum, job.fingerprint)
    checksum, fingerprint = job
    return (checksum, fingerprint)


def _outcome_name(job: JobKey, attempt: str) -> str:
    return f"{job_name(*job)}-{attempt}.json"


# -- tickets ---------------------------------------------------------------------------------


def _check_payload(_inst: ty.Any, _attr: ty.Any, payload: ty.Any) -> None:
    """A payload is a reference to a file read under the ownership walk, ``{codec, path,
    sha256}``, or JSON carried inline, ``{codec, sha256, json}``; never bytes or a pickle.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("codec"), str):
        raise ValueError(
            f"a payload is a JSON object naming its codec, got {payload!r}"
        )
    unknown = set(payload) - _PAYLOAD_KEYS
    if unknown:
        raise ValueError(f"a payload has no keys {sorted(unknown)}")
    sha = payload.get("sha256")
    if not isinstance(sha, str) or not FINGERPRINT.fullmatch(sha):
        raise TypeError(
            f"a payload's sha256 is 64 lower-case hex characters, got {sha!r}"
        )
    if "json" in payload:
        if "path" in payload or payload["codec"] == PRIVATE_PICKLE:
            raise ValueError(
                "an inline payload carries JSON only, never a path or a pickle codec"
            )
        jcs(payload["json"])  # raises for a value that is not JSON
    elif not isinstance(payload.get("path"), str):
        raise ValueError("a payload is either a path and its digest or inline JSON")


@attrs.frozen
class TicketJob:
    """One job of a ticket (RFC 0005 section 5.1).

    Attributes
    ----------
    checksum, fingerprint : str
        The job's address in the store; together they name the job in the follow index.
    payload : dict
        Where the job is: ``{"codec", "path", "sha256"}`` or ``{"codec", "sha256", "json"}``.
    attempt : str or None
        The pool attempt of the generation that put the job in this ticket. A claimant reads the
        job's attempt from its newest generation, never from here.
    runner_attempt : int or None
        The `GraphRunner`'s attempt, for reporting only.
    units : float or None
        The job's units, for batching.
    """

    checksum: str = attrs.field(validator=attrs.validators.instance_of(str))
    fingerprint: str = attrs.field(validator=attrs.validators.instance_of(str))
    payload: dict = attrs.field(validator=_check_payload, eq=True, hash=False)
    attempt: str | None = None
    runner_attempt: int | None = None
    units: float | None = None

    @property
    def key(self) -> JobKey:
        return (self.checksum, self.fingerprint)

    def to_json(self) -> dict:
        return {
            "checksum": self.checksum,
            "fingerprint": self.fingerprint,
            "attempt": self.attempt,
            "runner_attempt": self.runner_attempt,
            "payload": self.payload,
            "units": self.units,
        }

    @classmethod
    def from_json(cls, record: dict) -> "TicketJob":
        return cls(
            checksum=record["checksum"],
            fingerprint=record["fingerprint"],
            payload=record["payload"],
            attempt=record.get("attempt"),
            runner_attempt=record.get("runner_attempt"),
            units=record.get("units"),
        )


def _tuple(value: ty.Iterable) -> tuple:
    return tuple(value)


@attrs.frozen
class Ticket:
    """A queue item: up to ``quantum_jobs`` jobs of one node and one queue, as JSON, never a pickle.

    Attributes
    ----------
    queue : str
        The queue's name, ``<class>`` or ``<class>#<residency digest>``.
    node : str
    jobs : tuple of TicketJob
    residency : tuple of dict
        The resident state the jobs need (provider and environment keys).
    batch : int or None
    roots : tuple of str
        Binding roots the jobs read.
    ladder : tuple of str
        The rest of the jobs' placement order: the classes a claim holder moves them up to.
    time : float or None
        The longest time its jobs declare, in seconds; None when none declares one.
    """

    queue: str
    node: str
    jobs: tuple[TicketJob, ...] = attrs.field(converter=_tuple)
    residency: tuple[dict, ...] = attrs.field(default=(), converter=_tuple, hash=False)
    batch: int | None = None
    roots: tuple[str, ...] = attrs.field(default=(), converter=_tuple)
    ladder: tuple[str, ...] = attrs.field(default=(), converter=_tuple)
    time: float | None = None

    def __attrs_post_init__(self) -> None:
        queue_label(self.queue)
        if not self.jobs:
            raise ValueError("a ticket holds at least one job")
        keys = [j.key for j in self.jobs]
        if len(set(keys)) != len(keys):
            raise ValueError("a job is in a ticket once")
        for name in self.ladder:
            queue_label(name)

    def to_json(self) -> dict:
        return {
            "format": TICKET_FORMAT,
            "queue": self.queue,
            "node": self.node,
            "residency": list(self.residency),
            "batch": self.batch,
            "roots": list(self.roots),
            "ladder": list(self.ladder),
            "time": self.time,
            "jobs": [j.to_json() for j in self.jobs],
        }

    @classmethod
    def from_json(cls, record: dict) -> "Ticket":
        """The ticket `to_json` wrote.

        Raises
        ------
        CorruptTicket
            ``record`` is not a ticket.
        """
        try:
            if record.get("format") != TICKET_FORMAT:
                raise ValueError(f"format is {record.get('format')!r}")
            return cls(
                queue=record["queue"],
                node=record["node"],
                jobs=[TicketJob.from_json(j) for j in record["jobs"]],
                residency=record.get("residency", ()),
                batch=record.get("batch"),
                roots=record.get("roots", ()),
                ladder=record.get("ladder", ()),
                time=record.get("time"),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            raise CorruptTicket(f"not a ticket: {err}") from err

    def canonical(self) -> bytes:
        """The ticket's bytes on storage: its JSON as JCS (RFC 8785)."""
        return jcs(self.to_json())

    @property
    def digest(self) -> str:
        """The sha256 of `canonical`, which names the ticket."""
        return hashlib.sha256(self.canonical()).hexdigest()

    @property
    def residency_label(self) -> str:
        """The sha256 of the residency's JCS, the label a hold of it names."""
        return hashlib.sha256(jcs(list(self.residency))).hexdigest()


# -- storage primitives ----------------------------------------------------------------------


def _absent(err: OSError) -> bool:
    return isinstance(err, FileNotFoundError) or err.errno in (errno.ENOENT, _ESTALE)


def _read(d: StorageDir, name: str) -> bytes | None:
    """The bytes of ``name``, None when it is absent or its directory is stale. On Windows a
    record another process is linking into place refuses an open for a moment (a sharing
    violation): the read is tried again as `SHARING_RETRY` says."""
    tries = portable_store.SHARING_RETRY.tries()
    for _ in tries:
        try:
            with d.open_read(name) as f:
                return f.read()
        except PermissionError:
            if not IS_WINDOWS or tries.last:
                raise
        except OSError as err:
            if _absent(err):
                return None
            raise


def _names(d: StorageDir | None) -> list[str]:
    """The names in ``d`` other than staged files, listed with no per-entry stat; none when ``d`` is
    absent or stale. A caller never reads an empty listing as proof that something is absent.
    """
    if d is None:
        return []
    try:
        return [n for n in d.names() if not n.startswith(_STAGED)]
    except OSError as err:
        if _absent(err):
            return []
        raise


def _child(d: StorageDir, name: str) -> StorageDir | None:
    """The directory ``name`` in ``d``, None when absent or stale."""
    try:
        return d.child(name)
    except OSError as err:
        if _absent(err):
            return None
        raise


def _dir(d: StorageDir, name: str) -> StorageDir:
    """The directory ``name`` in ``d``, made ``0o700`` if absent, with ``d`` flushed after."""
    try:
        return d.child(name)
    except OSError as err:
        if not _absent(err):
            raise
    handle = d.child(name, create=True, mode=DIR_MODE)
    d.sync()
    return handle


def _create(
    d: StorageDir,
    name: str,
    data: bytes,
    guard: ty.Callable[[], bool] | None = None,
) -> str | None:
    """Create ``name`` in ``d`` holding ``data``, once: staged, flushed, linked, ``d`` flushed.

    Returns
    -------
    "created", "own", "exists" or None
        ``"own"``: the name existed with exactly ``data`` (a lost reply to this write, when ``data``
        carries a fresh uuid); ``"exists"``: it holds something else, or ``data`` is empty;
        None: ``guard`` answered False after staging and nothing was linked.
    """
    staged = f"{_STAGED}{uuid.uuid4().hex}"
    with d.open_new(staged, FILE_MODE) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        if guard is not None and not guard():
            return None
        try:
            if IS_WINDOWS:
                d.rename(staged, d, name)
            else:
                d.link(staged, name)
            outcome = "created"
        except FileExistsError:
            outcome = "own" if data and _read(d, name) == data else "exists"
        if outcome != "exists":
            d.sync()
        return outcome
    finally:
        with contextlib.suppress(FileNotFoundError):
            d.unlink(staged)


def _record(data: bytes | None) -> dict | None:
    """A JSON object from ``data``, None when absent or not one."""
    if data is None:
        return None
    try:
        record = json.loads(data)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _read_record(d: StorageDir, name: str) -> dict | None:
    """The record ``name``: read again once when it does not parse, then treated as absent."""
    record = _record(_read(d, name))
    if record is None:
        record = _record(_read(d, name))
    return record


def _count_outcomes(d: StorageDir | None) -> int:
    """The number of outcome records in a ticket's outcome directory."""
    return sum(1 for n in _names(d) if n.endswith(".json"))


def _generation_numbers(d: StorageDir | None) -> list[int]:
    """The generation numbers listed in a job's follow directory, ascending."""
    return sorted(int(n) for n in _names(d) if _GEN.fullmatch(n))


def _generation_of(attempt: ty.Any) -> int | None:
    """The generation number of a pool attempt ``<pool id>.<g>``, None for anything else."""
    if not isinstance(attempt, str):
        return None
    _, dot, g = attempt.rpartition(".")
    return int(g) if dot and _GEN.fullmatch(g) else None


def _seq(seq: int) -> str:
    return f"{seq:012d}"


def _slots(names: list[str]) -> tuple[list[int], set[int], dict[int, int]]:
    """The slots listed in a segment, in order, the resolved ones, and each slot's job count."""
    slots, resolved, jobs = [], set(), {}
    for name in names:
        if m := _SLOT.fullmatch(name):
            slots.append(int(m.group(1)))
        elif m := _RES.fullmatch(name):
            resolved.add(int(m.group(1)))
        elif m := _JOBS.fullmatch(name):
            jobs[int(m.group(1))] = int(m.group(2))
    return sorted(slots), resolved, jobs


# -- queues ----------------------------------------------------------------------------------


class _Queue:
    """A queue's open directories and what this process has read of it."""

    _SUBDIRS = (
        "tickets",
        "segments",
        "slotted",
        "claims",
        "losses",
        "outcome",
        "settled",
    )

    def __init__(self, pool: "PoolRoot", name: str):
        self.name = name
        self.label = queue_label(name)
        self.path = pool.path / "queues" / self.label
        with _dir(pool._root, "queues") as queues:
            self.dir = _dir(queues, self.label)
        if _read(self.dir, "name") is None:
            _create(self.dir, "name", name.encode())
        for sub in self._SUBDIRS:
            setattr(self, sub, _dir(self.dir, sub))
        self.head = 0  # the newest segment a put used
        self.oldest = 0  # the oldest segment that may be live
        self.targets: dict[tuple[int, int], str] = {}
        self.cached: dict[str, Ticket] = {}
        self.judged: dict[str, float] = {}  # tickets found under a live claim, when
        self.refused: dict[str, float] = (
            {}
        )  # claim leases the storage refuses, since when
        self.missing: dict[str, float] = {}  # tickets found missing or corrupt, when
        self.faults: list[tuple[str, str]] = []

    def close(self) -> None:
        for sub in self._SUBDIRS:
            getattr(self, sub).close()
        self.dir.close()


# -- the pool --------------------------------------------------------------------------------


class PoolRoot:
    """A pool directory, opened by `open`."""

    def __init__(
        self, path: Path, root: StorageDir, header: dict, settings: LeaseSettings
    ):
        self.path = path
        self._root = root
        self.header = header
        self.pool_id: str = header["pool_id"]
        self.segment_items: int = header["segment_items"]
        self.settings = settings
        self._lease_settings = attrs.evolve(settings, holder="anonymous")
        self._queues: dict[str, _Queue] = {}
        self._labels: dict[str, str] = {}
        self.monotonic: ty.Callable[[], float] = time.monotonic  # claims' local clock

    # -- opening -------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        root: os.PathLike,
        settings: LeaseSettings,
        *,
        create: bool = True,
        probe: bool = True,
        codec: str = PRIVATE_PICKLE,
        segment_items: int = SEGMENT_ITEMS,
    ) -> "PoolRoot":
        """Open the pool at ``root``, creating it (``0o700``) when ``create`` and it has no
        ``pool.json``.

        Parameters
        ----------
        root : os.PathLike
        settings : LeaseSettings
            This process's lease settings; compared with the recorded ones on all four staleness
            fields (``max_wait`` and ``holder`` are not compared).
        create : bool
        probe : bool
            Probe the storage (`pydra.workers.pool.probe.probe`) before creating the pool, refuse
            it on a missing primitive, and record the result in ``pool.json``.
        codec : str
            The payload codec recorded at creation.
        segment_items : int
            The slots per segment recorded at creation; an existing pool keeps its own.

        Raises
        ------
        PoolSettingsMismatch
            ``settings`` differ from the recorded ones.
        PoolOwnerMismatch
            Another user owns the pool.
        PoolStorageUnsupported
            The probe refused the storage.
        StoreUnavailable
            The root can be written by group or others.
        """
        path = Path(root).absolute()
        if not path.exists():
            if not create:
                raise FileNotFoundError(errno.ENOENT, "no pool here", str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(FileExistsError):
                os.mkdir(path, DIR_MODE)
                if not IS_WINDOWS:
                    os.chmod(path, DIR_MODE)
        handle = StorageDir.open_root(path, POLICY, trust=_TRUST)
        try:
            data = _read(handle, HEADER)
            if data is None:
                if not create:
                    raise FileNotFoundError(errno.ENOENT, "no pool.json", str(path))
                found = None
                if probe:
                    from pydra.workers.pool import probe as probing

                    result = probing.probe(path, limit=settings.attribute_delay + 1.0)
                    probing.require(result, path, settings.attribute_delay)
                    found = result.to_json()
                header = {
                    "format": POOL_FORMAT,
                    "pool_id": uuid.uuid4().hex[:12],
                    "lease": settings.to_json(),
                    "codec": codec,
                    "owner_uid": None if IS_WINDOWS else os.geteuid(),
                    "segment_items": segment_items,
                    "probe": found,
                    "uuid": uuid.uuid4().hex,
                }
                _create(handle, HEADER, jcs(header))
                data = _read(handle, HEADER)
            header = _record(data)
            if header is None or header.get("format") != POOL_FORMAT:
                raise StoreUnavailable(
                    errno.EINVAL, f"{HEADER} is not a {POOL_FORMAT} header", str(path)
                )
            cls._check(header, settings, path)
        except BaseException:
            handle.close()
            raise
        return cls(path, handle, header, settings)

    @staticmethod
    def _check(header: dict, settings: LeaseSettings, path: Path) -> None:
        uid = header.get("owner_uid")
        if uid is not None and not IS_WINDOWS and uid != os.geteuid():
            raise PoolOwnerMismatch(
                errno.EPERM,
                f"the pool belongs to uid {uid}; a pool is single-user",
                str(path),
            )
        recorded = header.get("lease") or {}
        mine = settings.to_json()
        differ = [k for k in mine if recorded.get(k) != mine[k]]
        if differ:
            shown = ", ".join(
                f"{k} {mine[k]!r} (pool: {recorded.get(k)!r})" for k in differ
            )
            raise PoolSettingsMismatch(
                f"lease settings differ from the pool's at {path}: {shown}", differ
            )

    def close(self) -> None:
        """Close every directory this pool holds open."""
        for queue in self._queues.values():
            queue.close()
        self._queues.clear()
        self._root.close()

    def __enter__(self) -> "PoolRoot":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PoolRoot({str(self.path)!r}, pool_id={self.pool_id!r})"

    @property
    def lease_settings(self) -> LeaseSettings:
        """The settings of every lease taken on this pool (anonymous holders)."""
        return self._lease_settings

    # -- queues and tickets --------------------------------------------------------------

    def queue(self, name: str) -> _Queue:
        """The queue ``name``, made if missing."""
        if name not in self._queues:
            self._queues[name] = _Queue(self, name)
            self._labels[self._queues[name].label] = name
        return self._queues[name]

    def _by_label(self, label: str) -> _Queue | None:
        """The queue whose directory is ``label``, None when there is none or it is unreadable."""
        if not isinstance(label, str) or not FINGERPRINT.fullmatch(label):
            return None
        if label in self._labels:
            return self._queues[self._labels[label]]
        with _dir(self._root, "queues") as queues:
            qdir = _child(queues, label)
            if qdir is None:
                return None
            with qdir:
                data = _read(qdir, "name")
        if data is None or queue_label(data.decode()) != label:
            return None
        return self.queue(data.decode())

    def ticket(self, queue: str, digest: str) -> Ticket | None:
        """The ticket ``digest`` of ``queue``, None when it is absent.

        Raises
        ------
        CorruptTicket
            The file is not the ticket its name digests.
        """
        return self._ticket(self.queue(queue), digest)

    def _ticket(self, q: _Queue, digest: str) -> Ticket | None:
        if digest in q.cached:
            return q.cached[digest]
        data = _read(q.tickets, f"{digest}.json")
        if data is None:
            return None
        if hashlib.sha256(data).hexdigest() != digest:
            raise CorruptTicket(f"ticket {digest} of {q.name!r} is not its digest's")
        try:
            ticket = Ticket.from_json(json.loads(data))
        except ValueError as err:
            raise CorruptTicket(f"ticket {digest} of {q.name!r}: {err}") from err
        q.cached[digest] = ticket
        return ticket

    def put(self, queue: str, ticket: Ticket) -> str:
        """Put ``ticket`` on ``queue``: the ticket file, a generation per job, then its slot.

        Each job's attempt is that of its next generation. A put is idempotent: jobs already held by
        the very ticket this put would write (an earlier put of the same work, or one interrupted)
        are not placed again, and a missing slot is written.

        Returns
        -------
        str
            The ticket's digest.

        Raises
        ------
        JobHeld
            A job is held by another live ticket; the coordinator follows it instead.
        ValueError
            ``queue`` is not ``ticket.queue``.
        """
        if ticket.queue != queue:
            raise ValueError(f"a ticket of queue {ticket.queue!r} is put on {queue!r}")
        q = self.queue(queue)
        held: dict[JobKey, str] = {}
        jobs = []
        for job in ticket.jobs:
            newest = self.newest(*job.key)
            holder = self._holder(newest, job.key)
            if holder is not None:
                held[job.key] = holder
                jobs.append(attrs.evolve(job, attempt=newest[1]["attempt"]))
            else:
                g = newest[0] + 1 if newest else 0
                jobs.append(attrs.evolve(job, attempt=f"{self.pool_id}.{g}"))
        candidate = attrs.evolve(ticket, jobs=jobs)
        digest = candidate.digest
        for job, holder in held.items():
            if holder != digest:
                raise JobHeld(job, holder)
        _create(q.tickets, f"{digest}.json", candidate.canonical())
        for job in candidate.jobs:
            if job.key not in held:
                self._allocate(job.key, ticket=digest, queue=q.label, cause="put")
        self._ensure_slot(q, digest, len(candidate.jobs))
        return digest

    # -- the index -----------------------------------------------------------------------

    def _ensure_slot(self, q: _Queue, digest: str, count: int) -> None:
        if _read(q.slotted, digest) is not None:
            return
        seq, n = self._take_slot(q, digest, count)
        _create(q.slotted, digest, f"{seq}/{n}".encode())

    def _take_slot(self, q: _Queue, digest: str, count: int) -> tuple[int, int]:
        """Make the next free slot of the newest segment name ``digest``, rolling to a new segment
        when it is full. A slot is only ever made below ``segment_items``, so a segment whose
        slots are all taken stays full and can be retired."""
        seq = q.head or max(
            (int(n) for n in _names(q.segments) if _SEQ.fullmatch(n)), default=0
        )
        while True:
            with _dir(q.segments, _seq(seq)) as segment:
                slots, _, _ = _slots(_names(segment))
                n = slots[-1] + 1 if slots else 0
                while n < self.segment_items:
                    if _create(segment, f"slot-{n}", digest.encode()) != "exists":
                        _create(segment, f"jobs-{n}-{count}", b"")
                        q.head = seq
                        return seq, n
                    n += 1
            seq += 1

    def _slot_of(self, q: _Queue, digest: str) -> tuple[int, int] | None:
        data = _read(q.slotted, digest)
        if data is None:
            return None
        seq, _, n = data.decode().partition("/")
        return int(seq), int(n)

    def _segments(
        self, q: _Queue
    ) -> ty.Iterator[tuple[int, StorageDir, list[int], set, dict[int, int]]]:
        """Each live segment, open, oldest first from the oldest this process saw live, with its
        slots, the resolved ones and each slot's job count; a full segment whose slots are all
        resolved is retired on the way."""
        seq, prefix = q.oldest, True
        while True:
            segment = _child(q.segments, _seq(seq))
            if segment is None:
                return
            with segment:
                names = _names(segment)
                live = "retired" not in names
                if live:
                    slots, resolved, jobs = _slots(names)
                    full = len(slots) >= self.segment_items
                    live = not (full and resolved.issuperset(slots))
                    if not live:
                        _create(segment, "retired", b"")
                if live:
                    prefix = False
                    yield seq, segment, slots, resolved, jobs
                elif prefix:
                    q.oldest = seq + 1
                    q.targets = {k: v for k, v in q.targets.items() if k[0] > seq}
                    q.cached = {
                        d: t for d, t in q.cached.items() if d in q.targets.values()
                    }
            seq += 1

    def _target(self, q: _Queue, segment: StorageDir, seq: int, n: int) -> str | None:
        if (seq, n) not in q.targets:
            data = _read(segment, f"slot-{n}")
            target = data.decode() if data else ""
            if not FINGERPRINT.fullmatch(target):
                return None
            q.targets[(seq, n)] = target
        return q.targets[(seq, n)]

    def oldest(self, queue: str) -> float | None:
        """The storage mtime of the oldest slot of ``queue`` not marked resolved, None when there
        is none: how long its oldest waiting ticket has waited (RFC 0005 section 4.2).
        """
        q = self.queue(queue)
        walk = self._segments(q)
        with contextlib.closing(walk):
            for _, segment, slots, resolved, _ in walk:
                for n in slots:
                    if n in resolved:
                        continue
                    try:
                        return segment.lstat(f"slot-{n}").st_mtime
                    except OSError as err:
                        if not _absent(err):
                            raise
        return None

    def queues(self) -> list[str]:
        """The names of the queues made on this pool."""
        with _dir(self._root, "queues") as queues:
            labels = _names(queues)
        found = []
        for label in labels:
            q = self._by_label(label)
            if q is not None:
                found.append(q.name)
        return sorted(found)

    def depth(self, queue: str) -> int:
        """An upper bound on the tickets of ``queue`` without a final outcome for every job."""
        return sum(1 for _ in self._waiting(queue))

    def work(self, queue: str) -> int:
        """An upper bound on the jobs of the tickets `depth` counts."""
        return sum(self._waiting(queue))

    def _waiting(self, queue: str) -> ty.Iterator[int]:
        q = self.queue(queue)
        for name in sorted(n for n in _names(q.segments) if _SEQ.fullmatch(n)):
            segment = _child(q.segments, name)
            if segment is None:
                continue
            with segment:
                names = _names(segment)
                if "retired" in names:
                    continue
                slots, resolved, jobs = _slots(names)
                for n in slots:
                    if n not in resolved:
                        yield jobs.get(n, 1)

    # -- follow generations --------------------------------------------------------------

    def _follow(self, job: JobKey, create: bool = False) -> StorageDir | None:
        name = job_name(*job)
        with _dir(self._root, "follow") as follow:
            return _dir(follow, name) if create else _child(follow, name)

    def generations(self, checksum: str, fingerprint: str) -> list[tuple[int, dict]]:
        """Every readable generation of a job, oldest first."""
        d = self._follow((checksum, fingerprint))
        if d is None:
            return []
        with d:
            found = []
            for g in _generation_numbers(d):
                record = _read_record(d, str(g))
                if record is not None:
                    found.append((g, record))
            return found

    def newest(
        self, checksum: str, fingerprint: str, *, hint: int | None = None
    ) -> tuple[int, dict] | None:
        """The newest readable generation of a job: the ticket that holds it and its attempt.

        A listing may be stale (an attribute cache) or empty (``ESTALE``), so the generations above
        the highest listed one, or above ``hint`` (a generation known to exist), are looked up by
        name until one is absent. None means unknown: nothing was listed or readable.
        """
        d = self._follow((checksum, fingerprint))
        if d is None:
            return None
        with d:
            numbers = _generation_numbers(d)
            top = max(numbers[-1] if numbers else -1, -1 if hint is None else hint)
            while _read(d, str(top + 1)) is not None:
                top += 1
            for g in range(top, -1, -1):
                record = _read_record(d, str(g))
                if record is not None:
                    return g, record
        return None

    def _generation(self, job: JobKey, g: int) -> dict | None:
        """Generation ``g`` of a job, read by name."""
        d = self._follow(job)
        if d is None:
            return None
        with d:
            return _read_record(d, str(g))

    def started(self, checksum: str, fingerprint: str, g: int) -> dict | None:
        """The started mark of a job's attempt ``g``: which worker started it."""
        d = self._follow((checksum, fingerprint))
        if d is None:
            return None
        with d:
            return _read_record(d, f"{g}.started")

    def drained(self, checksum: str, fingerprint: str, g: int) -> bool:
        """Whether a job's attempt ``g`` was cut short by its worker's drain (`Claim.drained`)."""
        d = self._follow((checksum, fingerprint))
        if d is None:
            return False
        with d:
            return _read_record(d, f"{g}.drained") is not None

    def charges(self, checksum: str, fingerprint: str) -> dict[str, int]:
        """A job's crash count (generations with ``charged``), its ``resource_moves``
        (``ladder`` generations at the ``call`` phase) and its infrastructure requeues
        (``restart`` generations at the ``call`` or ``load`` phase, `Claim.restart`)."""
        numbered = self.generations(checksum, fingerprint)
        gens = [r for _, r in numbered]
        return {
            "preemptions": sum(
                1 for g, _ in numbered if self.drained(checksum, fingerprint, g)
            ),
            "crashes": sum(1 for r in gens if r.get("charged") is True),
            "infrastructure": sum(
                1
                for r in gens
                if r.get("cause") == "restart" and r.get("phase") in ("call", "load")
            ),
            "resource_moves": sum(
                1
                for r in gens
                if r.get("cause") == "ladder" and r.get("phase") == "call"
            ),
        }

    def _allocate(
        self,
        job: JobKey,
        *,
        ticket: str,
        queue: str,
        cause: str,
        phase: str | None = None,
        prior: str | None = None,
        charged: bool = False,
    ) -> tuple[int, dict]:
        """Write a job's next generation by exclusive create, one more than the highest listed; on
        ``EEXIST`` list again and take the next number. Its attempt is ``<pool id>.<g>``.
        """
        if cause not in CAUSES or phase not in PHASES:
            raise ValueError(f"no generation cause {cause!r} at phase {phase!r}")
        with self._follow(job, create=True) as d:
            numbers = _generation_numbers(d)
            g = numbers[-1] + 1 if numbers else 0
            while True:
                record = {
                    "ticket": ticket,
                    "queue": queue,
                    "attempt": f"{self.pool_id}.{g}",
                    "cause": cause,
                    "phase": phase,
                    "prior": prior,
                    "charged": bool(charged),
                    "uuid": uuid.uuid4().hex,
                }
                if _create(d, str(g), jcs(record)) != "exists":
                    return g, record
                numbers = _generation_numbers(d)
                g = max(g + 1, numbers[-1] + 1 if numbers else 0)

    def _next_generation(self, job: JobKey) -> int:
        """One more than the highest generation listed for a job (0 when it has none)."""
        d = self._follow(job)
        if d is None:
            return 0
        with d:
            numbers = _generation_numbers(d)
        return numbers[-1] + 1 if numbers else 0

    def _holder(self, newest: tuple[int, dict] | None, job: JobKey) -> str | None:
        """The ticket holding a job whose newest generation is ``newest``: None when it has none,
        when that ticket was collected, or when the job has a final outcome there."""
        if newest is None:
            return None
        _, record = newest
        q = self._by_label(record.get("queue", ""))
        digest = record.get("ticket", "")
        if q is None or not FINGERPRINT.fullmatch(digest):
            return None
        if _read(q.tickets, f"{digest}.json") is None:
            return None
        found = self._read_outcome(q, digest, job, record["attempt"])
        if found is not None and found.get("outcome") in FINAL:
            return None
        return digest

    def follow(self, checksum: str, fingerprint: str) -> str | None:
        """The digest of the live ticket that holds a job, None when nothing does: the job has no
        generation, its newest names a collected ticket, or it has a final outcome there.
        """
        job = (checksum, fingerprint)
        return self._holder(self.newest(*job), job)

    def collect(self, payload_age: float = 86400.0) -> int:
        """Remove each settled job's generations and started marks except the highest generation
        and its mark, which keep numbering from restarting, and each payload no unsettled ticket
        names that is older than ``payload_age`` seconds (a payload is written before the ticket
        that names it); return how many files went.
        """
        removed = self._collect_payloads(payload_age)
        with _dir(self._root, "follow") as follow:
            for name in _names(follow):
                d = _child(follow, name)
                if d is None:
                    continue
                with d:
                    numbers = _generation_numbers(d)
                    if len(numbers) < 2:
                        continue
                    newest = _read_record(d, str(numbers[-1]))
                    if newest is None:
                        continue
                    job = self._job_of(newest, name)
                    if (
                        job is None
                        or self._holder((numbers[-1], newest), job) is not None
                    ):
                        continue
                    for g in numbers[:-1]:
                        for member in (f"{g}.started", f"{g}.drained", str(g)):
                            with contextlib.suppress(FileNotFoundError):
                                d.unlink(member)
                                removed += 1
        return removed

    def collect_settled(self, age: float = 86400.0) -> int:
        """Remove each ticket marked resolved more than ``age`` seconds ago, with its settled marks,
        outcomes, losses, claim lease and ``slotted`` entry (RFC 0005 section 4.1), so that a queue's
        listings stay as long as its live tickets. Its slot stays in the index (a segment is walked by
        its slot numbers); its jobs' newest generations stay, naming a ticket that is gone, which
        holds nothing. Return how many tickets went."""
        import shutil

        removed = 0
        now = time.time()
        with _dir(self._root, "queues") as queues:
            labels = _names(queues)
        for label in labels:
            q = self._by_label(label)
            if q is None:
                continue
            marks = _names(q.settled)
            for name in _names(q.tickets):
                digest = name[: -len(".json")] if name.endswith(".json") else ""
                if not FINGERPRINT.fullmatch(digest):
                    continue
                at = self._resolved_at(q, digest)
                if at is None or now - at < age:
                    continue
                for mark in marks:
                    if mark.startswith(f"{digest}-"):
                        with contextlib.suppress(OSError):
                            q.settled.unlink(mark)
                for sub in ("outcome", "losses"):
                    shutil.rmtree(q.path / sub / digest, ignore_errors=True)
                shutil.rmtree(q.path / "claims" / f"{digest}.lease", ignore_errors=True)
                for d, member in ((q.slotted, digest), (q.tickets, name)):
                    with contextlib.suppress(FileNotFoundError):
                        d.unlink(member)
                q.cached.pop(digest, None)
                removed += 1
        return removed

    def _resolved_at(self, q: _Queue, digest: str) -> float | None:
        """When ticket ``digest``'s slot was marked resolved (its ``res-<n>`` mtime), None if not."""
        slot = self._slot_of(q, digest)
        if slot is None:
            return None
        seq, n = slot
        segment = _child(q.segments, _seq(seq))
        if segment is None:
            return None
        with segment:
            try:
                return segment.lstat(f"res-{n}").st_mtime
            except OSError as err:
                if not _absent(err):
                    raise
                return None

    def _collect_payloads(self, age: float) -> int:
        payloads = _child(self._root, "payloads")
        if payloads is None:
            return 0
        with payloads:
            names = [n for n in _names(payloads) if FINGERPRINT.fullmatch(n)]
            if not names:
                return 0
            live = self._live_payloads()
            now = time.time()
            removed = 0
            for name in names:
                if name in live:
                    continue
                try:
                    if now - payloads.lstat(name).st_mtime < age:
                        continue
                    payloads.unlink(name)
                    removed += 1
                except OSError as err:
                    if not _absent(err):
                        raise
            return removed

    def _live_payloads(self) -> set[str]:
        """The payload names every ticket not marked resolved names (a ticket without a slot, or
        unreadable, keeps what it might name)."""
        live: set[str] = set()
        keep_all = False
        with _dir(self._root, "queues") as queues:
            labels = _names(queues)
        for label in labels:
            q = self._by_label(label)
            if q is None:
                continue
            for name in _names(q.tickets):
                digest = name[: -len(".json")] if name.endswith(".json") else ""
                if not FINGERPRINT.fullmatch(digest):
                    continue
                if self._resolved(q, digest):
                    continue
                try:
                    ticket = self._ticket(q, digest)
                except CorruptTicket:
                    ticket = None
                if ticket is None:
                    keep_all = True
                    continue
                for job in ticket.jobs:
                    path = job.payload.get("path")
                    if isinstance(path, str):
                        live.add(os.path.basename(path))
        if keep_all:
            live.add("*")
            return {"*"} | live | set(self._all_payloads())
        return live

    def _all_payloads(self) -> list[str]:
        payloads = _child(self._root, "payloads")
        if payloads is None:
            return []
        with payloads:
            return _names(payloads)

    def _resolved(self, q: _Queue, digest: str) -> bool:
        slot = self._slot_of(q, digest)
        if slot is None:
            return False
        seq, n = slot
        segment = _child(q.segments, _seq(seq))
        if segment is None:
            return False
        with segment:
            return _read(segment, f"res-{n}") is not None

    def _job_of(self, record: dict, name: str) -> JobKey | None:
        """The job a follow directory ``name`` is for, from the ticket its generation names."""
        q = self._by_label(record.get("queue", ""))
        if q is None:
            return None
        try:
            ticket = self._ticket(q, record.get("ticket", ""))
        except CorruptTicket:
            return None
        if ticket is None:
            return None
        for job in ticket.jobs:
            if job_name(*job.key) == name:
                return job.key
        return None

    # -- outcomes and settled marks ------------------------------------------------------

    def outcome(
        self,
        queue: str,
        digest: str,
        job: "TicketJob | JobKey",
        attempt: str,
        record: dict,
        *,
        lease: DirectoryLease | None = None,
        settle: bool = True,
    ) -> bool | None:
        """Write a job's outcome under ``attempt`` in ticket ``digest``, by exclusive create; the
        first writer wins.

        Parameters
        ----------
        record : dict
            ``{"outcome": "done" | "failed" | "held" | "withdrawn" | "expired", ...}``; a
            ``withdrawn`` one carries ``reason`` and, when the job moved, ``by``.
        lease : DirectoryLease, optional
            Proved after staging and before the link; nothing is written when it fails.
        settle : bool
            Write the ticket's settled mark after (`settle`).

        Returns
        -------
        bool or None
            True when the record is this write's (made now, or before a lost reply), False when
            another record stands, None when the proof failed.
        """
        q = self.queue(queue)
        return self._write_outcome(q, digest, _key(job), attempt, record, lease, settle)

    def _write_outcome(
        self,
        q: _Queue,
        digest: str,
        job: JobKey,
        attempt: str,
        record: dict,
        lease: DirectoryLease | None,
        settle: bool,
    ) -> bool | None:
        kind = record.get("outcome")
        if kind not in OUTCOMES:
            raise ValueError(f"an outcome is one of {OUTCOMES}, got {kind!r}")
        if kind == "withdrawn" and lease is None:
            raise ValueError(
                "a withdrawal is written only under the ticket's claim lease (RFC 0005 5.5): "
                "take it with PoolRoot.take and withdraw through the Claim"
            )
        if kind == "withdrawn" and record.get("reason") not in WITHDRAW_REASONS:
            raise ValueError(
                f"a withdrawal's reason is one of {WITHDRAW_REASONS}, got "
                f"{record.get('reason')!r}"
            )
        full = {
            **record,
            "format": OUTCOME_FORMAT,
            "checksum": job[0],
            "fingerprint": job[1],
            "attempt": attempt,
            "ticket": digest,
            "uuid": uuid.uuid4().hex,
        }
        with _dir(q.outcome, digest) as d:
            made = _create(
                d,
                _outcome_name(job, attempt),
                jcs(full),
                guard=lease.prove if lease is not None else None,
            )
        if made is None:
            return None
        if settle:
            self._settle(q, digest)
        return made != "exists"

    def read_outcome(
        self, queue: str, digest: str, job: "TicketJob | JobKey", attempt: str
    ) -> dict | None:
        """A job's outcome record under ``attempt`` in ticket ``digest``; None while it has none."""
        return self._read_outcome(self.queue(queue), digest, _key(job), attempt)

    def _read_outcome(
        self, q: _Queue, digest: str, job: JobKey, attempt: str
    ) -> dict | None:
        d = _child(q.outcome, digest)
        if d is None:
            return None
        with d:
            return _read_record(d, _outcome_name(job, attempt))

    def outcomes(
        self,
        queue: str,
        digest: str,
        *,
        expect: int | None = None,
        within: float | None = None,
    ) -> dict[str, dict]:
        """Every readable outcome record of ticket ``digest``, by file name.

        Parameters
        ----------
        expect : int, optional
            The count a settled mark showed. The directory is listed again, backing off, until that
            many records are read, so a listing cached behind the mark is not taken for the outcomes.
        within : float, optional
            Seconds to keep listing; default ``attribute_delay + renew_interval``, at least 1 s.

        Raises
        ------
        ListingBehind
            Fewer than ``expect`` records could be read within ``within``.
        """
        if within is None:
            within = max(
                1.0, self.settings.attribute_delay + self.settings.renew_interval
            )
        for _ in OUTCOME_LISTING.tries(within):
            found = self._outcomes(queue, digest)
            if expect is None or len(found) >= expect:
                return found
        raise ListingBehind(
            f"ticket {digest} of {queue!r} lists {len(found)} outcomes after {within:g} s, "
            f"its settled mark {expect}",
            found,
            expect,
        )

    def _outcomes(self, queue: str, digest: str) -> dict[str, dict]:
        q = self.queue(queue)
        d = _child(q.outcome, digest)
        if d is None:
            return {}
        with d:
            found = {}
            for name in _names(d):
                if name.endswith(".json"):
                    record = _read_record(d, name)
                    if record is not None:
                        found[name] = record
            return found

    def settle(self, queue: str, digest: str) -> int:
        """Write ``settled/<digest>-<count>`` for the ticket's outcome count; return the count marked.

        On ``EEXIST`` the writer counts again: an unchanged count means the mark it met covers its
        own outcome; a larger one is marked in turn, so the last count is always written.
        """
        return self._settle(self.queue(queue), digest)

    def _settle(self, q: _Queue, digest: str) -> int:
        d = _child(q.outcome, digest)
        try:
            count = _count_outcomes(d)
            while count:
                if _create(q.settled, f"{digest}-{count}", b"") != "exists":
                    # the follower reads the highest: one mark per ticket bounds its listing
                    if count > 1:
                        with contextlib.suppress(OSError):
                            q.settled.unlink(f"{digest}-{count - 1}")
                    return count
                again = _count_outcomes(d)
                if again <= count:
                    return count
                count = again
            return 0
        finally:
            if d is not None:
                d.close()

    def settled(self, queue: str) -> dict[str, int]:
        """The highest settled count of each ticket of ``queue``: one listing."""
        found: dict[str, int] = {}
        for name in _names(self.queue(queue).settled):
            if m := _SETTLED.fullmatch(name):
                found[m.group(1)] = max(found.get(m.group(1), 0), int(m.group(2)))
        return found

    def losses(self, queue: str, digest: str) -> list[dict]:
        """The takeovers of ticket ``digest``, in order."""
        d = _child(self.queue(queue).losses, digest)
        if d is None:
            return []
        with d:
            numbers = sorted(int(n) for n in _names(d) if _GEN.fullmatch(n))
            return [r for n in numbers if (r := _read_record(d, str(n))) is not None]

    def _record_loss(self, q: _Queue, digest: str, broke: list[dict]) -> int:
        lost = [
            {"token": (b.get("holder") or {}).get("token"), "age": b.get("age")}
            for b in broke
        ]
        with _dir(q.losses, digest) as d:
            numbers = sorted(int(n) for n in _names(d) if _GEN.fullmatch(n))
            n = numbers[-1] + 1 if numbers else 0
            while True:
                record = {"broke": lost, "uuid": uuid.uuid4().hex}
                if _create(d, str(n), jcs(record)) != "exists":
                    return n
                n += 1

    # -- worker classes and workers ------------------------------------------------------

    def put_class(self, name: str, record: dict) -> None:
        """Write ``classes/<name>.json``, the worker class as JCS, once; an existing one stays."""
        queue_label(name)
        with _dir(self._root, "classes") as classes:
            _create(classes, f"{name}.json", jcs(record))

    def worker_class(self, name: str) -> dict | None:
        """The record of worker class ``name``, None when the pool has none."""
        queue_label(name)
        with _dir(self._root, "classes") as classes:
            return _read_record(classes, f"{name}.json")

    def worker_dir(self, token: str) -> StorageDir:
        """``workers/<token>/``, made if missing."""
        if not _HEX32.fullmatch(token):
            raise ValueError(f"a worker token is 32 hex digits, not {token!r}")
        with _dir(self._root, "workers") as workers:
            return _dir(workers, token)

    def write_worker_record(self, token: str, name: str, record: dict) -> None:
        """Write the worker's record ``name`` (``ready.json``) once."""
        with self.worker_dir(token) as d:
            _create(d, name, jcs(record))

    def worker_record(self, token: str, name: str) -> dict | None:
        """The worker's record ``name``, None when absent."""
        with _dir(self._root, "workers") as workers:
            d = _child(workers, token)
            if d is None:
                return None
            with d:
                return _read_record(d, name)

    def write_call(self, token: str, seq: int, record: dict) -> None:
        """Write the call record ``workers/<token>/calls/<seq>.json`` by exclusive create, before
        the call runs (RFC 0005 section 4.3 step 4)."""
        with self.worker_dir(token) as w, _dir(w, "calls") as calls:
            if _create(calls, f"{_seq(seq)}.json", jcs(record)) == "exists":
                raise FileExistsError(
                    errno.EEXIST, "a call record is written once", seq
                )

    def call_done(self, token: str, seq: int) -> None:
        """Write ``<seq>.done`` once the call returned."""
        with self.worker_dir(token) as w, _dir(w, "calls") as calls:
            _create(calls, f"{_seq(seq)}.done", jcs({"uuid": uuid.uuid4().hex}))

    def calls(self, token: str) -> list[tuple[int, dict, bool]]:
        """Each readable call record of a worker, oldest first, with whether it is done."""
        with _dir(self._root, "workers") as workers:
            w = _child(workers, token)
            if w is None:
                return []
            with w:
                d = _child(w, "calls")
                if d is None:
                    return []
                with d:
                    names = set(_names(d))
                    found = []
                    for name in sorted(names):
                        stem, dot, ext = name.partition(".")
                        if ext != "json" or not _SEQ.fullmatch(stem):
                            continue
                        record = _read_record(d, name)
                        if record is not None:
                            done = (
                                f"{stem}.done" in names
                                or _read(d, f"{stem}.done") is not None
                            )
                            found.append((int(stem), record, done))
                    return found

    def in_flight(self, token: str) -> dict | None:
        """A worker's newest call record without ``done``: the call it was running when it died."""
        for _, record, done in reversed(self.calls(token)):
            return None if done else record
        return None

    # -- holds and coordinator epochs ----------------------------------------------------

    def hold(self, queue: str, epoch: int, label: str) -> None:
        """Hold the residency ``label`` on ``queue`` for the coordinator of ``epoch``: its tickets
        are not claimed while the hold counts (for that epoch or an older one)."""
        if not isinstance(epoch, int) or epoch < 1 or not FINGERPRINT.fullmatch(label):
            raise ValueError(
                f"a hold names an epoch from 1 and a residency label: {epoch}, {label}"
            )
        _create(self.queue(queue).dir, f"held-{epoch}-{label}", b"")

    def infrastructure_fault(
        self, queue: str, epoch: int, label: str, record: dict
    ) -> int:
        """Record an infrastructure fault of the residency ``label`` on ``queue`` in coordinator
        epoch ``epoch`` (``faults/<epoch>-<label>/<n>``, by exclusive create); return how many that
        residency has had in that epoch, this one included (RFC 0005 section 6)."""
        if not isinstance(epoch, int) or epoch < 1 or not FINGERPRINT.fullmatch(label):
            raise ValueError(
                f"a fault names an epoch from 1 and a residency label: {epoch}, {label}"
            )
        with _dir(self.queue(queue).dir, "faults") as faults:
            with _dir(faults, f"{epoch}-{label}") as d:
                numbers = sorted(int(n) for n in _names(d) if _GEN.fullmatch(n))
                n = numbers[-1] + 1 if numbers else 0
                while True:
                    data = jcs({**record, "uuid": uuid.uuid4().hex})
                    if _create(d, str(n), data) != "exists":
                        return n + 1
                    n += 1

    def holds(self, queue: str, epoch: int | None = None) -> dict[str, int]:
        """Each residency label held on ``queue`` by ``epoch`` (default: the newest epoch) or a newer
        one, with the newest such epoch."""
        current = self._current(epoch)
        found: dict[str, int] = {}
        for name in _names(self.queue(queue).dir):
            if (m := _HELD.fullmatch(name)) and int(m.group(1)) >= current:
                found[m.group(2)] = max(found.get(m.group(2), 0), int(m.group(1)))
        return found

    def release_holds(self, queue: str, label: str | None = None) -> int:
        """Lift the holds of ``queue`` (of the residency ``label`` only, when given), whatever their
        epoch; return how many were lifted (``pydracli pool release-hold``)."""
        q = self.queue(queue)
        lifted = 0
        for name in _names(q.dir):
            m = _HELD.fullmatch(name)
            if m and (label is None or m.group(2) == label):
                with contextlib.suppress(FileNotFoundError):
                    q.dir.unlink(name)
                    lifted += 1
        return lifted

    def _current(self, epoch: int | None) -> int:
        if epoch is not None:
            return epoch
        epochs = self.epochs()
        return epochs[-1] if epochs else 0

    def epochs(self) -> list[int]:
        """The coordinator epochs taken on this pool, ascending."""
        with _dir(self._root, "coordinator") as coordinator:
            return sorted(int(n) for n in _names(coordinator) if _EPOCH.fullmatch(n))

    def take_epoch(self) -> int:
        """Take the next coordinator epoch by ``mkdir``; never one taken before."""
        with _dir(self._root, "coordinator") as coordinator:
            epochs = sorted(int(n) for n in _names(coordinator) if _EPOCH.fullmatch(n))
            epoch = epochs[-1] + 1 if epochs else 1
            while True:
                path = self.path / "coordinator" / str(epoch)
                try:
                    os.mkdir(path, DIR_MODE)
                except FileExistsError:
                    epoch += 1
                    continue
                if not IS_WINDOWS:
                    os.chmod(path, DIR_MODE)
                coordinator.sync()
                return epoch

    def fenced(self, epoch: int) -> bool:
        """Whether a newer coordinator epoch than ``epoch`` was taken."""
        epochs = self.epochs()
        return bool(epochs) and epochs[-1] > epoch

    def end_epoch(self, epoch: int) -> None:
        """Write epoch ``epoch``'s ``ended`` mark."""
        with _dir(self._root, "coordinator") as coordinator:
            with coordinator.child(str(epoch)) as d:
                _create(d, "ended", b"")

    def write_faults(self, epoch: int, record: dict) -> None:
        """Write coordinator epoch ``epoch``'s fault budgets and classifier reference,
        ``coordinator/<epoch>/faults.json``, once (RFC 0005 section 6): every worker reads them
        (`faults`), whoever started it."""
        with _dir(self._root, "coordinator") as coordinator:
            with coordinator.child(str(epoch)) as d:
                _create(d, "faults.json", jcs(record))

    def faults(self, epoch: int | None = None) -> tuple[int, dict] | None:
        """The newest coordinator epoch that wrote a fault record (only ``epoch``, when given), and
        the record; None when none did."""
        with _dir(self._root, "coordinator") as coordinator:
            for epoch in [epoch] if epoch is not None else reversed(self.epochs()):
                d = _child(coordinator, str(epoch))
                if d is None:
                    continue
                with d:
                    record = _read_record(d, "faults.json")
                if record is not None:
                    return epoch, record
        return None

    def coordinator_lease(
        self, epoch: int, *, beat: ty.Literal["thread", "external"] = "thread"
    ) -> DirectoryLease:
        """The lease of the coordinator of ``epoch``, untaken."""
        return DirectoryLease.at(
            self.path / "coordinator" / str(epoch) / "lease",
            self._lease_settings,
            beat=beat,
        )

    # -- claims --------------------------------------------------------------------------

    def claim(
        self,
        queue: str,
        *,
        beat: ty.Literal["thread", "external"] = "thread",
        epoch: int | None = None,
        max_jobs: int | None = None,
        max_time: float | None = None,
    ) -> "Claim | None":
        """Claim the oldest claimable ticket of ``queue``; None when there is none.

        Slots are walked oldest first from the oldest live segment. A ticket under a live claim is
        passed over (and not judged again for one renewal interval), as is a missing or corrupt
        one, and one whose residency is held (`holds` for ``epoch``). A stale claim is taken over,
        and the takeover recorded in ``losses/``. A ticket whose jobs all have a final outcome, or
        moved, is marked resolved and passed over; a moved job's new ticket gets its slot if it has
        none.

        Parameters
        ----------
        beat : "thread" or "external"
            Who beats the claim lease (`DirectoryLease.at`). A worker with a beater passes
            ``"external"`` and hands ``claim.lease`` to its `BeaterClient` at once.
        epoch : int, optional
            The coordinator epoch whose holds count; default the newest.
        max_jobs : int, optional
            Pass over tickets of more jobs than this (a warm worker gathering one-job tickets).
        max_time : float, optional
            Pass over tickets declaring a longer time than this, in seconds (the time left in
            the claimant's allocation).
        """
        q = self.queue(queue)
        # judged ticket by ticket: a class's queue may carry several residencies (a
        # residency="ignore" job shares it with plain ones), so one held residency never skips
        # it; a routed queue (``<class>#<digest>``) carries one, so any hold there skips it
        held = self.holds(queue, epoch)
        if held and "#" in queue:
            return None
        now = self.monotonic()
        interval = self.settings.renew_interval
        q.judged = {d: at for d, at in q.judged.items() if now - at < interval}
        q.missing = {d: at for d, at in q.missing.items() if now - at < interval}
        walk = self._segments(q)
        with contextlib.closing(walk):
            for seq, segment, slots, resolved, counts in walk:
                for n in slots:
                    if n in resolved:
                        continue
                    if max_jobs is not None and counts.get(n, 1) > max_jobs:
                        continue
                    digest = self._target(q, segment, seq, n)
                    if digest is None or digest in q.judged or digest in q.missing:
                        continue
                    ticket = self._claimable(q, digest)
                    if ticket is None or (held and ticket.residency_label in held):
                        continue
                    if max_time is not None and (ticket.time or 0) > max_time:
                        continue
                    claim = self._take(q, digest, ticket, (seq, n), beat)
                    if claim is not None:
                        return claim
        return None

    def _claimable(self, q: _Queue, digest: str) -> Ticket | None:
        try:
            ticket = self._ticket(q, digest)
        except CorruptTicket as err:
            q.faults.append((digest, f"corrupt: {err}"))
            q.missing[digest] = self.monotonic()
            return None
        if ticket is None:
            q.faults.append((digest, "missing"))
            q.missing[digest] = self.monotonic()
        return ticket

    def take(
        self,
        queue: str,
        digest: str,
        *,
        beat: ty.Literal["thread", "external"] = "thread",
    ) -> "Claim | None":
        """Take the claim of ticket ``digest`` by one `try_take`: None when its claim is live, the
        ticket is absent, or all its jobs have a final outcome. A coordinator withdraws a job (cancel,
        a re-placement on resume) only under such a claim."""
        q = self.queue(queue)
        ticket = self._ticket(q, digest)
        if ticket is None:
            return None
        return self._take(q, digest, ticket, self._slot_of(q, digest), beat)

    def _take(
        self,
        q: _Queue,
        digest: str,
        ticket: Ticket,
        slot: tuple[int, int] | None,
        beat: ty.Literal["thread", "external"],
    ) -> "Claim | None":
        lease = DirectoryLease.at(
            q.path / "claims" / f"{digest}.lease",
            self._lease_settings,
            beat=beat,
            monotonic=self.monotonic,
            refused_since=q.refused.get(digest),
        )
        try:
            wait = lease.try_take()
        except LeaseTimeout:
            wait = 0.0
        if lease.refusing_since is None:
            q.refused.pop(digest, None)
        else:
            q.refused[digest] = lease.refusing_since
        if wait is not None:
            q.judged[digest] = self.monotonic()
            return None
        claim = Claim(self, q, digest, ticket, lease, slot)
        try:
            if lease.broke:
                claim.loss = self._record_loss(q, digest, lease.broke)
            self._settle(q, digest)  # an outcome whose writer died before its mark
            for job in ticket.jobs:
                claim._state(job, repair=True)
            if claim.resolved():
                claim.release()
                return None
        except BaseException:
            lease.__exit__(None, None, None)
            raise
        return claim

    def _repair(self, record: dict) -> None:
        """Give the ticket a generation names its slot, if a move died before writing it."""
        q = self._by_label(record.get("queue", ""))
        digest = record.get("ticket", "")
        if q is None or not FINGERPRINT.fullmatch(digest):
            return
        if _read(q.slotted, digest) is not None:
            return
        try:
            ticket = self._ticket(q, digest)
        except CorruptTicket:
            return
        if ticket is not None:
            self._ensure_slot(q, digest, len(ticket.jobs))


class Claim:
    """A ticket held under its claim lease, made by `PoolRoot.claim` or `PoolRoot.take`.

    Attributes
    ----------
    digest : str
    ticket : Ticket
    lease : DirectoryLease
        The claim lease, held. For ``beat="external"`` the holder's beater beats it; release it
        from the beater before `release`.
    queue : str
    slot : tuple of int or None
        ``(segment, slot)`` the ticket was claimed from.
    loss : int or None
        The number of the ``losses/`` record this claim wrote when it took a stale claim over.
    """

    def __init__(
        self,
        pool: PoolRoot,
        q: _Queue,
        digest: str,
        ticket: Ticket,
        lease: DirectoryLease,
        slot: tuple[int, int] | None,
    ):
        self.pool = pool
        self._q = q
        self.queue = q.name
        self.digest = digest
        self.ticket = ticket
        self.lease = lease
        self.slot = slot
        self.loss: int | None = None

    @property
    def took_over(self) -> bool:
        """Whether taking this claim took a stale one over."""
        return self.loss is not None

    def __repr__(self) -> str:
        return f"Claim({self.queue!r}, {self.digest})"

    # -- reading -------------------------------------------------------------------------

    def _state(
        self, job: TicketJob, repair: bool = False
    ) -> tuple[str, int | None, dict | None]:
        """What became of a job of this ticket, on positive evidence only.

        ``("open", g, generation)``: its newest generation names this ticket and it has no outcome
        under that attempt. The outcome's kind with ``(g, outcome record)``: it has one here. When a
        newer generation names another ticket, the job's outcome here under this ticket's attempt
        decides; without a final one it is ``"moved"``, and with ``repair`` (a claimant) the move's
        withdrawal here is written first, naming the new ticket, and its slot if it has none.
        ``"unknown"``: no generation could be read; never taken as moved.
        """
        newest = self.pool.newest(*job.key, hint=_generation_of(job.attempt))
        if newest is None:
            return "unknown", None, None
        g, record = newest
        if record.get("ticket") == self.digest:
            found = self.pool._read_outcome(
                self._q, self.digest, job.key, record["attempt"]
            )
            if found is not None:
                return str(found.get("outcome")), g, found
            return "open", g, record
        mine = self._own_attempt(job, g)
        attempt = mine if mine is not None else job.attempt
        found = None
        if attempt is not None:
            found = self.pool._read_outcome(self._q, self.digest, job.key, attempt)
        if found is not None and found.get("outcome") in FINAL:
            return str(found.get("outcome")), g, found
        # without a generation of its own here, the ticket's own attempt
        if repair and attempt is not None:
            self.pool._repair(record)
            reason = "split" if record.get("cause") == "split" else "replaced"
            self._withdraw(job, attempt, reason, record.get("ticket"))
            found = self.pool._read_outcome(self._q, self.digest, job.key, attempt)
            if found is not None and found.get("outcome") in FINAL:
                return str(found.get("outcome")), g, found
        return "moved", g, record

    def _own_attempt(self, job: TicketJob, above: int) -> str | None:
        """The attempt of the job's newest generation below ``above`` that names this ticket, read
        by name; None when none can be read."""
        for g in range(above - 1, -1, -1):
            record = self.pool._generation(job.key, g)
            if record is not None and record.get("ticket") == self.digest:
                return record.get("attempt")
        return None

    def current(self, job: TicketJob) -> tuple[int, dict] | None:
        """The job's newest generation when it names this ticket: its attempt is the job's."""
        newest = self.pool.newest(*job.key, hint=_generation_of(job.attempt))
        if newest is None or newest[1].get("ticket") != self.digest:
            return None
        return newest

    def resolved(self) -> bool:
        """Whether every job of the ticket has a final outcome written in this ticket."""
        return all(self._state(job)[0] in FINAL for job in self.ticket.jobs)

    def open_jobs(self) -> list[TicketJob]:
        """The jobs this ticket holds that have no outcome under their attempt."""
        return [j for j in self.ticket.jobs if self._state(j)[0] == "open"]

    # -- running -------------------------------------------------------------------------

    def start(self, job: TicketJob, worker: str) -> str | None:
        """Mark the job's attempt started and return it; None to skip the job (its newest
        generation names another ticket, it has an outcome under that attempt, or no generation can
        be read).

        The attempt is read from the newest generation, never from the ticket file. When that
        attempt's started mark exists, the attempt has run once already, so a ``restart``
        generation naming this ticket is written first and its attempt started instead.
        """
        state, g, record = self._state(job)
        if state != "open":
            return None
        with self.pool._follow(job.key, create=True) as d:
            while True:
                mark = jcs({"worker": worker, "uuid": uuid.uuid4().hex})
                if _create(d, f"{g}.started", mark) != "exists":
                    return record["attempt"]
                g, record = self.pool._allocate(
                    job.key,
                    ticket=self.digest,
                    queue=self._q.label,
                    cause="restart",
                    prior=record["attempt"],
                )

    def restart(self, job: TicketJob, *, phase: str = "call") -> str:
        """Write the job's next generation, ``restart`` at ``phase`` naming this ticket, before an
        infrastructure fault releases it, so the requeue is counted (`PoolRoot.charges`); return
        the new attempt."""
        current = self.current(job)
        if current is None:
            raise ValueError(f"job {job.checksum} is not held by ticket {self.digest}")
        _, record = self.pool._allocate(
            job.key,
            ticket=self.digest,
            queue=self._q.label,
            cause="restart",
            phase=phase,
            prior=current[1]["attempt"],
        )
        return record["attempt"]

    def outcome(
        self, job: TicketJob, attempt: str, kind: str, **fields: ty.Any
    ) -> bool | None:
        """Write the job's outcome ``kind`` under ``attempt``, proving the claim lease after staging
        and before the link, then the ticket's settled mark.

        Returns
        -------
        bool or None
            True: written (or a lost reply of this write); False: another outcome stands; None: the
            claim was lost and nothing was written.
        """
        return self.pool._write_outcome(
            self._q,
            self.digest,
            job.key,
            attempt,
            {"outcome": kind, **fields},
            self.lease,
            True,
        )

    def withdraw(
        self, job: TicketJob, reason: str, *, by: str | None = None
    ) -> bool | None:
        """Withdraw a job from this ticket under its current attempt (the attempt of its newest
        generation naming this ticket), with ``reason`` and, when it moved, ``by``."""
        if reason not in WITHDRAW_REASONS:
            raise ValueError(f"a withdrawal's reason is one of {WITHDRAW_REASONS}")
        current = self.current(job)
        if current is None:
            raise ValueError(f"job {job.checksum} is not held by ticket {self.digest}")
        return self._withdraw(job, current[1]["attempt"], reason, by)

    def _withdraw(
        self, job: TicketJob, attempt: str, reason: str, by: str | None
    ) -> bool | None:
        fields: dict[str, ty.Any] = {"reason": reason}
        if by is not None:
            fields["by"] = by
        return self.outcome(job, attempt, "withdrawn", **fields)

    def drained(self, job: TicketJob, worker: str) -> None:
        """Mark the job's current attempt cut short by this worker's drain (its scheduler preempted
        or stopped it): its taker places it again uncharged, within the preemption budget.
        """
        current = self.current(job)
        if current is None:
            return
        with self.pool._follow(job.key, create=True) as d:
            _create(d, f"{current[0]}.drained", jcs({"worker": worker}))

    def takeover(self, job: TicketJob, *, charged: bool = False) -> str:
        """Place a lost job again in this ticket under a fresh ``takeover`` generation, ``prior`` the
        lost attempt; return the new attempt. The caller looks the job up in the store first.
        """
        current = self.current(job)
        if current is None:
            raise ValueError(f"job {job.checksum} is not held by ticket {self.digest}")
        _, record = self.pool._allocate(
            job.key,
            ticket=self.digest,
            queue=self._q.label,
            cause="takeover",
            prior=current[1]["attempt"],
            charged=charged,
        )
        return record["attempt"]

    # -- moving --------------------------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        """Called after each of the five steps of a move."""

    def move(
        self,
        groups: ty.Sequence[ty.Sequence[TicketJob]],
        queue: str,
        *,
        cause: str,
        phase: str | None = None,
        ladder: ty.Sequence[str] | None = None,
        charged: bool | ty.Mapping[JobKey, bool] = False,
        reason: str | None = None,
    ) -> list[str]:
        """Move jobs of this ticket into new tickets on ``queue``, one per group, in the order that
        keeps every job held (RFC 0005 section 5.5): (1) the new ticket files, with no slot;
        (2) each job's next generation naming its new ticket, with a fresh attempt; (3) the new
        tickets' slots; (4) the ``withdrawn`` outcomes here, under each job's attempt before the
        move, naming the new ticket in ``by``. Step (5), releasing this claim, is the caller's.

        Parameters
        ----------
        groups : sequence of sequence of TicketJob
            The jobs of each new ticket; each job must be held by this ticket.
        queue : str
            The queue of the new tickets.
        cause : {"replaced", "split", "ladder"}
        phase : {"call", "load", None}
            For a ``ladder`` move, where the fault happened.
        ladder : sequence of str, optional
            The new tickets' ladder; default this ticket's.
        charged : bool or mapping
            Whether each job's new generation is charged as a crash.
        reason : str, optional
            The withdrawal's reason; default ``"split"`` for a split, else ``"replaced"``.

        Returns
        -------
        list of str
            The new tickets' digests, in the order of ``groups``.
        """
        if cause not in ("replaced", "split", "ladder"):
            raise ValueError(
                f"a move's cause is replaced, split or ladder, not {cause!r}"
            )
        reason = reason or ("split" if cause == "split" else "replaced")
        if reason not in WITHDRAW_REASONS:
            raise ValueError(f"a withdrawal's reason is one of {WITHDRAW_REASONS}")
        target = self.pool.queue(queue)
        before: dict[JobKey, str] = {}
        for group in groups:
            if not group:
                raise ValueError("a new ticket holds at least one job")
            for job in group:
                current = self.current(job)
                if current is None:
                    raise ValueError(f"job {job.checksum} is not held by {self.digest}")
                before[job.key] = current[1]["attempt"]
        made: list[tuple[str, Ticket]] = []
        for group in groups:  # (1)
            jobs = []
            for job in group:
                g = self.pool._next_generation(job.key)
                jobs.append(attrs.evolve(job, attempt=f"{self.pool.pool_id}.{g}"))
            new = attrs.evolve(
                self.ticket,
                queue=queue,
                jobs=jobs,
                ladder=self.ticket.ladder if ladder is None else tuple(ladder),
            )
            _create(target.tickets, f"{new.digest}.json", new.canonical())
            made.append((new.digest, new))
        self._checkpoint(1)
        for digest, new in made:  # (2)
            for job in new.jobs:
                self.pool._allocate(
                    job.key,
                    ticket=digest,
                    queue=target.label,
                    cause=cause,
                    phase=phase,
                    prior=before[job.key],
                    charged=(
                        charged.get(job.key, False)
                        if isinstance(charged, ty.Mapping)
                        else charged
                    ),
                )
        self._checkpoint(2)
        for digest, new in made:  # (3)
            self.pool._ensure_slot(target, digest, len(new.jobs))
        self._checkpoint(3)
        for digest, new in made:  # (4)
            for job in new.jobs:
                self._withdraw(job, before[job.key], reason, digest)
        self._checkpoint(4)
        self._checkpoint(5)
        return [digest for digest, _ in made]

    def split(self, *, charged: bool | ty.Mapping[JobKey, bool] = False) -> list[str]:
        """Split this ticket's open jobs into two half-tickets on its queue (cause ``split``), each
        job under a fresh generation; release this claim after.

        Raises
        ------
        ValueError
            Fewer than two jobs are open.
        """
        jobs = self.open_jobs()
        if len(jobs) < 2:
            raise ValueError(
                f"ticket {self.digest} has {len(jobs)} open jobs: nothing to split"
            )
        half = (len(jobs) + 1) // 2
        return self.move(
            [jobs[:half], jobs[half:]], self.queue, cause="split", charged=charged
        )

    # -- releasing -----------------------------------------------------------------------

    def release(self) -> None:
        """Write the ticket's settled mark for its outcome count, mark its slot resolved when every
        job has a final outcome in this ticket, then release the claim lease. Releasing again does
        nothing. The ticket leaves the queue's judged set, so that this process (which passed over
        it while holding it) may claim it again at once."""
        if self.lease.token_dir is None:
            return
        self._q.judged.pop(self.digest, None)
        try:
            if self.lease.held:
                self.pool._settle(self._q, self.digest)
            if self.slot is not None and self.lease.held and self.resolved():
                seq, n = self.slot
                segment = _child(self._q.segments, _seq(seq))
                if segment is not None:
                    with segment:
                        if self.pool._target(self._q, segment, seq, n) == self.digest:
                            _create(segment, f"res-{n}", b"")
        finally:
            self.lease.__exit__(None, None, None)

    def __enter__(self) -> "Claim":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
