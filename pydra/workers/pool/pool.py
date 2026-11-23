"""`WarmPool`: the executor that runs a runner's jobs on warm workers (RFC 0005 sections 4.1, 4.2,
4.5, 5.5, 6 and 7).

It places each job on a worker class and queue at `submit` (a job that fits no class is reported
``Failed``), writes tickets of at most ``quantum_jobs`` jobs for the pool's workers, follows their
outcomes and reports them, and scales each class's workers through its substrate. One coordinator
is live per pool: a new one waits up to ``coordinator_wait`` (then `PoolBusy`), and one that sees a
newer epoch reports its open jobs ``NotRun`` (`PoolFenced`). A class whose workers keep failing to
start is broken and its jobs held until `release_hold`. A pool outlives the runs driven on it;
whoever made it closes it. It refuses a store whose identity would split slots by pool facts
(`PoolFactsWouldSplitSlots`).
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
import queue as queuelib
import threading
import time
import types
import typing as ty
import uuid
import warnings
from pathlib import Path

import attrs

from pydra.engine.executor import Handle
from pydra.engine.lease import LeaseTimeout
from pydra.engine.runner import Failed, NotRun
from pydra.utils.digest import jcs
from pydra.utils.general import is_workflow
from pydra.utils.messenger import AuditFlag
from pydra.utils.resources import ComputeResources
from pydra.workers.pool.calibration import (
    DirectoryCalibration,
    ParityRecord,
    residency_key,
)
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.codec import (
    JobCodec,
    PrivatePickle,
    encode_payload,
    store_payload,
)
from pydra.workers.pool import faults as _faults
from pydra.workers.pool.faults import Classifier, FaultBudgets
from pydra.workers.pool.fit import MeasuredFit
from pydra.workers.pool.placement import (
    ClassTried,
    DeclaredFit,
    Fit,
    Placement,
    Unplaceable,
    _candidate_reason,
    _order,
    fit_request,
    place,
)
from pydra.workers.pool.settings import PoolSettings
from pydra.utils.storagedir import StoreUnavailable
from pydra.workers.pool.storage import (
    CorruptTicket,
    JobHeld,
    ListingBehind,
    STOP_MARK,
    PoolRoot,
    Ticket,
    TicketJob,
    _outcome_name,
)
from pydra.workers.pool.substrate import (
    Attachment,
    ExternalSubstrate,
    Substrate,
    SubstrateError,
    WorkerHandle,
)

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.runner import Outcome

__all__ = [
    "PoolBusy",
    "PoolFactsWouldSplitSlots",
    "PoolFenced",
    "PoolJobFailed",
    "PoolSettings",
    "UnplaceableJob",
    "WarmPool",
]

logger = logging.getLogger("pydra.pool")

JobKey = tuple[str, str]


class PoolBusy(RuntimeError):
    """Another coordinator's lease on the pool stayed live for ``coordinator_wait`` seconds."""


class PoolFactsWouldSplitSlots(ValueError):
    """The store's identity would key the ``pool`` section of the execute context: every worker and
    every batch size would split one result into many slots (RFC 0005 section 8)."""


POOL_FACTS_SECTION = "pool"
"""The execute-context section a worker adds (``pool-facts@1``)."""


def pool_facts_recorded(store: ty.Any) -> bool:
    """Whether the pool facts enter the execute context of a job published to ``store``: True when its
    identity lists ``pool`` in ``recorded_only``; False under a store with no identity (`PickleStore`),
    whose facts reach the audit record and ``Result.runtime`` only.

    Raises
    ------
    PoolFactsWouldSplitSlots
        An identity that does not list ``pool`` in ``recorded_only``: it would key the facts.
    """
    identity = store.identity
    if identity is None:
        return False
    if POOL_FACTS_SECTION in identity.recorded_only:
        return True
    name = identity.name
    raise PoolFactsWouldSplitSlots(
        f"the store's identity {name!r} does not list {POOL_FACTS_SECTION!r} in recorded_only, so "
        "the pool facts of each worker and batch would split one result into many slots; list it "
        "in the identity's recorded_only"
    )


class PoolFenced(RuntimeError):
    """A newer coordinator epoch was taken on the pool; this coordinator writes nothing more."""


class PoolJobFailed(RuntimeError):
    """A job the pool settled ``failed`` without an errored result of its own (a crash, a resource
    or infrastructure fault, or a corrupt payload).

    Attributes
    ----------
    fault : dict
        The fault record of the outcome.
    """

    def __init__(self, message: str, fault: dict):
        super().__init__(message)
        self.fault = fault

    def __reduce__(self):
        return (type(self), (self.args[0], self.fault))


class UnplaceableJob(RuntimeError):
    """A job that fits no worker class of the pool.

    Attributes
    ----------
    unplaceable : Unplaceable
        Every class tried, with its verdict and numbers.
    """

    def __init__(self, unplaceable: Unplaceable):
        super().__init__(unplaceable.message)
        self.unplaceable = unplaceable

    def __reduce__(self):
        return (type(self), (self.unplaceable,))


def _never_reusable(job: "Job", lease: ty.Any, looked_up: bool = False) -> None:
    """`Job._reusable` of a cold re-check: it runs, never serving the entry it verifies against."""
    return None


@attrs.define(eq=False)
class _Job:
    """A job this coordinator was handed and has not reported."""

    job: "Job"
    handle: Handle | None
    key: JobKey
    queue: str  # where it is (or will be) placed
    ladder: tuple[str, ...]
    residency: tuple[dict, ...]
    ticket: "_Ticket | None" = (
        None  # the ticket that holds it, once written or followed
    )
    # where its outcome goes: the binding of the run that submitted it (None: the latest binding)
    report: "ty.Callable[[Job, Outcome], None] | None" = None
    # jobs of other runs with its key, submitted while it was open: they settle with it
    twins: "list[_Job]" = attrs.field(factory=list)
    twin_of: "_Job | None" = None


def ticket_time(jobs: ty.Iterable["Job"]) -> float | None:
    """The longest time ``jobs`` declare (``resources.time``); None when none declares one."""
    times = [
        j.resources.time
        for j in jobs
        if j.resources is not None and j.resources.time is not None
    ]
    return max(times, default=None)


@attrs.define(eq=False)
class _Ticket:
    """A ticket some open job is held by: what the follower has seen of it."""

    queue: str
    digest: str
    jobs: dict[JobKey, _Job] = attrs.field(factory=dict)
    seen: int = 0
    moved: float = attrs.field(factory=time.monotonic)
    judged: float = 0.0
    recheck: bool = False
    foreign: bool = False  # of a class this coordinator does not start


def _queue_class(queue: str) -> str:
    return queue.split("#", 1)[0]


def _reference(obj: ty.Any) -> str:
    if isinstance(obj, str):
        return obj
    kind = obj if isinstance(obj, type) else type(obj)
    return f"{kind.__module__}:{kind.__qualname__}"


def _label(residency: ty.Sequence[dict]) -> str:
    """A residency's label, as `Ticket.residency_label` computes it."""
    return hashlib.sha256(jcs(list(residency))).hexdigest()


def _classifier_reference(obj: ty.Any) -> str:
    """The ``module:qualname`` a worker imports the classifier ``obj`` by: a module-level function, or
    a class it makes with no arguments, or such a reference itself.

    Raises
    ------
    ValueError
        ``obj`` does not round-trip through its reference (a ``functools.partial``, a bound method,
        a lambda, a nested function, an instance), or the reference does not resolve to a callable.
    """
    from pydra.workers.pool.worker import _resolve

    if isinstance(obj, str):
        reference = obj
        try:
            resolved = _resolve(reference)
        except Exception as err:
            raise ValueError(
                f"the classifier {reference!r} cannot be imported: {err!r}"
            ) from err
        if not callable(resolved):
            raise ValueError(f"the classifier {reference!r} is not callable")
        return reference
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise ValueError(
            f"the classifier {obj!r} has no importable name: workers import it by "
            "module:qualname; pass a module-level function or class"
        )
    reference = f"{module}:{qualname}"
    try:
        found: ty.Any = importlib.import_module(module)
        for part in qualname.split("."):
            found = getattr(found, part)
    except Exception:
        found = None
    if found is not obj:
        raise ValueError(
            f"the classifier {obj!r} does not round-trip through {reference!r} (a partial, a "
            "bound method, a lambda or a nested function cannot reach a worker); pass a "
            "module-level function or class"
        )
    return reference


def _sampled(checksum: str, fraction: float) -> bool:
    """Whether the job of ``checksum`` is re-checked under ``recheck_fraction``: the same answer for
    a checksum every time, and for about ``fraction`` of checksums."""
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    return (
        int(hashlib.sha256(checksum.encode()).hexdigest()[:16], 16) < fraction * 2**64
    )


def _fingerprint(job: "Job") -> str:
    identity = job.store.identity
    if identity is not None:
        return identity.fingerprint(job.context)
    return hashlib.sha256(jcs(job.context)).hexdigest()


_RUN: contextvars.ContextVar[tuple[object, object] | None] = contextvars.ContextVar(
    "pydra_pool_run", default=None
)
"""The run a `bind` was made in: ``(token, owner)``, the owner being the asyncio task or thread that
set it. Each task and thread has its own copy of the context, so two runs on one event loop are told
apart; a copy inherited from a parent carries its token to code that does not bind (a hop to a
thread), and is replaced by the first `bind` of its own."""


def _owner() -> object:
    """The asyncio task running this code, or its thread outside one."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return task if task is not None else threading.current_thread()


def _run_token(create: bool = False) -> object | None:
    """This context's run token. With ``create`` (a `bind`), the token is this task's or thread's
    own: one inherited from a parent context is replaced by a fresh one."""
    held = _RUN.get()
    if not create:
        return None if held is None else held[0]
    owner = _owner()
    if held is not None and held[1] is owner:
        return held[0]
    token = object()
    _RUN.set((token, owner))
    return token


def _newest_mtime(lease: Path) -> float | None:
    """The newest mtime of a lease directory, its entries and theirs (a beater's heartbeats);
    None when it is missing."""
    try:
        newest = os.lstat(lease).st_mtime
        for entry in os.scandir(lease):
            newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
            if entry.is_dir(follow_symlinks=False):
                for sub in os.scandir(entry.path):
                    newest = max(newest, sub.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return None
    return newest


class WarmPool:
    """An executor (RFC 0002 section 3.2) that runs jobs on warm workers of a pool.

    Parameters
    ----------
    root : Path
        The pool directory, on storage every worker and this process see; per pool, not per run.
    classes : sequence of WorkerClass
        The classes this coordinator places jobs on and starts workers of.
    substrate : Substrate
        Starts the workers (`LocalSubstrate`, or `ExternalSubstrate` for workers started by hand).
    store : ResultStore
        The store the jobs publish to; the follower reads results back through ``Job.result()``.
    fit : Fit
        Judges a job's request against a class (default `DeclaredFit`).
    calibration : Calibration, optional
        Where measurements go; None: `DirectoryCalibration` at ``root / "calibration"``. A
        `DirectoryCalibration`'s directory is where the workers write samples and unfit
        records; any other `Calibration` receives only the coordinator's `ParityRecord`s,
        with a warning, as does a `MeasuredFit` whose calibration is not the workers'.
    providers : sequence of Provider or "module:qualname"
        Providers the workers this pool starts load before the jobs' own.
    faults : FaultBudgets
        Written into the pool for this coordinator's epoch, where every worker reads them, whoever
        started it; ``start_failures`` is this coordinator's.
    classify : callable
        ``classify(exc, phase) -> FaultRecord``, written into the pool as a ``module:qualname``
        reference (a module-level function, or a class made with no arguments), which every worker
        imports: it must be importable where the workers run.
    settings : PoolSettings
        Written for the workers this pool starts; its lease settings must equal the pool's.
    codec : JobCodec
        How a job travels to a worker (default `PrivatePickle`).
    probe : bool
        Probe the storage when the pool is created.

    Raises
    ------
    PoolBusy
        Another coordinator stayed live for ``settings.coordinator_wait`` seconds.
    PoolFactsWouldSplitSlots
        The store's identity would key the ``pool`` section of the execute context.

    Notes
    -----
    Several runs may share a pool, one after another or at once (two `Submitter`s on two threads,
    or two ``drive`` calls on one event loop): each job's outcome goes to the run that submitted it
    (`bind` is per run, that is per asyncio task or thread, through a context variable), a job
    another run already has open here settles with it, and a run's ``cancel`` touches only its own
    jobs. All runs share its workers and their resident state. A twin of a job run with
    ``rerun=True`` settles with the other run's attempt rather than computing again.
    """

    checks_store: ty.ClassVar[bool] = True
    # its workers run jobs side by side: the driver never feeds it one at a time
    serial: ty.ClassVar[bool] = False
    inline: ty.ClassVar[bool] = False
    recheck_queue: ty.ClassVar[int] = 64
    """Re-checks waiting for the re-check thread; one more is dropped and counted."""

    def __init__(
        self,
        root: os.PathLike,
        classes: ty.Sequence[WorkerClass],
        substrate: Substrate,
        *,
        store: ty.Any,
        fit: Fit = DeclaredFit(),
        calibration: ty.Any = None,
        providers: ty.Sequence[ty.Any] = (),
        faults: FaultBudgets = FaultBudgets(),
        classify: Classifier = _faults.classify,
        settings: PoolSettings = PoolSettings(),
        codec: JobCodec = PrivatePickle(),
        probe: bool = True,
    ):
        pool_facts_recorded(store)  # before the pool is touched
        self.batches = True
        classify_ref = (
            None if classify is _faults.classify else _classifier_reference(classify)
        )
        self.root = Path(root).absolute()
        self.classes = tuple(classes)
        names = [c.name for c in self.classes]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"a pool needs classes with distinct names, got {names}")
        self._class = {c.name: c for c in self.classes}
        self.substrate = substrate
        self.store = store
        self.fit = fit
        self.calibration = (
            calibration
            if calibration is not None
            else DirectoryCalibration(self.root / "calibration")
        )
        written = (
            self.calibration.root
            if isinstance(self.calibration, DirectoryCalibration)
            else self.root / "calibration"
        )
        if not isinstance(self.calibration, DirectoryCalibration):
            warnings.warn(
                f"workers write their samples and unfit records to {written}, not to "
                f"{self.calibration!r}, which receives only the re-checks' parity records",
                UserWarning,
                stacklevel=2,
            )
        read = (
            fit.calibration.root
            if isinstance(fit, MeasuredFit)
            and isinstance(fit.calibration, DirectoryCalibration)
            else None
        )
        if read is not None and Path(read).absolute() != Path(written).absolute():
            warnings.warn(
                f"the fit reads the calibration at {read}, but workers write to {written}: "
                "no out-of-memory verdict of this pool reaches the fit",
                UserWarning,
                stacklevel=2,
            )
        self._calibration_root = Path(written).absolute()
        self.providers = tuple(_reference(p) for p in providers)
        self._provider_objects: tuple | None = None
        self._parity_reported: set[str] = set()
        self._recheck_queue: queuelib.Queue = queuelib.Queue(maxsize=self.recheck_queue)
        self._recheck_thread: threading.Thread | None = None
        self._recheck_stop = threading.Event()
        self.rechecks_dropped = 0
        self.faults = faults
        self.classify = classify
        self.settings = settings
        self.codec = codec
        # each run's binding, by its run token (`_RUN`): the report of the jobs it submits
        self._reports: dict[object, ty.Callable[["Job", "Outcome"], None]] = {}
        self._by_key: dict[JobKey, _Job] = {}  # the open job followed for each key
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # ordered sets: a settle removes its job in constant time, whatever the backlog
        self._waiting: dict[str, dict[_Job, None]] = {}
        self._unchecked: dict[_Job, None] = {}
        self._tickets: dict[tuple[str, str], _Ticket] = {}
        self._queues: dict[str, set[tuple[str, str]]] = {}
        self._open: dict["Job", _Job] = {}
        self._started: dict[str, WorkerHandle] = {}
        self._ready: set[str] = set()
        self._stopped: set[str] = set()
        self._failures: dict[str, int] = collections.defaultdict(int)
        self._failed_at: dict[str, float] = {}
        # class -> when its starts began failing in a row
        self._failing_since: dict[str, float] = {}
        # queue -> when its ticket writes began failing in a row, how many, and the latest error
        self._write_failing: dict[str, tuple[float, int, str]] = {}
        self._cause: dict[str, str] = {}
        self._broken: dict[str, str] = {}
        # class -> the jobs held because it broke: a retry of one of them lifts the break
        self._break_held: dict[str, set[JobKey]] = collections.defaultdict(set)
        self._listing_failures: dict[str, int] = collections.defaultdict(int)
        # when the substrate was last listed (or, never listed, first failed to be), and the
        # classes a ticket of which settled since then
        self._listed_at: float | None = None
        self._progress: dict[str, float] = (
            {}
        )  # class -> when a ticket of it last settled
        # token -> (newest mtime of its worker lease, when the follower saw it change)
        self._lease_seen: dict[str, tuple[float | None, float]] = {}
        # (queue, residency label) a worker held in this epoch, and why
        self._held: dict[tuple[str, str], str] = {}
        self._idle: dict[str, int] = collections.defaultdict(int)
        self._fenced: str | None = None
        self._closed = False
        self.epoch: int | None = None
        self._lease = None
        substrate.check_root(self.root)  # before the pool's tree is made
        self.pool = PoolRoot.open(
            self.root, settings.lease, create=True, probe=probe, codec=codec.name
        )
        try:
            from pydra.workers.pool.worker import class_record

            for cls in self.classes:
                self.pool.put_class(cls.name, class_record(cls))
            private = self.root / "private"
            private.mkdir(mode=0o700, exist_ok=True)
            self.settings_path = private / f"settings-{uuid.uuid4().hex}.json"
            fd = os.open(
                self.settings_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(fd, "w") as f:
                json.dump(settings.to_json(), f)
            codec_ref = None if isinstance(codec, PrivatePickle) else _reference(codec)
            attachment = Attachment(
                self.root,
                self.settings_path,
                self.providers,
                codec_ref,
                pool_id=self.pool.pool_id,
                classes=self.classes,
                pool_settings=settings,
            )
            substrate.attach(attachment)
            self._await_coordinators()
            # the epoch is taken after every fallible step
            self.epoch = self.pool.take_epoch()
            attachment.epoch = self.epoch
            # every worker reads the budgets and the classifier from the pool, however started
            self.pool.write_faults(
                self.epoch,
                {
                    "budgets": faults.to_json(),
                    "classify": classify_ref,
                    "calibration": str(self._calibration_root),
                },
            )
            lease = self.pool.coordinator_lease(self.epoch)
            lease.__enter__()
            self._lease = lease
        except BaseException:
            self.pool.close()
            raise
        logger.info("coordinator epoch %d of the pool at %s", self.epoch, self.root)

    def __repr__(self) -> str:
        return f"WarmPool({str(self.root)!r}, epoch={self.epoch})"

    # -- one live coordinator ------------------------------------------------------------------

    def _live_coordinator(self) -> tuple[int, dict | None] | None:
        """The newest coordinator epoch whose lease is live, with its holder; None when none is."""
        for epoch in reversed(self.pool.epochs()):
            if (self.root / "coordinator" / str(epoch) / "ended").exists():
                continue
            lease = self.pool.coordinator_lease(epoch, beat="external")
            try:
                wait = lease.try_take()
            except LeaseTimeout:
                wait = 0.0
            if wait is None:  # free or stale: not live
                lease.__exit__(None, None, None)
                continue
            return epoch, lease.holder()
        return None

    def _await_coordinators(self) -> None:
        began, logged = time.monotonic(), False
        while True:
            live = self._live_coordinator()
            if live is None:
                return
            epoch, holder = live
            if not logged:
                logger.warning(
                    "coordinator epoch %d of %s is live (%s): waiting",
                    epoch,
                    self.root,
                    holder,
                )
                logged = True
            limit = self.settings.coordinator_wait
            if limit is not None and time.monotonic() - began >= limit:
                raise PoolBusy(
                    f"coordinator epoch {epoch} of the pool at {self.root} is live "
                    f"(holder {holder}) after {limit:g} s; a run that must proceed at the "
                    "same time uses its own pool"
                )
            time.sleep(min(self.settings.poll_s, self.settings.lease.renew_interval))

    # -- the executor protocol -----------------------------------------------------------------

    def bind(self, report: ty.Callable[["Job", "Outcome"], None]) -> None:
        """Set the callable that receives the outcomes (from the follower thread) of the jobs this
        run submits from here on. A run is the context it binds in (`contextvars`: each asyncio
        task and each thread has its own), so runs on other tasks or threads keep their own.
        """
        with self._lock:
            self._reports[_run_token(create=True)] = report

    def unbind(self, report: ty.Callable[["Job", "Outcome"], None]) -> None:
        """Forget the binding of the run calling it, if it is ``report``; its jobs still open are
        reported to it."""
        with self._lock:
            token = _run_token()
            if token is not None and self._reports.get(token) is report:
                del self._reports[token]

    def _sole_report(self) -> "ty.Callable[[Job, Outcome], None] | None":
        """The binding when exactly one run is bound, else None."""
        if len(self._reports) == 1:
            return next(iter(self._reports.values()))
        return None

    def submit(self, jobs: ty.Sequence["Job"]) -> Handle:
        """Place the jobs and queue them for the follower to write as tickets; never blocks on
        capacity. A job that fits no class is reported ``Failed`` before this returns.
        """
        with self._lock:
            token = _run_token()
            report = self._reports.get(token) if token is not None else None
            if report is None:
                report = self._sole_report()
            bound = len(self._reports)
        if report is None:
            raise RuntimeError(
                "bind() a report callable in this run (task or thread) before submitting jobs: "
                f"{bound} runs are bound, none here, so the outcomes would have nowhere to go"
            )
        handle = Handle(self, tuple(jobs), set(jobs))
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self!r} is closed")
            if self._fenced is not None:
                reports = [(j, None, NotRun(self._fenced, held=True)) for j in jobs]
            else:
                reports = self._queue_jobs(jobs, handle, report)
        self._deliver(reports, handle, report)
        self._ensure_thread()
        self._wake.set()
        return handle

    def cancel(self, handle: Handle) -> None:
        """Withdraw the handle's unsettled jobs (reason ``cancel``) from every ticket whose claim this
        process can take (unclaimed or stale); a ticket under a live claim is left to its holder,
        whose results stay in the store. Every cancelled job is reported ``NotRun(held=False)``.
        """
        with self._lock:
            found = [self._open[j] for j in handle.jobs if j in self._open]
            cancelled: list[_Job] = []
            by_ticket: dict[tuple[str, str], list[_Job]] = {}
            for j in found:
                gone, withdraw = self._cancel_one(j)
                cancelled.append(gone)
                if withdraw:
                    by_ticket.setdefault(
                        (gone.ticket.queue, gone.ticket.digest), []
                    ).append(gone)
                gone.ticket = None
        for (queue, digest), group in by_ticket.items():
            try:
                claim = self.pool.take(queue, digest)
            except Exception as err:  # the storage: left to the ticket's holder
                logger.warning("cancelling on ticket %s failed: %s", digest, err)
                continue
            if claim is None:
                continue
            try:
                entries = {e.key: e for e in claim.ticket.jobs}
                for j in group:
                    entry = entries.get(j.key)
                    if entry is not None and claim.current(entry) is not None:
                        claim.withdraw(entry, "cancel")
            finally:
                claim.release()
        self._deliver(
            [(j.job, j, NotRun("cancelled", held=False)) for j in cancelled], handle
        )

    def close(self) -> None:
        """Stop following, write this coordinator's ``ended`` mark and, when it still holds the
        newest epoch, stop the workers the substrate lists; release the coordinator lease.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        self._stop_rechecks()
        try:
            if self.epoch is not None:
                self.pool.end_epoch(self.epoch)
                if self._fenced is None and not self.pool.fenced(self.epoch):
                    try:
                        live = list(self.substrate.live())
                    except Exception as err:  # stop what this coordinator started
                        logger.warning(
                            "listing the substrate's workers failed at close: %s", err
                        )
                        live = list(self._started.values())
                    if live:
                        self._stop_workers(live)
        finally:
            try:
                if self._lease is not None:
                    self._lease.__exit__(None, None, None)
            finally:
                self.pool.close()

    def _stop_workers(self, handles: ty.Sequence[WorkerHandle]) -> None:
        """Write each worker's `STOP_MARK`, then stop them with the substrate."""
        for h in handles:
            try:
                self.pool.write_worker_record(h.token, STOP_MARK, {"at": time.time()})
            except (OSError, ValueError) as err:
                logger.warning(
                    "writing the stop mark of worker %s failed: %s", h.token, err
                )
        self.substrate.stop(handles)

    def place(self, jobs: ty.Sequence["Job"]) -> list[Placement | Unplaceable]:
        """Each job's placement, as `submit` would make it, for ``plan()`` (RFC 0005 section 7.4)."""
        startable = self._startable()
        return [self._placement(job, startable=startable)[0] for job in jobs]

    # -- placement -----------------------------------------------------------------------------

    def _startable(self) -> ty.Callable[[WorkerClass], "bool | str"]:
        """For placement: a class with a live worker can start; otherwise the substrate says. The
        substrate is listed once, and asked once per class, per call of this function.
        """
        live = {h.worker_class for h in self._live()}
        verdicts: dict[str, "bool | str"] = {}

        def can_start(cls: WorkerClass) -> "bool | str":
            if cls.name not in verdicts:
                verdicts[cls.name] = (
                    True if cls.name in live else self.substrate.can_start(cls)
                )
            return verdicts[cls.name]

        return can_start

    def _providers(self) -> tuple:
        """The pool's providers as its workers make them, for the code a placement judges by; one
        that cannot be imported here is left out."""
        if self._provider_objects is None:
            from pydra.workers.pool.worker import _resolve

            made = []
            for reference in self.providers:
                try:
                    made.append(_resolve(reference))
                except Exception as err:
                    logger.warning(
                        "the provider %s cannot be made here: %s", reference, err
                    )
            self._provider_objects = tuple(made)
        return self._provider_objects

    def _live(self) -> list[WorkerHandle]:
        try:
            return list(self.substrate.live())
        except Exception as err:
            cause = f"{type(err).__name__}: {err}"
            logger.warning(
                "listing the substrate's workers failed: %s",
                cause,
                extra={"event": "listing_failed", "cause": cause},
            )
            return []

    def _placement(
        self,
        job: "Job",
        above: str | None = None,
        startable: ty.Callable[[WorkerClass], "bool | str"] | None = None,
    ) -> tuple[Placement | Unplaceable, tuple[str, ...], tuple[dict, ...]]:
        """The job's placement, the rest of its ladder (as queue names) and its residency."""
        from pydra.engine import resident

        can_start = startable or self._startable()
        classes = self.classes
        if above is not None:  # re-placed from the class after ``above``
            ordered = sorted(self.classes, key=_order)
            names = [c.name for c in ordered]
            if above in names:
                classes = tuple(ordered[names.index(above) + 1 :])
        request = fit_request(job, self._providers())
        placement = place(job, classes, self.fit, can_start, request=request)
        if isinstance(placement, Unplaceable):
            return placement, (), ()
        res = job.resources if job.resources is not None else ComputeResources()
        fields = resident.provided_fields(job.task)
        residency = tuple(
            attrs.asdict(resident.spec_key(job.task, f, p)) for f, p in fields
        )
        suffix = ""
        digest = resident.residency_digest(job)
        if digest is not None and res.residency == "route":
            suffix = f"#{digest}"
        candidates = [
            c for c in sorted(classes, key=_order) if _candidate_reason(res, c) is None
        ]
        names = [c.name for c in candidates]
        rest = candidates[names.index(placement.worker_class) + 1 :]
        ladder = tuple(c.name + suffix for c in rest if can_start(c) is True)
        return (
            attrs.evolve(placement, queue=placement.worker_class + suffix),
            ladder,
            residency,
        )

    def _queue_jobs(
        self,
        jobs: ty.Sequence["Job"],
        handle: Handle | None,
        report: "ty.Callable[[Job, Outcome], None] | None" = None,
    ) -> list[tuple["Job", _Job | None, "Outcome"]]:
        """Place each job and queue it for the follower, which checks it against the pool (a live
        ticket that holds it, a refusal) before writing it; no storage is read here. Return the
        reports due now: jobs that cannot be placed, and jobs placed on a broken class.
        """
        reports: list[tuple["Job", _Job | None, "Outcome"]] = []
        report = report or self._sole_report()
        startable = self._startable()
        busy = {
            _queue_class(j.queue if j.ticket is None else j.ticket.queue)
            for j in self._open.values()
            if j.twin_of is None
        }
        for job in jobs:
            if is_workflow(job.task):
                reports.append(
                    (
                        job,
                        None,
                        Failed(
                            TypeError(
                                f"a WarmPool runs leaf jobs; route the workflow job "
                                f"{job.name!r} to another executor"
                            )
                        ),
                    )
                )
                continue
            try:
                key = (job.checksum, _fingerprint(job))
                primary = self._by_key.get(key)
                if primary is not None and primary.job in self._open:
                    # another run's job of this key is open here: settle with it
                    twin = _Job(
                        job,
                        handle,
                        key,
                        primary.queue,
                        primary.ladder,
                        primary.residency,
                        report=report,
                        twin_of=primary,
                    )
                    primary.twins.append(twin)
                    self._open[job] = twin
                    continue
                placement, ladder, residency = self._placement(job, startable=startable)
            except Exception as exc:
                reports.append((job, None, Failed(exc)))
                continue
            if isinstance(placement, Unplaceable):
                reports.append((job, None, Failed(UnplaceableJob(placement))))
                continue
            j = _Job(
                job, handle, key, placement.queue, ladder, residency, report=report
            )
            name = placement.worker_class
            if name in self._broken and key in self._break_held.get(name, ()):
                self.release_hold(name)  # a retry of a job the break held
            broken = self._broken.get(name) or self._held_reason(
                placement.queue, job, residency
            )
            if broken is not None:
                if name in self._broken:
                    self._break_held[name].add(key)
                reports.append((job, j, NotRun(broken, held=True)))
                continue
            if name not in busy:  # work arrives at an idle class: a fresh streak
                self._end_streaks(name)
                busy.add(name)
            self._open[job] = j
            self._by_key[key] = j
            self._unchecked[j] = None
        return reports

    def _check(self, j: _Job) -> tuple:
        """What the pool says of a queued job, read without the lock: ``("put",)``, ``("follow",
        queue, ticket)``, ``("move", queue, ticket)`` or ``("refused", ticket)``."""
        newest = self.pool.newest(*j.key)
        holder = self.pool._holder(newest, j.key)
        if holder is None:
            refused = self._refused(j.key, newest)
            return ("put",) if refused is None else ("refused", refused)
        q = self.pool._by_label(newest[1].get("queue", ""))
        if q is None:
            return ("put",)
        return ("follow" if self._starts(q.name) else "move", q.name, holder)

    def _resume(self) -> None:
        """Check each newly queued job against the pool: follow it where a live ticket holds it (or
        move it, when that ticket's class is not one this coordinator starts and its claim can be
        taken), re-place it above a class that refused it, else leave it to be written.
        """
        with self._lock:
            unchecked, self._unchecked = list(self._unchecked), {}
        if not unchecked:
            return
        decided = []
        for j in unchecked:
            try:
                decided.append((j, self._check(j)))
            except (
                Exception
            ) as err:  # the storage: written as new; a held job is followed
                logger.warning(
                    "checking job %s against the pool failed: %s", j.key[0], err
                )
                decided.append((j, ("put",)))
        reports: list = []
        moves: dict[tuple[str, str], list[_Job]] = {}
        with self._lock:
            for j, verdict in decided:
                if j.job not in self._open:
                    continue  # cancelled meanwhile
                if verdict[0] == "follow":
                    self._hold(j, verdict[1], verdict[2])
                    continue
                if verdict[0] == "move":
                    moves.setdefault((verdict[1], verdict[2]), []).append(j)
                    continue
                if verdict[0] == "refused" and not self._climb(j, verdict[1], reports):
                    continue
                self._enqueue(j)
        for (queue, digest), group in moves.items():
            self._move_foreign(queue, digest, group)
        self._deliver(reports)

    def _enqueue(self, j: _Job) -> None:
        """Put ``j`` last among its queue's waiting jobs."""
        self._waiting.setdefault(j.queue, {})[j] = None

    def _unwait(self, j: _Job) -> None:
        self._waiting.get(j.queue, {}).pop(j, None)
        self._unchecked.pop(j, None)

    def _starts(self, queue: str) -> bool:
        """Whether this coordinator starts the workers of ``queue``'s class (or none at all)."""
        return isinstance(self.substrate, ExternalSubstrate) or (
            _queue_class(queue) in self._class
        )

    def _refused(self, key: JobKey, newest: tuple[int, dict] | None) -> Ticket | None:
        """The ticket in which the job's newest attempt was withdrawn ``refused``, if it was."""
        if newest is None:
            return None
        record = newest[1]
        q = self.pool._by_label(record.get("queue", ""))
        if q is None:
            return None
        found = self.pool.read_outcome(q.name, record["ticket"], key, record["attempt"])
        if (found or {}).get("outcome") != "withdrawn" or found.get(
            "reason"
        ) != "refused":
            return None
        try:
            return self.pool.ticket(q.name, record["ticket"])
        except (CorruptTicket, StoreUnavailable) as err:
            logger.warning(
                "the ticket %s that refused job %s cannot be read (%s): it is placed as a new job",
                record["ticket"],
                key[0],
                err,
            )
            return None

    def _climb(self, j: _Job, refused: Ticket, reports: list) -> bool:
        """Re-place a job a worker refused one class up the refused ticket's ladder, under a fresh
        attempt (RFC 0005 sections 3.3, 6); False (reported ``Failed``) when no class is left.
        """
        if not refused.ladder:
            self._forget(j)
            tried = ClassTried(
                _queue_class(refused.queue),
                None,
                "a worker of the class refused it, and no larger class is left",
            )
            unplaceable = Unplaceable(
                tried=(tried,), request=j.job.resources or ComputeResources()
            )
            reports.append((j.job, j, Failed(UnplaceableJob(unplaceable))))
            return False
        j.queue, j.ladder = refused.ladder[0], tuple(refused.ladder[1:])
        logger.info(
            "job %s was refused on %r; re-placed on %r",
            j.key[0],
            refused.queue,
            j.queue,
        )
        return True

    # -- tracking ------------------------------------------------------------------------------

    def _ticket(self, queue: str, digest: str) -> _Ticket:
        tid = (queue, digest)
        t = self._tickets.get(tid)
        if t is None:
            t = _Ticket(queue, digest, foreign=not self._starts(queue))
            self._tickets[tid] = t
            self._queues.setdefault(queue, set()).add(tid)
        return t

    def _hold(self, j: _Job, queue: str, digest: str) -> None:
        """Follow ``j`` in ticket ``digest`` of ``queue``."""
        if j.ticket is not None and (j.ticket.queue, j.ticket.digest) == (
            queue,
            digest,
        ):
            return  # already followed there: its record (when it last moved) stays
        if j.ticket is not None:
            self._release(j)
        t = self._ticket(queue, digest)
        t.jobs[j.key] = j
        t.recheck = True
        j.ticket = t

    def _follow_key(self, queue: str, digest: str, key: JobKey, job: ty.Any) -> None:
        """Follow a job by its key alone (for tests of the follower's cost)."""
        j = _Job(job, None, key, queue, (), ())
        t = self._ticket(queue, digest)
        t.jobs[key] = j
        j.ticket = t

    def _release(self, j: _Job) -> None:
        t, j.ticket = j.ticket, None
        if t is None:
            return
        t.jobs.pop(j.key, None)
        if not t.jobs:
            tid = (t.queue, t.digest)
            self._tickets.pop(tid, None)
            queue = self._queues.get(t.queue)
            if queue is not None:
                queue.discard(tid)
                if not queue:
                    del self._queues[t.queue]

    def _forget(self, j: _Job) -> None:
        """Stop following ``j``. A twin leaves its primary; a primary's twins leave the open jobs
        with it and are reported with it (`_deliver`)."""
        if j.twin_of is not None:
            if j in j.twin_of.twins:
                j.twin_of.twins.remove(j)
            j.twin_of = None
            self._open.pop(j.job, None)
            return
        self._release(j)
        if self._by_key.get(j.key) is j:
            del self._by_key[j.key]
        if j.job is not None:
            self._open.pop(j.job, None)
        for twin in j.twins:
            self._open.pop(twin.job, None)

    def _cancel_one(self, j: _Job) -> tuple[_Job, bool]:
        """Stop following the cancelled job ``j``: the `_Job` to report ``NotRun``, and whether its
        ticket entry may be withdrawn. A primary with twins hands its place, ticket and all, to its
        first twin and is not withdrawn; a twin just leaves its primary."""
        if j.twin_of is not None:
            self._forget(j)
            return j, False
        if not j.twins:
            self._unwait(j)
            ticket = j.ticket
            self._forget(j)
            j.ticket = ticket  # read by the withdrawal, then cleared
            return j, ticket is not None
        heir = j.twins.pop(0)
        gone = _Job(
            j.job, j.handle, j.key, j.queue, j.ladder, j.residency, report=j.report
        )
        self._open.pop(j.job, None)
        j.job, j.handle, j.report = heir.job, heir.handle, heir.report
        for twin in j.twins:
            twin.twin_of = j
        self._open[j.job] = j
        return gone, False

    def _queued(self, queue: str) -> int:
        return sum(len(self._tickets[tid].jobs) for tid in self._queues.get(queue, ()))

    def _max_queued(self, queue: str) -> int:
        if self.settings.max_queued_jobs is not None:
            return self.settings.max_queued_jobs
        cls = self._class.get(_queue_class(queue))
        if cls is None:
            return 4 * 64
        return 4 * cls.quantum_jobs * cls.max_workers

    # -- moving foreign tickets ----------------------------------------------------------------

    def _move_foreign(self, queue: str, digest: str, group: list[_Job]) -> None:
        """Take ticket ``digest`` of ``queue`` (a class this coordinator does not start) by
        ``try_take`` and move ``group`` to their placements; follow them there when its claim is
        live. Called without the lock: the storage is written outside it."""
        held: list[tuple[_Job, str, str]] = []
        claim = None
        with contextlib.suppress(LeaseTimeout):
            claim = self.pool.take(queue, digest)
        if claim is not None:
            try:
                entries = {e.key: e for e in claim.ticket.jobs}
                by_queue: dict[str, list[tuple[_Job, TicketJob]]] = {}
                for j in group:
                    entry = entries.get(j.key)
                    if entry is None or claim.current(entry) is None:
                        continue  # settled or moved meanwhile: the follower finds where
                    by_queue.setdefault(j.queue, []).append((j, entry))
                for target, pairs in by_queue.items():
                    quantum = self._class[_queue_class(target)].quantum_jobs
                    groups = [
                        pairs[i : i + quantum] for i in range(0, len(pairs), quantum)
                    ]
                    made = claim.move(
                        [[e for _, e in g] for g in groups],
                        target,
                        cause="replaced",
                        ladder=pairs[0][0].ladder,
                    )
                    for g, new in zip(groups, made):
                        held.extend((j, target, new) for j, _ in g)
                    logger.info(
                        "moved %d job(s) of ticket %s on %r to %r",
                        len(pairs),
                        digest,
                        queue,
                        target,
                    )
            finally:
                claim.release()
        moved = {id(j) for j, _, _ in held}
        held.extend((j, queue, digest) for j in group if id(j) not in moved)
        with self._lock:
            for j, where, ticket in held:
                if j.job is None or j.job in self._open:
                    self._hold(j, where, ticket)

    # -- writing tickets -----------------------------------------------------------------------

    def _write(self) -> None:
        """Write waiting jobs as tickets, up to ``max_queued_jobs`` unsettled jobs per queue."""
        self._resume()
        with self._lock:
            batches = []
            for queue, waiting in self._waiting.items():
                if _queue_class(queue) in self._broken:
                    continue  # held by `_hold_broken`, as are jobs of a held residency
                room = self._max_queued(queue) - self._queued(queue)
                taken = []
                while waiting and room > 0:
                    first = next(iter(waiting))
                    del waiting[first]
                    taken.append(first)
                    room -= 1
                if taken:
                    batches.append((queue, taken))
            self._waiting = {q: w for q, w in self._waiting.items() if w}
            writing = {queue for queue, _ in batches}
            for queue in list(self._write_failing):
                if queue not in writing:  # nothing waits to be written: the streak ends
                    self._write_failing.pop(queue, None)
        for queue, taken in batches:
            quantum = self._class[_queue_class(queue)].quantum_jobs
            groups: list[list[_Job]] = []
            for j in taken:
                if (
                    not groups
                    or len(groups[-1]) >= quantum
                    or groups[-1][0].job.name != j.job.name
                ):
                    groups.append([])
                groups[-1].append(j)
            for i, group in enumerate(groups):
                try:
                    self._put(queue, group)
                except Exception as err:
                    self._write_failed(queue, [j for g in groups[i:] for j in g], err)
                    break
            else:
                with self._lock:
                    self._write_failing.pop(queue, None)

    def _write_failed(self, queue: str, jobs: list[_Job], err: Exception) -> None:
        """Put the jobs of a failed ticket write, and those after them, back at the head of their
        queue's waiting jobs, never dropped; once the queue's streak of failed writes has
        ``start_deadline_failures`` failures over ``start_deadline_s``, break its class so that its
        jobs are held with the error."""
        now = time.monotonic()
        cause = f"{type(err).__name__}: {err}"
        with self._lock:
            back = [j for j in jobs if j.job in self._open and j.ticket is None]
            self._waiting[queue] = {
                **dict.fromkeys(back),
                **self._waiting.get(queue, {}),
            }
            since, failures, _ = self._write_failing.get(queue, (now, 0, cause))
            self._write_failing[queue] = (since, failures + 1, cause)
        logger.warning(
            "writing a ticket on %r failed (%s): %d job(s) wait for the next write",
            queue,
            cause,
            len(back),
            extra={"event": "write_failed", "queue": queue, "cause": cause},
        )
        cls = self._class.get(_queue_class(queue))
        if (
            cls is not None
            and cls.name not in self._broken
            and self._deadline_passed(since, failures + 1, now)
        ):
            self._break(
                cls,
                f"writing tickets on queue {queue!r} failed {failures + 1} times over "
                f"{now - since:.0f}s; the latest: {cause}",
            )

    def _deadline_passed(self, since: float, failures: int, now: float) -> bool:
        """Whether a streak of ``failures`` failed attempts since ``since`` breaks its class."""
        limit = self.settings.start_deadline_s
        return (
            limit is not None
            and failures >= self.settings.start_deadline_failures
            and now - since >= limit
        )

    def _put(self, queue: str, group: list[_Job]) -> None:
        """Write ``group`` as one ticket. A job its codec cannot encode is reported ``Failed`` on
        its own; an error writing the payloads or the ticket is raised, for `_write` to requeue.
        """
        group = [j for j in group if j.job in self._open]
        encoded, failed = [], []
        for j in group:
            try:
                encoded.append((j, encode_payload(self.codec, j.job)))
            except Exception as err:
                logger.warning(
                    "job %s cannot be encoded by %s: %s",
                    j.job.name,
                    self.codec.name,
                    err,
                    extra={
                        "event": "encode_failed",
                        "cause": f"{type(err).__name__}: {err}",
                    },
                )
                failed.append((j, err))
        if failed:
            with self._lock:
                for j, _ in failed:
                    self._forget(j)
            self._deliver([(j.job, j, Failed(err)) for j, err in failed])
        if not encoded:
            return
        group = [j for j, _ in encoded]
        entries = [
            TicketJob(
                checksum=j.key[0],
                fingerprint=j.key[1],
                payload=store_payload(self.codec, data, media, self.root),
            )
            for j, (data, media) in encoded
        ]
        while entries:
            ticket = Ticket(
                queue=queue,
                node=group[0].job.name,
                jobs=entries,
                residency=group[0].residency,
                ladder=group[0].ladder,
                time=ticket_time(j.job for j in group),
            )
            try:
                digest = self.pool.put(queue, ticket)
            except JobHeld as held:  # another process put it meanwhile: follow it
                newest = self.pool.newest(*held.job)
                q = self.pool._by_label((newest or (0, {}))[1].get("queue", ""))
                with self._lock:
                    for j in group:
                        if j.key == held.job and q is not None:
                            self._hold(j, q.name, held.ticket)
                entries = [e for e in entries if e.key != held.job]
                group = [j for j in group if j.key != held.job]
                continue
            with self._lock:
                for j in group:
                    if j.job in self._open:
                        self._hold(j, queue, digest)
            return

    # -- the follower --------------------------------------------------------------------------

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(
                    target=self._follow_loop, name="pydra-pool-follower", daemon=True
                )
                self._thread.start()

    def _follow_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll()
            except Exception as err:  # a storage error: the next poll tries again
                logger.exception(
                    "the pool's follower failed a poll; retrying",
                    extra={
                        "event": "poll_failed",
                        "cause": f"{type(err).__name__}: {err}",
                    },
                )
            self._wake.wait(self.settings.poll_s)
            self._wake.clear()

    def _poll(self, now: float | None = None) -> None:
        """One pass: the fence, broken classes' jobs held, the waiting jobs written, each queue's
        settled marks listed once and the outcomes of tickets whose count moved read, followed claims
        judged, the workers scaled. Storage and results are read without the lock."""
        if self._fenced is not None or self.epoch is None:
            return
        if self.pool.fenced(self.epoch):
            self._fence()
            return
        now = time.monotonic() if now is None else now
        self._hold_broken()
        self._write()
        reports: list[tuple["Job", _Job | None, "Outcome"]] = []
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            settled = self.pool.settled(queue)
            with self._lock:
                tickets = [self._tickets[t] for t in self._queues.get(queue, ())]
            for t in tickets:
                count = settled.get(t.digest, 0)
                if count > t.seen:
                    self._progress[_queue_class(queue)] = now
                if count > t.seen or (t.recheck and count > 0):
                    try:
                        found = self.pool.outcomes(
                            queue, t.digest, expect=count, within=0.0
                        )
                    except ListingBehind:
                        continue  # read at the next poll
                    with self._lock:
                        t.seen, t.recheck, t.moved = count, False, now
                    self._read(t, found, reports)
                elif t.foreign:
                    self._judge(t, now)
        self._scale(now)
        self._hold_broken()
        self._deliver(reports)

    def _read(self, t: _Ticket, found: dict[str, dict], reports: list) -> None:
        """Settle or retarget each followed job of ``t`` that has an outcome record in it (a job that
        moved has its withdrawal here); the others have nothing new, so their generations are not
        read. Generations and results are read without the lock; what they say is applied under it,
        to jobs still followed in ``t``."""
        recorded = {(r.get("checksum"), r.get("fingerprint")) for r in found.values()}
        with self._lock:
            jobs = [j for j in t.jobs.values() if j.key in recorded]
        for j in jobs:
            newest = self.pool.newest(*j.key)
            if newest is None:
                continue
            _, record = newest
            if record.get("ticket") != t.digest:
                q = self.pool._by_label(record.get("queue", ""))
                if q is not None:
                    with self._lock:
                        if j.ticket is t:
                            self._hold(j, q.name, record["ticket"])
                continue
            outcome = found.get(_outcome_name(j.key, record["attempt"]))
            if outcome is None:
                continue
            verdict = self._verdict(t, outcome, record["attempt"], j)
            if verdict is None:
                continue
            with self._lock:
                if j.ticket is not t:
                    continue  # cancelled or moved meanwhile
                if verdict[0] == "climb":
                    self._release(j)
                    if self._climb(j, verdict[1], reports):
                        self._enqueue(j)
                    continue
                if verdict[0] == "held":
                    # the job's residency (or, named by ``hold``, its warm environment) is held on
                    # its queue for this epoch: its other jobs too
                    label = outcome.get("hold") or _label(j.residency)
                    self._held.setdefault((t.queue, label), verdict[1].reason)
                self._forget(j)
                reports.append((j.job, j, verdict[1]))
                if outcome.get("outcome") == "done" and _sampled(
                    j.key[0], self.settings.recheck_fraction
                ):
                    self._queue_recheck((j, verdict[1], outcome))

    # -- the sampled cold re-check -------------------------------------------------------------

    def _queue_recheck(self, item: tuple) -> bool:
        """Queue a re-check for the re-check thread (started on first use); a full queue drops it,
        counted in ``rechecks_dropped``. Never blocks the follower."""
        if self._recheck_stop.is_set():
            return False
        try:
            self._recheck_queue.put_nowait(item)
        except queuelib.Full:
            with self._lock:
                self.rechecks_dropped += 1
            logger.warning(
                "the re-check queue is full (%d): a re-check is dropped (%d so far)",
                self.recheck_queue,
                self.rechecks_dropped,
            )
            return False
        with self._lock:
            if self._recheck_thread is None:
                self._recheck_thread = threading.Thread(
                    target=self._recheck_loop, name="pydra-pool-recheck", daemon=True
                )
                self._recheck_thread.start()
        return True

    def _recheck_loop(self) -> None:
        while True:
            if self._recheck_stop.is_set() and self._recheck_queue.empty():
                return  # closed past its bound: nothing left to drop
            item = self._recheck_queue.get()
            if item is None:
                return
            if self._recheck_stop.is_set():
                with self._lock:
                    self.rechecks_dropped += 1
                continue
            j, result, outcome = item
            try:
                self._recheck(j, result, outcome)
            except Exception:  # a re-check is a measurement: it never fails the run
                logger.exception("re-checking %s failed", j.job.name)

    def _stop_rechecks(self) -> None:
        """Let the queued re-checks finish within ``settings.drain_s``; past it, drop the rest
        (logged) and leave a running one to its daemon thread."""
        thread = self._recheck_thread
        if thread is None:
            return
        bound = self.settings.drain_s
        deadline = time.monotonic() + bound
        try:
            self._recheck_queue.put(None, timeout=bound)
        except queuelib.Full:
            pass
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            self._recheck_stop.set()
            left = self._recheck_queue.qsize()
            logger.warning(
                "re-checks still running after %g s at close: %d queued re-check(s) dropped",
                bound,
                left,
            )

    def _recheck(self, j: _Job, result: ty.Any, outcome: dict) -> None:
        """Run ``j``'s job again alone on a fresh residency against its committed entry and write
        the `ParityRecord`; report the first mismatch of the residency through the audit.
        """
        from pydra.engine.portable.reuse import Candidate

        job = j.job
        if result is None or isinstance(result, Failed) or result.errored:
            return
        committed = result.committed
        key = residency_key(j.residency) or "none"
        batch = int((outcome.get("sample") or {}).get("batch_size") or 1)
        if committed is None:
            self._not_run(
                job, key, batch, "the store has no committed entry to verify against"
            )
            return
        entry = committed.entry
        candidate = next(
            (
                Candidate(ref, context, digest, 0, "recheck")
                for ref, context, digest in job.store.stored(job.checksum)
                if os.path.abspath(ref.root) == os.path.abspath(entry.root)
                and (ref.cdir, ref.fingerprint) == (entry.cdir, entry.fingerprint)
            ),
            None,
        )
        if candidate is None:
            self._not_run(job, key, batch, "the committed entry cannot be read")
            return
        fresh, against, reason, ran = self._fresh_run(job, candidate)
        equal: bool | None
        if not ran:
            equal = None  # no statement about the task: never audited
        else:
            equal = not reason and fresh is not None and fresh == against
            if not reason and not equal:
                reason = "the output digests differ"
        record = ParityRecord(
            key, job.checksum, (batch, 1), equal, against, fresh, reason
        )
        try:
            self.calibration.record(record)
        except Exception as err:
            logger.warning("could not record the re-check of %s: %s", job.name, err)
        if equal is not False or key in self._parity_reported:
            if equal is None:
                logger.warning("%s could not be re-checked here: %s", job.name, reason)
            return
        self._parity_reported.add(key)
        logger.warning(
            "the re-check of %s on a fresh residency differs from its committed entry "
            "(residency %s): %s",
            job.name,
            key,
            reason,
        )
        audit = job.audit
        if audit is not None and audit.audit_check(AuditFlag.PROV):
            try:
                audit.audit_message(
                    {
                        "@type": "pool-recheck",
                        "residency": key,
                        "checksum": job.checksum,
                        "batch_sizes": [batch, 1],
                        "committed": against,
                        "fresh": fresh,
                        "reason": reason,
                    },
                    AuditFlag.PROV,
                )
            except Exception as err:
                logger.warning("could not audit the re-check of %s: %s", job.name, err)

    def _not_run(self, job: "Job", key: str, batch: int, reason: str) -> None:
        logger.warning("%s could not be re-checked here: %s", job.name, reason)
        try:
            self.calibration.record(
                ParityRecord(key, job.checksum, (batch, 1), None, reason=reason)
            )
        except Exception as err:
            logger.warning("could not record the re-check of %s: %s", job.name, err)

    def _fresh_run(
        self, job: "Job", candidate: ty.Any
    ) -> tuple[str | None, str | None, str, bool]:
        """Run a copy of ``job`` (decoded as a worker decodes it) alone, never served from the
        store, on a resident set of its own, marked to verify against ``candidate``; its publish
        loses to the committed entry. Return its output digest, the committed entry's, why it
        did not reach them ("" when it did), and whether the task ran to its outputs (False when
        the task, a provider's load or the device failed here; a refusal by the store's
        comparison comes after the task ran)."""
        data, _ = self.codec.encode(job)
        copy = self.codec.decode(data, job.store)
        copy.attempt = f"recheck-{uuid.uuid4().hex}"
        copy._work_dir_memo = None
        copy.claim = None
        copy.chdir = (
            False  # the follower is a thread: never change the process's directory
        )
        copy.verify = candidate
        copy.looked_up = False  # a fresh attempt: no lookup made before it counts
        copy._reusable = types.MethodType(_never_reusable, copy)
        ran = {"task": False}
        inner = copy._run_task

        def run_task(rerun: bool = False) -> None:
            inner(rerun)
            ran["task"] = True

        copy._run_task = run_task
        submitter = copy.submitter
        if submitter is not None:  # a resident set of its own: a fresh residency
            submitter._worker_copy = False
            submitter._resident_set = None
        try:
            copy.run()
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {exc}"[:2000], ran["task"]
        finally:
            resident_set = submitter._resident_set if submitter is not None else None
            if resident_set is not None:
                resident_set.close()
        published = copy.committed
        if published is None:
            return None, None, "the re-check published nothing to compare", False
        return published.own_digest, published.committed_digest, "", True

    def _verdict(
        self, t: _Ticket, outcome: dict, attempt: str, j: _Job
    ) -> tuple | None:
        """What a job's outcome means, read without the lock: ``("report", outcome)``,
        ``("held", outcome)`` (its residency held), ``("climb", refused ticket)``, or None (an
        ``expired`` attempt waits for its taker; a moved job is followed by its generation).
        """
        kind = outcome.get("outcome")
        if kind == "withdrawn":
            reason = outcome.get("reason")
            if reason == "refused":
                try:
                    ticket = self.pool.ticket(t.queue, t.digest)
                except Exception:
                    ticket = None
                return None if ticket is None else ("climb", ticket)
            if reason != "cancel":
                return None
            return ("report", NotRun("cancelled", held=False))
        if kind == "held":
            why = outcome.get("reason") or (outcome.get("fault") or {}).get("message")
            return ("held", NotRun(f"held: {why}", held=True))
        if kind == "done":
            # a result a taker found already committed is not in this attempt's work directory
            result = self._result(j.job, attempt, settled=not outcome.get("served"))
            if result is None:
                result = self._result(j.job, attempt, settled=True)
            return ("report", result)
        if kind == "failed":
            fault = outcome.get("fault") or {}
            found = None
            if fault.get("kind") == "content":
                found = self._result(j.job, attempt, settled=False)
                if found is None or isinstance(found, Failed) or not found.errored:
                    found = None
            return (
                "report",
                found
                or Failed(
                    PoolJobFailed(
                        f"{fault.get('type', 'failed')}: {fault.get('message', '')}",
                        fault,
                    )
                ),
            )
        return None

    def _result(self, job: "Job", attempt: str, settled: bool) -> "Outcome":
        job.attempt = attempt
        job._work_dir_memo = None
        try:
            return job.result(settled=settled)
        except Exception as exc:
            return Failed(exc)

    def _judge(self, t: _Ticket, now: float) -> None:
        """Take a followed ticket of a class this coordinator does not start once it has not moved
        for ``stale_after`` (at most once per ``stale_after``) and its claim is stale, and move its
        jobs to classes this coordinator starts."""
        stale = self.settings.lease.stale_after
        if now - t.moved < stale or now - t.judged < stale:
            return
        t.judged = now
        with self._lock:
            group = list(t.jobs.values())
        if group:
            self._move_foreign(t.queue, t.digest, group)

    def _fence(self) -> None:
        epochs = self.pool.epochs()
        why = (
            f"PoolFenced: coordinator epoch {epochs[-1] if epochs else '?'} took the pool at "
            f"{self.root} from epoch {self.epoch}; resubmit the run"
        )
        logger.warning(why)
        with self._lock:
            self._fenced = why
            primaries = [j for j in self._open.values() if j.twin_of is None]
            reports = [(j.job, j, NotRun(why, held=True)) for j in primaries]
            for j in primaries:
                self._forget(j)
            self._waiting.clear()
        self._deliver(reports)

    def _deliver(
        self,
        reports: list,
        handle: Handle | None = None,
        report: "ty.Callable[[Job, Outcome], None] | None" = None,
    ) -> None:
        """Report each outcome to the run that submitted the job (its `_Job`'s binding; else
        ``report``, else the one binding when exactly one run is bound), and a primary's outcome to
        its twins too.
        """
        expanded = []
        for job, j, outcome in reports:
            expanded.append((job, j, outcome))
            if j is not None and j.twin_of is None and j.twins:
                expanded.extend((t.job, t, outcome) for t in list(j.twins))
                j.twins.clear()
        for job, j, outcome in expanded:
            if job is None:
                continue
            owner = j.handle if j is not None else handle
            if owner is not None:
                owner.open.discard(job)
            to = (j.report if j is not None else None) or report or self._sole_report()
            if to is None:
                logger.error(
                    "the outcome of %s has no run to go to: it was submitted outside a bound run "
                    "while several runs are bound",
                    job.name,
                )
                continue
            try:
                to(job, outcome)
            except Exception as err:  # the driver is gone
                logger.warning("reporting %s failed: %s", job.name, err)

    # -- scaling and failed starts -------------------------------------------------------------

    def _demand(self) -> dict[str, int]:
        """The workers each class's waiting and queued jobs want (RFC 0005 section 4.5)."""
        with self._lock:
            jobs: dict[str, int] = collections.Counter()
            for queue, waiting in self._waiting.items():
                jobs[queue] += len(waiting)
            for queue in self._queues:
                jobs[queue] += self._queued(queue)
        wanted: dict[str, int] = collections.Counter()
        for queue, n in jobs.items():
            cls = self._class.get(_queue_class(queue))
            if cls is None or not n:
                continue
            per_queue = min(
                self.settings.unmeasured_workers,
                math.ceil(n / cls.quantum_jobs),
                cls.max_workers,
            )
            wanted[cls.name] = min(wanted[cls.name] + per_queue, cls.max_workers)
        return wanted

    def _scale(self, now: float) -> None:
        try:
            live = list(self.substrate.live())
        except Exception as err:
            # not read as every worker gone: the next poll lists again, within a budget
            self._listing_failed(err, now)
            return
        self._listing_failures.clear()
        self._listed_at = now
        live_ids = {id(h) for h in live}
        failed_now: set[str] = set()  # classes with a failed start in this pass
        for token, h in list(self._started.items()):
            if id(h) in live_ids:
                if token not in self._ready and self.pool.worker_record(
                    token, "ready.json"
                ):
                    self._ready.add(token)
                    self._failures[h.worker_class] = 0
                    self._failing_since.pop(h.worker_class, None)
                continue
            del self._started[token]
            if token in self._stopped:  # stopped by this pool: not a failed start
                self._stopped.discard(token)
                continue
            if token in self._ready or self.pool.worker_record(token, "ready.json"):
                self._ready.discard(token)
                continue
            self._start_failed(
                h.worker_class,
                h.cause or "no ready mark",
                now,
                f"worker {token} left without a ready mark",
            )
            failed_now.add(h.worker_class)
        wanted = self._demand()
        for cls in self.classes:
            name = cls.name
            if name in self._broken:
                continue
            if self._failures[name] >= self.faults.start_failures:
                self._break(cls)
                continue
            mine = [h for h in live if h.worker_class == name]
            want = wanted.get(name, 0)
            if not want:  # no waiting work: a failure streak never outlives it
                self._end_streaks(name)
            self._idle[name] = 0 if want else self._idle[name] + 1
            if len(mine) < want and self._may_start(name, now):
                for _ in range(want - len(mine)):
                    if self.substrate.can_start(cls) is not True:
                        break
                    try:
                        handles = self.substrate.start(cls, 1)
                    except Exception as err:  # a refused submission: a failed start
                        started = err.started if isinstance(err, SubstrateError) else ()
                        for h in started:
                            self._started[h.token] = h
                        self._start_failed(
                            name,
                            f"{type(err).__name__}: {err}",
                            now,
                            "starting a worker failed",
                        )
                        if not started:
                            failed_now.add(name)
                        break
                    for h in handles:
                        self._started[h.token] = h
            if (
                want
                and not mine
                and name in failed_now
                and self._deadline_due(name, now)
            ):
                self._break(
                    cls,
                    f"no worker of the class became ready: {self._failures[name]} starts "
                    f"failed in a row over {now - self._failing_since[name]:.0f}s; the "
                    f"latest: {self._cause.get(name)}",
                )
                continue
            if len(mine) > want:
                pending = [h for h in mine if h.token not in self._ready]
                surplus = pending[: len(mine) - want]
                if not want and self._idle[name] >= self.settings.idle_polls:
                    surplus = mine
                if surplus:
                    self._stopped.update(h.token for h in surplus)
                    self._stop_workers(surplus)

    def _listing_failed(self, err: Exception, now: float) -> None:
        """Count a failed listing against each class with work that shows no progress: no live
        ready worker (one whose worker lease moved within ``stale_after``) and no ticket settled
        within ``stale_after``. Once ``listing_failures`` polls in a row have failed and
        ``stale_after`` has passed since the last good listing, such a class is broken, so its
        jobs are held rather than left waiting. Everything is read from storage."""
        cause = f"{type(err).__name__}: {err}"
        logger.warning(
            "listing the substrate's workers failed: %s",
            cause,
            extra={"event": "listing_failed", "cause": cause},
        )
        stale = self.settings.lease.stale_after
        if self._listed_at is None:
            self._listed_at = now
        ready = set()
        for token, h in self._started.items():
            if token not in self._ready:
                if not self.pool.worker_record(token, "ready.json"):
                    continue
                self._ready.add(token)
            if self._lease_moved(token, now) < stale:
                ready.add(h.worker_class)
        for token in set(self._lease_seen) - set(self._started):
            del self._lease_seen[token]
        wanted = self._demand()
        started = {h.worker_class for h in self._started.values()}
        for cls in self.classes:
            name = cls.name
            if name in self._broken or not (wanted.get(name) or name in started):
                continue
            settled = now - self._progress.get(name, float("-inf")) < stale
            if name in ready or settled:  # progressing: the run is not stuck
                self._listing_failures.pop(name, None)
                continue
            self._listing_failures[name] += 1
            n = self._listing_failures[name]
            down = now - self._listed_at
            if n >= self.faults.listing_failures and down >= stale:
                self._break(
                    cls,
                    f"listing the substrate's workers failed {n} times in a row over "
                    f"{down:.0f}s: {type(err).__name__}: {err}",
                )

    def _lease_moved(self, token: str, now: float) -> float:
        """Seconds, on the follower's clock, since the worker's lease was last seen to move; a
        missing lease never moved. Mtimes are compared only with each other."""
        newest = _newest_mtime(self.root / "workers" / token / "lease")
        last = self._lease_seen.get(token)
        if newest is None:
            return float("inf")
        if last is None or last[0] != newest:
            self._lease_seen[token] = (newest, now)
            return 0.0
        return now - last[1]

    def _may_start(self, name: str, now: float) -> bool:
        failures = self._failures[name]
        if not failures:
            return True
        backoff = self.settings.poll_s * 2 ** (failures - 1)
        return now - self._failed_at.get(name, 0.0) >= backoff

    def _start_failed(self, name: str, cause: str, now: float, what: str) -> None:
        """Count a failed start of class ``name`` and emit it as a ``start_failed`` pool event."""
        self._failures[name] += 1
        self._failed_at[name] = now
        self._failing_since.setdefault(name, now)
        self._cause[name] = cause
        logger.warning(
            "%s (class %r, %s): failed start %d",
            what,
            name,
            cause,
            self._failures[name],
            extra={
                "event": "start_failed",
                "worker_class": name,
                "cause": cause,
                "failures": self._failures[name],
            },
        )

    def _deadline_due(self, name: str, now: float) -> bool:
        """Whether the streak of failed starts of class ``name`` breaks it (`_deadline_passed`)."""
        since = self._failing_since.get(name)
        return since is not None and self._deadline_passed(
            since, self._failures[name], now
        )

    def _end_streaks(self, name: str) -> None:
        """End the failure streaks of class ``name`` and of its queues' ticket writes: the next
        attempt is a fresh one, not backed off. Called from the follower and from `submit`.
        """
        with self._lock:
            self._failures[name] = 0
            self._failing_since.pop(name, None)
            self._failed_at.pop(name, None)
            for queue in list(self._write_failing):
                if _queue_class(queue) == name:
                    self._write_failing.pop(queue, None)

    def release_hold(self, worker_class: str | None = None) -> list[str]:
        """Lift this coordinator's break of ``worker_class`` (of every broken class when None) and
        end its failure streaks, so the next start is a fresh attempt; its jobs held meanwhile are
        handed out again by the runner's ``retry``. A ``retry`` of a job the break held lifts it
        too. Residency holds on storage are lifted by `release_hold` of this module
        (``pydracli pool release-hold``). Return the classes released.
        """
        with self._lock:
            names = [
                n for n in self._broken if worker_class is None or n == worker_class
            ]
            for name in names:
                del self._broken[name]
                self._break_held.pop(name, None)
                self._end_streaks(name)
                logger.info(
                    "worker class %r is released: its next start is a fresh attempt",
                    name,
                    extra={"event": "class_released", "worker_class": name},
                )
        self._wake.set()
        return names

    def _break(self, cls: WorkerClass, cause: str | None = None) -> None:
        from pydra.workers.pool.worker import class_record

        if cause is None:
            cause = (
                f"{self._failures[cls.name]} consecutive failed starts; the latest: "
                f"{self._cause.get(cls.name)}"
            )
        why = (
            f"worker class {cls.name!r} is broken: {cause}; class "
            f"{json.dumps(class_record(cls), sort_keys=True)}"
        )
        logger.error(
            why,
            extra={"event": "class_broken", "worker_class": cls.name, "cause": cause},
        )
        self._broken[cls.name] = why
        self._hold_broken()

    def _held_reason(
        self, queue: str, job: "Job", residency: ty.Sequence[dict]
    ) -> str | None:
        """Why a worker held ``job`` on ``queue`` in this epoch: its residency is held, or its warm
        environment (`pydra.workers.pool.placement.warm_environment_label`) is; None otherwise. A
        held environment never holds a job of another environment or of none."""
        why = self._held.get((queue, _label(residency)))
        if why is None and self._held:
            from pydra.workers.pool.placement import warm_environment_label

            label = warm_environment_label(job)
            if label is not None:
                why = self._held.get((queue, label))
        return why

    def _hold_broken(self) -> None:
        """Report every open job placed on a broken class, or of a residency a worker held on its
        queue in this epoch, ``NotRun(held=True)``: queued, followed, or re-placed onto it since.
        Jobs of other residencies on the same queue (``residency="ignore"``) are not held. A hold is
        kept for the epoch, also after ``pydracli pool release-hold`` lifts its mark: a later job of
        it is reported held until a new coordinator (a new epoch) takes the pool.
        """
        if not self._broken and not self._held:
            return
        with self._lock:
            held = []
            for j in list(self._open.values()):
                if j.twin_of is not None:
                    continue  # reported with its primary
                queue = j.queue if j.ticket is None else j.ticket.queue
                why = self._broken.get(_queue_class(queue)) or self._held_reason(
                    queue, j.job, j.residency
                )
                if why is None:
                    continue
                name = _queue_class(queue)
                if name in self._broken:
                    self._break_held[name].add(j.key)
                    self._break_held[name].update(t.key for t in j.twins)
                self._unwait(j)
                self._forget(j)
                held.append((j.job, j, NotRun(why, held=True)))
        self._deliver(held)


# -- pydracli pool status, release-hold, collect -----------------------------------------------


def _open_existing(root: os.PathLike) -> PoolRoot:
    """The pool at ``root``, opened with the lease settings its ``pool.json`` records."""
    from pydra.engine.lease import LeaseSettings

    header = json.loads((Path(root) / "pool.json").read_bytes())
    return PoolRoot.open(
        root, LeaseSettings.from_json(header["lease"]), create=False, probe=False
    )


def pool_status(root: os.PathLike) -> dict:
    """What a pool holds: its coordinator epochs (and which ended), each queue's waiting tickets and
    jobs (upper bounds), settled tickets and holds, and each worker's ready mark."""
    with _open_existing(root) as pool:
        base = pool.path
        epochs = [
            {"epoch": e, "ended": (base / "coordinator" / str(e) / "ended").exists()}
            for e in pool.epochs()
        ]
        queues = {}
        for name in pool.queues():
            queues[name] = {
                "waiting_tickets": pool.depth(name),
                "waiting_jobs": pool.work(name),
                "settled_tickets": len(pool.settled(name)),
                "holds": pool.holds(name, epoch=0),
            }
        workers = []
        for d in (
            sorted((base / "workers").glob("*")) if (base / "workers").is_dir() else []
        ):
            ready = pool.worker_record(d.name, "ready.json")
            workers.append(
                {
                    "token": d.name,
                    "class": (ready or {}).get("class"),
                    "ready": ready is not None,
                }
            )
        return {
            "root": str(base),
            "pool_id": pool.pool_id,
            "epochs": epochs,
            "queues": queues,
            "workers": workers,
        }


def release_hold(root: os.PathLike, queue: str, residency: str | None = None) -> int:
    """Lift the holds of ``queue`` (of one residency label, when given); return how many."""
    with _open_existing(root) as pool:
        return pool.release_holds(queue, residency)


def collect_pool(root: os.PathLike, payload_age: float = 86400.0) -> dict:
    """`PoolRoot.collect` (settled jobs' older generations, payloads no unsettled ticket names),
    `PoolRoot.collect_settled` (tickets resolved more than ``payload_age`` ago, with their marks and
    outcomes), then the directories of ended coordinators below the newest epoch, which stays
    so that no epoch number is taken twice, and the directories of workers stopped before they
    started (holding only a `STOP_MARK` older than ``payload_age`` and than the pool's
    ``max_honoured_stale_after``)."""
    import shutil

    with _open_existing(root) as pool:
        removed = pool.collect(payload_age=payload_age)
        tickets = pool.collect_settled(age=payload_age)
        epochs = pool.epochs()
        gone = []
        for epoch in epochs[:-1]:
            where = pool.path / "coordinator" / str(epoch)
            if (where / "ended").exists():
                shutil.rmtree(where, ignore_errors=True)
                gone.append(epoch)
        swept = []
        mark_age = max(payload_age, pool.settings.max_honoured_stale_after)
        workers = pool.path / "workers"
        for d in sorted(workers.iterdir()) if workers.is_dir() else []:
            mark = d / STOP_MARK
            try:
                only_mark = [e.name for e in d.iterdir()] == [STOP_MARK]
                old = time.time() - mark.stat().st_mtime >= mark_age
            except OSError:
                continue
            if only_mark and old:
                shutil.rmtree(d, ignore_errors=True)
                swept.append(d.name)
        return {
            "files": removed,
            "tickets": tickets,
            "coordinators": gone,
            "workers": swept,
        }
