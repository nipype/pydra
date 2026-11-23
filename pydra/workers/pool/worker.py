"""The warm worker of a pool (RFC 0005 sections 4.3, 5.3-5.5).

``pydracli pool worker --root <pool> --class <name>`` (or ``python -m
pydra.workers.pool.worker``) runs one worker of a class. It refuses to start when the pool's
lease settings differ from its own in a staleness field (`EXIT_SETTINGS`) or the hardware does
not match the class (`EXIT_HARDWARE`), then writes ``ready.json``, claims tickets of the class's
queues, loads each ticket's residency once, runs its jobs through ``Job.run``, and settles each
job's sample and outcome. A call that fails goes through the bisection engine
(`pydra.workers.pool.bisect`) and the pool's fault budgets.

It exits after ``idle_polls`` empty polls, on SIGTERM once finished jobs are settled and the
rest released, after ``orphan_after_s`` without a live coordinator, and recycles
(`EXIT_RECYCLE`) after an expired call or a dead beater.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import functools
import hashlib
import importlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.engine.lease import (
    BeaterClient,
    BeaterDead,
    DirectoryLease,
    LeaseLost,
    newest_mark,
)
from pydra.utils.storagedir import StoreUnavailable
from pydra.workers.pool.beater import expiry_payload, pool_beater
from pydra.workers.pool.classes import WorkerClass, parse_size
from pydra.workers.pool.codec import JobCodec, PrivatePickle, read_payload
from pydra.workers.pool import bisect as engine
from pydra.workers.pool import faults as _faults
from pydra.workers.pool.bisect import Abandon
from pydra.workers.pool.faults import (
    Classifier,
    FaultBudgets,
    FaultRecord,
    attach,
    attached,
    fault_of,
)
from pydra.workers.pool.samples import (
    HardwareClass,
    Sample,
    device_memory,
    host_memory,
    read_hardware,
)
from pydra.workers.pool.settings import PoolSettings
from pydra.workers.pool.storage import (
    EXIT_MARK,
    FINAL,
    Claim,
    PoolOwnerMismatch,
    PoolRoot,
    PoolSettingsMismatch,
    STOP_MARK,
    TicketJob,
)

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.resident import Provider

__all__ = [
    "EXIT_FAULTS",
    "EXIT_HARDWARE",
    "EXIT_OK",
    "EXIT_RECYCLE",
    "EXIT_SETTINGS",
    "EXIT_USAGE",
    "Worker",
    "WrongHardware",
    "check_hardware",
    "class_from_record",
    "class_record",
    "main",
    "run_worker",
]

logger = logging.getLogger("pydra.pool")

EXIT_OK = 0
"""Idle, orphaned, or drained after SIGTERM."""
EXIT_USAGE = 2
"""No pool, another user's pool, or no such class."""
EXIT_SETTINGS = 3
"""``PoolSettingsMismatch``: the lease settings differ from the pool's."""
EXIT_HARDWARE = 4
"""``WrongHardware``: the grant does not match the class."""
EXIT_RECYCLE = 5
"""A call's deadline passed or the beater died: start a fresh worker."""
EXIT_FAULTS = 6
"""The pool's classifier reference cannot be imported here or is not callable."""


class WrongHardware(RuntimeError):
    """The hardware a worker reads does not match its class's ``gpu_type`` or ``gpus``."""


# -- classes -----------------------------------------------------------------------------------


def class_record(cls: WorkerClass) -> dict:
    """``classes/<name>.json``: the worker class as JSON."""
    record = attrs.asdict(cls, recurse=False)
    record["mounts"] = [str(m) for m in cls.mounts]
    record["tags"] = sorted(cls.tags)
    record["substrate_args"] = dict(cls.substrate_args)
    return record


def class_from_record(record: dict) -> WorkerClass:
    """The worker class `class_record` wrote."""
    return WorkerClass(**record)


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _gpu_type_matches(gpu_type: str, product: str) -> bool:
    """Whether ``gpu_type``'s tokens appear, whole and in order (not necessarily adjacent), among
    the product name's: ``h100`` matches "NVIDIA H100 80GB HBM3", ``a100-80gb`` matches "NVIDIA
    A100-SXM4-80GB", and ``a10`` does not match it.
    """
    want, have = _tokens(gpu_type), iter(_tokens(product))
    return bool(want) and all(token in have for token in want)


def check_hardware(cls: WorkerClass, hardware: HardwareClass) -> WorkerClass:
    """The class with ``gpu_mem`` filled from ``hardware`` when it is None.

    Raises
    ------
    WrongHardware
        Fewer devices than ``gpus`` (or none on a CUDA class), or a device whose product is not the
        class's ``gpu_type``.
    """
    wanted = cls.gpus or (1 if cls.device == "cuda" else 0)
    if not wanted:
        return cls
    if len(hardware.gpus) < wanted:
        raise WrongHardware(
            f"class {cls.name!r} needs {wanted} GPU(s); this worker was granted "
            f"{len(hardware.gpus)}"
        )
    granted = hardware.gpus[:wanted]
    if cls.gpu_type is not None:
        other = [
            g.product for g in granted if not _gpu_type_matches(cls.gpu_type, g.product)
        ]
        if other:
            raise WrongHardware(
                f"class {cls.name!r} is gpu_type {cls.gpu_type!r}; this worker was granted "
                f"{', '.join(other)}"
            )
    if cls.gpu_mem is None:
        known = [g.memory for g in granted if g.memory is not None]
        if known and len(known) == len(granted):
            cls = attrs.evolve(cls, gpu_mem=min(known))
    return cls


def _host_token() -> str | None:
    """A digest of this boot's random id, never a host name (RFC 0005 section 5.2)."""
    try:
        with open("/proc/sys/kernel/random/boot_id", "rb") as f:
            return hashlib.sha256(f.read().strip()).hexdigest()[:16]
    except OSError:
        return None


def _resolve(reference: ty.Any) -> ty.Any:
    if not isinstance(reference, str):
        return reference
    module, _, qualname = reference.partition(":")
    obj: ty.Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj() if isinstance(obj, type) else obj


def _machinery(exc: Exception) -> Exception:
    """``exc``, raised by the pool's machinery around a call, as an infrastructure fault of the
    residency (a storage error is never the job's): an attached record, unless it carries one.
    """
    if attached(exc) is None:
        attach(
            exc,
            FaultRecord(
                "infrastructure", type(exc).__name__, str(exc)[:2000], "pool", gap=True
            ),
        )
    return exc


# -- the worker --------------------------------------------------------------------------------


class _Call:
    """One call's bookkeeping while it runs."""

    def __init__(self) -> None:
        self.lost = False
        self.expired = False
        self.abandoned = False
        self.result: ty.Any = None
        self.error: BaseException | None = None
        self.wall = 0.0


def _squeue(argv: list[str], timeout: float) -> str:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=True
    ).stdout


def allocation_end(
    environ: ty.Mapping[str, str] = os.environ,
    squeue: ty.Callable[[list[str]], str] | None = None,
    timeout: float = PoolSettings().squeue_timeout,
) -> float | None:
    """The time (``time.time``) at which this process's Slurm allocation ends:
    ``SLURM_JOB_END_TIME``, else the end ``squeue -h -j $SLURM_JOB_ID -o %e`` reports within
    ``timeout`` seconds; None when neither gives one (no end is assumed)"""
    if squeue is None:
        squeue = functools.partial(_squeue, timeout=timeout)
    try:
        return float(environ["SLURM_JOB_END_TIME"])
    except (KeyError, ValueError):
        pass
    job = environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        said = squeue(["squeue", "-h", "-j", job, "-o", "%e"]).strip()
        return datetime.datetime.fromisoformat(said).timestamp()
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def claims_fit(
    ends_at: float | None, now: float, quantum_s: float, call_timeout: float | None
) -> bool:
    """Whether a worker whose allocation ends at ``ends_at`` may claim more: the time left
    holds a quantum and a call's timeout"""
    if ends_at is None:
        return True
    return ends_at - now >= max(quantum_s, call_timeout or 0)


class Worker:
    """A pool worker of one class; `run` is the loop of RFC 0005 section 4.3.

    Parameters
    ----------
    root : os.PathLike
        The pool directory.
    cls : str or WorkerClass
        The class, or its name in the pool's ``classes/``.
    providers : sequence of Provider or "module:qualname"
        Providers of provided fields, before the jobs' own and the entry points; references are
        imported after the beater starts.
    classify : callable, optional
        ``classify(exc, phase) -> FaultRecord``; a record attached to an exception or its chain
        overrides it (`pydra.workers.pool.faults.fault_of`). None: the reference the newest
        coordinator wrote into the pool, else the default.
    settings : PoolSettings
    faults : FaultBudgets, optional
        None: the budgets the newest coordinator wrote into the pool, else the defaults.
    substrate : bool
        Started by a substrate: exit after ``orphan_after_s`` without a live coordinator.
    hardware, memory, device_memory : callable
        Read the hardware, this process's ``(current, peak)`` host bytes and device-wide bytes.
    codec : JobCodec
    token : str, optional
        The worker's token (32 hex digits), named by the substrate that started it so that its
        coordinator finds its ready mark; default a fresh one.
    """

    def __init__(
        self,
        root: os.PathLike,
        cls: "str | WorkerClass",
        *,
        providers: ty.Sequence["Provider | str"] = (),
        classify: Classifier | None = None,
        settings: PoolSettings = PoolSettings(),
        faults: FaultBudgets | None = None,
        substrate: bool = False,
        hardware: ty.Callable[[], HardwareClass] = read_hardware,
        memory: ty.Callable[[], tuple[int | None, int | None]] = host_memory,
        device_memory: ty.Callable[[], int | None] = device_memory,
        codec: JobCodec | None = None,
        token: str | None = None,
    ):
        self.root = Path(root)
        self._token_arg = token
        self._cls_arg = cls
        self._provider_args = tuple(providers)
        self.classify: Classifier = classify or _faults.classify
        self._classify_arg = classify
        self.settings = settings
        self.ends_at = allocation_end(timeout=self.settings.squeue_timeout)
        self.faults = faults or FaultBudgets()
        self._faults_arg = faults
        self.substrate = substrate
        self.hardware = hardware
        self.memory = memory
        self.device_memory = device_memory
        self.codec = codec or PrivatePickle()
        self.pool: PoolRoot | None = None
        self.client: BeaterClient | None = None
        self.cls: WorkerClass | None = None
        self.token: str | None = None
        self.providers: tuple = ()
        self.polls = 0
        self.live_polls = (
            0  # loop passes on which a coordinator was judged live (substrate)
        )
        self.orphan_polls = (
            0  # loop passes since then with none live, before the orphan exit
        )
        self.served: list[tuple[str, str]] = []
        self.claims: list[int] = []
        self.stop = threading.Event()
        self.recycle: str | None = None
        self._wake = threading.Event()
        self._held: dict[Path, Claim] = {}
        self._lost: set[Path] = set()
        self._expired: set[Path] = set()
        self._warm: dict[str, int] = {}
        self._seq = 0
        self._filled_gpu_mem = False
        self._baseline_bytes: int | None = None
        self._device_baseline: int | None = None
        self._device_read = False
        self._worker_lease: DirectoryLease | None = None
        self._gaps: set[str] = set()
        # residencies this worker holds: (queue, label) -> (coordinator epoch, reason, latest fault);
        # honoured only while that epoch is the newest and the hold mark stands
        self._holds: dict[tuple[str, str], tuple[int, str, FaultRecord]] = {}
        self._epoch: int | None = None  # the newest coordinator epoch this worker read
        self._faults_epoch: int | None = None  # the epoch whose fault record it read
        self._recorded = (
            False  # whether any coordinator of the pool wrote a fault record
        )
        self._scanned = (
            False  # whether every epoch was searched since the newest changed
        )
        self._last_live = time.monotonic()
        self._pinged = 0.0

    # -- the run -------------------------------------------------------------------------------

    def run(self) -> int:
        """Run the worker until it exits; return its exit code."""
        try:
            self.pool = PoolRoot.open(
                self.root, self.settings.lease, create=False, probe=False
            )
        except PoolSettingsMismatch as err:
            logger.error("refusing to start: %s", err)
            return EXIT_SETTINGS
        except (PoolOwnerMismatch, FileNotFoundError, StoreUnavailable) as err:
            logger.error("refusing to start: %s", err)
            return EXIT_USAGE
        try:
            if not self._refresh():
                return EXIT_FAULTS
            return self._run()
        finally:
            if self.token is not None:  # a worker refused before it took a token
                try:
                    self.pool.write_worker_record(
                        self.token, EXIT_MARK, {"at": time.time()}
                    )
                except OSError as err:
                    logger.warning("could not write the exit mark: %s", err)
            self.pool.close()

    def _refresh(self) -> bool:
        """When the newest coordinator epoch has changed, forget the holds of older epochs; until
        that epoch's fault record has been read (its coordinator writes it after taking the epoch),
        read the newest record (`_read_faults`). False when the classifier is unusable.
        """
        epochs = self.pool.epochs()
        newest = epochs[-1] if epochs else 0
        if newest != self._epoch:
            self._epoch = newest
            self._scanned = False
            self._holds = {k: v for k, v in self._holds.items() if v[0] == newest}
        elif self._faults_epoch == newest:
            return True
        return self._read_faults()

    def _awaiting_faults(self) -> bool:
        """Whether the pool's coordinators write fault records and the newest epoch's has not been
        read: claims made now are not run (its coordinator writes it after taking the epoch).
        """
        return self._recorded and self._faults_epoch != self._epoch

    def _read_faults(self) -> bool:
        """Take the budgets and the classifier the newest coordinator wrote into the pool, unless
        given; False (logged) when the classifier reference cannot be imported here or names
        something not callable: the worker then exits `EXIT_FAULTS` rather than type faults by
        another rule."""
        found = self.pool.faults(self._epoch) if self._epoch else None
        if found is None and not self._scanned:
            # an older epoch's record, read once per epoch change, not on every pass
            self._scanned = True
            found = self.pool.faults()
        if found is None and self._recorded:
            return True  # the record in force stays until the newest epoch's is written
        if found is not None:
            self._recorded = True
        self._faults_epoch, record = found if found is not None else (None, {})
        if self._faults_arg is None:
            budgets = record.get("budgets")
            self.faults = (
                FaultBudgets.from_json(budgets)
                if isinstance(budgets, dict)
                else FaultBudgets()
            )
        if self._classify_arg is not None:
            return True
        reference = record.get("classify")
        if reference is None:
            self.classify = _faults.classify
            return True
        try:
            classify = _resolve(reference)
        except Exception as err:
            logger.error(
                "the pool's classifier %r cannot be imported here: %r", reference, err
            )
            return False
        if not callable(classify):
            logger.error("the pool's classifier %r is not callable", reference)
            return False
        self.classify = classify
        return True

    def beater_alive(self) -> bool:
        """Whether the worker's beater process still runs."""
        return (
            self.client is not None
            and self.client.process.poll() is None
            and self.client.alive
        )

    def _run(self) -> int:
        cls = self._class()
        if cls is None:
            return EXIT_USAGE
        self._baseline_bytes = self.memory()[0]
        from pydra.engine.procgroup import process_baseline

        # OOM attribution is per worker: a kill recorded before this start is not this worker's
        process_baseline().reset()
        self.client = pool_beater(
            self.settings.timeout_grace,
            on_lost=self._on_lost,
            on_expired=self._on_expired,
            on_dead=self._on_dead,
        )
        try:
            self.providers = tuple(_resolve(p) for p in self._provider_args)
            self.token = self._token_arg or uuid.uuid4().hex
            self.pool.worker_dir(self.token).close()
            lease = DirectoryLease.at(
                self.pool.path / "workers" / self.token / "lease",
                self.pool.lease_settings,
                beat="external",
            )
            if lease.try_take() is not None:
                raise RuntimeError(f"the new worker lease {lease.lockfile} is held")
            self._worker_lease = lease
            self.client.hold(lease)
            hardware = self.hardware()
            try:
                self.cls = check_hardware(cls, hardware)
            except WrongHardware as err:
                logger.error("refusing to start: %s", err)
                return EXIT_HARDWARE
            self._filled_gpu_mem = cls.gpu_mem is None and self.cls.gpu_mem is not None
            from pydra.engine import resident

            resident.set_process_device(self._device())
            self.pool.write_worker_record(
                self.token,
                "ready.json",
                {
                    "class": self.cls.name,
                    "gpu_mem": self.cls.gpu_mem,
                    "hardware": hardware.to_json(),
                    "host": _host_token(),
                    "started": time.time(),
                    "uuid": uuid.uuid4().hex,
                },
            )
            logger.info(
                "worker %s of class %r ready on %s",
                self.token,
                self.cls.name,
                self.pool.path,
                extra={"event": "ready"},
            )
            with self._signals():
                return self._loop()
        except BeaterDead as err:
            logger.error("the beater died: %s", err)
            return EXIT_RECYCLE
        finally:
            self._shutdown()

    def _class(self) -> WorkerClass | None:
        if isinstance(self._cls_arg, WorkerClass):
            return self._cls_arg
        record = self.pool.worker_class(self._cls_arg)
        if record is None:
            logger.error("refusing to start: the pool has no class %r", self._cls_arg)
            return None
        return class_from_record(record)

    @contextlib.contextmanager
    def _signals(self) -> ty.Iterator[None]:
        """Drain on SIGTERM (in the main thread) and on the worker's `STOP_MARK`, which a thread
        looks for every ``poll_s``: either sets ``stop`` and wakes the worker."""

        def drain(signum=None, frame=None):
            self.stop.set()
            self._wake.set()

        done = threading.Event()
        watcher = threading.Thread(
            target=self._watch_stop_mark,
            args=(done, drain),
            name="pydra-pool-stop-mark",
            daemon=True,
        )
        watcher.start()
        main = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, drain) if main else None
        try:
            yield
        finally:
            if main:
                signal.signal(signal.SIGTERM, previous)
            done.set()
            watcher.join(self.settings.poll_s)  # a read stuck on hung storage is left

    def _watch_stop_mark(self, done: threading.Event, drain) -> None:
        failing = False
        while not self.stop.is_set():
            try:
                marked = self.pool.worker_record(self.token, STOP_MARK) is not None
            except OSError as err:
                if not failing:
                    logger.warning(
                        "reading the stop mark of worker %s failed, retrying: %s",
                        self.token,
                        err,
                    )
                failing, marked = True, False
            else:
                failing = False
            if marked:
                logger.info("worker %s was told to stop: draining", self.token)
                drain()
                return
            if done.wait(self.settings.poll_s):
                return

    def _device(self):
        """The device providers load on: the first granted GPU on a CUDA class (an index into
        this process's ``CUDA_VISIBLE_DEVICES``), else the CPU."""
        from pydra.engine.resident import Device

        if self.cls.device == "cuda":
            return Device("cuda", 0)
        return Device()

    def _shutdown(self) -> None:
        for claim in list(self._held.values()):
            self._release(claim)
        try:
            from pydra.engine import resident

            resident.close_process_set()
            resident.set_process_device(None)
        finally:
            if self._worker_lease is not None:
                self._beater_release(self._worker_lease.token_dir)
                self._worker_lease.__exit__(None, None, None)
            if self.client is not None:
                self.client.close()

    # -- beater events -------------------------------------------------------------------------

    def _on_lost(self, token_dir: Path, reason: str) -> None:
        self._lost.add(Path(token_dir))
        self._wake.set()

    def _on_expired(self, token_dir: Path) -> None:
        self._expired.add(Path(token_dir))
        self._wake.set()

    def _on_dead(self, reason: str) -> None:
        self.recycle = self.recycle or f"the beater died: {reason}"
        self._wake.set()

    def _beater_release(self, token_dir: Path | None) -> None:
        if token_dir is None or self.client is None:
            return
        with contextlib.suppress(BeaterDead, ValueError):
            self.client.release(token_dir)

    # -- the claim loop ------------------------------------------------------------------------

    def _loop(self) -> int:
        quantum_jobs, quantum_s = self.cls.quantum_jobs, self.cls.quantum_s
        current: str | None = None
        spent, began, idle = 0, time.monotonic(), 0
        self._last_live = time.monotonic()
        while not self.stop.is_set() and self.recycle is None:
            if not self._refresh():
                return EXIT_FAULTS
            if self.substrate and self._orphaned():
                logger.info(
                    "no live coordinator for %ss: exiting", self.settings.orphan_after_s
                )
                break
            if not claims_fit(
                self.ends_at, time.time(), quantum_s, self.settings.call_timeout
            ):  # infrastructure: no call is cut at the limit, none is charged
                self.recycle = self.recycle or "its allocation ends before a quantum"
                break
            ended = current is not None and (
                spent >= quantum_jobs or time.monotonic() - began >= quantum_s
            )
            prefer = None if ended else current
            asked = time.monotonic()
            held = self._next(prefer, quantum_jobs - spent if prefer else quantum_jobs)
            if not held and prefer is not None:
                prefer = None  # nothing more fits the quantum: a new one begins
                held = self._next(None, quantum_jobs)
            if not held:
                current = None
                self.polls += 1
                idle += 1
                if idle >= self.settings.idle_polls:
                    break
                self.stop.wait(self.settings.poll_s)
                continue
            if (
                not self._refresh()
            ):  # a newer epoch may have begun since this pass began
                for claim in held:
                    self._release(claim)
                return EXIT_FAULTS
            if self._awaiting_faults():
                logger.info(
                    "coordinator epoch %s has not written its fault budgets yet: released "
                    "%d claim(s) unrun",
                    self._epoch,
                    len(held),
                )
                for claim in held:
                    self._release(claim)
                self.stop.wait(self.settings.poll_s)
                continue
            idle = 0
            queue = held[0].queue
            if queue != current or prefer is None:
                current, spent, began = queue, 0, time.monotonic()
            jobs = sum(len(c.ticket.jobs) for c in held)
            spent += jobs
            self.claims.append(jobs)
            self.served.extend((c.queue, c.digest) for c in held)
            self._run_held(held, time.monotonic() - asked)
        if self.recycle is not None:
            logger.warning("worker %s recycles: %s", self.token, self.recycle)
            return EXIT_RECYCLE
        return EXIT_OK

    def _queues(self) -> list[str]:
        name = self.cls.name
        return [q for q in self.pool.queues() if q == name or q.startswith(name + "#")]

    def _next(self, prefer: str | None, left: int) -> list[Claim]:
        """Claims from ``prefer`` (up to ``left`` jobs), else from the queue whose oldest waiting
        ticket is oldest; one ticket on a queue not yet served, one-job tickets gathered once warm.
        """
        names = self._queues()
        ages = {n: self.pool.oldest(n) for n in names if n != prefer}
        order = [prefer] if prefer in names else []
        order += sorted((n for n, a in ages.items() if a is not None), key=ages.get)
        for name in order:
            size = max(left if name == prefer else self.cls.quantum_jobs, 1)
            warm = name in self._warm
            first = self._claim(name, size if warm else None)
            if first is None:
                if self.recycle is not None:
                    return []
                continue
            held = [first]
            if warm and len(first.ticket.jobs) == 1:
                while len(held) < size and self.recycle is None:
                    more = self._claim(name, 1)
                    if more is None:
                        break
                    held.append(more)
            return held
        return []

    def _claim(self, queue: str, max_jobs: int | None) -> Claim | None:
        left = None if self.ends_at is None else self.ends_at - time.time()
        claim = self.pool.claim(
            queue, beat="external", max_jobs=max_jobs, max_time=left
        )
        if claim is None:
            return None
        try:
            self.client.hold(claim.lease)
        except BeaterDead as err:
            self.recycle = self.recycle or f"the beater died: {err}"
            claim.release()
            return None
        self._held[claim.lease.token_dir] = claim
        if claim.took_over:
            losses = self.pool.losses(queue, claim.digest)
            broke = losses[-1].get("broke", []) if losses else []
            tokens = [b.get("token") for b in broke]
            age = max((b.get("age") or 0 for b in broke), default=None)
            logger.info(
                "reclaimed ticket %s of %r from %s, %ss stale",
                claim.digest,
                queue,
                tokens,
                age,
                extra={
                    "event": "reclaimed",
                    "ticket": claim.digest,
                    "tokens": tokens,
                    "age": age,
                },
            )
        logger.info(
            "claimed ticket %s of %r (%d jobs)",
            claim.digest,
            queue,
            len(claim.ticket.jobs),
            extra={"event": "claimed", "ticket": claim.digest},
        )
        return claim

    def _release(self, claim: Claim) -> None:
        """Stop beating the claim and release it; an expired claim is left to go stale, so that
        the next claimant takes it over and types the timeout (RFC 0005 section 5.3)."""
        token_dir = claim.lease.token_dir
        self._held.pop(token_dir, None)
        self._beater_release(token_dir)
        if token_dir in self._expired:
            return
        try:
            claim.release()
        except OSError as err:
            logger.warning("releasing %s failed: %s", claim, err)
        if token_dir is not None:
            self._lost.discard(token_dir)
            self._expired.discard(token_dir)

    def _lost_claim(self, claim: Claim) -> bool:
        token_dir = claim.lease.token_dir
        return (
            token_dir is None or token_dir in self._lost or claim.lease.lost is not None
        )

    def _orphaned(self) -> bool:
        if self._coordinator_live():
            self._last_live = time.monotonic()
            self.live_polls += 1
            return False
        if time.monotonic() - self._last_live > self.settings.orphan_after_s:
            return True
        self.orphan_polls += 1
        return False

    def _coordinator_live(self) -> bool:
        """Whether the newest coordinator epoch has not ended and its lease was beaten within
        ``stale_after``, on the storage's clock."""
        epochs = self.pool.epochs()
        if not epochs:
            return False
        where = self.pool.path / "coordinator" / str(epochs[-1])
        if (where / "ended").exists():
            return False
        newest = newest_mark(where / "lease")
        if newest is None:
            return False
        clock = self.pool.path / "workers" / self.token / "clock"
        with contextlib.suppress(FileExistsError):
            os.close(os.open(clock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        os.utime(clock)
        return os.stat(clock).st_mtime - newest < self.settings.lease.stale_after

    # -- running claims ------------------------------------------------------------------------

    def _run_held(self, held: list[Claim], claim_s: float) -> None:
        group: dict[str, ty.Any] = {"claim_s": claim_s, "device": None, "read": False}
        if self.cls.device == "cuda" and not self._device_read:
            self._device_baseline = self.device_memory()
            self._device_read = True
        try:
            for claim in held:
                if self.stop.is_set() or self.recycle is not None:
                    break
                if self._lost_claim(claim) or self._held_here(claim):
                    continue
                group.pop("stop", None)
                self._run_ticket(claim, group)
        finally:
            for claim in held:
                if claim.lease.token_dir in self._held:
                    self._release(claim)

    def _run_ticket(self, claim: Claim, group: dict) -> None:
        if claim.took_over:
            if not self._take_over(claim):
                return
        else:
            self._expired_pass(claim)
        entries = claim.open_jobs()
        jobs: list[tuple[TicketJob, "Job"]] = []
        for entry in entries:
            job = self._decode(claim, entry)
            if job is not None:
                jobs.append((entry, job))
        if not jobs:
            return
        if self._refuse(claim, jobs):
            return
        try:
            load_s, loaded = self._load(jobs[0][1])
        except Exception as exc:
            self._load_fault(claim, [e for e, _ in jobs], exc)
            return
        queue = claim.queue
        cold = queue not in self._warm or loaded
        if cold:
            self._warm[queue] = 0
        group["load_s"], group["cold"], group["reload"] = load_s, cold, False
        self._run_lone(claim, jobs, group)

    def _fits(self, job: "Job") -> bool:
        """Whether the time left in the allocation holds ``job``'s declared time, or else the
        call timeout; True when no end is known or neither is set"""
        if self.ends_at is None:
            return True
        resources = job.resources
        need = (
            resources.time
            if resources is not None and resources.time is not None
            else self.settings.call_timeout
        )
        return need is None or self.ends_at - time.time() >= need

    def _run_lone(
        self, claim: Claim, jobs: list[tuple[TicketJob, "Job"]], group: dict
    ) -> None:
        """Run the ticket's jobs one call each; before each, the checks a call has, and a job the
        time left does not hold recycles the worker, leaving it and the rest unrun"""
        for entry, job in jobs:
            if (
                self.stop.is_set()
                or self.recycle is not None
                or self._lost_claim(claim)
            ):
                return
            if not claim.lease.prove():
                self._abandon(claim, "its proof failed")
                return
            if not self._within_crash_budget(claim, entry):
                continue
            if self._held_here(claim):
                return
            if not self._fits(job):
                self.recycle = f"the allocation ends before {job.name} could finish"
                return
            self._run_job(claim, entry, job, group)
            if self._lost_claim(claim) or group.get("stop"):
                if self._lost_claim(claim):
                    self._abandon(claim, "it was lost")
                return
            group["load_s"], group["cold"] = 0.0, False

    def _decode(
        self, claim: Claim, entry: TicketJob, attempt: str | None = None
    ) -> "Job | None":
        """The job a ticket entry names; None when its payload cannot be read. A storage error
        (``ESTALE``, ``EIO``) requeues the job as an infrastructure fault, bounded; any other
        (a digest mismatch, an untrusted directory, a payload that does not decode) is ``failed``.
        """
        from pydra.engine.portable.errors import EntryNotTrusted

        try:
            return read_payload(self.codec, entry.payload, None)
        except Exception as exc:
            logger.error("the payload of %s is unreadable: %r", entry.checksum, exc)
            fault = fault_of(exc, "pool", self.classify)
            if isinstance(exc, OSError) and not isinstance(exc, EntryNotTrusted):
                self._infrastructure(claim, [entry], fault, {"stop": False})
                return None
            current = claim.current(entry)
            if attempt is None and current is not None:
                attempt = current[1]["attempt"]
            if attempt is not None:
                claim.outcome(entry, attempt, "failed", fault=fault.to_json())
            return None

    def _refuse(self, claim: Claim, jobs: list[tuple[TicketJob, "Job"]]) -> bool:
        """Withdraw the ticket's jobs (reason ``refused``) when a class whose ``gpu_mem`` was unknown
        reads less device memory than one of them declares (RFC 0005 sections 3.3, 6).
        """
        if not self._filled_gpu_mem:
            return False
        share = self.cls.gpu_mem // self.cls.workers_per_device
        need = max(
            (
                parse_size(job.resources.gpu_mem) or 0
                for _, job in jobs
                if job.resources is not None and job.resources.gpus
            ),
            default=0,
        )
        if need <= share:
            return False
        logger.warning(
            "ticket %s needs %d bytes of device memory; this worker has %d: refused",
            claim.digest,
            need,
            share,
        )
        for entry, _ in jobs:
            current = claim.current(entry)
            if current is not None:
                claim.outcome(
                    entry,
                    current[1]["attempt"],
                    "withdrawn",
                    reason="refused",
                    memory=self.cls.gpu_mem,
                    need=need,
                )
        return True

    def _load(self, job: "Job") -> tuple[float, bool]:
        """Load the job's provided state into the process's resident set (and leave it resident);
        return the seconds taken and whether anything was loaded."""
        from pydra.engine import resident

        fields = resident.provided_fields(job.task)
        if not fields:
            return 0.0, False
        self._add_providers(job)
        rs = resident.process_set()
        began, loaded = time.monotonic(), False
        for field, name in fields:
            provider = resident.find_provider(job, name)
            spec = getattr(job.task, field)
            key = resident.spec_key(job.task, field, name)
            loaded = loaded or key not in rs
            rs.get(
                key,
                lambda p=provider, s=spec: p.load(s, resident.process_device()),
                close=provider.close,
                footprint=provider.footprint(spec),
            )
            rs.release(key)
        return time.monotonic() - began, loaded

    def _add_providers(self, job: "Job") -> None:
        submitter = job.submitter
        if submitter is not None and self.providers:
            own = tuple(submitter.providers)
            if not own or own[: len(self.providers)] != self.providers:
                submitter.providers = (*self.providers, *own)

    def _load_fault(
        self, claim: Claim, entries: list[TicketJob], exc: BaseException
    ) -> None:
        """A load failed: a resource fault moves the ticket's jobs up the ladder (`_resource`);
        anything else is one infrastructure fault of the residency (`_infrastructure`). The claim is
        released."""
        fault = fault_of(exc, "load", self.classify)
        logger.error("loading the residency of %s failed: %r", claim.digest, exc)
        if fault.kind == "resource":
            self._resource(claim, entries, fault, "load")
        else:
            self._infrastructure(claim, entries, fault, {"stop": False}, phase="load")
        self._release(claim)

    def _gap(self, fault: FaultRecord) -> None:
        """Report an unclassified exception from the pool's machinery, once per type."""
        if fault.gap and fault.type not in self._gaps:
            self._gaps.add(fault.type)
            logger.warning(
                "unclassified %s at %s, typed an infrastructure fault (a gap in the "
                "classifier): %s",
                fault.type,
                fault.phase,
                fault.message,
                extra={"event": "gap", "type": fault.type},
            )

    def _infrastructure(
        self,
        claim: Claim,
        entries: list[TicketJob],
        fault: FaultRecord,
        group: dict,
        sample: Sample | None = None,
        phase: str = "call",
    ) -> str:
        """One infrastructure fault of the ticket's residency at ``phase`` (``call`` or ``load``):
        below the budget its jobs are requeued, uncharged, each under a ``restart`` generation
        written now; at it the residency is held (`_count_fault`). The caller releases the ticket.
        Return ``"held"`` or ``"stopped"``."""
        group["stop"] = True
        entries = [e for e in entries if claim.current(e) is not None]
        if not entries:
            return "stopped"
        return self._count_fault(
            claim, fault, phase, requeue=entries, group=group, sample=sample
        )

    def _count_fault(
        self,
        claim: Claim,
        fault: FaultRecord,
        phase: str,
        *,
        requeue: list[TicketJob],
        group: dict,
        sample: Sample | None = None,
    ) -> str:
        """Record an infrastructure fault of the ticket's residency for the newest coordinator epoch.
        Below ``infrastructure_replacements`` faults in that epoch, ``requeue``'s jobs are requeued
        under ``restart`` generations; at it the residency is held on its queue
        (``held-<epoch>-<label>``), remembered by this worker so that no gathered ticket of it runs,
        and every open job of the ticket is settled ``held`` with a reason naming the residency and
        the latest cause (RFC 0005 section 6)."""
        self._gap(fault)
        epochs = self.pool.epochs()
        epoch = epochs[-1] if epochs else 1
        label = claim.ticket.residency_label
        budget = self.faults.infrastructure_replacements
        count = self.pool.infrastructure_fault(
            claim.queue,
            epoch,
            label,
            {
                "fault": fault.to_json(),
                "phase": phase,
                "ticket": claim.digest,
                "worker": self.token,
            },
        )
        if count < budget:
            for entry in requeue:
                claim.restart(entry, phase=phase)
            logger.warning(
                "infrastructure fault %d of residency %s on %r in epoch %d at %s: %s: %s%s",
                count,
                label,
                claim.queue,
                epoch,
                phase,
                fault.type,
                fault.message,
                "; requeued" if requeue else "",
            )
            return "stopped"
        self.pool.hold(claim.queue, epoch, label)
        reason = (
            f"residency {label} held on queue {claim.queue!r} after {count} infrastructure "
            f"faults in coordinator epoch {epoch} (the budget is {budget}); the latest, at "
            f"{phase}: {fault.type}: {fault.message}"
        )
        logger.error(reason, extra={"event": "held", "residency": label})
        self._holds[(claim.queue, label)] = (epoch, reason, fault)
        group["stop"] = True
        faulted = {e.key for e in requeue}
        for entry in claim.open_jobs():
            current = claim.current(entry)
            if current is None:
                continue
            fields = {"reason": reason, "fault": fault.to_json()}
            if sample is not None and entry.key in faulted:
                self._settle(
                    claim, entry, current[1]["attempt"], "held", sample, **fields
                )
            else:
                claim.outcome(entry, current[1]["attempt"], "held", **fields)
        return "held"

    def _held_here(self, claim: Claim) -> bool:
        """Whether this worker held the claim's residency in the newest coordinator epoch and the
        hold mark still stands (``pydracli pool release-hold`` lifts it); if so settle the claim's
        open jobs ``held``, unrun and unloaded."""
        key = (claim.queue, claim.ticket.residency_label)
        found = self._holds.get(key)
        if found is None:
            return False
        epoch, reason, fault = found
        if key[1] not in self.pool.holds(claim.queue, epoch):  # lifted by release-hold
            del self._holds[key]
            return False
        for entry in claim.open_jobs():
            current = claim.current(entry)
            if current is not None:
                claim.outcome(
                    entry,
                    current[1]["attempt"],
                    "held",
                    reason=reason,
                    fault=fault.to_json(),
                )
        return True

    def _stored(self, entry: TicketJob) -> bool:
        """Whether the store holds a committed result for the job (RFC 0005 section 5.5: a taker
        looks the job up first)."""
        job = None
        with contextlib.suppress(Exception):
            job = read_payload(self.codec, entry.payload, None)
        if job is None or job.store is None or not job.store.finds:
            return False
        try:
            found = job.store.find(job)
        except Exception:
            return False
        return found.result is not None

    def _within_crash_budget(self, claim: Claim, entry: TicketJob) -> bool:
        crashes = self.pool.charges(*entry.key)["crashes"]
        if crashes < self.faults.crashes_charged:
            return True
        current = claim.current(entry)
        if current is not None and self._stored(entry):
            claim.outcome(entry, current[1]["attempt"], "done", served=True)
            return False
        if current is not None:
            claim.outcome(
                entry,
                current[1]["attempt"],
                "failed",
                fault=FaultRecord(
                    "content",
                    "JobCrashedWorker",
                    f"the job's workers were lost {crashes} times",
                    "call",
                ).to_json(),
            )
        return False

    # -- one call ------------------------------------------------------------------------------

    def _run_job(self, claim: Claim, entry: TicketJob, job: "Job", group: dict) -> str:
        """Run one job as a call of one through the bisection engine, retried up to
        ``content_attempts``, and settle it; return what became of it: ``done``, ``failed``,
        ``moved``, ``held``, ``lost``, ``expired`` or ``stopped`` (released unsettled).

        Only the task's own exception and a load's are typed by the classifier. An `Exception` from
        the pool's machinery around the call (starting the attempt, the call record, the sample) is
        an infrastructure fault of the residency; a `BaseException` from it propagates.
        """
        tries = 0
        while True:
            tries += 1
            state: dict[str, ty.Any] = {
                # every attempt after the first runs a freshly decoded job
                "fresh": tries > 1,
                "attempt": None,
                "sample": None,
                "verdict": None,
            }

            def call(batch: list[TicketJob], state=state) -> list:
                [item] = batch
                try:
                    attempt = claim.start(item, self.token)
                    if attempt is None:  # superseded or withdrawn: nothing is written
                        raise Abandon("superseded")
                    current = job
                    if state["fresh"]:
                        current = self._decode(claim, item, attempt)
                        if current is None:
                            raise Abandon("unreadable")
                    state["fresh"], state["attempt"] = True, attempt
                except Abandon:
                    raise
                except Exception as exc:
                    raise _machinery(exc)
                try:
                    self._load(current)
                except Exception as exc:
                    raise attach(exc, fault_of(exc, "load", self.classify))
                try:
                    ran = self._invoke(claim, item, current, attempt, group)
                except Exception as exc:
                    raise _machinery(exc)
                if isinstance(ran, str):
                    state["verdict"] = ran
                    raise Abandon(ran)
                done, state["sample"] = ran
                if done.error is None:
                    return [done.result]
                if isinstance(done.error, Exception):
                    raise done.error
                # the task's own KeyboardInterrupt or SystemExit is its content fault
                raised = RuntimeError(f"{type(done.error).__name__}: {done.error}")
                raise raised from done.error

            survived: list[FaultRecord] = []
            try:
                [settled] = engine.bisect(
                    [entry],
                    call,
                    classify=self._classify_call,
                    reload=lambda fault: self._reload(group),
                    lost=lambda item: self.pool.charges(*item.key)["crashes"],
                    crashes_charged=self.faults.crashes_charged,
                    survived=survived.append,
                )
            except Abandon:
                return state["verdict"] or "stopped"
            except Exception as exc:
                record = attached(exc)
                if record is None or record.phase != "load":
                    raise
                # a load inside the call: never split or retried; every open job of the ticket
                self._load_fault(claim, claim.open_jobs() or [entry], exc)
                group["stop"] = True
                return "stopped"
            verdict = self._settle_job(claim, entry, settled, state, group, tries)
            for fault in survived:
                self._survived(claim, fault, group)
            if verdict != "retry":
                return verdict

    def _settle_job(
        self,
        claim: Claim,
        entry: TicketJob,
        settled: "engine.Settled",
        state: dict,
        group: dict,
        tries: int,
    ) -> str:
        """Settle one job the engine settled; ``retry`` for a content fault with attempts left."""
        attempt, sample, fault = state["attempt"], state["sample"], settled.fault
        if fault is None:
            return self._settle(claim, entry, attempt, "done", sample)
        if settled.charged or fault.kind not in (
            "content",
            "resource",
            "infrastructure",
        ):
            return self._settle(
                claim, entry, attempt, "failed", sample, fault=fault.to_json()
            )
        if fault.kind == "content":
            if tries < self.faults.content_attempts:
                return "retry"
            return self._settle(
                claim, entry, attempt, "failed", sample, fault=fault.to_json()
            )
        if fault.kind == "resource":
            self._resource(claim, [entry], fault, "call")
            return "moved"
        return self._infrastructure(claim, [entry], fault, group, sample)

    def _survived(self, claim: Claim, fault: FaultRecord, group: dict) -> None:
        """A crash whose job completed afterwards: an infrastructure fault of the residency, charged
        to no job; at the budget the residency is held."""
        self._count_fault(claim, fault, "call", requeue=[], group=group)

    def _invoke(
        self, claim: Claim, entry: TicketJob, job: "Job", attempt: str, group: dict
    ) -> "tuple[_Call, Sample] | str":
        """Run one job as one call under ``attempt``: the call record before, ``done`` after, its
        deadline sent to the beater. Return the call and its sample, or what ended it without a
        verdict: ``lost``, ``expired`` or ``stopped`` (released unsettled)."""
        job.attempt = attempt
        job._work_dir_memo = None
        job.chdir = False
        job.claim = claim.lease
        self._add_providers(job)
        self._seq += 1
        seq = self._seq
        token_dir = claim.lease.token_dir
        listed = [
            {
                "checksum": entry.checksum,
                "fingerprint": entry.fingerprint,
                "attempt": attempt,
            }
        ]
        limit = None
        if job.resources is not None and job.resources.time:
            limit = float(job.resources.time)
        elif self.settings.call_timeout is not None:
            limit = self.settings.call_timeout
        deadline = (
            None if limit is None else time.time() + limit + self.settings.timeout_grace
        )
        payload = None
        if deadline is not None:
            payload = expiry_payload(
                self.pool.path, self.settings.lease, claim.queue, claim.digest, listed
            )
        self.pool.write_call(
            self.token,
            seq,
            {
                "seq": seq,
                "queue": claim.queue,
                "ticket": claim.digest,
                "jobs": listed,
                "deadline": deadline,
                "uuid": uuid.uuid4().hex,
            },
        )
        if deadline is not None:
            with contextlib.suppress(BeaterDead):  # then the loop enforces it
                self.client.deadline(token_dir, deadline, payload)
        call = self._run_call(claim, job, deadline, payload)
        if call.abandoned and self.stop.is_set() and not call.expired:
            claim.drained(entry, self.token)
        if not call.abandoned:
            self.pool.call_done(self.token, seq)
            if deadline is not None and not call.expired:
                with contextlib.suppress(BeaterDead, ValueError):
                    self.client.deadline(token_dir, None)
        if call.abandoned or (self.stop.is_set() and call.error is not None):
            group["stop"] = True
            return "stopped"
        if call.expired:
            return "expired"
        if call.lost or isinstance(call.error, LeaseLost) or self._lost_claim(claim):
            self._lost.add(token_dir)
            return "lost"
        self._warm[claim.queue] = self._warm.get(claim.queue, 0) + 1
        return call, self._sample(claim, entry, call.wall, group)

    def _run_call(
        self,
        claim: Claim,
        job: "Job",
        deadline: float | None = None,
        payload: dict | None = None,
    ) -> _Call:
        """Run ``job.run`` on a thread; meanwhile prove the claim every renewal interval, act on the
        beater's ``lost`` and ``expired``, and on SIGTERM wait ``drain_s`` before abandoning.

        The beater is pinged every renewal interval. While the beater is dead the claim is not proved,
        and once the call's ``deadline`` passes its ``expired`` outcomes are written here
        (`pool_expire`) and the call is abandoned, so the worker recycles.
        """
        call = _Call()
        done = threading.Event()
        token_dir = claim.lease.token_dir

        def target() -> None:
            try:
                call.result = job.run(rerun=job.rerun)
            except BaseException as exc:  # noqa: B036 - reported to the worker's loop
                call.error = exc
            finally:
                done.set()
                self._wake.set()

        began = time.monotonic()
        thread = threading.Thread(target=target, name="pydra-pool-call", daemon=True)
        thread.start()
        interval = self.settings.lease.renew_interval
        proved, drain_until = time.monotonic(), None
        while not done.is_set():
            wait = interval
            if drain_until is not None:  # wake when the drain is over
                wait = max(0.0, min(interval, drain_until - time.monotonic()))
            self._wake.wait(wait)
            self._wake.clear()
            if done.is_set():
                break
            beaten = self._ping(interval)
            if not call.lost and self._lost_claim(claim):
                call.lost = True
                self._stop_call()
            elif not call.lost and beaten and time.monotonic() - proved >= interval:
                proved = time.monotonic()
                if not claim.lease.prove():
                    call.lost = True
                    self._stop_call()
            if (
                not beaten
                and not call.expired
                and deadline is not None
                and time.time() >= deadline
            ):
                self._expire_here(token_dir, payload)
                call.expired = True
                call.abandoned = True
                break
            if not call.expired and token_dir in self._expired:
                call.expired = True
                self.recycle = self.recycle or "a call passed its deadline"
                if self._stop_call():
                    self._ack_expired(token_dir)
            if self.stop.is_set():
                if drain_until is None:
                    drain_until = time.monotonic() + self.settings.drain_s
                elif time.monotonic() >= drain_until:
                    self._stop_call()  # the drain is over: stop what can be stopped
                    call.abandoned = True
                    break
        call.wall = time.monotonic() - began
        if token_dir in self._expired:
            call.expired = True
            if done.is_set():
                self._ack_expired(
                    token_dir
                )  # the call has ended: nothing for the beater to kill
        return call

    def _ping(self, interval: float) -> bool:
        """Whether the beater answers (at most one ping per renewal interval)."""
        if self.client is None or not self.client.alive:
            return False
        now = time.monotonic()
        if now - self._pinged < interval:
            return True
        self._pinged = now
        try:
            self.client.ping(
                interval
            )  # a miss is not a death: PING_MISSES in a row are
        except BeaterDead as err:
            self.recycle = self.recycle or f"the beater died: {err}"
            return False
        return True

    def _expire_here(self, token_dir: Path, payload: dict | None) -> None:
        """The beater is dead and the call passed its deadline: write its ``expired`` outcomes in
        this process, leave the claim unproved to go stale, and recycle."""
        from pydra.workers.pool.beater import pool_expire

        if payload is not None:
            try:
                pool_expire(token_dir, payload)
            except Exception as err:  # the claim goes stale all the same
                logger.error("writing the expired outcomes failed: %s", err)
        self._expired.add(Path(token_dir))
        self.recycle = self.recycle or "a call passed its deadline with the beater dead"
        self._stop_call()

    def _ack_expired(self, token_dir: Path) -> None:
        with contextlib.suppress(BeaterDead, ValueError):
            self.client.ack_expired(token_dir)

    def _stop_call(self) -> bool:
        """Stop the call in flight where it runs in a process-owning provider's process: stop each
        such provider's group and evict it from the process's resident set, so the next job loads
        afresh rather than reaching a dead process. An in-process call cannot be stopped. Return
        whether any provider was stopped."""
        if "pydra.engine.procgroup" not in sys.modules:
            return False
        from pydra.engine import resident
        from pydra.engine.procgroup import ProcessHandle

        rs = resident.existing_process_set()
        if rs is None:
            return False
        stopped = False
        for key, obj in rs.items():
            if isinstance(obj, ProcessHandle):
                with contextlib.suppress(Exception):
                    obj.group.stop()
                rs.evict(key)
                stopped = True
        if stopped:
            self._warm.clear()
        return stopped

    def _classify_call(self, error: BaseException, phase: str) -> FaultRecord:
        """The fault of a call that raised: the classifier's, or a resource fault when this
        worker's cgroup recorded an OOM kill since its latest baseline (taken at start and after
        each provider start), so a sibling's earlier kill is not this call's."""
        fault = self.classify(error, "call")
        if fault.kind != "resource" and "pydra.engine.procgroup" in sys.modules:
            from pydra.engine.procgroup import process_baseline

            if process_baseline().take():
                fault = FaultRecord(
                    "resource",
                    type(error).__name__,
                    str(error)[:2000],
                    "call",
                    reason="oom",
                )
        return fault

    def _reload(self, group: dict) -> None:
        """After a resource fault the resident state is not trusted: close it; the next job loads."""
        from pydra.engine import resident

        resident.close_process_set()
        self._warm.clear()
        group["reload"] = True

    # -- settling ------------------------------------------------------------------------------

    def _sample(
        self, claim: Claim, entry: TicketJob, wall: float, group: dict
    ) -> Sample:
        current, peak = self.memory()
        device = None
        if self.cls.device == "cuda":
            if not group["read"]:
                group["device"] = self.device_memory()
                group["read"] = True
            device = group["device"]
        since = self._warm.get(claim.queue, 1) - 1
        return Sample(
            wall_s=wall,
            claim_s=group["claim_s"],
            batch_size=1,
            units=entry.units,
            baseline_bytes=self._baseline_bytes,
            resident_bytes=current,
            peak_bytes=peak,
            device_baseline_bytes=None if device is None else self._device_baseline,
            device_bytes=device,
            load_s=group.get("load_s", 0.0),
            cold=bool(group.get("cold", False)),
            since_load=max(since, 0),
            reload=bool(group.get("reload", False)),
            worker=self.token,
            ended=time.time(),
        )

    def _settle(
        self,
        claim: Claim,
        entry: TicketJob,
        attempt: str,
        kind: str,
        sample: Sample | None,
        **fields: ty.Any,
    ) -> str:
        """The sample first (when the call ran), then the outcome (proved), then the settled mark."""
        if sample is not None:
            record = sample.to_json()
            self._append_sample(
                {
                    "ticket": claim.digest,
                    "checksum": entry.checksum,
                    "attempt": attempt,
                    **record,
                }
            )
            fields["sample"] = record
        written = claim.outcome(entry, attempt, kind, **fields)
        if written is None:
            self._lost.add(claim.lease.token_dir)
            return "lost"
        return kind

    def _append_sample(self, record: dict) -> None:
        path = self.pool.path / "workers" / self.token / "samples.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
            try:
                os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
            finally:
                os.close(fd)
        except OSError as err:
            logger.warning("could not record a sample: %s", err)

    def _abandon(self, claim: Claim, why: str) -> None:
        logger.warning("abandoning ticket %s: %s", claim.digest, why)
        self._release(claim)

    # -- faults that move jobs -----------------------------------------------------------------

    def _resource(
        self, claim: Claim, entries: list[TicketJob], fault: FaultRecord, phase: str
    ) -> None:
        """Move the jobs up the ticket's ladder (``cause: "ladder"``), or fail them when the ladder or
        the budget is spent: ``ResourceExhausted`` at a call, ``Unplaceable`` at a load.
        """
        ladder = list(claim.ticket.ladder)
        if fault.reason == "timeout":
            ladder = self._timeout_ladder(ladder)
        within = phase == "load" or all(
            self.pool.charges(*e.key)["resource_moves"] < self.faults.resource_moves
            for e in entries
        )
        movable = [e for e in entries if claim.current(e) is not None]
        if ladder and within and movable:
            claim.move(
                [movable],
                ladder[0],
                cause="ladder",
                phase="call" if phase == "call" else "load",
                ladder=ladder[1:],
            )
            logger.warning(
                "moved %d job(s) of %s to %r", len(movable), claim.digest, ladder[0]
            )
            return
        kind = "Unplaceable" if phase == "load" else "ResourceExhausted"
        record = FaultRecord("resource", kind, fault.message, fault.phase, fault.reason)
        for entry in movable:
            claim.outcome(
                entry,
                claim.current(entry)[1]["attempt"],
                "failed",
                fault=record.to_json(),
                cause=fault.to_json(),
            )

    def _timeout_ladder(self, ladder: list[str]) -> list[str]:
        """The ladder from its first class whose ``time`` is larger than this class's or whose
        ``gpu_type`` differs (a timeout moves nowhere else)."""
        for i, name in enumerate(ladder):
            record = self.pool.worker_class(name.split("#", 1)[0])
            if record is None:
                continue
            longer = record.get("time") is not None and (
                self.cls.time is None or record["time"] > self.cls.time
            )
            if longer or record.get("gpu_type") != self.cls.gpu_type:
                return ladder[i:]
        return []

    def _take_over(self, claim: Claim) -> bool:
        """Place the jobs of a ticket taken over from a lost holder again, under fresh attempts (RFC
        0005 section 5.5); return whether this worker runs it (False: split and released).
        """
        open_entries: list[TicketJob] = []
        charged: dict[tuple[str, str], bool] = {}
        for entry in claim.ticket.jobs:
            current = claim.current(entry)
            if current is None:
                continue
            g, record = current
            attempt = record["attempt"]
            found = self.pool.read_outcome(claim.queue, claim.digest, entry, attempt)
            kind = None if found is None else found.get("outcome")
            if kind in FINAL:
                continue
            if kind == "expired":
                self._expired_job(claim, entry)
                continue
            if kind is not None:
                continue
            mark = self.pool.started(*entry.key, g)
            charged[entry.key] = (
                mark is not None
                and not self._preempted(entry, g)
                and self._was_alone_in_call(mark.get("worker"), entry, attempt)
            )
            open_entries.append(entry)
        if len(open_entries) >= 2:
            claim.split(charged=charged)
            self._release(claim)
            return False
        for entry in open_entries:
            if self._stored(entry):  # a committed result settles done, uncharged
                attempt = claim.takeover(entry, charged=False)
                claim.outcome(entry, attempt, "done", served=True)
                continue
            claim.takeover(entry, charged=charged.get(entry.key, False))
        return True

    def _expired_pass(self, claim: Claim) -> None:
        """Type each job whose current attempt in this ticket expired (a claim released after its
        deadline passed, rather than left to go stale)."""
        for entry in claim.ticket.jobs:
            current = claim.current(entry)
            if current is None:
                continue
            found = self.pool.read_outcome(
                claim.queue, claim.digest, entry, current[1]["attempt"]
            )
            if found is not None and found.get("outcome") == "expired":
                self._expired_job(claim, entry)

    def _preempted(self, entry: TicketJob, g: int) -> bool:
        """Whether attempt ``g`` was cut short by its worker's drain within the job's
        ``preemption_budget``: an infrastructure event, retried uncharged."""
        if not self.pool.drained(*entry.key, g):
            return False
        preemptions = self.pool.charges(*entry.key)["preemptions"]
        return preemptions <= self.settings.preemption_budget

    def _was_alone_in_call(
        self, token: str | None, entry: TicketJob, attempt: str
    ) -> bool:
        """Whether the lost attempt was the only job of its worker's in-flight call."""
        if not isinstance(token, str) or len(token) != 32:
            return False
        try:
            record = self.pool.in_flight(token)
        except (OSError, ValueError):
            return False
        if record is None:
            return False
        jobs = record.get("jobs") or []
        return len(jobs) == 1 and jobs[0] == {
            "checksum": entry.checksum,
            "fingerprint": entry.fingerprint,
            "attempt": attempt,
        }

    def _expired_job(self, claim: Claim, entry: TicketJob) -> None:
        """A job whose attempt passed its deadline: under a fresh, uncharged ``takeover`` attempt,
        settle a result its call published, else type the timeout a resource fault."""
        attempt = claim.takeover(entry, charged=False)
        if self._stored(entry):
            claim.outcome(entry, attempt, "done", served=True)
            return
        fault = FaultRecord(
            "resource",
            "Timeout",
            "the call passed its deadline",
            "call",
            reason="timeout",
        )
        self._resource(claim, [entry], fault, "call")


def run_worker(
    root: os.PathLike,
    cls: "str | WorkerClass",
    *,
    providers: ty.Sequence["Provider | str"] = (),
    classify: Classifier | None = None,
    settings: PoolSettings = PoolSettings(),
    **kwargs: ty.Any,
) -> int:
    """Run a worker of class ``cls`` on the pool at ``root`` (see `Worker`); return its exit code:
    `EXIT_OK`, `EXIT_USAGE`, `EXIT_SETTINGS` (``PoolSettingsMismatch``), `EXIT_HARDWARE`
    (``WrongHardware``) or `EXIT_RECYCLE`."""
    return Worker(
        root, cls, providers=providers, classify=classify, settings=settings, **kwargs
    ).run()


def main(argv: ty.Sequence[str] | None = None) -> int:
    """``python -m pydra.workers.pool.worker --root <pool> --class <name>``."""
    parser = argparse.ArgumentParser(prog="python -m pydra.workers.pool.worker")
    parser.add_argument("--root", required=True, help="the pool directory")
    parser.add_argument("--class", dest="cls", required=True, help="the worker class")
    parser.add_argument("--settings", help="a JSON file of PoolSettings")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="a provider as module:qualname (repeatable)",
    )
    parser.add_argument(
        "--substrate",
        action="store_true",
        help="started by a substrate: exit when no coordinator is live",
    )
    parser.add_argument(
        "--token", help="the worker's token (32 hex digits), named by its substrate"
    )
    parser.add_argument(
        "--codec", help="the payload codec as module:qualname; default PrivatePickle"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    settings = PoolSettings()
    if args.settings:
        settings = PoolSettings.from_json(json.loads(Path(args.settings).read_text()))
    return run_worker(
        args.root,
        args.cls,
        providers=args.provider,
        settings=settings,
        substrate=args.substrate,
        token=args.token,
        codec=load_codec(args.codec),
    )


def load_codec(reference: str | None) -> JobCodec | None:
    """The codec ``module:qualname`` names, made with no arguments; None for None."""
    if reference is None:
        return None
    codec = _resolve(reference)
    return codec() if isinstance(codec, type) else codec


if __name__ == "__main__":
    sys.exit(main())
