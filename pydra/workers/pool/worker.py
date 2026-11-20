"""The warm worker of a pool (RFC 0005 sections 4.3, 5.3-5.5).

``pydracli pool worker --root <pool> --class <name>`` (or ``python -m pydra.workers.pool.worker``) runs
one loop:

1. **Start.** Open the pool and refuse to start (exit `EXIT_SETTINGS`) when its lease settings differ
   from the worker's in any of the four staleness fields. Start the beater (a `LeaseBeater` subprocess)
   before any provider is imported, take the worker lease with ``beat="external"`` and hand it to the
   beater. Read the hardware and refuse to start (exit `EXIT_HARDWARE`) when it does not match the
   class's ``gpu_type`` or ``gpus``; fill a ``gpu_mem`` of None from it. Write ``ready.json``.
2. **Claim** tickets of the class's queues, oldest waiting work first, preferring the queue served last
   while its quantum (``quantum_jobs`` jobs or ``quantum_s`` seconds) lasts. A queue not yet served
   gets one claim of one ticket; once warm, one-job tickets are gathered up to the quantum. Each claim
   is handed to the beater before anything runs under it. A claim that took a stale one over places
   its jobs again under fresh attempts first: a lone open job under a ``takeover`` generation, charged
   as a crash only when it was alone in the dead worker's in-flight call; two or more are split; a job
   whose attempt expired is looked up in the store, then typed a timeout (a resource fault).
3. **Re-judge** the ticket against the hardware read (a class whose ``gpu_mem`` was unknown): a ticket
   needing more device memory is withdrawn, reason ``refused``, and nothing runs. **Load** its
   residency through the process's `ResidentSet`, on the granted device (the process device, which
   a job's own binding uses too); a failure there is a load fault, never a job's.
4. **Run** each job through ``Job.run`` on a thread, with ``chdir=False``, under the attempt of its
   newest follow generation (a ``restart`` generation first when that attempt already started), with
   the claim as its de-duplication lease, proved before ``publish``. The call record is written before
   the call and its deadline sent to the beater; ``done`` is written after it. Each call goes through
   the bisection engine (`pydra.workers.pool.bisect`): a resource fault or a crash reloads the
   residency; a lone job out of memory moves up the ladder, and one that crashes is run again under a
   charged ``restart`` generation until ``crashes_charged``. An infrastructure fault requeues the job;
   at ``infrastructure_replacements`` faults of the residency in the coordinator's epoch the residency
   is held on its queue and the ticket's open jobs are settled ``held``.
5. **Settle** each job as it finishes: its `Sample`, then its outcome record (proved), then the ticket's
   settled mark.
6. **Prove** every held claim each renewal interval while a call runs; a lost claim is abandoned (the
   call in flight stopped when a provider's process runs it, its outcome not written).
7. **Exit** after ``idle_polls`` empty polls; on SIGTERM after settling finished jobs and releasing the
   rest; for a substrate-started worker after ``orphan_after_s`` without a live coordinator; and,
   recycling (exit `EXIT_RECYCLE`), after a call's deadline passed or the beater died. Resident state is
   closed on exit, process groups included.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.engine.lease import BeaterClient, BeaterDead, DirectoryLease, LeaseLost
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
    FINAL,
    Claim,
    PoolOwnerMismatch,
    PoolRoot,
    PoolSettingsMismatch,
    TicketJob,
)

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.resident import Provider

__all__ = [
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
    classify : callable
        ``classify(exc, phase) -> FaultRecord``; a record attached to an exception or its chain
        overrides it (`pydra.workers.pool.faults.fault_of`).
    settings : PoolSettings
    faults : FaultBudgets
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
        classify: Classifier = _faults.classify,
        settings: PoolSettings = PoolSettings(),
        faults: FaultBudgets = FaultBudgets(),
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
        self.classify = classify
        self.settings = settings
        self.faults = faults
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
            return self._run()
        finally:
            self.pool.close()

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
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        def drain(signum, frame):
            self.stop.set()
            self._wake.set()

        previous = signal.signal(signal.SIGTERM, drain)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)

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
            if self.substrate and self._orphaned():
                logger.info(
                    "no live coordinator for %ss: exiting", self.settings.orphan_after_s
                )
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
        claim = self.pool.claim(queue, beat="external", max_jobs=max_jobs)
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
        newest = None
        for base in [
            where / "lease",
            *(where / "lease").glob("*"),
            *(where / "lease").glob("*/*"),
        ]:
            with contextlib.suppress(OSError):
                mtime = os.stat(base).st_mtime
                newest = mtime if newest is None else max(newest, mtime)
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
                if self._lost_claim(claim):
                    continue
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
        submitter = getattr(job, "submitter", None)
        if submitter is not None and self.providers:
            own = tuple(getattr(submitter, "providers", ()))
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
        """One infrastructure fault of the ticket's residency, at ``phase`` (``call`` or ``load``),
        recorded for the newest coordinator epoch. Below ``infrastructure_replacements`` such faults
        in that epoch the jobs are requeued, uncharged, each under a ``restart`` generation written
        now; at it the residency is held on its queue (``held-<epoch>-<label>``) and every open job of
        the ticket is settled ``held``, its reason naming the residency and the latest cause (RFC 0005
        section 6). The caller releases the ticket. Return ``"held"`` or ``"stopped"``.
        """
        group["stop"] = True
        entries = [e for e in entries if claim.current(e) is not None]
        if not entries:
            return "stopped"
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
            for entry in entries:
                claim.restart(entry, phase=phase)
            logger.warning(
                "infrastructure fault %d of residency %s on %r in epoch %d at %s: %s: %s; "
                "requeued",
                count,
                label,
                claim.queue,
                epoch,
                phase,
                fault.type,
                fault.message,
            )
            return "stopped"
        self.pool.hold(claim.queue, epoch, label)
        reason = (
            f"residency {label} held on queue {claim.queue!r} after {count} infrastructure "
            f"faults in coordinator epoch {epoch} (the budget is {budget}); the latest, at "
            f"{phase}: {fault.type}: {fault.message}"
        )
        logger.error(reason, extra={"event": "held", "residency": label})
        faulted = {e.key for e in entries}
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

    def _stored(self, entry: TicketJob) -> bool:
        """Whether the store holds a committed result for the job (RFC 0005 section 5.5: a taker
        looks the job up first)."""
        job = None
        with contextlib.suppress(Exception):
            job = read_payload(self.codec, entry.payload, None)
        find = getattr(getattr(job, "store", None), "find", None)
        if job is None or find is None:
            return False
        try:
            found = find(job)
        except Exception:
            return False
        return getattr(found, "result", None) is not None

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
        """Run one job as a call of one through the bisection engine (W10 batches calls), retried up
        to ``content_attempts``, and settle it; return what became of it: ``done``, ``failed``,
        ``moved``, ``held``, ``lost``, ``expired`` or ``stopped`` (released unsettled).
        """
        tries = 0
        while True:
            tries += 1
            state: dict[str, ty.Any] = {
                "fresh": tries
                > 1,  # every attempt after the first runs a freshly decoded job
                "attempt": None,
                "sample": None,
                "verdict": None,
            }

            def call(batch: list[TicketJob], state=state) -> list:
                [item] = batch
                attempt = claim.start(item, self.token)
                if attempt is None:  # superseded or withdrawn: nothing is written
                    raise Abandon("superseded")
                current = job
                if state["fresh"]:
                    current = self._decode(claim, item, attempt)
                    if current is None:
                        raise Abandon("unreadable")
                state["fresh"], state["attempt"] = True, attempt
                ran = self._invoke(claim, item, current, attempt, group)
                if isinstance(ran, str):
                    state["verdict"] = ran
                    raise Abandon(ran)
                done, state["sample"] = ran
                if done.error is not None:
                    raise done.error
                return [done.result]

            try:
                [settled] = engine.bisect(
                    [entry],
                    call,
                    classify=self._classify_call,
                    reload=lambda fault: self._reload(group),
                    lost=lambda item: self._crashed(claim, item),
                    crashes_charged=self.faults.crashes_charged,
                )
            except Abandon:
                return state["verdict"] or "stopped"
            except Exception as exc:  # a load inside the call: never split or retried
                self._load_fault(claim, [entry], exc)
                group["stop"] = True
                return "stopped"
            attempt, sample, fault = state["attempt"], state["sample"], settled.fault
            if fault is None:
                return self._settle(claim, entry, attempt, "done", sample)
            if settled.charged:
                current = claim.current(entry)
                if current is None:
                    return "stopped"
                return self._settle(
                    claim,
                    entry,
                    current[1]["attempt"],
                    "failed",
                    sample,
                    fault=fault.to_json(),
                )
            if fault.kind == "content":
                if tries < self.faults.content_attempts:
                    continue
                return self._settle(
                    claim, entry, attempt, "failed", sample, fault=fault.to_json()
                )
            if fault.kind == "resource":
                self._resource(claim, [entry], fault, "call")
                return "moved"
            if fault.kind == "infrastructure":
                return self._infrastructure(claim, [entry], fault, group, sample)
            return self._settle(
                claim, entry, attempt, "failed", sample, fault=fault.to_json()
            )

    def _crashed(self, claim: Claim, entry: TicketJob) -> int:
        """A job's call crashed with the job alone in it: write its next generation, a charged
        ``restart``, and return its crash count (`PoolRoot.charges`)."""
        claim.restart(entry, phase="call", charged=True)
        return self.pool.charges(*entry.key)["crashes"]

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

        The beater is pinged every renewal interval, so a hung one is found. While the beater is
        dead the claim is not proved (a proof renews it, and nothing else beats it), and the call's
        ``deadline`` is the loop's: once it passes, the call's ``expired`` outcomes are written here
        (`pool_expire`) and the call is abandoned, so the worker recycles and the claim goes stale.
        """
        call = _Call()
        done = threading.Event()
        token_dir = claim.lease.token_dir

        def target() -> None:
            try:
                call.result = job.run(rerun=bool(getattr(job, "rerun", False)))
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
            self._wake.wait(interval)
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
        sample: Sample,
        **fields: ty.Any,
    ) -> str:
        """The sample first, then the outcome (proved), then the settled mark."""
        record = sample.to_json()
        self._append_sample(
            {
                "ticket": claim.digest,
                "checksum": entry.checksum,
                "attempt": attempt,
                **record,
            }
        )
        written = claim.outcome(entry, attempt, kind, sample=record, **fields)
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
            charged[entry.key] = mark is not None and self._was_alone_in_call(
                mark.get("worker"), entry, attempt
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
    classify: Classifier = _faults.classify,
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
    parser.add_argument("--faults", help="a JSON file of FaultBudgets")
    parser.add_argument(
        "--classify",
        help="the classifier as module:qualname; default pydra.workers.pool.faults:classify",
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
        classify=load_classifier(args.classify),
        settings=settings,
        faults=load_faults(args.faults),
        substrate=args.substrate,
        token=args.token,
        codec=load_codec(args.codec),
    )


def load_classifier(reference: str | None) -> Classifier:
    """The classifier ``module:qualname`` names; the default classifier for None."""
    if reference is None:
        return _faults.classify
    return _resolve(reference)


def load_faults(path: str | os.PathLike | None) -> FaultBudgets:
    """The `FaultBudgets` of the JSON file at ``path``; the defaults for None."""
    if path is None:
        return FaultBudgets()
    return FaultBudgets.from_json(json.loads(Path(path).read_text()))


def load_codec(reference: str | None) -> JobCodec | None:
    """The codec ``module:qualname`` names, made with no arguments; None for None."""
    if reference is None:
        return None
    codec = _resolve(reference)
    return codec() if isinstance(codec, type) else codec


if __name__ == "__main__":
    sys.exit(main())
