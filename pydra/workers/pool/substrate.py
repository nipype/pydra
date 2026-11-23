"""Substrates: what starts a pool's workers (RFC 0005 section 4.5).

A substrate starts workers of a class, lists the ones it started that are pending or running, stops
them, and says whether one more may start. `LocalSubstrate` starts worker processes on this host;
`SlurmSubstrate` submits one Slurm job per worker; `ExternalSubstrate` starts nothing, for workers
started by hand or by another system.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.utils.storagedir import IS_WINDOWS
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.storage import EXIT_MARK

if ty.TYPE_CHECKING:
    from pydra.workers.pool.settings import PoolSettings

__all__ = [
    "ExternalSubstrate",
    "LocalSubstrate",
    "SlurmSubstrate",
    "Substrate",
    "SubstrateError",
    "WorkerHandle",
]

logger = logging.getLogger("pydra.pool")


@attrs.define(eq=False)
class WorkerHandle:
    """A worker a substrate started.

    Attributes
    ----------
    token : str
        The worker's token: its directory is ``workers/<token>/`` and its ready mark
        ``workers/<token>/ready.json``.
    worker_class : str
    started : float
        When it was started (``time.monotonic``).
    devices : tuple of str
        The devices it was given (``CUDA_VISIBLE_DEVICES``).
    process : subprocess.Popen, optional
        The worker process, for a local substrate.
    log : Path, optional
        Where its output goes.
    cpus : int
        The CPUs it holds against a substrate's CPU cap.
    job_id : str, optional
        Its scheduler job, for a batch substrate.
    ended : str, optional
        Why it ended, set by a substrate that has no process to ask.
    """

    token: str
    worker_class: str
    started: float = attrs.field(factory=time.monotonic)
    devices: tuple[str, ...] = ()
    process: subprocess.Popen | None = attrs.field(default=None, repr=False)
    log: Path | None = None
    cpus: int = 1
    job_id: str | None = None
    ended: str | None = None

    @property
    def cause(self) -> str | None:
        """Why the worker ended, when its substrate can tell: its exit code, or what its substrate
        saw, and the last line of its log."""
        if self.process is not None and self.process.poll() is not None:
            why = f"exit code {self.process.returncode}"
        elif self.process is None and self.ended is not None:
            why = self.ended
        else:
            return None
        last = _last_line(self.log)
        return f"{why}: {last}" if last else why


class SubstrateError(RuntimeError):
    """A substrate's command failed; ``started`` holds the workers it started before it did."""

    def __init__(self, message: str, started: ty.Sequence[WorkerHandle] = ()):
        super().__init__(message)
        self.started = tuple(started)


def _last_line(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [ln for ln in f.read().decode(errors="replace").splitlines() if ln]
    except OSError:
        return None
    return lines[-1][-500:] if lines else None


class Substrate(ty.Protocol):
    """What starts a pool's workers."""

    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]:
        """Start ``n`` workers of ``cls``; never blocks on a queue."""

    def live(self) -> list[WorkerHandle]:
        """The workers this substrate started that are pending or running: one listing per call."""

    def stop(self, workers: ty.Sequence[WorkerHandle]) -> None:
        """Stop the workers."""

    def can_start(self, cls: WorkerClass) -> "bool | str":
        """True when one more worker of ``cls`` may start now, else why not."""

    def attach(self, attachment: "Attachment") -> None:
        """Take what it needs of the pool it starts workers for, before it starts any."""


@attrs.define
class Attachment:
    """What a substrate needs of the pool it starts workers for (`WarmPool` passes it)."""

    root: Path
    settings: Path
    providers: tuple[str, ...] = ()
    codec: str | None = None
    pool_id: str = ""
    epoch: int | None = None  # the coordinator's epoch, set once it is taken
    classes: tuple[WorkerClass, ...] = ()  # the classes the pool places jobs on
    pool_settings: PoolSettings | None = None  # the settings the pool's workers run on


def worker_argv(python: str, at: Attachment, cls: WorkerClass, token: str) -> list[str]:
    """The command that runs one substrate-started worker of ``cls`` for the pool ``at`` names."""
    argv = [
        python,
        "-m",
        "pydra.workers.pool.worker",
        "--root",
        str(at.root),
        "--class",
        cls.name,
        "--settings",
        str(at.settings),
        "--substrate",
        "--token",
        token,
    ]
    for provider in at.providers:
        argv += ["--provider", provider]
    if at.codec is not None:
        argv += ["--codec", at.codec]
    return argv


def worker_tokens(token: str, n: int) -> tuple[str, ...]:
    """The tokens of the ``n`` workers of a job named after ``token``: ``token`` first, then
    32 hex digits derived from it."""
    return (token,) + tuple(
        hashlib.blake2b(f"{token}:{i}".encode(), digest_size=16).hexdigest()
        for i in range(1, n)
    )


def _logs(root: Path) -> Path:
    logs = root / "private" / "logs"
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    return logs


class ExternalSubstrate:
    """Starts nothing: workers are started by hand or by another system and attach by pointing at
    the pool's root (``pydracli pool worker --root <pool> --class <name>``). Such a worker exits on
    idle only, and a coordinator's ``close`` never stops it."""

    def attach(self, attachment: Attachment) -> None:
        """Nothing to keep."""

    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]:
        return []

    def live(self) -> list[WorkerHandle]:
        return []

    def stop(self, workers: ty.Sequence[WorkerHandle]) -> None:
        """Workers started by hand are left running."""

    def can_start(self, cls: WorkerClass) -> "bool | str":
        return True


def visible_devices(timeout: float = 20.0) -> list[str]:
    """The indices ``nvidia-smi`` lists, none without it."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def drain_bound(settings: "PoolSettings") -> float:
    """Most seconds a worker takes to stop once its stop mark is written: it looks for the mark
    every ``poll_s``, waits ``drain_s`` for a call in flight, then releases its claims, retried
    for `release_patience`."""
    from pydra.engine.lease import release_patience

    return settings.poll_s + settings.drain_s + release_patience(settings.lease)


def shutdown_bound(settings: "PoolSettings") -> float:
    """Most seconds a worker takes to shut down once drained: a provider's process group stopped
    on the default timeouts, then its worker lease released."""
    from pydra.engine.lease import release_patience
    from pydra.engine.procgroup import KILL_TIMEOUT, TERM_TIMEOUT

    return TERM_TIMEOUT + KILL_TIMEOUT + release_patience(settings.lease)


class LocalSubstrate:
    """Starts worker processes on this host, each in its own session, under the submitting process.

    A ``cuda`` class's worker is given ``max(gpus, 1)`` devices through ``CUDA_VISIBLE_DEVICES``,
    at most ``workers_per_device`` workers per device, so a host with four GPUs runs four one-GPU
    workers or two two-GPU workers.

    Parameters
    ----------
    devices : sequence of int or str, optional
        The devices workers may be given; default the indices ``nvidia-smi`` lists (read once).
    per_user_cpu_cap : int, optional
        The most CPUs (``cls.cpus``, 1 when unset) the live workers may hold together.
    python : str
        The interpreter that runs the workers.
    stop_wait : float, optional
        Seconds `stop` waits for a worker to drain and exit before it kills its process group;
        at least `drain_bound` of the pool's settings. Default `drain_bound` plus
        `shutdown_bound`.
    """

    def __init__(
        self,
        devices: ty.Sequence[int | str] | None = None,
        *,
        per_user_cpu_cap: int | None = None,
        python: str = sys.executable,
        stop_wait: float | None = None,
    ):
        self._devices = None if devices is None else [str(d) for d in devices]
        self.per_user_cpu_cap = per_user_cpu_cap
        self.python = python
        self._stop_wait_arg = stop_wait
        self.stop_wait = stop_wait
        self.handles: list[WorkerHandle] = []
        self._attachment: Attachment | None = None

    def attach(self, attachment: Attachment) -> None:
        """Start workers for the pool ``attachment`` names.

        Raises
        ------
        ValueError
            The attachment carries no ``pool_settings``, or ``stop_wait`` is shorter than
            their `drain_bound`.
        """
        from pydra.engine.lease import release_patience

        settings = attachment.pool_settings
        if settings is None:
            raise ValueError(
                "a LocalSubstrate needs the pool's settings on its attachment, to bound "
                "stop_wait"
            )
        self.stop_wait = self._stop_wait_arg
        if self.stop_wait is None:
            self.stop_wait = drain_bound(settings) + shutdown_bound(settings)
        if self.stop_wait < drain_bound(settings):
            raise ValueError(
                f"stop_wait {self.stop_wait:g}s is shorter than a worker may need to stop: "
                f"poll_s {settings.poll_s:g} + drain_s {settings.drain_s:g} + release "
                f"{release_patience(settings.lease):g} = {drain_bound(settings):g}s"
            )
        self._attachment = attachment

    @property
    def devices(self) -> list[str]:
        if self._devices is None:
            self._devices = visible_devices()
        return self._devices

    def live(self) -> list[WorkerHandle]:
        return [
            h
            for h in self.handles
            if h.process is not None and h.process.poll() is None
        ]

    def _free(self, cls: WorkerClass, live: list[WorkerHandle]) -> list[str]:
        used: dict[str, int] = {}
        for h in live:
            for d in h.devices:
                used[d] = used.get(d, 0) + 1
        return [d for d in self.devices if used.get(d, 0) < cls.workers_per_device]

    def can_start(self, cls: WorkerClass) -> "bool | str":
        live = self.live()
        if cls.device == "cuda":
            want = max(cls.gpus, 1)
            if len(self.devices) < want:
                return (
                    f"class {cls.name!r} needs {want} GPU(s); this host lists "
                    f"{len(self.devices)}"
                )
            free = self._free(cls, live)
            if len(free) < want:
                return (
                    f"class {cls.name!r} needs {want} free GPU(s) at "
                    f"{cls.workers_per_device} worker(s) per device; {len(free)} free"
                )
        if self.per_user_cpu_cap is not None:
            held = sum(h.cpus for h in live)
            if held + (cls.cpus or 1) > self.per_user_cpu_cap:
                return (
                    f"{held} CPUs held by live workers; one more of class {cls.name!r} "
                    f"({cls.cpus or 1}) exceeds the cap of {self.per_user_cpu_cap}"
                )
        return True

    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]:
        if self._attachment is None:
            raise RuntimeError(
                "a LocalSubstrate starts workers once a WarmPool attaches it"
            )
        started = []
        for _ in range(n):
            if self.can_start(cls) is not True:
                break
            devices: tuple[str, ...] = ()
            if cls.device == "cuda":
                devices = tuple(self._free(cls, self.live())[: max(cls.gpus, 1)])
            started.append(self._start(cls, devices))
        return started

    def _start(self, cls: WorkerClass, devices: tuple[str, ...]) -> WorkerHandle:
        at = self._attachment
        token = uuid.uuid4().hex
        argv = worker_argv(self.python, at, cls, token)
        env = dict(os.environ)
        if cls.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        log = _logs(at.root) / f"{token}.log"
        with open(log, "ab") as out:
            process = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=not IS_WINDOWS,
            )
        handle = WorkerHandle(
            token=token,
            worker_class=cls.name,
            devices=devices,
            process=process,
            log=log,
            cpus=cls.cpus or 1,
        )
        self.handles.append(handle)
        logger.info(
            "started worker %s of class %r (pid %d%s)",
            token,
            cls.name,
            process.pid,
            f", devices {','.join(devices)}" if devices else "",
        )
        return handle

    def stop(self, workers: ty.Sequence[WorkerHandle]) -> None:
        """Let each worker drain on its stop mark (`WarmPool` writes it first) and, on POSIX, on
        SIGTERM; kill the process group of any still running after ``stop_wait``. No signal is
        sent on Windows."""
        running = [
            h for h in workers if h.process is not None and h.process.poll() is None
        ]
        if not IS_WINDOWS:
            for h in running:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    h.process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + self.stop_wait
        for h in running:
            try:
                h.process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for h in running:
            if IS_WINDOWS:
                if h.process.poll() is None:
                    h.process.kill()
            else:
                # the group: a worker's beater and providers' processes go with it
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(h.process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                h.process.wait(self.stop_wait)


# options a class's ``substrate_args`` may not set: those the class's fields give, and those that
# identify and wrap the job
_SLURM_OWNED = {
    "gres": "set from the class's gpus and gpu_type",
    "gpus": "set from the class's gpus and gpu_type",
    "G": "set from the class's gpus and gpu_type",
    "cpus-per-task": "set from the class's cpus",
    "c": "set from the class's cpus",
    "mem": "set from the class's mem",
    "time": "set from the class's time",
    "t": "set from the class's time",
    "job-name": "the substrate names the job",
    "J": "the substrate names the job",
    "comment": "the substrate's record of the job's class",
    "output": "worker logs go under the pool's private directory",
    "o": "worker logs go under the pool's private directory",
    "error": "worker logs go under the pool's private directory",
    "e": "worker logs go under the pool's private directory",
    "wrap": "the substrate wraps the worker command",
    "parsable": "the substrate reads the job id",
    "requeue": "the substrate passes --no-requeue: a worker's token serves one process",
    "no-requeue": "the substrate passes --no-requeue: a worker's token serves one process",
    "array": "one job per worker",
    "a": "one job per worker",
    "clusters": "a job on another cluster is never listed by this one's squeue",
    "M": "a job on another cluster is never listed by this one's squeue",
    "gpus-per-node": "set from the class's gpus and gpu_type",
    "gpus-per-task": "set from the class's gpus and gpu_type",
    "gpus-per-socket": "set from the class's gpus and gpu_type",
    "tres-per-task": "set from the class's gpus and cpus",
    "cpus-per-gpu": "set from the class's cpus",
    "mem-per-gpu": "set from the class's mem",
    "mem-per-cpu": "set from the class's mem",
}


# squeue states of a job that has not started running: plain or held pending, and requeue-held
_NOT_STARTED = frozenset({"PENDING", "REQUEUE_HOLD", "REQUEUE_FED"})


class SlurmSubstrate:
    """Submits one Slurm job per worker, never one per pydra job.

    A worker of a class is ``sbatch --parsable`` with ``--gres=gpu:<type>:<n>`` (``gpu:<n>`` with no
    ``gpu_type``; a ``cuda`` class asks for at least one), ``--cpus-per-task``, ``--mem`` and
    ``--time`` from the class's fields where set, then the class's ``substrate_args`` (each ``key:
    value`` as ``--key=value``, a one-letter key as ``-k value``, an empty value as a bare flag),
    and ``--wrap`` running the worker command. The job is named ``pydra-<pool
    id>-e<epoch>-<token>``, carries its class as ``--comment``, is ``--no-requeue`` (a preempted
    worker is lost, its claims taken over once stale) and writes its output to
    ``<pool>/private/logs/<token>.log``. A ``--qos`` is passed only as ``substrate_args`` give it; a
    class naming an option the substrate sets itself cannot start. The substrate does not set
    ``CUDA_VISIBLE_DEVICES``. Up to ``workers_per_device`` workers share one job, which asks the
    class's GPUs once and its CPUs and memory per worker.

    `live` lists this pool's jobs from the process's shared queue listing, including jobs it did not
    record (a late submission, a worker an earlier coordinator started), which count toward the
    class's workers and the CPU cap. Its first listing cancels earlier epochs' jobs that never
    started. A submission whose ``sbatch`` times out is looked up by name, twice,
    ``attribute_delay`` apart, and submitted again under a new token only when not found.

    Parameters
    ----------
    per_user_cpu_cap : int, optional
        The most CPUs (``cls.cpus``, 1 when unset) the live workers may hold together; None: no
        cap. A site with GPU, submit or other caps overrides `can_start`.
    python : str
        The interpreter that runs the workers, as the nodes see it.
    submit_timeout : float
        Seconds one ``sbatch`` may take before it counts as timed out.
    squeue_timeout : float
        Seconds one ``squeue`` or ``scancel`` may take.
    attribute_delay : float, optional
        Seconds between the two lookups of a timed-out submission; default the pool's
        ``LeaseSettings.attribute_delay``.
    max_age : float, optional
        Seconds a queue listing serves `live`; default the pool's ``poll_s``.
    listing : QueueListing, optional
        The queue listing; default the process's, shared with `SlurmWorker`.
    clock, sleep : callable
        Monotonic seconds, and the wait before a resubmission.
    """

    def __init__(
        self,
        *,
        per_user_cpu_cap: int | None = None,
        python: str = sys.executable,
        submit_timeout: float = 60.0,
        squeue_timeout: float = 60.0,
        attribute_delay: float | None = None,
        max_age: float | None = None,
        listing: ty.Any = None,
        clock: ty.Callable[[], float] = time.monotonic,
        sleep: ty.Callable[[float], None] = time.sleep,
    ):
        self.per_user_cpu_cap = per_user_cpu_cap
        self.python = python
        self.submit_timeout = submit_timeout
        self.squeue_timeout = squeue_timeout
        self.attribute_delay = attribute_delay
        self.max_age = max_age
        self._given_listing = listing
        self._clock = clock
        self._sleep = sleep
        self.handles: list[WorkerHandle] = []
        self._jobs: dict[str, list[WorkerHandle]] = {}
        self._cpus: dict[str, int] = {}
        self._attachment: Attachment | None = None
        self._swept = False
        self._held = 0
        self._stopping: set[str] = set()  # cancelled jobs Slurm may still list

    @property
    def _listing(self):
        from pydra.workers import slurm

        return self._given_listing or slurm.shared_listing()

    def attach(self, attachment: Attachment) -> None:
        """Submit workers for the pool ``attachment`` names; unset waits come from its settings."""
        from pydra.engine.lease import ATTRIBUTE_DELAY
        from pydra.workers.pool.settings import PoolSettings

        self._attachment = attachment
        self._swept = False
        for cls in attachment.classes:
            self._cpus[cls.name] = cls.cpus or 1
        if self.attribute_delay is None or self.max_age is None:
            try:
                settings = PoolSettings.from_json(
                    json.loads(Path(attachment.settings).read_text())
                )
            except (OSError, ValueError):
                settings = None
            if self.attribute_delay is None:
                self.attribute_delay = (
                    settings.lease.attribute_delay if settings else ATTRIBUTE_DELAY
                )
            if self.max_age is None:
                self.max_age = settings.poll_s if settings else PoolSettings().poll_s

    def _require(self) -> Attachment:
        if self._attachment is None:
            raise RuntimeError(
                "a SlurmSubstrate submits workers once a WarmPool attaches it"
            )
        return self._attachment

    @property
    def pool_prefix(self) -> str:
        """The start of the name of every job of this pool, whatever its epoch."""
        at = self._require()
        pool_id = at.pool_id or uuid.uuid5(uuid.NAMESPACE_URL, str(at.root)).hex[:12]
        return f"pydra-{pool_id}-e"

    @property
    def prefix(self) -> str:
        """The start of the name of every job of this pool and coordinator epoch."""
        return f"{self.pool_prefix}{self._require().epoch or 0}-"

    def _epoch_token(self, name: str) -> "tuple[int, tuple[str, ...]] | None":
        """The epoch and worker tokens of a job of this pool named ``name``; None for any other
        job."""
        rest = (
            name[len(self.pool_prefix) :] if name.startswith(self.pool_prefix) else ""
        )
        epoch, _, token = rest.partition("-")
        token, _, n = token.partition("x")
        if not epoch.isdigit() or not token or not (n == "" or n.isdigit() and int(n)):
            return None
        return int(epoch), worker_tokens(token, int(n or 1))

    def job_name(self, token: str, n: int = 1) -> str:
        return f"{self.prefix}{token}" + (f"x{n}" if n > 1 else "")

    @staticmethod
    def _refused(cls: WorkerClass) -> str | None:
        for key in cls.substrate_args:
            why = _SLURM_OWNED.get(str(key).lstrip("-"))
            if why is not None:
                return f"class {cls.name!r} sets substrate_args {key!r}: {why}"
        return None

    def sbatch_args(self, cls: WorkerClass, token: str, workers: int = 1) -> list[str]:
        """The ``sbatch`` command line of a job of ``workers`` workers of ``cls``, the first under
        ``token``."""
        at = self._require()
        tokens = worker_tokens(token, workers)
        log = at.root / "private" / "logs" / f"{token}.log"
        args = [
            "sbatch",
            "--parsable",
            f"--job-name={self.job_name(token, workers)}",
            f"--comment={cls.name}",
            f"--output={log}",
            "--no-requeue",
        ]
        if cls.device == "cuda":
            n = max(cls.gpus, 1)
            args.append(
                f"--gres=gpu:{cls.gpu_type}:{n}" if cls.gpu_type else f"--gres=gpu:{n}"
            )
        if cls.cpus is not None:
            args.append(f"--cpus-per-task={cls.cpus * workers}")
        if cls.mem is not None:
            args.append(f"--mem={math.ceil(cls.mem * workers / 2**20)}M")
        if cls.time is not None:
            args.append(f"--time={max(1, math.ceil(cls.time / 60))}")
        for key, value in cls.substrate_args.items():
            key = str(key).lstrip("-")
            if len(key) == 1:
                args += [f"-{key}", str(value)] if value != "" else [f"-{key}"]
            else:
                args.append(f"--{key}={value}" if value != "" else f"--{key}")
        commands = [shlex.join(worker_argv(self.python, at, cls, t)) for t in tokens]
        if workers == 1:
            return [*args, "--wrap", commands[0]]
        # the workers run side by side; a signal to the job's shell reaches each of them
        script = "; ".join(f'{c} & p="$p $!"' for c in commands)
        wrap = f"p=; {script}; trap 'kill -TERM $p 2>/dev/null; wait' TERM INT; wait"
        return [*args, "--wrap", wrap]

    def can_start(self, cls: WorkerClass) -> "bool | str":
        why = self._refused(cls)
        if why is not None:
            return why
        if self.per_user_cpu_cap is None:
            return True
        room = self._room(cls)
        if room is not None and room < 1:
            return (
                f"{self._held} CPUs held by live workers; one more of class {cls.name!r} "
                f"({cls.cpus or 1}) exceeds the cap of {self.per_user_cpu_cap}"
            )
        return True

    def _room(self, cls: WorkerClass) -> int | None:
        """How many more workers of ``cls`` the CPU cap allows; None without a cap or when the
        queue cannot be listed (scaling lists again before starting)."""
        if self.per_user_cpu_cap is None:
            return None
        try:
            live = self.live()
        except Exception:
            return None
        self._held = sum(h.cpus for h in live)
        return (self.per_user_cpu_cap - self._held) // (cls.cpus or 1)

    def live(self) -> list[WorkerHandle]:
        """This pool's jobs of this epoch and of earlier ones, from the shared listing. At the
        first listing after attaching, earlier epochs' jobs still pending are cancelled; their
        running ones are adopted as unrecorded workers, so they count and scaling can stop them.
        """
        at = self._require()
        snapshot = self._listing.read(self.max_age, self.squeue_timeout)
        epoch = at.epoch or 0
        listed, stale, stopping = set(), [], set()
        for job_id, (state, name, comment) in snapshot.rows.items():
            found = self._epoch_token(name)
            if found is None or found[0] > epoch:
                continue
            if (
                job_id in self._stopping
            ):  # cancelled by `stop`, still stopping: not a worker
                stopping.add(job_id)
                continue
            if job_id not in self._jobs:
                if found[0] < epoch and not self._swept and state in _NOT_STARTED:
                    stale.append(job_id)
                    continue
                self._adopt(job_id, found, comment, snapshot.taken)
            listed.add(job_id)
        if not self._swept:
            if stale and not self._cancel(stale):
                for job_id in stale:  # left queued: count them, try again next listing
                    _, name, comment = snapshot.rows[job_id]
                    self._adopt(
                        job_id, self._epoch_token(name), comment, snapshot.taken
                    )
                    listed.add(job_id)
            else:
                self._swept = True
                if stale:
                    logger.info(
                        "cancelled pending Slurm jobs of earlier epochs: %s",
                        " ".join(stale),
                    )
        self._stopping &= stopping
        live = []
        for job_id, handles in list(self._jobs.items()):
            if job_id in listed or handles[0].started >= snapshot.taken:
                for handle in handles:
                    if handle.ended is None and self._exited(at.root, handle.token):
                        handle.ended = "the worker exited"
                live.extend(h for h in handles if h.ended is None)
            else:
                for handle in handles:
                    handle.ended = f"Slurm job {job_id} left the Slurm queue"
                del self._jobs[job_id]
        return live

    @staticmethod
    def _exited(root: Path, token: str) -> bool:
        """Whether the worker ``token`` wrote its exit mark: a job it shares stays listed"""
        return os.path.exists(os.path.join(root, "workers", token, EXIT_MARK))

    def _adopt(self, job_id: str, found, comment: str, taken: float) -> None:
        job_epoch, tokens = found
        handles = [
            WorkerHandle(
                token=token,
                worker_class=comment,
                started=taken,
                log=self._require().root / "private" / "logs" / f"{tokens[0]}.log",
                cpus=self._cpus.get(comment, 1),
                job_id=job_id,
            )
            for token in tokens
        ]
        self._jobs[job_id] = handles
        self.handles.extend(handles)
        logger.info(
            "Slurm job %s (epoch %d, class %r) of this pool is adopted as an unrecorded worker",
            job_id,
            job_epoch,
            comment,
        )

    def _cancel(self, job_ids: ty.Sequence[str]) -> bool:
        """``scancel`` the jobs; False when it failed (logged)."""
        argv = ["scancel", *job_ids]
        try:
            done = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.squeue_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as err:
            logger.warning("%s failed: %s", " ".join(argv), err)
            return False
        if done.returncode:
            logger.warning(
                "%s exited %d: %s", " ".join(argv), done.returncode, done.stderr.strip()
            )
            return False
        return True

    def start(self, cls: WorkerClass, n: int) -> list[WorkerHandle]:
        """Submit up to ``n`` workers of ``cls`` while `can_start` allows.

        Raises
        ------
        ValueError
            The class sets an option the substrate sets.
        SubstrateError
            An ``sbatch`` failed; it carries the workers submitted before it (``started``).
        """
        at = self._require()
        why = self._refused(cls)
        if why is not None:
            raise ValueError(why)
        _logs(at.root)
        self._cpus[cls.name] = cls.cpus or 1
        started: list[WorkerHandle] = []
        while len(started) < n:
            room = self._room(cls)
            share = min(cls.workers_per_device, n - len(started))
            if room is not None:
                share = min(share, room)
            if share < 1:
                break
            try:
                started.extend(self._submit(cls, share))
            except SubstrateError as err:
                err.started = tuple(started)
                raise
        return started

    def _sbatch(self, argv: list[str]) -> str:
        """Run ``sbatch``; its output. A timeout is `subprocess.TimeoutExpired`."""
        try:
            done = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.submit_timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except OSError as err:
            raise SubstrateError(f"could not run sbatch: {err}") from err
        if done.returncode:
            said = (done.stderr or done.stdout).strip()
            raise SubstrateError(f"sbatch exited {done.returncode}: {said}")
        return done.stdout

    @staticmethod
    def _job_id(stdout: str) -> str:
        first = stdout.strip().splitlines()[0] if stdout.strip() else ""
        job_id = first.split(";")[0].strip()
        if not job_id.isdigit():
            raise SubstrateError(f"no job id in sbatch output {stdout!r}")
        return job_id

    def _lookup(self, name: str) -> str | None:
        """The job named ``name``, in any state; None when there is none."""
        from pydra.workers import slurm

        cmd = slurm.squeue_command("--states=all", f"--name={name}")
        try:
            done = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.squeue_timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as err:
            raise SubstrateError(
                f"the lookup of timed-out submission {name} failed: {err}"
            ) from err
        if done.returncode or "slurm_load_jobs error" in done.stderr:
            raise SubstrateError(
                f"the lookup of timed-out submission {name} failed: squeue exited "
                f"{done.returncode}: {done.stderr.strip()}"
            )
        rows = slurm.parse_squeue(done.stdout)
        return next((j for j, (_, n, _) in rows.items() if n == name), None)

    def _submit(self, cls: WorkerClass, n: int = 1) -> list[WorkerHandle]:
        token = uuid.uuid4().hex
        try:
            job_id = self._job_id(self._sbatch(self.sbatch_args(cls, token, n)))
        except subprocess.TimeoutExpired:
            name = self.job_name(token, n)
            job_id = self._lookup(name)
            if job_id is None:
                self._sleep(self.attribute_delay)
                job_id = self._lookup(name)
            if job_id is None:
                logger.warning(
                    "sbatch of %s timed out and no job is named so after %ss: submitting "
                    "once more",
                    name,
                    self.attribute_delay,
                )
                token = uuid.uuid4().hex
                try:
                    job_id = self._job_id(self._sbatch(self.sbatch_args(cls, token, n)))
                except subprocess.TimeoutExpired:
                    job_id = self._lookup(self.job_name(token, n))
                    if job_id is None:
                        raise SubstrateError(
                            f"sbatch of a worker of class {cls.name!r} timed out twice "
                            f"after {self.submit_timeout}s and neither job was found"
                        ) from None
        handles = [
            WorkerHandle(
                token=t,
                worker_class=cls.name,
                started=self._clock(),
                log=self._require().root / "private" / "logs" / f"{token}.log",
                cpus=cls.cpus or 1,
                job_id=job_id,
            )
            for t in worker_tokens(token, n)
        ]
        self._jobs[job_id] = handles
        self.handles.extend(handles)
        logger.info(
            "submitted %d worker(s) %s of class %r as Slurm job %s",
            n,
            token,
            cls.name,
            job_id,
        )
        return handles

    def stop(self, workers: ty.Sequence[WorkerHandle]) -> None:
        """``scancel`` the jobs all of whose workers are stopped; a worker given SIGTERM by Slurm
        settles what finished and releases the rest. A worker sharing its job with one still
        running stops on the stop mark its pool wrote. A failed ``scancel`` is logged and the
        workers stay listed. A cancelled job that Slurm still lists while it stops is neither live
        nor adopted again.
        """
        stopping = {id(h) for h in workers if h.job_id is not None}
        whole, part = [], []
        for job_id, handles in self._jobs.items():
            mine = [h for h in handles if id(h) in stopping]
            if mine:
                (whole if len(mine) == len(handles) else part).append((job_id, mine))
        if whole and not self._cancel([job_id for job_id, _ in whole]):
            whole = []
        self._stopping.update(job_id for job_id, _ in whole)
        for job_id, mine in whole + part:
            for h in mine:
                h.ended = "stopped"
            left = [h for h in self._jobs[job_id] if h not in mine]
            if left:
                self._jobs[job_id] = left
            else:
                del self._jobs[job_id]
