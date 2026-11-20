"""Substrates: what starts a pool's workers (RFC 0005 section 4.5).

A substrate starts workers of a class, lists the ones it started that are pending or running, stops
them, and says whether one more may start. `LocalSubstrate` starts worker processes on this host;
`ExternalSubstrate` starts nothing, for workers started by hand or by another system.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.workers.pool.classes import WorkerClass

__all__ = ["ExternalSubstrate", "LocalSubstrate", "Substrate", "WorkerHandle"]

logger = logging.getLogger("pydra.pool")

IS_WINDOWS = sys.platform == "win32"


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
    """

    token: str
    worker_class: str
    started: float = attrs.field(factory=time.monotonic)
    devices: tuple[str, ...] = ()
    process: subprocess.Popen | None = attrs.field(default=None, repr=False)
    log: Path | None = None
    cpus: int = 1

    @property
    def cause(self) -> str | None:
        """Why the worker ended, when its substrate can tell: its exit code and last log line."""
        if self.process is None or self.process.poll() is None:
            return None
        why = f"exit code {self.process.returncode}"
        last = _last_line(self.log)
        return f"{why}: {last}" if last else why


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


@attrs.define
class Attachment:
    """What a substrate needs of the pool it starts workers for (`WarmPool` passes it)."""

    root: Path
    settings: Path
    providers: tuple[str, ...] = ()
    codec: str | None = None
    faults: Path | None = None
    classify: str | None = None


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
    stop_wait : float
        Seconds `stop` waits for a worker given SIGTERM before it kills its process group.
    """

    def __init__(
        self,
        devices: ty.Sequence[int | str] | None = None,
        *,
        per_user_cpu_cap: int | None = None,
        python: str = sys.executable,
        stop_wait: float = 30.0,
    ):
        self._devices = None if devices is None else [str(d) for d in devices]
        self.per_user_cpu_cap = per_user_cpu_cap
        self.python = python
        self.stop_wait = stop_wait
        self.handles: list[WorkerHandle] = []
        self._attachment: Attachment | None = None

    def attach(self, attachment: Attachment) -> None:
        """Start workers for the pool ``attachment`` names."""
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
        argv = [
            self.python,
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
        if at.faults is not None:
            argv += ["--faults", str(at.faults)]
        if at.classify is not None:
            argv += ["--classify", at.classify]
        env = dict(os.environ)
        if cls.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        logs = at.root / "private" / "logs"
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        log = logs / f"{token}.log"
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
        """SIGTERM each worker (it settles what finished and releases the rest), then kill the
        process group of any still running after ``stop_wait``."""
        running = [
            h for h in workers if h.process is not None and h.process.poll() is None
        ]
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
