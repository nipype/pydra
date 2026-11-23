"""What a pool worker measures: its hardware and a `Sample` per settled job (RFC 0005 section 4.3)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import typing as ty

import attrs

__all__ = [
    "Gpu",
    "HardwareClass",
    "Sample",
    "device_memory",
    "host_memory",
    "read_hardware",
]


@attrs.frozen
class Gpu:
    """One device the worker was granted."""

    product: str
    memory: int | None  # bytes
    compute_capability: str | None = None


@attrs.frozen
class HardwareClass:
    """The hardware a worker reads at start.

    Attributes
    ----------
    device : {"cpu", "cuda"}
    gpus : tuple of Gpu
        The visible devices, in ``CUDA_VISIBLE_DEVICES`` order.
    cpus : int or None
        The CPUs this process may run on.
    """

    device: str = "cpu"
    gpus: tuple[Gpu, ...] = attrs.field(default=(), converter=tuple)
    cpus: int | None = None

    @property
    def gpu_mem(self) -> int | None:
        """The smallest device memory of the granted devices, None when unknown or no device."""
        known = [g.memory for g in self.gpus if g.memory is not None]
        return min(known) if known and len(known) == len(self.gpus) else None

    def to_json(self) -> dict:
        return attrs.asdict(self)


def _cpus() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return len(affinity(0))
        except OSError:
            pass
    return os.cpu_count()


def _visible(rows: list[list[str]]) -> list[list[str]]:
    """The rows (index first) that ``CUDA_VISIBLE_DEVICES`` grants, in its order; all when unset or
    when it names devices by UUID."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return rows
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if not entries:
        return []
    if not all(e.isdigit() for e in entries):
        return rows
    by_index = {r[0]: r for r in rows}
    return [by_index[e] for e in entries if e in by_index]


def read_hardware(timeout: float = 20.0) -> HardwareClass:
    """The devices from ``nvidia-smi`` (none without it) and the CPUs this process may use."""
    exe = shutil.which("nvidia-smi")
    gpus: list[Gpu] = []
    if exe is not None:
        try:
            out = subprocess.run(
                [
                    exe,
                    "--query-gpu=index,name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        rows = [[c.strip() for c in line.split(",")] for line in out.splitlines()]
        for row in _visible([r for r in rows if len(r) >= 3]):
            try:
                memory = int(float(row[2])) << 20  # MiB
            except ValueError:
                memory = None
            capability = (
                row[3] if len(row) > 3 and row[3] not in ("", "[N/A]") else None
            )
            gpus.append(Gpu(row[1], memory, capability))
    return HardwareClass("cuda" if gpus else "cpu", tuple(gpus), _cpus())


def host_memory() -> tuple[int | None, int | None]:
    """This process's resident bytes now and at its peak; None where the host does not say."""
    current = peak = None
    try:
        import psutil  # optional

        info = psutil.Process().memory_info()
        current = info.rss
        peak = getattr(info, "peak_wset", None)
    except Exception:
        if sys.platform.startswith("linux"):
            try:
                with open("/proc/self/statm") as f:
                    current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
    if peak is None:
        try:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = maxrss if sys.platform == "darwin" else maxrss * 1024
        except (ImportError, OSError):
            pass
    return current, peak


def device_memory() -> int | None:
    """Device bytes held by every compute process on the visible devices (the co-tenancy baseline),
    None without ``nvidia-smi``."""
    from pydra.engine.procgroup import gpu_memory_by_pid

    held = gpu_memory_by_pid()
    return None if held is None else sum(held.values())


@attrs.frozen
class Sample:
    """What a worker measured for one settled job.

    Attributes
    ----------
    wall_s : float
        Seconds of the call that ran the job.
    claim_s : float
        Seconds the worker spent claiming the tickets it ran with.
    batch_size : int
        Jobs in the call.
    units : float, optional
        The job's units (padded units of the call, when batched).
    baseline_bytes : int, optional
        The worker's resident bytes at start, before any provider loaded.
    resident_bytes, peak_bytes : int, optional
        The worker's resident bytes after the call, and its peak so far.
    device_baseline_bytes : int, optional
        Device-wide bytes before the worker's first claim on a CUDA class.
    device_bytes : int, optional
        Device-wide bytes, read once per claim on a CUDA class.
    load_s : float
        Seconds loading the residency for this job (0 when it was resident).
    cold : bool
        Whether this job's residency was loaded for it.
    since_load : int
        Jobs this worker ran on the residency since it was loaded, before this one.
    reload : bool
        Whether the residency was reloaded after a resource fault.
    worker : str
        The worker's token.
    ended : float
        ``time.time()`` when the job settled.
    """

    wall_s: float
    claim_s: float
    batch_size: int
    units: float | None
    baseline_bytes: int | None
    resident_bytes: int | None
    peak_bytes: int | None
    device_baseline_bytes: int | None
    device_bytes: int | None
    load_s: float
    cold: bool
    since_load: int
    reload: bool
    worker: str
    ended: float

    def to_json(self) -> dict[str, ty.Any]:
        return attrs.asdict(self)
