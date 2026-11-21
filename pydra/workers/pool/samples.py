"""What a pool worker measures: its hardware and a `Sample` per settled job (RFC 0005 section 4.3)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import typing as ty

import attrs

logger = logging.getLogger(__name__)

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
    uuid: str | None = None
    index: int | None = (
        None  # nvidia-smi's index; the logical ordinal is the position in ``gpus``
    )


@attrs.frozen
class HardwareClass:
    """The hardware a worker reads at start.

    Attributes
    ----------
    device : {"cpu", "cuda"}
    gpus : tuple of Gpu
        The visible devices, in ``CUDA_VISIBLE_DEVICES`` order; position is the logical ordinal.
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


class _Unmapped(ValueError):
    """``CUDA_VISIBLE_DEVICES`` names a device ``nvidia-smi`` does not list."""


def _is_uuid(entry: str) -> bool:
    return entry.upper().startswith(("GPU-", "MIG-"))


def _visible(rows: list[list[str]]) -> list[list[str]]:
    """The rows ``[index, uuid, name, memory, compute_cap]`` that ``CUDA_VISIBLE_DEVICES`` grants,
    in its order (position is the logical ordinal torch sees); all rows when it is unset.

    An entry beginning ``GPU-`` or ``MIG-`` is a UUID (an unambiguous prefix suffices); any other
    entry is an ``nvidia-smi`` index. Under a cgroup that constrains devices ``nvidia-smi`` lists
    only the granted devices, renumbered from 0, so when every entry is an index and one is not
    listed the rows are taken in the order ``nvidia-smi`` reports them, which is the order CUDA
    enumerates them in; that needs as many entries as listed devices.

    Raises
    ------
    _Unmapped
        When an entry matches no listed device or more than one, or names a device twice.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return rows
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if not entries:
        return []
    by_index = {r[0]: r for r in rows}
    if not any(_is_uuid(e) for e in entries) and not all(
        e in by_index for e in entries
    ):
        if all(e.isdigit() for e in entries) and len(entries) == len(rows):
            return list(rows)
        raise _Unmapped(
            f"CUDA_VISIBLE_DEVICES={visible!r} does not match nvidia-smi's "
            f"devices {sorted(by_index)}"
        )
    granted = []
    for e in entries:
        if _is_uuid(e):
            hits = [r for r in rows if r[1].upper().startswith(e.upper())]
        else:
            hits = [by_index[e]] if e in by_index else []
        if len(hits) != 1:
            raise _Unmapped(
                f"CUDA_VISIBLE_DEVICES entry {e!r} matches {len(hits)} of nvidia-smi's devices"
            )
        granted.append(hits[0])
    if len({r[0] for r in granted}) != len(granted):
        raise _Unmapped(f"CUDA_VISIBLE_DEVICES={visible!r} names a device twice")
    return granted


def read_hardware(timeout: float = 20.0) -> HardwareClass:
    """The devices from ``nvidia-smi`` (none without it) and the CPUs this process may use.

    A ``CUDA_VISIBLE_DEVICES`` that cannot be mapped to the devices ``nvidia-smi`` lists is logged
    as a warning and grants no device, never a guessed one.
    """
    exe = shutil.which("nvidia-smi")
    gpus: list[Gpu] = []
    if exe is not None:
        try:
            out = subprocess.run(
                [
                    exe,
                    "--query-gpu=index,uuid,name,memory.total,compute_cap",
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
        try:
            granted = _visible([r for r in rows if len(r) >= 4])
        except _Unmapped as exc:
            logger.warning("no GPU granted: %s", exc)
            granted = []
        for row in granted:
            try:
                memory = int(float(row[3])) << 20  # MiB
            except ValueError:
                memory = None
            capability = (
                row[4] if len(row) > 4 and row[4] not in ("", "[N/A]") else None
            )
            index = int(row[0]) if row[0].isdigit() else None
            gpus.append(Gpu(row[2], memory, capability, row[1], index))
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
    """What a worker measured for one settled job (senselab's ``Sample`` without its audio fields).

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
