"""What a pool worker measures: its hardware and a `Sample` per settled job (RFC 0005 section 4.3)."""

from __future__ import annotations

import logging
import os
import re
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
    uuid: str | None = attrs.field(default=None, eq=False)
    # nvidia-smi's index; the logical ordinal is the position in ``gpus``
    index: int | None = attrs.field(default=None, eq=False)


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


def _mig_parents(exe: str, timeout: float) -> dict[str, str]:
    """Each ``MIG-`` UUID that ``nvidia-smi -L`` lists, mapped to its parent device's UUID (the
    ``--query-gpu`` rows carry only the parent's)."""
    try:
        out = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, timeout=timeout, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    parents: dict[str, str] = {}
    parent = None
    for line in out.splitlines():
        m = re.search(r"\(UUID: ((?:GPU|MIG)-[^)\s]+)\)", line)
        if m is None:
            continue
        if line.lstrip().startswith("GPU"):
            parent = m.group(1)
        elif parent is not None:
            parents[m.group(1)] = parent
    return parents


def _visible(
    rows: list[list[str]], mig: dict[str, str] | None = None
) -> list[list[str]]:
    """The rows ``[index, uuid, name, memory, compute_cap]`` that ``CUDA_VISIBLE_DEVICES`` grants,
    in its order (position is the logical ordinal torch sees); all rows when it is unset, none
    when it is empty, ``-1`` or ``NoDevFiles``.

    An entry beginning ``GPU-`` or ``MIG-`` is a UUID (an unambiguous prefix suffices); any other
    entry is an ``nvidia-smi`` index. A ``MIG-`` entry is the row of its parent (``mig`` maps it
    there) with its own UUID and no memory. Under a
    cgroup that constrains devices Slurm sets local indices ``0..k-1``, which ``nvidia-smi`` lists
    under the same numbers; host indices that it does not list are not mapped.

    Raises
    ------
    _Unmapped
        When an entry matches no listed device or more than one, or names a device twice.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return rows
    if visible.strip() in ("", "-1", "NoDevFiles"):
        return []
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    by_index = {r[0]: r for r in rows}
    granted = []
    for e in entries:
        if e.upper().startswith("MIG-"):
            full = [u for u in (mig or {}) if u.upper().startswith(e.upper())]
            parent = [r for r in rows if len(full) == 1 and r[1] == mig[full[0]]]
            hits = (
                [[parent[0][0], full[0], parent[0][2], "", *parent[0][4:]]]
                if parent
                else []
            )
        elif e.upper().startswith("GPU-"):
            hits = [r for r in rows if r[1].upper().startswith(e.upper())]
        else:
            hits = [by_index[e]] if e in by_index else []
        if len(hits) != 1:
            raise _Unmapped(
                f"CUDA_VISIBLE_DEVICES entry {e!r} matches {len(hits)} of nvidia-smi's "
                f"devices {sorted(by_index)}"
            )
        granted.append(hits[0])
    if len({r[1] for r in granted}) != len(granted):
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
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        mig = _mig_parents(exe, timeout) if "MIG-" in visible.upper() else {}
        try:
            granted = _visible([r for r in rows if len(r) >= 4], mig)
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
