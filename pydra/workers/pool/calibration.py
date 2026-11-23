"""Where a pool's measurements live (RFC 0005 section 3.3).

A pool's workers record a `Sample` per settled job and an `UnfitRecord` per load that ran out of
memory; its coordinator records a `ParityRecord` per sampled cold re-check (section 4.4). A
`Calibration` keeps them, keyed by the residency digest (`residency_key`) and, for samples, by the
hardware class (`hardware_of`), and gives back the `MemoryFit`, the unfit records and the per-batch
times `MeasuredFit` judges by. `DirectoryCalibration` keeps them as JSON files any later run reads.
"""

from __future__ import annotations

import hashlib
import functools
import json
import logging
import os
import statistics
import time
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.utils.general import atomic_write
from pydra.workers.pool.samples import Gpu, HardwareClass

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.workers.pool.classes import WorkerClass

logger = logging.getLogger(__name__)

__all__ = [
    "Calibration",
    "Code",
    "DirectoryCalibration",
    "MemoryFit",
    "ParityRecord",
    "Sample",
    "TimeRecord",
    "UnfitRecord",
    "code_of",
    "hardware_key",
    "hardware_of",
    "residency_key",
]


def _hardware_json(hardware: HardwareClass | None) -> dict | None:
    return None if hardware is None else hardware.to_json()


def _hardware_from(record: dict | None) -> HardwareClass | None:
    if record is None:
        return None
    gpus = tuple(Gpu(**g) for g in record.get("gpus", ()))
    return HardwareClass(record.get("device", "cpu"), gpus, record.get("cpus"))


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
    resident_bytes : int, optional
        The host bytes of this residency: ``baseline_bytes`` plus the worker's resident-set growth
        across the load of this residency; None when either was not read.
    peak_bytes : int, optional
        ``resident_bytes`` plus the call's host peak above the resident set just before the call;
        None when the host gives no per-call peak.
    device_baseline_bytes : int, optional
        Device bytes of the worker's own processes before its first load, on a CUDA class.
    device_bytes : int, optional
        Device bytes of the worker's own processes gained across the load of this residency.
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
    key : str, optional
        The residency digest (`residency_key`); None for a job with no resident state.
    hardware : HardwareClass, optional
        The hardware class of the worker's class (`hardware_of`).
    device_peak_bytes : int, optional
        ``device_bytes`` plus the call's device allocator peak above what was allocated before the
        call; None when not read (a provider whose state is another process).
    others : int
        Resident keys of other residencies the worker held during the call; a sample with any is
        not fitted (their bytes cannot be told apart).
    task : str, optional
        The code identity of the call's task (`task_key`): times and memory are read per task.
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
    key: str | None = None
    hardware: HardwareClass | None = None
    device_peak_bytes: int | None = None
    others: int = 0
    task: str | None = None

    def to_json(self) -> dict[str, ty.Any]:
        record = attrs.asdict(self, recurse=False)
        record["hardware"] = _hardware_json(self.hardware)
        return record

    @classmethod
    def from_json(cls, record: dict) -> "Sample":
        fields = {f.name for f in attrs.fields(cls)}
        values = {k: v for k, v in record.items() if k in fields}
        values["hardware"] = _hardware_from(record.get("hardware"))
        return cls(**values)


@attrs.frozen
class Code:
    """The code a load ran under: each provider's ``version()`` by name, and the warm environment's
    resolved reference (None without one)."""

    versions: tuple[tuple[str, str], ...] = attrs.field(
        default=(), converter=lambda v: tuple(sorted((str(a), str(b)) for a, b in v))
    )
    environment: str | None = None


@attrs.frozen
class UnfitRecord:
    """A load of residency ``key`` that ran out of ``resource`` memory in a grant of ``grant`` bytes.

    It binds only the code that wrote it (`binds`): a change to a provider's ``version()`` or to the
    environment's resolved reference gives a record that no longer binds, as a
    ``current_unfit`` does.

    Attributes
    ----------
    key : str
        The residency digest.
    resource : {"device", "host"}
        The memory that ran out.
    grant : int, optional
        The bytes the worker had of ``resource`` (per worker on a device); None when its class
        states none, and the record then binds its own class only.
    worker_class : str
        The class of the worker whose load ran out.
    code : Code
        The providers' versions and the environment's reference the load ran under.
    worker : str, optional
        The worker's token.
    ended : float
        ``time.time()`` when it was recorded.
    """

    key: str
    resource: ty.Literal["device", "host"]
    grant: int | None
    worker_class: str
    code: Code = attrs.field(factory=Code)
    worker: str | None = None
    ended: float = attrs.field(factory=time.time)

    def binds(self, code: Code | None) -> bool:
        """Whether the record holds for a load under ``code``; an unknown code binds nothing."""
        return code is not None and code == self.code

    def to_json(self) -> dict[str, ty.Any]:
        record = attrs.asdict(self, recurse=False)
        record["code"] = {
            "versions": [list(v) for v in self.code.versions],
            "environment": self.code.environment,
        }
        return record

    @classmethod
    def from_json(cls, record: dict) -> "UnfitRecord":
        known = {f.name for f in attrs.fields(cls)}
        values = {k: v for k, v in record.items() if k in known}
        code = values.pop("code", None) or {}
        return cls(
            code=Code(code.get("versions", ()), code.get("environment")), **values
        )


@attrs.frozen
class TimeRecord:
    """Per-job seconds of warm calls of ``batch`` jobs, the median of ``n``."""

    batch: int
    per_job_s: float
    n: int


@attrs.frozen
class ParityRecord:
    """One sampled cold re-check (RFC 0005 section 4.4): job ``checksum`` of residency ``key`` run
    again alone on a fresh residency, and whether its output digest equals the committed entry's.

    Attributes
    ----------
    key : str
        The residency digest, ``"none"`` for a job with no resident state.
    checksum : str
        The job's checksum.
    batch_sizes : tuple of int
        The batch size of the committed run, then of the re-check (1).
    equal : bool or None
        The output digests are equal (True) or not, or the store's comparison refused the fresh
        outputs (False); None when the re-check could not run (the task, a provider or the
        device failed in the coordinator), which is no statement about the task.
    committed, fresh : str, optional
        The committed entry's output digest and the re-check's; None when one was not reached.
    reason : str
        Why ``equal`` is not true ("" when it is).
    ended : float
        ``time.time()`` when it was recorded.
    """

    key: str
    checksum: str
    batch_sizes: tuple[int, int] = attrs.field(
        converter=lambda v: tuple(int(b) for b in v)
    )
    equal: bool | None
    committed: str | None = None
    fresh: str | None = None
    reason: str = ""
    ended: float = attrs.field(factory=time.time)

    def to_json(self) -> dict[str, ty.Any]:
        record = attrs.asdict(self, recurse=False)
        record["batch_sizes"] = list(self.batch_sizes)
        return record

    @classmethod
    def from_json(cls, record: dict) -> "ParityRecord":
        known = {f.name for f in attrs.fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in known})


@attrs.frozen
class MemoryFit:
    """Memory of one kind: the resident bytes, and the peak beyond them as ``m0 + m1 × units``.

    ``m1`` is None when the samples carry fewer than two distinct units: the per-unit memory is then
    unknown, not zero.
    """

    resident: int
    m0: float
    m1: float | None
    n: int


class Calibration(ty.Protocol):
    """Where measurements live; ``key`` is the residency digest."""

    def memory(
        self, key: str, hardware: HardwareClass, task: str | None = None
    ) -> MemoryFit | None:
        """The memory fit of ``key`` on ``hardware``; None when not measured."""

    def unfit(self, key: str) -> ty.Sequence[UnfitRecord]:
        """Every unfit record of ``key``, whichever code wrote it."""

    def times(
        self, key: str, hardware: HardwareClass, task: str | None = None
    ) -> ty.Sequence[TimeRecord]:
        """Per-job times of ``key`` on ``hardware``, one per batch size measured."""

    def parity(self, key: str) -> ty.Sequence[ParityRecord]:
        """Every re-check of ``key``."""

    def record(self, record: Sample | UnfitRecord | ParityRecord) -> None:
        """Keep ``record``."""


@functools.lru_cache(maxsize=None)
def _class_key(klass: type) -> str:
    import marshal

    from pydra.compose.python import PythonTask
    from pydra.utils.general import get_fields

    name = f"{klass.__module__}.{klass.__qualname__}"
    if not issubclass(klass, PythonTask):
        return name
    functions = [get_fields(klass)[klass._executor_name].default]
    code = hashlib.sha256(
        b"".join(marshal.dumps(f.__code__) for f in functions if f is not None)
    ).hexdigest()
    return f"{name}:{code[:16]}"


def task_key(task: ty.Any) -> str:
    """The identity of ``task``'s class, which keys its samples: its qualified name and, for a
    Python task, a digest of the compiled code of its function"""
    return _class_key(type(task))


def hardware_of(cls: "WorkerClass") -> HardwareClass:
    """The hardware class a worker of ``cls`` measures on, as both a worker and a coordinator name
    it: its device, and per GPU the class's ``gpu_type`` and per-device memory."""
    gpus = tuple(Gpu(cls.gpu_type or "", cls.gpu_mem) for _ in range(cls.gpus))
    return HardwareClass(cls.device, gpus, cls.cpus)


def hardware_key(hardware: HardwareClass) -> str:
    """A file name for ``hardware``: the device, the GPU count, type and smallest memory."""
    if not hardware.gpus:
        return hardware.device
    products = sorted({g.product for g in hardware.gpus})
    name = "-".join(p for p in products if p) or "gpu"
    memory = hardware.gpu_mem
    return _safe(
        f"{hardware.device}-{len(hardware.gpus)}x{name}-"
        f"{'unknown' if memory is None else memory}"
    )


def _safe(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return cleaned or hashlib.sha256(name.encode()).hexdigest()[:16]


def residency_key(residency: ty.Sequence[ty.Any]) -> str | None:
    """The residency digest of ``residency`` (`ResidentKey`s, or their ``attrs.asdict``, as a ticket
    carries them), equal to `pydra.engine.resident.residency_digest` of the job; None when empty.
    """
    from pydra.engine.resident import ResidentKey

    if not residency:
        return None
    digests = [
        (k if isinstance(k, ResidentKey) else ResidentKey(**k)).digest()
        for k in residency
    ]
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def code_of(job: "Job", providers: ty.Sequence[ty.Any] = ()) -> Code | None:
    """The code a load of ``job``'s residency runs under: the ``version()`` of the provider of each
    provided field (``providers`` first, as a pool's workers add them, then the job's own); None
    when a provider cannot be found or its version read here."""
    from pydra.engine import resident

    try:
        fields = resident.provided_fields(job.task)
    except Exception:
        return None
    versions = []
    for _, name in fields:
        provider = next((p for p in providers if p.name == name), None)
        try:
            if provider is None:
                provider = resident.find_provider(job, name)
            versions.append((name, str(provider.version())))
        except Exception as err:
            logger.debug("no version for provider %r: %s", name, err)
            return None
    return Code(tuple(versions))


def _median(values: ty.Sequence[float]) -> float:
    return float(statistics.median(values))


def _theil_sen(points: ty.Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """``(intercept, slope)`` of the Theil–Sen line through ``points``; slope 0 when every ``x`` is
    equal; None without points."""
    if not points:
        return None
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1 :]
        if x2 != x1
    ]
    slope = _median(slopes) if slopes else 0.0
    intercept = _median([y - slope * x for x, y in points])
    return intercept, slope


class DirectoryCalibration:
    """A `Calibration` kept as JSON files under ``root``, written by a pool's workers and its
    coordinator and read by any later run.

    Layout::

        <root>/<key>/<hardware key>/samples/<worker token>.jsonl   one writer per file, appended
        <root>/<key>/unfit/<time>-<uuid>.json                       one file per record
        <root>/<key>/parity/<time>-<uuid>.json                      one file per record

    A file of one record is written under a temporary name and renamed, so a reader never sees it
    half written; a samples file has one writer (its worker), and a reader skips a last line that
    is not yet whole.

    Parameters
    ----------
    root : Path
    min_samples : int
        Warm samples a memory fit needs; fewer give None (unmeasured).
    """

    def __init__(self, root: os.PathLike, min_samples: int = 3):
        if min_samples < 1:
            raise ValueError(f"min_samples must be positive, not {min_samples!r}")
        self.root = Path(root)
        self.min_samples = min_samples

    def __repr__(self) -> str:
        return f"DirectoryCalibration({str(self.root)!r})"

    # -- writing -------------------------------------------------------------------------------

    def record(self, record: Sample | UnfitRecord | ParityRecord) -> None:
        if isinstance(record, Sample):
            if record.key is None or record.hardware is None:
                raise ValueError("a sample is recorded with its key and hardware")
            where = (
                self.root
                / _safe(record.key)
                / hardware_key(record.hardware)
                / "samples"
            )
            where.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record.to_json(), sort_keys=True) + "\n"
            flags = (
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(where / f"{_safe(record.worker)}.jsonl", flags, 0o644)
            try:
                data = line.encode()
                while data:
                    data = data[os.write(fd, data) :]
            finally:
                os.close(fd)
            return
        if isinstance(record, UnfitRecord):
            kind = "unfit"
        elif isinstance(record, ParityRecord):
            kind = "parity"
        else:
            raise TypeError(f"not a calibration record: {record!r}")
        where = self.root / _safe(record.key) / kind
        where.mkdir(parents=True, exist_ok=True)
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex}.json"
        with atomic_write(where / name, "w") as f:
            json.dump(record.to_json(), f, sort_keys=True)

    # -- reading -------------------------------------------------------------------------------

    def _records(self, key: str, kind: str) -> list[dict]:
        where = self.root / _safe(key) / kind
        found = []
        try:
            names = sorted(n for n in os.listdir(where) if n.endswith(".json"))
        except FileNotFoundError:
            return []
        for name in names:
            try:
                found.append(json.loads((where / name).read_text()))
            except (OSError, ValueError) as err:
                logger.warning(
                    "skipping the calibration record %s: %s", where / name, err
                )
        return found

    def _parsed(self, key: str, kind: str, parse: ty.Callable[[dict], ty.Any]) -> list:
        found = []
        for record in self._records(key, kind):
            try:
                found.append(parse(record))
            except (TypeError, ValueError, AttributeError) as err:
                logger.warning("skipping a %s record of %s: %s", kind, key, err)
        return found

    def unfit(self, key: str) -> list[UnfitRecord]:
        return self._parsed(key, "unfit", UnfitRecord.from_json)

    def parity(self, key: str) -> list[ParityRecord]:
        return self._parsed(key, "parity", ParityRecord.from_json)

    def samples(self, key: str, hardware: HardwareClass) -> list[Sample]:
        """Every sample of ``key`` on ``hardware``."""
        where = self.root / _safe(key) / hardware_key(hardware) / "samples"
        try:
            names = sorted(n for n in os.listdir(where) if n.endswith(".jsonl"))
        except FileNotFoundError:
            return []
        found = []
        for name in names:
            try:
                text = (where / name).read_text()
            except OSError as err:
                logger.warning("skipping the samples %s: %s", where / name, err)
                continue
            for line in text.splitlines(keepends=True):
                if not line.endswith("\n"):
                    break  # being written
                try:
                    found.append(Sample.from_json(json.loads(line)))
                except (ValueError, TypeError) as err:
                    logger.warning("skipping a sample in %s: %s", where / name, err)
        return found

    def _warm(
        self, key: str, hardware: HardwareClass, task: str | None = None
    ) -> list[Sample]:
        """Samples of calls on a residency loaded before the call, alone in its worker; of
        ``task`` (and those that record no task), when given."""
        return [
            s
            for s in self.samples(key, hardware)
            if not s.cold
            and not s.reload
            and s.others == 0
            and (task is None or s.task is None or s.task == task)
        ]

    def memory(
        self, key: str, hardware: HardwareClass, task: str | None = None
    ) -> MemoryFit | None:
        """The fit of peak bytes beyond the resident ones against units over the warm samples: device
        memory on a GPU class, host memory otherwise.

        A sample without a measured peak is not used, and with none there is no fit. The resident
        bytes are the least seen. ``m1`` is the Theil–Sen slope over the samples whose units are known,
        None with fewer than two distinct units (``m0`` is then the median excess); a negative slope,
        or fewer than ``min_samples`` samples, gives None.
        """
        rows: list[tuple[float | None, int, int]] = []
        for s in self._warm(key, hardware, task):
            if hardware.gpus:
                if s.device_bytes is None or s.device_peak_bytes is None:
                    continue
                base = s.device_baseline_bytes or 0
                after, peak = base + s.device_bytes, base + s.device_peak_bytes
            else:
                if s.resident_bytes is None or s.peak_bytes is None:
                    continue
                after, peak = s.resident_bytes, s.peak_bytes
            rows.append((s.units, peak, after))
        if len(rows) < self.min_samples:
            return None
        resident = min(after for _, _, after in rows)
        known = [
            (float(u), float(peak - after)) for u, peak, after in rows if u is not None
        ]
        if len({u for u, _ in known}) >= 2:
            fit = _theil_sen(known)
            if fit is None:
                return None
            m0, m1 = fit
            if m1 < 0:
                return None
            return MemoryFit(resident=resident, m0=m0, m1=m1, n=len(known))
        m0 = _median([float(peak - after) for _, peak, after in rows])
        return MemoryFit(resident=resident, m0=m0, m1=None, n=len(rows))

    def times(
        self, key: str, hardware: HardwareClass, task: str | None = None
    ) -> list[TimeRecord]:
        """The median per-job seconds of warm calls, per batch size."""
        by_batch: dict[int, list[float]] = {}
        for s in self._warm(key, hardware, task):
            if s.batch_size >= 1:
                by_batch.setdefault(s.batch_size, []).append(s.wall_s / s.batch_size)
        return [
            TimeRecord(batch, _median(times), len(times))
            for batch, times in sorted(by_batch.items())
        ]
