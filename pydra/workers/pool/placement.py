"""Placement of a job on a worker class, and the fit that judges it."""

import typing as ty
from collections.abc import Callable, Sequence
from pathlib import Path
import attrs
from pydra.utils.resources import ComputeResources
from .classes import WorkerClass, parse_size

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

Verdict = ty.Literal["fits", "too small", "unfit", "unknown"]
CanStart = Callable[[WorkerClass], "bool | str"]


def warm_environment_label(job: ty.Any) -> str | None:
    """The hold label of a warm shell job's environment: the residency label (as
    `pydra.workers.pool.storage.Ticket.residency_label` computes it) of the environment's resident
    key alone, which names the environment and the reference the job carries. None for any other
    job, or one whose reference cannot be read here. Warm-start faults are counted, and held, under
    it."""
    import hashlib

    from pydra.compose.shell.task import Task as ShellTask
    from pydra.engine.resident import ResidentKey
    from pydra.utils.digest import jcs

    if not (
        isinstance(job.task, ShellTask)
        and job.environment.warm_requested
        and job.resolves_reference
    ):
        return None
    try:
        key = ResidentKey(
            "environment", job.environment.plugin_name(), job.resolved.key
        )
    except Exception:
        return None
    return hashlib.sha256(jcs([attrs.asdict(key)])).hexdigest()


@attrs.frozen
class ClassTried:
    """One class the walk considered.

    ``verdict`` is ``None`` when the class was not judged: it did not satisfy the
    request, or the substrate said it cannot start it (``reason`` says which).
    """

    worker_class: str
    verdict: Verdict | None
    reason: str = ""


@attrs.frozen
class FitRequest:
    """What a fit judges: resident state, resources and the job's size.

    Attributes
    ----------
    residency : tuple of ResidentKey
        The job's resident keys, in field order.
    resources : ComputeResources
    units : float, optional
        The job's size in the task's own units, when it declares one.
    batch : int, optional
        The task's declared batch (``python.define(batch=)``): the most jobs of one batched call
        its author states; None when it declares none.
    key : str, optional
        The residency digest (`pydra.engine.resident.residency_digest`); None without resident
        state.
    code : Code, optional
        The code a load of the residency runs under (`pydra.workers.pool.calibration.code_of`);
        None when it cannot be read where the job is placed.
    task : str, optional
        The code identity of the job's task (`pydra.workers.pool.calibration.task_key`), which
        selects the samples a measured fit reads.
    """

    residency: tuple[ty.Any, ...]
    resources: ComputeResources
    units: float | None = None
    key: str | None = None
    code: ty.Any = None
    task: str | None = None
    batch: int | None = None


def fit_request(job: "Job", providers: Sequence[ty.Any] = ()) -> FitRequest:
    """The request ``job`` makes of a class: its resources, resident keys, residency digest, the
    code its providers load under (``providers`` before the job's own) and its units (the task's
    ``units``, `pydra.compose.python.units_of`). A job whose resident keys cannot be read yet (a
    stand-in of ``plan()``, its inputs lazy) has none, nor units."""
    from pydra.engine import resident
    from pydra.compose.python import PythonTask
    from .calibration import code_of, residency_key, task_key

    res = job.resources if job.resources is not None else ComputeResources()
    try:
        residency = tuple(
            resident.spec_key(job.task, f, p)
            for f, p in resident.provided_fields(job.task)
        )
    except Exception:
        return FitRequest(residency=(), resources=res)
    units = job_units(job)
    batch = type(job.task)._batch if isinstance(job.task, PythonTask) else None
    if not residency:
        return FitRequest(residency=(), resources=res, units=units, batch=batch)
    return FitRequest(
        residency=residency,
        resources=res,
        units=units,
        batch=batch,
        key=residency_key(residency),
        code=code_of(job, providers),
        task=task_key(job.task),
    )


def job_units(job: "Job") -> float | None:
    """The job's size in its task's own units (`pydra.compose.python.units_of`); None for a task
    that declares no ``units``, or whose inputs are not all set yet."""
    from pydra.compose.python import PythonTask, units_of
    from pydra.utils.typing import is_lazy

    task = job.task
    if not isinstance(task, PythonTask) or type(task)._units is None:
        return None
    if any(is_lazy(v) for v in attrs.asdict(task, recurse=False).values()):
        return None
    return units_of(task)


class Fit(ty.Protocol):
    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried:
        """Judge ``request`` against ``cls``."""

    def batch(
        self, request: FitRequest, cls: WorkerClass, units: Sequence[float | None]
    ) -> int:
        """The most jobs of this residency one call takes."""

    def memory_bound(
        self, request: FitRequest, cls: WorkerClass, units: Sequence[float | None]
    ) -> int:
        """The most jobs one call holds in memory; 1 when not measured."""

    def probing(self, request: FitRequest, cls: WorkerClass) -> bool:
        """Whether `batch` is a ceiling a worker probes under, call by call."""


@attrs.frozen
class Placement:
    """Where a job runs. ``batch`` is the most jobs of its residency one call takes (what
    ``plan()`` reports): a declared batch, else, with ``probe``, the ceiling a worker probes
    under. ``memory_bound`` is the fit's memory bound (1 when not measured).
    """

    worker_class: str
    queue: str
    tried: tuple[ClassTried, ...]
    batch: int
    roots: frozenset[Path] = frozenset()
    memory_bound: int | None = None
    probe: bool = False


@attrs.frozen
class Unplaceable:
    """A job that fits no class; ``tried`` names every class and why."""

    tried: tuple[ClassTried, ...]
    request: ComputeResources

    @property
    def message(self) -> str:
        lines = [f"no worker class fits the request {self.request}:"]
        for t in self.tried:
            lines.append(f"  {t.worker_class}: {t.verdict or 'skipped'}: {t.reason}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.message


def _fmt(n: "int | None") -> str:
    return "unknown" if n is None else f"{n} bytes"


class DeclaredFit:
    """Judge from declared figures: device memory, host memory and time.

    No figure to compare gives ``"unknown"``. ``batch`` returns the task's declared batch, else 1.
    """

    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried:
        res = request.resources
        compared = False
        if res.time is not None and cls.time is not None:
            if res.time > cls.time:
                return ClassTried(
                    cls.name,
                    "unfit",
                    f"time {res.time}s exceeds the class limit of {cls.time}s",
                )
            compared = True
        need_host = parse_size(res.mem)
        if need_host is not None and cls.mem is not None:
            if need_host > cls.mem:
                return ClassTried(
                    cls.name,
                    "too small",
                    f"mem {need_host} bytes exceeds host memory {cls.mem} bytes",
                )
            compared = True
        if res.gpus:
            need = parse_size(res.gpu_mem)
            if need is None or cls.gpu_mem is None:
                return ClassTried(
                    cls.name,
                    "unknown",
                    f"gpu_mem needed {_fmt(need)}, class offers {_fmt(cls.gpu_mem)}",
                )
            share = cls.gpu_mem // cls.workers_per_device
            if need > share:
                return ClassTried(
                    cls.name,
                    "too small",
                    f"gpu_mem {need} bytes exceeds {share} bytes per worker "
                    f"({cls.gpu_mem} per device over {cls.workers_per_device})",
                )
            compared = True
        if not compared:
            return ClassTried(cls.name, "unknown", "no declared figure to compare")
        return ClassTried(cls.name, "fits", "declared figures fit")

    def batch(self, request, cls, units) -> int:
        return request.batch or 1

    def memory_bound(self, request, cls, units) -> int:
        return 1

    def probing(self, request, cls) -> bool:
        return False


def _candidate_reason(res: ComputeResources, cls: WorkerClass) -> "str | None":
    """Why ``cls`` cannot take ``res``, or ``None`` when it can."""
    if res.gpus:
        if cls.device == "cpu" or cls.gpus < res.gpus:
            return f"needs {res.gpus} gpu(s), class offers {cls.gpus} on {cls.device}"
        if res.gpu_type is not None and cls.gpu_type != res.gpu_type:
            return f"gpu_type {res.gpu_type!r} requested, class is {cls.gpu_type!r}"
    else:
        if cls.device != "cpu":
            return "a request without gpus is placed on cpu classes only"
    if cls.tags and res.tags and not cls.tags & set(res.tags):
        return f"class tags {sorted(cls.tags)} share none of the request's {sorted(res.tags)}"
    if res.cpus is not None and cls.cpus is not None and res.cpus > cls.cpus:
        return f"cpus {res.cpus} exceeds {cls.cpus}"
    need = parse_size(res.mem)
    if need is not None and cls.mem is not None and need > cls.mem:
        return f"mem {need} bytes exceeds {cls.mem} bytes"
    return None


def _order(cls: WorkerClass):
    # classes of unknown size sort last: they are the last resort of the ladder
    return (
        cls.gpu_mem is None,
        cls.gpu_mem or 0,
        cls.mem is None,
        cls.mem or 0,
        cls.name,
    )


def place(
    job: "Job",
    classes: Sequence[WorkerClass],
    fit: Fit,
    can_start: CanStart,
    *,
    request: FitRequest | None = None,
) -> "Placement | Unplaceable":
    """Map a job to a worker class and a queue, before anything is written.

    Reads the job's resources and resident keys (``request``, default `fit_request`); the job and
    its checksum are untouched. A routed residency's queue is ``<class>#<residency digest>``.
    """
    if request is None:
        request = fit_request(job)
    res = request.resources
    tried: list[ClassTried] = []
    candidates = []
    for cls in classes:
        reason = _candidate_reason(res, cls)
        if reason is None:
            candidates.append(cls)
        else:
            tried.append(ClassTried(cls.name, None, reason))
    walked: list[ClassTried] = []
    for cls in sorted(candidates, key=_order):
        verdict = can_start(cls)
        if verdict is not True:
            reason = verdict if isinstance(verdict, str) and verdict else "refused"
            walked.append(ClassTried(cls.name, None, f"cannot start: {reason}"))
            continue
        result = fit.judge(request, cls)
        walked.append(result)
        if result.verdict in ("fits", "unknown"):
            queue = cls.name
            if request.key is not None and res.residency == "route":
                queue = f"{cls.name}#{request.key}"
            units = [request.units]
            return Placement(
                worker_class=cls.name,
                queue=queue,
                tried=tuple(walked),
                batch=fit.batch(request, cls, units),
                memory_bound=fit.memory_bound(request, cls, units),
                probe=fit.probing(request, cls),
            )
    return Unplaceable(tried=tuple(tried + walked), request=res)
