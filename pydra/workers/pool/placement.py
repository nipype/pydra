"""Placement of a job on a worker class, and the fit that judges it."""

import hashlib
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
    """What a fit judges: resident state, resources and the job's size."""

    residency: tuple[ty.Any, ...]
    resources: ComputeResources
    units: float | None = None


class Fit(ty.Protocol):
    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried:
        """Judge ``request`` against ``cls``."""

    def batch(
        self, request: FitRequest, cls: WorkerClass, units: Sequence[float | None]
    ) -> int:
        """The most jobs of this residency one call takes."""


@attrs.frozen
class Placement:
    worker_class: str
    queue: str
    tried: tuple[ClassTried, ...]
    batch: int
    roots: frozenset[Path] = frozenset()


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

    No figure to compare gives ``"unknown"``. ``batch`` returns 1.
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
        return 1


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


def residency_digest(residency: Sequence[ty.Any]) -> str:
    joined = "\n".join(sorted(str(k) for k in residency))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def place(
    job: "Job",
    classes: Sequence[WorkerClass],
    fit: Fit,
    can_start: CanStart,
) -> "Placement | Unplaceable":
    """Map a job to a worker class and a queue, before anything is written.

    Reads the job's resources only; the job and its checksum are untouched.
    """
    res = job.resources if job.resources is not None else ComputeResources()
    request = FitRequest(residency=(), resources=res, units=None)
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
            if request.residency and res.residency == "route":
                queue = f"{cls.name}#{residency_digest(request.residency)}"
            return Placement(
                worker_class=cls.name,
                queue=queue,
                tried=tuple(walked),
                batch=fit.batch(request, cls, [request.units]),
            )
    return Unplaceable(tried=tuple(tried + walked), request=res)
