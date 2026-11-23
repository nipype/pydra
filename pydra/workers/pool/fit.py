"""`MeasuredFit`: placement judged from a pool's measurements (RFC 0005 section 3.3)."""

from __future__ import annotations

import math
import typing as ty

from pydra.workers.pool.calibration import Calibration, MemoryFit, hardware_of
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.placement import ClassTried, DeclaredFit, FitRequest

__all__ = ["MeasuredFit"]


def _grant(cls: WorkerClass) -> tuple[ty.Literal["device", "host"], int | None]:
    """The memory a worker of ``cls`` is judged against: its share of a device on a GPU class (the
    per-device memory over ``workers_per_device``), its host memory otherwise."""
    if cls.gpus:
        share = None if cls.gpu_mem is None else cls.gpu_mem // cls.workers_per_device
        return "device", share
    return "host", cls.mem


def _grants(cls: WorkerClass) -> dict[str, int | None]:
    """Every memory a worker of ``cls`` holds, by resource: its host memory, and on a GPU class its
    share of a device as well."""
    grants: dict[str, int | None] = {"host": cls.mem}
    if cls.gpus:
        grants["device"] = _grant(cls)[1]
    return grants


class MeasuredFit:
    """The measured fit rule over a `Calibration`.

    `judge`, in order:

    1. A class whose ``time`` is below the request's is ``"unfit"`` (as under every `Fit`).
    2. A class is ``"unfit"`` when a load of the residency ran out of memory, under the code it
       runs under now, in a grant at least as large as the class's of the same resource (or, when
       either grant is unknown, on that class): host records against host memory on every class,
       device records against the device share on a GPU class.
    3. A GPU class with ``gpu_mem=None`` is ``"unknown"``.
    4. With a memory fit of the residency on the class's hardware, the class is ``"too small"``
       when ``resident + m0 > headroom × grant``, otherwise it ``"fits"``.
    5. With no fit, the verdict is `DeclaredFit`'s.

    `batch`: ``max(1, floor((headroom × share − resident − m0) / (m1 × length)))``, ``length`` the
    longest of ``units``, at most the class's ``quantum_jobs``: the memory bound. It is 1 when ``m1``
    is unknown, or when ``m1 > 0`` and no unit is known, and ``quantum_jobs`` when ``m1`` was
    measured as 0. Then, when per-job times at two or more
    batch sizes within the bound are recorded, the fastest of them.

    Parameters
    ----------
    calibration : Calibration
    headroom : float
        The fraction of a grant a residency may use; required, with no default.
    """

    def __init__(self, calibration: Calibration, headroom: float):
        if not 0 < headroom <= 1:
            raise ValueError(f"headroom is in (0, 1], not {headroom!r}")
        self.calibration = calibration
        self.headroom = headroom
        self._declared = DeclaredFit()

    def __repr__(self) -> str:
        return f"MeasuredFit({self.calibration!r}, headroom={self.headroom})"

    def _memory(self, request: FitRequest, cls: WorkerClass) -> MemoryFit | None:
        if request.key is None:
            return None
        return self.calibration.memory(request.key, hardware_of(cls), request.task)

    def _unfit(self, request: FitRequest, cls: WorkerClass) -> ClassTried | None:
        if request.key is None:
            return None
        grants = _grants(cls)
        held = []
        for r in self.calibration.unfit(request.key):
            if not r.binds(request.code) or r.resource not in grants:
                continue
            grant = grants[r.resource]
            if (
                r.worker_class == cls.name
                if r.grant is None or grant is None
                else r.grant >= grant
            ):
                held.append((r, grant))
        if not held:
            return None
        worst, grant = max(held, key=lambda h: -1 if h[0].grant is None else h[0].grant)
        where = (
            f"on class {worst.worker_class}"
            if worst.grant is None
            else f"in a grant of {worst.grant} bytes on class {worst.worker_class}"
        )
        return ClassTried(
            cls.name,
            "unfit",
            f"a load ran out of {worst.resource} memory {where} "
            f"(this class's {worst.resource} grant: "
            f"{'unknown' if grant is None else f'{grant} bytes'})",
        )

    def judge(self, request: FitRequest, cls: WorkerClass) -> ClassTried:
        res = request.resources
        if res.time is not None and cls.time is not None and res.time > cls.time:
            return ClassTried(
                cls.name,
                "unfit",
                f"time {res.time}s exceeds the class limit of {cls.time}s",
            )
        unfit = self._unfit(request, cls)
        if unfit is not None:
            return unfit
        resource, grant = _grant(cls)
        if cls.gpus and grant is None:
            return ClassTried(
                cls.name, "unknown", "the class's device memory is unknown"
            )
        fit = self._memory(request, cls)
        if fit is None or grant is None:
            return self._declared.judge(request, cls)
        limit = self.headroom * grant
        if fit.resident + fit.m0 > limit:
            return ClassTried(
                cls.name,
                "too small",
                f"resident {fit.resident} + m0 {fit.m0:g} bytes exceed {limit:.0f} bytes, "
                f"{self.headroom:g} of the {resource} grant of {grant} bytes",
            )
        return ClassTried(
            cls.name,
            "fits",
            f"resident {fit.resident} + m0 {fit.m0:g} bytes within {limit:.0f} bytes, "
            f"{self.headroom:g} of the {resource} grant of {grant} bytes",
        )

    def batch(
        self,
        request: FitRequest,
        cls: WorkerClass,
        units: ty.Sequence[float | None],
    ) -> int:
        fit = self._memory(request, cls)
        _, share = _grant(cls)
        if fit is None or share is None or fit.m1 is None:
            return 1
        room = self.headroom * share - fit.resident - fit.m0
        known = [u for u in units if u is not None]
        if room < 0:
            bound = 1
        elif fit.m1 <= 0:
            bound = cls.quantum_jobs
        elif not known:
            bound = 1  # the per-job memory is measured, this job's length is not
        else:
            per_job = fit.m1 * max(known)
            if per_job <= 0:
                bound = cls.quantum_jobs
            else:
                bound = max(1, min(cls.quantum_jobs, math.floor(room / per_job)))
        if request.key is None:
            return bound
        times = [
            t
            for t in self.calibration.times(request.key, hardware_of(cls), request.task)
            if t.batch <= bound
        ]
        if len({t.batch for t in times}) >= 2:
            return min(times, key=lambda t: (t.per_job_s, t.batch)).batch
        return bound
