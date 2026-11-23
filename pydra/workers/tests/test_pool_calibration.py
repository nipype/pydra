"""Calibration, `MeasuredFit`, the cold re-check and ``place`` for ``plan()`` (RFC 0005 sections 3.3,
4.4, 7.4; W9).

The fit's contract (an oversized residency moves up, one that fits no class is unplaceable with its
numbers, an unfit record skips its class and every smaller one, nothing measured leaves the placement
alone, the one judging rule at its boundary, measured figures stay out of the placement and the
checksum) and a load out of memory at run time (kept as a record, so the next placement avoids the
class; one that fails on every class records each and the next placement is unplaceable). A client
that measures writes the same records (`DirectoryCalibration.record`).
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import attrs
import pytest

from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.submitter import Submitter
from pydra.workers.pool.calibration import (
    Code,
    DirectoryCalibration,
    MemoryFit,
    ParityRecord,
    Sample,
    TimeRecord,
    UnfitRecord,
    hardware_of,
    residency_key,
)
from pydra.workers.pool.classes import WorkerClass
from pydra.workers.pool.fit import MeasuredFit
from pydra.workers.pool.placement import (
    FitRequest,
    Placement,
    Unplaceable,
    fit_request,
    place,
)
from pydra.utils.resources import ComputeResources

POSIX = sys.platform != "win32"
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails
GB = 10**9
G = 1024**3
SMALL, LARGE = 1 * G, 8 * G
HEADROOM = 0.9  # a site's headroom; MeasuredFit has none of its own

TASKS = """
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.engine.resident import Provided
from pydra.utils.resources import ComputeResources

VERSION = {"sized": "1", "cuda": "1"}
FAIL = {"load": False}
LOADS = []
MEM = {"rss": 1000, "device": 50}
SIZES = {"a": 100, "b": 300}


class OutOfMemoryError(RuntimeError):
    # stands for torch's: the classifier knows it by its qualified name
    pass


OutOfMemoryError.__module__ = "torch.cuda"


class Recorder:
    # an in-process provider that records each load
    name = "recorder"

    def load(self, spec, device):
        LOADS.append("recorder")
        return spec

    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


class Sized(Recorder):
    # a provider whose load runs out of host memory while FAIL["load"] is set
    name = "sized"

    def load(self, spec, device):
        LOADS.append("sized")
        if FAIL["load"]:
            raise MemoryError("out of memory loading the model")
        return spec

    def version(self):
        return VERSION["sized"]


class Cuda(Recorder):
    # a GPU provider: FAIL["load"] "cuda" raises CUDA out of memory, "host" a MemoryError
    name = "cuda"

    def load(self, spec, device):
        LOADS.append("cuda")
        if FAIL["load"] == "cuda":
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if FAIL["load"] == "host":
            raise MemoryError("out of host memory")
        return spec

    def version(self):
        return VERSION["cuda"]


class Grows(Recorder):
    # a provider whose load grows the fake host and device memory by SIZES[spec]
    name = "grows"

    def load(self, spec, device):
        MEM["rss"] += SIZES[spec]
        MEM["device"] += 10 * SIZES[spec]
        return spec


class _Group:
    def members(self):  # live members only: a dead child is never measured
        return [CHILD]


class _Handle:
    # what a process-owning provider's load returns: its process group
    group = _Group()


CHILD = 4_000_000  # a pid no process has: its /proc reads are faked


class Child(Recorder):
    # a provider whose model lives in a child process
    name = "child"

    def load(self, spec, device):
        LOADS.append("child")
        return _Handle()


@python.define
def UseSized(model: Provided[object, "sized"], x: int) -> int:
    return x


@python.define
def UseChild(model: Provided[object, "child"], x: int) -> int:
    return x


@python.define
def UseCuda(model: Provided[object, "cuda"], x: int) -> int:
    return x


@python.define
def UseGrows(model: Provided[object, "grows"], x: int) -> int:
    return x


@workflow.define
def Chain(x: int) -> int:
    # three residencies, so no node's result is another's
    a = workflow.add(Same(model="a", x=x), name="A")
    b = workflow.add(Same(model="b", x=a.out), name="B")
    return workflow.add(Same(model="c", x=b.out), name="C").out


@workflow.define
def UseSizeds(xs: list) -> list:
    return workflow.add(UseSized(model="m").split(x=xs), name="U").out


@python.define
def Fresh(model: Provided[str, "recorder"], x: int) -> str:
    # a different output on every run: a re-check never matches
    return uuid4().hex


@python.define
def Same(model: Provided[str, "recorder"], x: int) -> int:
    return x


@workflow.define
def Freshes(xs: list) -> list:
    return workflow.add(Fresh(model="m").split(x=xs), name="F").out


@workflow.define
def Sames(xs: list) -> list:
    return workflow.add(Same(model="m").split(x=xs), name="S").out


@workflow.define
def Longs(xs: list) -> list:
    long = ComputeResources(time=3600)
    return workflow.add(Same(model="m").split(x=xs), name="S", resources=long).out
"""


# -- fixtures and helpers ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_calibration_tasks")
    (where / "_pydra_pool_calibration_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_calibration_tasks

    return _pydra_pool_calibration_tasks


@pytest.fixture(autouse=True)
def clean(monkeypatch, tasks):
    """Release every claim a test leaves held and reset the task module, pass or fail."""
    from pydra.workers.pool.storage import Claim

    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    from pydra.engine import resident

    monkeypatch.setitem(tasks.VERSION, "sized", "1")
    monkeypatch.setitem(tasks.VERSION, "cuda", "1")
    monkeypatch.setitem(tasks.FAIL, "load", False)
    monkeypatch.setitem(tasks.MEM, "rss", 1000)
    monkeypatch.setitem(tasks.MEM, "device", 50)
    device = resident.process_device()
    tasks.LOADS.clear()
    yield
    tasks.LOADS.clear()
    resident.set_process_device(device)
    resident.close_process_set()
    for claim in made:
        claim.lease.__exit__(None, None, None)


@pytest.fixture
def pools():
    """Every `WarmPool` a test makes is closed, every worker its substrate started is killed, and no
    process of theirs is left."""
    made = []
    yield made
    groups = []
    for pool in made:
        try:
            pool.close()
        finally:
            for handle in getattr(pool.substrate, "handles", []):
                if handle.process is not None:
                    groups.append(handle.process.pid)
                    _kill_group(handle.process)
    if POSIX:
        for pid in groups:
            found = subprocess.run(
                ["pgrep", "-g", str(pid)], capture_output=True, text=True
            )
            assert found.stdout.strip() == "", f"processes left in group {pid}"


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            if POSIX:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(BOUND)
    except subprocess.TimeoutExpired:
        pass


def _settings(**kwargs):
    from pydra.workers.pool.settings import PoolSettings

    fields = dict(lease=LEASE, poll_s=0.05, idle_polls=2, timeout_grace=60.0)
    fields.update(kwargs)
    return PoolSettings(**fields)


def _class(name="cpu", **kwargs):
    fields = dict(name=name, device="cpu", quantum_jobs=4, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


def ok(cls):
    return True


class Pool:
    """A pool, a store and the jobs put on the pool, for one test (as in ``test_pool_faults``)."""

    def __init__(self, tmp_path: Path, tasks):
        from pydra.workers.pool.storage import PoolRoot

        self.pool = PoolRoot.open(tmp_path / "pool", LEASE, probe=False)
        self.submitter = Submitter(cache_root=tmp_path / "store", worker="debug")
        self.submitter.providers = (tasks.Recorder(), tasks.Sized())
        self.store = self.submitter.store
        self.tickets: list[tuple[str, str]] = []

    @property
    def path(self) -> Path:
        return self.pool.path

    @property
    def calibration(self) -> DirectoryCalibration:
        return DirectoryCalibration(self.path / "calibration")

    def job(self, task, name: str, *, resources=None) -> Job:
        return Job(task=task, submitter=self.submitter, name=name, resources=resources)

    def put(self, jobs, queue, **kwargs) -> str:
        import attrs

        from pydra.engine import resident
        from pydra.workers.pool.codec import PrivatePickle, write_payload
        from pydra.workers.pool.storage import Ticket, TicketJob

        entries = []
        for job in jobs:
            fingerprint = self.store.identity.fingerprint(job.context)
            g = self.pool._next_generation((job.checksum, fingerprint))
            job.attempt = f"{self.pool.pool_id}.{g}"
            entries.append(
                TicketJob(
                    checksum=job.checksum,
                    fingerprint=fingerprint,
                    payload=write_payload(PrivatePickle(), job, self.path),
                    attempt=job.attempt,
                    runner_attempt=0,
                    units=1.0,
                )
            )
        residency = tuple(
            attrs.asdict(resident.spec_key(jobs[0].task, f, p))
            for f, p in resident.provided_fields(jobs[0].task)
        )
        ticket = Ticket(
            queue=queue, node="wf/node", jobs=entries, residency=residency, **kwargs
        )
        digest = self.pool.put(queue, ticket)
        self.tickets.append((queue, digest))
        return digest

    def newest(self, job: Job) -> tuple[str, dict | None]:
        """The queue of the job's newest generation's ticket and its outcome there."""
        key = (job.checksum, self.store.identity.fingerprint(job.context))
        _, record = self.pool.newest(*key)
        queue = self.pool._by_label(record["queue"]).name
        found = self.pool.read_outcome(queue, record["ticket"], key, record["attempt"])
        return queue, found

    def classes(self, *classes) -> None:
        from pydra.workers.pool.worker import class_record

        for cls in classes:
            self.pool.put_class(cls.name, class_record(cls))


@pytest.fixture
def pool(tmp_path, tasks):
    p = Pool(tmp_path, tasks)
    yield p
    p.pool.close()


def _run(pool: Pool, cls):
    from pydra.workers.pool.worker import Worker

    worker = Worker(pool.path, cls, settings=_settings(), device_memory=lambda: None)
    return worker, worker.run()


def _warm(tmp_path, pools, sub, classes=None, substrate=None, **kwargs):
    from pydra.workers.pool.pool import WarmPool
    from pydra.workers.pool.substrate import LocalSubstrate

    kwargs.setdefault("settings", _settings(idle_polls=40))
    pool = WarmPool(
        tmp_path / "pool",
        classes or [_class()],
        substrate if substrate is not None else LocalSubstrate(),
        store=sub.store,
        **kwargs,
    )
    pools.append(pool)
    return pool


def _drive(sub, task, executor):
    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner

    runner = GraphRunner(task, sub.store, on_error="raise_at_end", submitter=sub)
    asyncio.run(
        drive(
            runner,
            {"pool": executor},
            lambda job: "pool",
            on_error="raise_at_end",
        )
    )
    if runner.errors:
        raise AssertionError("\n".join(runner.errors))
    return runner


def _seed(cal, key, cls, *, resident, m0=0.0, m1=0.0, n=4):
    """Warm samples whose host fit is exactly ``resident``, ``m0`` and ``m1``."""
    for i in range(n):
        units = float(i + 1)
        cal.record(
            Sample(
                wall_s=0.5,
                claim_s=0.0,
                batch_size=1,
                units=units,
                baseline_bytes=None,
                resident_bytes=resident,
                peak_bytes=int(resident + m0 + m1 * units),
                device_baseline_bytes=None,
                device_bytes=None,
                load_s=0.0,
                cold=False,
                since_load=i + 1,
                reload=False,
                worker="seed",
                ended=0.0,
                key=key,
                hardware=hardware_of(cls),
            )
        )


CPU = _class("cpu", mem=1 * GB)
CPU_LARGE = _class("cpu-large", mem=8 * GB)


@pytest.fixture
def measured(tmp_path, tasks):
    """A job with resident state, its request and an empty calibration."""
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Recorder(), tasks.Sized())
    job = Job(task=tasks.Same(model="m", x=1), submitter=sub, name="j")
    return job, fit_request(job), DirectoryCalibration(tmp_path / "calibration")


# -- the calibration store ---------------------------------------------------------------------


def test_residency_key_is_the_residency_digest(measured):
    """The calibration key from a ticket's residency (as `attrs.asdict`) is the job's residency
    digest, the one a routed queue is named by."""
    import attrs

    from pydra.engine import resident

    job, request, _ = measured
    digest = resident.residency_digest(job)
    assert request.key == digest
    assert residency_key([attrs.asdict(k) for k in request.residency]) == digest
    assert residency_key(()) is None
    got = place(
        job, [CPU], MeasuredFit(DirectoryCalibration("unused"), headroom=HEADROOM), ok
    )
    assert got.queue == f"cpu#{digest}"


def test_directory_calibration_round_trip(tmp_path):
    """Records persist for a later instance; the memory fit is Theil–Sen (an outlier moves nothing);
    a last line still being written is skipped; fewer than ``min_samples`` samples are unmeasured.
    """
    cal = DirectoryCalibration(tmp_path / "cal")
    cls = _class("cpu", mem=8 * GB)
    _seed(cal, "k", cls, resident=1000, m0=100.0, m1=10.0, n=4)
    # an outlier: Theil-Sen keeps the line
    cal.record(
        attrs_evolve_sample(
            resident=1000, peak=1000 + 10_000, units=2.5, key="k", cls=cls
        )
    )
    unfit = UnfitRecord("k", "host", 8 * GB, "cpu", Code((("p", "1"),)))
    parity = ParityRecord("k", "c" * 64, (4, 1), False, "a", "b", "differs")
    cal.record(unfit)
    cal.record(parity)
    samples = cal.root / "k" / "cpu" / "samples" / "seed.jsonl"
    with samples.open("a") as f:
        f.write('{"wall_s": 1.0, "trunc')
    later = DirectoryCalibration(tmp_path / "cal")
    fit = later.memory("k", hardware_of(cls))
    assert (fit.resident, round(fit.m0, 6), round(fit.m1, 6), fit.n) == (
        1000,
        100.0,
        10.0,
        5,
    )
    assert later.unfit("k") == [unfit] and later.parity("k") == [parity]
    assert later.times("k", hardware_of(cls)) == [TimeRecord(1, 0.5, 5)]
    assert (
        DirectoryCalibration(tmp_path / "cal", min_samples=6).memory(
            "k", hardware_of(cls)
        )
        is None
    )
    assert later.memory("other", hardware_of(cls)) is None
    gpu = WorkerClass("h", "cuda", gpus=1, gpu_type="h100", gpu_mem=80 * G)
    assert later.memory("k", hardware_of(gpu)) is None  # another hardware class
    with pytest.raises(ValueError, match="key and hardware"):
        cal.record(attrs_evolve_sample(resident=1, peak=1, units=1, key=None, cls=cls))


def test_memory_fit_takes_warm_lone_measured_samples(tmp_path):
    """The resident bytes are the least seen; cold, reloaded and shared-worker samples, and samples
    without a measured peak, are not fitted; a negative slope is no fit; samples with fewer than
    two distinct units give an unknown ``m1``, not 0; a record that is not this version's is
    skipped, not fatal."""
    import attrs

    cls = _class("cpu", mem=8 * GB)
    hw = hardware_of(cls)
    cal = DirectoryCalibration(tmp_path / "cal")
    for resident, units in ((1000, 1.0), (1200, 2.0), (1100, 3.0)):
        cal.record(
            attrs_evolve_sample(
                resident=resident,
                peak=resident + 10 * int(units),
                units=units,
                key="k",
                cls=cls,
            )
        )
    base = attrs_evolve_sample(resident=10, peak=10**9, units=4.0, key="k", cls=cls)
    for skipped in (
        attrs.evolve(base, cold=True),
        attrs.evolve(base, reload=True),
        attrs.evolve(base, others=1),
        attrs.evolve(base, peak_bytes=None),
    ):
        cal.record(skipped)
    fit = cal.memory("k", hw)
    assert (fit.resident, round(fit.m0, 6), round(fit.m1, 6), fit.n) == (
        1000,
        0.0,
        10.0,
        3,
    )
    falling = DirectoryCalibration(tmp_path / "falling")
    for units in (1.0, 2.0, 3.0):
        falling.record(
            attrs_evolve_sample(
                resident=1000,
                peak=1000 + 100 - 10 * int(units),
                units=units,
                key="k",
                cls=cls,
            )
        )
    assert falling.memory("k", hw) is None
    flat = DirectoryCalibration(tmp_path / "flat")
    for peak in (1100, 1200, 1300):
        flat.record(
            attrs_evolve_sample(resident=1000, peak=peak, units=None, key="k", cls=cls)
        )
    assert flat.memory("k", hw) == MemoryFit(resident=1000, m0=200.0, m1=None, n=3)
    (flat.root / "k" / "unfit").mkdir()
    (flat.root / "k" / "unfit" / "1-x.json").write_text('{"key": "k", "resource": 3}')
    assert flat.unfit("k") == []


def test_no_device_peak_is_no_device_fit(measured, tmp_path):
    """A GPU worker's samples without a measured device peak give no device fit: a job declaring 30G is not placed on a 24G L4 because ten gigabytes
    were seen resident there; with measured peaks the fit judges."""
    import attrs

    job, request, _ = measured
    big = attrs.evolve(request, resources=ComputeResources(gpus=1, gpu_mem="30G"))
    l4 = WorkerClass("l4", "cuda", gpus=1, gpu_type="l4", gpu_mem=24 * G)
    a100 = WorkerClass("a100", "cuda", gpus=1, gpu_type="a100", gpu_mem=40 * G)
    cal = DirectoryCalibration(tmp_path / "cal")
    for i in range(4):
        sample = attrs_evolve_sample(
            resident=None, peak=None, units=1.0 + i, key=request.key, cls=l4
        )
        cal.record(attrs.evolve(sample, device_bytes=10 * G, device_baseline_bytes=0))
    assert cal.memory(request.key, hardware_of(l4)) is None
    got = place(job, [l4, a100], MeasuredFit(cal, headroom=HEADROOM), ok, request=big)
    assert got.worker_class == "a100"
    assert [(t.worker_class, t.verdict) for t in got.tried] == [
        ("l4", "too small"),
        ("a100", "fits"),
    ]
    assert MeasuredFit(cal, headroom=HEADROOM).batch(big, l4, [None]) == 1
    peaked = DirectoryCalibration(tmp_path / "peaked")
    for i in range(4):
        sample = attrs_evolve_sample(
            resident=None, peak=None, units=1.0 + i, key=request.key, cls=l4
        )
        peaked.record(
            attrs.evolve(
                sample,
                device_bytes=10 * G,
                device_baseline_bytes=0,
                device_peak_bytes=10 * G + G * (i + 1),
            )
        )
    fit = peaked.memory(request.key, hardware_of(l4))
    assert (fit.resident, round(fit.m0), round(fit.m1)) == (10 * G, 0, G)
    assert (
        place(
            job, [l4, a100], MeasuredFit(peaked, headroom=HEADROOM), ok, request=big
        ).worker_class
        == "l4"
    )


def test_batch_through_place_with_unknown_length_is_one(measured, tmp_path):
    """Through `place`, whose request carries no units: a measured per-unit memory gives a batch of
    1, an unknown one too, and only a per-unit memory measured as 0 over real units gives the
    ceiling."""
    job, request, _ = measured
    cls = _class("cpu", mem=8 * GB, quantum_jobs=64)
    sloped = DirectoryCalibration(tmp_path / "sloped")
    _seed(sloped, request.key, cls, resident=1 * GB, m1=1.0 * GB)
    assert sloped.memory(request.key, hardware_of(cls)).m1 == pytest.approx(GB)
    assert place(job, [cls], MeasuredFit(sloped, headroom=HEADROOM), ok).batch == 1
    flat = DirectoryCalibration(tmp_path / "flat")
    _seed(flat, request.key, cls, resident=1 * GB, m1=0.0)
    assert place(job, [cls], MeasuredFit(flat, headroom=HEADROOM), ok).batch == 64
    unknown = DirectoryCalibration(tmp_path / "unknown")
    for _ in range(3):
        unknown.record(
            attrs_evolve_sample(
                resident=GB, peak=GB, units=None, key=request.key, cls=cls
            )
        )
    assert unknown.memory(request.key, hardware_of(cls)).m1 is None
    assert place(job, [cls], MeasuredFit(unknown, headroom=HEADROOM), ok).batch == 1


def attrs_evolve_sample(*, resident, peak, units, key, cls):
    return Sample(
        wall_s=0.5,
        claim_s=0.0,
        batch_size=1,
        units=units,
        baseline_bytes=None,
        resident_bytes=resident,
        peak_bytes=peak,
        device_baseline_bytes=None,
        device_bytes=None,
        load_s=0.0,
        cold=False,
        since_load=3,
        reload=False,
        worker="seed",
        ended=0.0,
        key=key,
        hardware=hardware_of(cls),
    )


def test_times_and_memory_are_read_per_task(tmp_path):
    """Two tasks over one residency: each reads only its own samples"""
    from pydra.workers.pool.classes import WorkerClass

    cls = WorkerClass(name="cpu", device="cpu")
    calibration = DirectoryCalibration(tmp_path / "cal")
    for task, wall, peak in (("task-a", 1.0, 2 * GB), ("task-b", 9.0, 8 * GB)):
        for _ in range(3):
            calibration.record(
                attrs.evolve(
                    attrs_evolve_sample(
                        resident=GB, peak=peak, units=None, key="k", cls=cls
                    ),
                    wall_s=wall,
                    task=task,
                )
            )
    hardware = hardware_of(cls)
    (a,) = calibration.times("k", hardware, task="task-a")
    (b,) = calibration.times("k", hardware, task="task-b")
    assert (a.per_job_s, b.per_job_s) == (1.0, 9.0)
    assert calibration.memory("k", hardware, task="task-a").m0 == GB
    assert calibration.memory("k", hardware, task="task-b").m0 == 7 * GB


def test_unfit_record_binds_only_its_code():
    """A record binds the code that wrote it: a provider's version or the environment changed, or a
    code that cannot be read, binds nothing."""
    record = UnfitRecord("k", "device", 40 * G, "a100", Code((("m", "1"),)))
    assert record.binds(Code((("m", "1"),)))
    assert not record.binds(Code((("m", "2"),)))
    assert not record.binds(Code((("m", "1"),), environment="docker:abc"))
    assert not record.binds(None)
    assert UnfitRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_an_unfit_record_of_unknown_grant_binds_its_own_class_only(measured):
    """A record whose grant is unknown holds for the class it was written on, never another; a
    host record holds on a GPU class too, against its host memory."""
    _, request, _ = measured
    code = request.code
    record = UnfitRecord(request.key, "host", None, "cpu", code)

    class Fixed:
        records = [record]

        def memory(self, key, hardware, task=None):
            return None

        def unfit(self, key):
            return self.records

        def times(self, key, hardware, task=None):
            return ()

    fit = MeasuredFit(Fixed(), headroom=HEADROOM)
    assert fit.judge(request, _class("cpu")).verdict == "unfit"
    assert fit.judge(request, _class("other")).verdict != "unfit"
    gpu = WorkerClass("l4", "cuda", gpus=1, gpu_mem=24 * G, mem=32 * G)
    Fixed.records = [UnfitRecord(request.key, "host", 64 * G, "elsewhere", code)]
    assert fit.judge(request, gpu).verdict == "unfit"
    Fixed.records = [UnfitRecord(request.key, "host", 32 * G, "elsewhere", code)]
    assert fit.judge(request, gpu).verdict == "unfit"  # a grant as large binds
    Fixed.records = [UnfitRecord(request.key, "host", 16 * G, "elsewhere", code)]
    assert fit.judge(request, gpu).verdict != "unfit"
    anon = WorkerClass("anon", "cuda", gpus=1)
    Fixed.records = [UnfitRecord(request.key, "device", 24 * G, "anon", code)]
    assert fit.judge(request, anon).verdict == "unfit"  # before "unknown"


# -- MeasuredFit ------------------------------------------


def test_an_oversized_residency_is_placed_on_the_larger_class(measured):
    """Resident plus m0 over the headroom of a class's grant skips it; the next larger class that
    fits is taken, and the class tried names the numbers."""
    job, request, cal = measured
    assert (
        place(
            job, [CPU, CPU_LARGE], MeasuredFit(cal, headroom=HEADROOM), ok
        ).worker_class
        == "cpu"
    )
    _seed(cal, request.key, CPU, resident=950_000_000, m0=1000.0)
    got = place(job, [CPU_LARGE, CPU], MeasuredFit(cal, headroom=HEADROOM), ok)
    assert isinstance(got, Placement) and got.worker_class == "cpu-large"
    assert [(t.worker_class, t.verdict) for t in got.tried] == [
        ("cpu", "too small"),
        ("cpu-large", "fits"),
    ]
    assert "resident 950000000 + m0 1000" in got.tried[0].reason
    assert "900000000" in got.tried[0].reason
    small = DirectoryCalibration(cal.root.parent / "small")
    _seed(small, request.key, CPU, resident=100_000_000)
    assert (
        place(
            job, [CPU, CPU_LARGE], MeasuredFit(small, headroom=HEADROOM), ok
        ).worker_class
        == "cpu"
    )


def test_a_residency_that_fits_no_class_is_unplaceable_and_names_the_numbers(measured):
    """With no larger class the placement stops; the message names each class tried and the
    figures."""
    job, request, cal = measured
    _seed(cal, request.key, CPU, resident=950_000_000, m0=1000.0)
    got = place(job, [CPU], MeasuredFit(cal, headroom=HEADROOM), ok)
    assert isinstance(got, Unplaceable)
    [tried] = got.tried
    assert (tried.worker_class, tried.verdict) == ("cpu", "too small")
    assert "950000000" in str(got) and "900000000" in str(got)


def test_an_unfit_record_skips_the_class_it_names_and_every_smaller_one(measured):
    """A load that ran out of memory in a grant makes that grant and any below it unfit, whatever
    was measured; a record of the other resource does not."""
    job, request, cal = measured
    _seed(cal, request.key, CPU, resident=10_000_000)
    cal.record(UnfitRecord(request.key, "host", 1 * GB, "cpu", request.code))
    got = place(job, [CPU, CPU_LARGE], MeasuredFit(cal, headroom=HEADROOM), ok)
    assert got.worker_class == "cpu-large"
    assert got.tried[0].verdict == "unfit"
    assert "ran out of host memory in a grant of 1000000000" in got.tried[0].reason
    both = DirectoryCalibration(cal.root.parent / "both")
    both.record(UnfitRecord(request.key, "host", 8 * GB, "cpu-large", request.code))
    got = place(job, [CPU, CPU_LARGE], MeasuredFit(both, headroom=HEADROOM), ok)
    assert isinstance(got, Unplaceable)
    assert [t.verdict for t in got.tried] == ["unfit", "unfit"]
    other = DirectoryCalibration(cal.root.parent / "other")
    other.record(UnfitRecord(request.key, "device", 8 * GB, "gpu", request.code))
    assert (
        place(
            job, [CPU, CPU_LARGE], MeasuredFit(other, headroom=HEADROOM), ok
        ).worker_class
        == "cpu"
    )


def test_a_residency_with_no_measure_and_no_record_stays_where_it_is(measured):
    """Nothing to judge by: the placement is `DeclaredFit`'s."""
    from pydra.workers.pool.placement import DeclaredFit

    job, _, cal = measured
    classes = [CPU, CPU_LARGE]
    assert place(job, classes, MeasuredFit(cal, headroom=HEADROOM), ok) == place(
        job, classes, DeclaredFit(), ok
    )


def test_judge_is_the_one_rule(measured):
    """At the boundary resident + m0 equal to the limit fits; above it is too small; a GPU class of
    unknown memory is unknown; a time over the class's is unfit."""
    _, request, _ = measured

    class Fixed:
        def memory(self, key, hardware, task=None):
            return MemoryFit(resident=90, m0=0.0, m1=1.0, n=3)

        def unfit(self, key):
            return ()

        def times(self, key, hardware, task=None):
            return ()

    fit = MeasuredFit(Fixed(), headroom=HEADROOM)
    assert fit.judge(request, _class("a", mem=100)).verdict == "fits"
    assert fit.judge(request, _class("b", mem=99)).verdict == "too small"
    unknown = WorkerClass("g", "cuda", gpus=1)
    assert fit.judge(request, unknown).verdict == "unknown"
    import attrs

    slow = attrs.evolve(request, resources=ComputeResources(time=7200))
    assert fit.judge(slow, _class("c", mem=100, time=3600)).verdict == "unfit"


def test_measured_figures_stay_out_of_the_placement_and_the_checksum(
    measured, tmp_path
):
    """Two fits whose resident memory differs by a little place alike, and the job's checksum is
    the one it had before any placement."""
    job, request, _ = measured
    before = job.checksum
    placements = []
    for resident in (950_000_000, 960_000_000):
        cal = DirectoryCalibration(tmp_path / f"cal-{resident}")
        _seed(cal, request.key, CPU, resident=resident)
        got = place(job, [CPU, CPU_LARGE], MeasuredFit(cal, headroom=HEADROOM), ok)
        placements.append((got.worker_class, got.queue, got.batch))
    assert placements[0] == placements[1] and placements[0][0] == "cpu-large"
    assert job.checksum == before


def test_measured_fit_batch_rule(measured):
    """``max(1, floor((headroom x share - resident - m0) / (m1 x length)))`` on fixed figures, with
    ``share`` the device memory over ``workers_per_device`` and ``length`` the longest units; then
    the fastest recorded batch at or below that bound."""
    _, request, _ = measured
    figures = {"fit": MemoryFit(resident=10 * G, m0=2 * G, m1=0.5 * G, n=6)}
    figures["times"] = ()

    class Fixed:
        def memory(self, key, hardware, task=None):
            return figures["fit"]

        def unfit(self, key):
            return ()

        def times(self, key, hardware, task=None):
            return figures["times"]

    h100 = WorkerClass(
        "h100",
        "cuda",
        gpus=1,
        gpu_type="h100",
        gpu_mem=80 * G,
        workers_per_device=2,
        quantum_jobs=64,
    )
    fit = MeasuredFit(Fixed(), headroom=HEADROOM)
    # share 40G; 0.9 x 40G - 10G - 2G = 24G; m1 x length = 0.5G x 4 = 2G; floor(12)
    assert fit.batch(request, h100, [1.0, 4.0, None]) == 12
    figures["times"] = (
        TimeRecord(1, 0.5, 3),
        TimeRecord(8, 0.1, 3),
        TimeRecord(12, 0.2, 3),
        TimeRecord(16, 0.01, 3),  # above the bound: not taken
    )
    assert fit.batch(request, h100, [1.0, 4.0]) == 8
    figures["times"] = (TimeRecord(4, 0.1, 3),)  # one batch size: the bound stands
    assert fit.batch(request, h100, [4.0]) == 12
    figures["times"] = ()
    figures["fit"] = MemoryFit(resident=40 * G, m0=0.0, m1=0.5 * G, n=6)
    assert fit.batch(request, h100, [4.0]) == 1  # no room
    figures["fit"] = MemoryFit(resident=1 * G, m0=0.0, m1=0.0, n=6)
    assert fit.batch(request, h100, [4.0]) == 64  # no per-job memory: a ticket's worth
    figures["fit"] = MemoryFit(resident=1 * G, m0=0.0, m1=1.0, n=6)
    assert (
        fit.batch(request, h100, [4.0]) == 64
    )  # room for millions: capped at a ticket
    figures["fit"] = MemoryFit(resident=10 * G, m0=2 * G, m1=0.5 * G, n=6)
    assert (
        fit.batch(request, h100, [None]) == 1
    )  # per-unit memory measured, length unknown
    figures["fit"] = MemoryFit(resident=10 * G, m0=2 * G, m1=None, n=6)
    assert fit.batch(request, h100, [4.0]) == 1  # per-unit memory unknown
    figures["fit"] = None
    assert fit.batch(request, h100, [4.0]) == 1  # unmeasured


# -- a load out of memory at run time ---------------


def test_a_runtime_load_oom_records_unfit_and_the_replan_avoids_the_class(pool, tasks):
    """A load runs out of memory on the small class: the jobs move to the large one and run there,
    one `UnfitRecord` is kept, and the next placement starts on the large class."""
    small, large = _class("small", mem=SMALL), _class("large", mem=LARGE)
    pool.classes(small, large)
    jobs = [pool.job(tasks.UseSized(model="m", x=i), f"j{i}") for i in range(2)]
    key = fit_request(jobs[0]).key
    pool.put(jobs, "small", ladder=("large",))
    tasks.FAIL["load"] = True
    _, code = _run(pool, small)
    assert code == 0
    [record] = pool.calibration.unfit(key)
    assert (record.resource, record.grant, record.worker_class) == (
        "host",
        SMALL,
        "small",
    )
    assert record.code == Code((("sized", "1"),))
    tasks.FAIL["load"] = False
    _, code = _run(pool, large)
    assert code == 0
    assert [pool.newest(j)[0] for j in jobs] == ["large", "large"]
    assert [pool.newest(j)[1]["outcome"] for j in jobs] == ["done", "done"]
    third = pool.job(tasks.UseSized(model="m", x=3), "j3")
    got = place(
        third, [small, large], MeasuredFit(pool.calibration, headroom=HEADROOM), ok
    )
    assert got.worker_class == "large" and got.tried[0].verdict == "unfit"
    assert len(pool.calibration.unfit(key)) == 1


def test_a_load_that_fails_on_every_class_records_each_and_the_replan_is_unplaceable(
    pool, tasks
):
    """The moved jobs' load fails on the large class too: it records its own fact, the jobs fail
    ``Unplaceable``, and no class is left for the next placement."""
    small, large = _class("small", mem=SMALL), _class("large", mem=LARGE)
    pool.classes(small, large)
    job = pool.job(tasks.UseSized(model="m", x=1), "j")
    key = fit_request(job).key
    pool.put([job], "small", ladder=("large",))
    tasks.FAIL["load"] = True
    _run(pool, small)
    _run(pool, large)
    queue, failed = pool.newest(job)
    assert queue == "large"
    assert (failed["outcome"], failed["fault"]["type"]) == ("failed", "Unplaceable")
    assert sorted((r.resource, r.grant) for r in pool.calibration.unfit(key)) == [
        ("host", SMALL),
        ("host", LARGE),
    ]
    got = place(
        job, [small, large], MeasuredFit(pool.calibration, headroom=HEADROOM), ok
    )
    assert isinstance(got, Unplaceable)
    assert [t.verdict for t in got.tried] == ["unfit", "unfit"]


def test_unfit_record_unbinds_after_provider_version_change(pool, tasks):
    """An out-of-memory verdict persists across calibration instances while the provider's
    ``version()`` is the one the load ran under, and stops binding once it changes: the next
    placement goes back to the small class."""
    small, large = _class("small", mem=SMALL), _class("large", mem=LARGE)
    pool.classes(small, large)
    job = pool.job(tasks.UseSized(model="m", x=1), "j")
    pool.put([job], "small", ladder=("large",))
    tasks.FAIL["load"] = True
    _run(pool, small)
    again = pool.job(tasks.UseSized(model="m", x=2), "again")
    got = place(
        again, [small, large], MeasuredFit(pool.calibration, headroom=HEADROOM), ok
    )
    assert got.worker_class == "large"
    assert [(t.worker_class, t.verdict) for t in got.tried] == [
        ("small", "unfit"),
        ("large", "unknown"),
    ]
    tasks.VERSION["sized"] = "2"
    later = pool.job(tasks.UseSized(model="m", x=2), "later")
    got = place(
        later, [small, large], MeasuredFit(pool.calibration, headroom=HEADROOM), ok
    )
    assert got.worker_class == "small"
    [record] = pool.calibration.unfit(fit_request(later).key)  # kept, no longer binding
    assert record.code == Code((("sized", "1"),))
    assert fit_request(later).code == Code((("sized", "2"),))


def test_unplaceable_by_measure_fails_at_submit_and_writes_nothing(
    tmp_path, tasks, pools
):
    """A residency measured too large for every class is ``Failed(UnplaceableJob)`` at submit with
    its numbers; no ticket is written and no worker started."""
    from pydra.engine.runner import Failed
    from pydra.workers.pool.pool import UnplaceableJob

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Recorder(),)
    job = Job(task=tasks.Same(model="m", x=1), submitter=sub, name="j")
    cal = DirectoryCalibration(tmp_path / "pool" / "calibration")
    resident = 9 * GB
    _seed(cal, fit_request(job).key, CPU_LARGE, resident=resident)
    warm = _warm(
        tmp_path, pools, sub, [CPU, CPU_LARGE], fit=MeasuredFit(cal, headroom=HEADROOM)
    )
    reports = []
    warm.bind(lambda j, outcome: reports.append(outcome))
    warm.submit([job])
    [outcome] = reports
    assert isinstance(outcome, Failed) and isinstance(outcome.error, UnplaceableJob)
    assert f"resident {resident}" in str(outcome.error)
    assert list((tmp_path / "pool").rglob("tickets/*.json")) == []
    assert warm.substrate.handles == []


# -- plan() shows the classes tried (through RFC 0002 R8's place hook) --------------------------


def test_plan_shows_classes_tried(tmp_path, tasks, pools):
    """``GraphRunner.plan()`` asks the `WarmPool` where each node's elements would run: the
    placement carries every class tried with its verdict and numbers, and a recorded unfit load
    moves the next plan to the larger class. Nothing is written or started."""
    from pydra.engine.runner import GraphRunner

    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Recorder(),)
    small = _class("small", mem="1G", time=60)
    large = _class("large", mem="4G", time=7200)
    warm = _warm(tmp_path, pools, sub, [small, large])
    runner = GraphRunner(
        tasks.Longs(xs=[1, 2]), sub.store, submitter=sub, on_error="raise_at_end"
    )
    [node] = runner.plan(executors={"default": warm}).nodes
    assert node.path == "S" and len(node.placement) == 2
    for placement in node.placement:
        assert placement.worker_class == "large"
        assert [(t.worker_class, t.verdict) for t in placement.tried] == [
            ("small", "unfit"),
            ("large", "fits"),
        ]
        assert "time 3600s exceeds the class limit of 60s" in placement.tried[0].reason
        assert placement.queue.startswith("large#")
    assert warm.substrate.handles == []
    assert list((tmp_path / "pool").rglob("tickets/*.json")) == []

    measured = DirectoryCalibration(tmp_path / "measured")
    job = Job(task=tasks.Same(model="m", x=1), submitter=sub, name="probe")
    request = fit_request(job)
    measured.record(UnfitRecord(request.key, "host", 1 * G, "small", request.code))
    plain = _class("small", mem="1G"), _class("large", mem="4G")
    with pytest.warns(UserWarning, match="no out-of-memory verdict"):
        warm2 = _warm(
            tmp_path / "second",
            pools,
            sub,
            list(plain),
            fit=MeasuredFit(measured, headroom=HEADROOM),
        )
    runner = GraphRunner(
        tasks.Sames(xs=[1]), sub.store, submitter=sub, on_error="raise_at_end"
    )
    [node] = runner.plan(executors={"default": warm2}).nodes
    [placement] = node.placement
    assert placement.worker_class == "large"
    assert [(t.worker_class, t.verdict) for t in placement.tried] == [
        ("small", "unfit"),
        ("large", "unknown"),
    ]


# -- the sampled cold re-check -----------------------------------------------------------------


def test_recheck_sampling_is_deterministic():
    """A job is sampled by its checksum against ``recheck_fraction``: 0 never, 1 always, and the
    same answer every time."""
    from pydra.workers.pool.pool import _sampled

    checksums = [f"python-{i:064x}" for i in range(200)]
    assert not any(_sampled(c, 0.0) for c in checksums)
    assert all(_sampled(c, 1.0) for c in checksums)
    half = [_sampled(c, 0.5) for c in checksums]
    assert half == [_sampled(c, 0.5) for c in checksums] and 0 < sum(half) < 200
    from pydra.workers.pool.settings import PoolSettings

    with pytest.raises(ValueError):
        PoolSettings(recheck_fraction=1.5)


def _recheck_run(tmp_path, pools, tasks, workflow, fraction, model="recorder"):
    from pydra.utils.messenger import AuditFlag, FileMessenger

    messages = tmp_path / "messages"
    sub = Submitter(
        cache_root=tmp_path / "store",
        worker="debug",
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
        messenger_args={"message_dir": str(messages)},
    )
    sub.providers = (tasks.Recorder(), tasks.Sized())
    warm = _warm(
        tmp_path,
        pools,
        sub,
        settings=_settings(idle_polls=40, recheck_fraction=fraction),
    )
    runner = _drive(sub, workflow(xs=[1, 2]), warm)
    warm.close()
    task = tasks.Same if model == "recorder" else tasks.UseSized
    job = Job(task=task(model="m", x=1), submitter=sub, name="k")
    key = fit_request(job).key
    parity = DirectoryCalibration(tmp_path / "pool" / "calibration").parity(key)
    audits = []
    for path in messages.glob("*.jsonld") if messages.exists() else []:
        message = json.loads(path.read_text())
        if message.get("@type") == "pool-recheck":
            audits.append(message)
    return runner, parity, audits, sub


def test_recheck_mismatch_writes_parity_and_one_audit(tmp_path, tasks, pools):
    """With ``recheck_fraction=1`` each settled job runs again alone on a fresh residency, through
    the store's verify path: a task whose output differs on every run writes a `ParityRecord` per job
    with both digests, and one audit message for the residency; the committed results stand.
    """
    runner, parity, audits, sub = _recheck_run(
        tmp_path, pools, tasks, tasks.Freshes, 1.0
    )
    assert runner.done and not runner.failed
    assert all(len(sub.store.stored(r.checksum)) == 1 for r in parity)
    assert len(parity) == 2 and not any(r.equal for r in parity)
    assert all(r.batch_sizes == (1, 1) for r in parity)
    assert all(r.committed and r.fresh and r.committed != r.fresh for r in parity)
    assert len(audits) == 1
    assert audits[0]["residency"] == parity[0].key
    assert audits[0]["checksum"] in {r.checksum for r in parity}


def test_recheck_equal_writes_parity_and_no_audit(tmp_path, tasks, pools):
    """A deterministic task's re-check writes equal `ParityRecord`s and no audit message; with the
    default fraction of 0 nothing is re-checked."""
    _, parity, audits, _ = _recheck_run(tmp_path, pools, tasks, tasks.Sames, 1.0)
    assert len(parity) == 2 and all(r.equal and r.committed == r.fresh for r in parity)
    assert audits == []
    _, none, _, _ = _recheck_run(tmp_path / "off", pools, tasks, tasks.Sames, 0.0)
    assert none == []


def test_recheck_goes_through_the_verify_path(tmp_path, tasks, pools, monkeypatch):
    """The re-check is a run marked to verify against the committed entry: the store hands the
    entry and the fresh digest to its reuse hook's ``compare``, whose refusal makes the re-check
    unequal with the hook's reason. (Patched here, in the coordinator's process only: the workers'
    own runs verify nothing.)"""
    from pydra.engine.portable.reuse import PolicyReuse

    seen = []

    def compare(self, candidate, stored, fresh, fresh_digest, task):
        seen.append((candidate.admitted_by, stored.digest, fresh_digest))
        raise ValueError("the hook refuses")

    monkeypatch.setattr(PolicyReuse, "compare", compare)
    _, parity, audits, sub = _recheck_run(tmp_path, pools, tasks, tasks.Sames, 1.0)
    assert len(seen) == 2
    assert all(by == "recheck" and stored == fresh for by, stored, fresh in seen)
    assert len(parity) == 2 and not any(r.equal for r in parity)
    assert all("the hook refuses" in r.reason for r in parity)
    assert len(audits) == 1
    assert all(len(sub.store.stored(r.checksum)) == 1 for r in parity)


def test_recheck_of_a_looked_up_job_still_verifies(tmp_path, tasks, pools, monkeypatch):
    """A job marked ``looked_up`` by the runner's lookup before dispatch is re-checked as a
    fresh attempt: the copy runs under the store's ``_reusable(lease, looked_up)`` call without
    being served, and is verified against the committed entry.
    """
    from pydra.engine.portable.reuse import PolicyReuse
    from pydra.workers.pool.codec import PrivatePickle

    seen = []

    def compare(self, candidate, stored, fresh, fresh_digest, task):
        seen.append(candidate.admitted_by)
        raise ValueError("the hook refuses")

    decode = PrivatePickle.decode

    def decode_looked_up(self, data, store):
        job = decode(self, data, store)
        job.looked_up = True  # as if this process had looked it up before dispatch
        return job

    monkeypatch.setattr(PolicyReuse, "compare", compare)
    monkeypatch.setattr(PrivatePickle, "decode", decode_looked_up)
    _, parity, _, _ = _recheck_run(tmp_path, pools, tasks, tasks.Sames, 1.0)
    assert seen == ["recheck", "recheck"]
    assert len(parity) == 2 and all("the hook refuses" in r.reason for r in parity)


# -- GPU verdicts across pools, attribution, the re-check thread, homes --------------


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    import time

    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _gpu(name, product, memory, **kwargs):
    from pydra.workers.pool.samples import Gpu, HardwareClass

    fields = dict(name=name, device="cuda", gpus=1, quantum_jobs=4, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields), HardwareClass("cuda", (Gpu(product, memory),), 4)


def _hand_run(tmp_path, pools, sub, classes, job, runs):
    """A `WarmPool` on an `ExternalSubstrate`: ``job`` submitted, then for each ``(class, hardware,
    failure)`` of ``runs`` a worker of that class started by hand in this process on that hardware
    once the job is queued there, the provider failing as ``failure`` says. Returns the outcome.
    """
    from pydra.workers.pool.substrate import ExternalSubstrate
    from pydra.workers.pool.worker import Worker

    warm = _warm(tmp_path, pools, sub, classes, substrate=ExternalSubstrate())
    reports = []
    warm.bind(lambda j, outcome: reports.append(outcome))
    warm.submit([job])
    key = (job.checksum, sub.store.identity.fingerprint(job.context))
    for cls, hardware, failure in runs:

        def queued(name=cls.name):
            newest = warm.pool.newest(*key)
            if newest is None:
                return False
            q = warm.pool._by_label(newest[1].get("queue", ""))
            return q is not None and q.name.split("#")[0] == name

        _wait(queued, f"the job queued on {cls.name}")
        job.submitter  # the provider reads FAIL in this process
        tasks_module = sys.modules["_pydra_pool_calibration_tasks"]
        tasks_module.FAIL["load"] = failure
        worker = Worker(
            warm.root,
            cls,
            settings=_settings(),
            device_memory=lambda: None,
            hardware=lambda h=hardware: h,
        )
        assert worker.run() == 0
    _wait(lambda: reports, "the job's outcome")
    warm.close()
    return reports[0]


def _gpu_sub(tmp_path, tasks):
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Cuda(),)
    return sub


def test_gpu_load_oom_persists_across_warm_pools(tmp_path, tasks, pools):
    """A "CUDA out of memory" load on an L4 shared by two workers is a device record of the
    worker's share (12G), the job runs on the A100, and the next pool's placement starts there;
    a new provider version unbinds it."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    l4, l4_hw = _gpu(
        "l4", "NVIDIA L4", 24 * G, gpu_type="l4", gpu_mem=24 * G, workers_per_device=2
    )
    a100, a100_hw = _gpu(
        "a100", "NVIDIA A100-SXM4-40GB", 40 * G, gpu_type="a100", gpu_mem=40 * G
    )
    sub = _gpu_sub(tmp_path, tasks)
    job = Job(
        task=tasks.UseCuda(model="m", x=1),
        submitter=sub,
        name="j",
        resources=ComputeResources(gpus=1),
    )
    outcome = _hand_run(
        tmp_path,
        pools,
        sub,
        [l4, a100],
        job,
        [(l4, l4_hw, "cuda"), (a100, a100_hw, False)],
    )
    assert not getattr(outcome, "errored", True)
    key = fit_request(job).key
    [record] = DirectoryCalibration(tmp_path / "pool" / "calibration").unfit(key)
    assert (record.resource, record.grant, record.worker_class) == (
        "device",
        12 * G,
        "l4",
    )
    measured = MeasuredFit(
        DirectoryCalibration(tmp_path / "pool" / "calibration"), headroom=HEADROOM
    )
    second = _warm(
        tmp_path, pools, sub, [l4, a100], substrate=ExternalSubstrate(), fit=measured
    )
    [got] = second.place([job])
    assert got.worker_class == "a100"
    assert [(t.worker_class, t.verdict) for t in got.tried] == [
        ("l4", "unfit"),
        ("a100", "unknown"),
    ]
    tasks.VERSION["cuda"] = "2"
    [got] = second.place([job])
    assert got.worker_class == "l4"


@pytest.mark.parametrize(
    "case", ["unknown device memory", "host out of memory on a GPU class"]
)
def test_gpu_oom_verdict_holds(tmp_path, tasks, pools, case):
    """The verdict holds for a class whose device memory was unknown at placement (its worker read
    24G: the record names it, and the class is unfit before it is unknown), and for a host
    ``MemoryError`` on a GPU class (a host record against its host memory)."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    if case == "unknown device memory":
        cls, hw = _gpu("anon", "NVIDIA L4", 24 * G)
        failure, expect = "cuda", ("device", 24 * G)
    else:
        cls, hw = _gpu(
            "l4", "NVIDIA L4", 24 * G, gpu_type="l4", gpu_mem=24 * G, mem=32 * G
        )
        failure, expect = "host", ("host", 32 * G)
    sub = _gpu_sub(tmp_path, tasks)
    job = Job(
        task=tasks.UseCuda(model="m", x=1),
        submitter=sub,
        name="j",
        resources=ComputeResources(gpus=1),
    )
    _hand_run(tmp_path, pools, sub, [cls], job, [(cls, hw, failure)])
    calibration = DirectoryCalibration(tmp_path / "pool" / "calibration")
    [record] = calibration.unfit(fit_request(job).key)
    assert (record.resource, record.grant, record.worker_class) == (*expect, cls.name)
    second = _warm(
        tmp_path,
        pools,
        sub,
        [cls],
        substrate=ExternalSubstrate(),
        fit=MeasuredFit(calibration, headroom=HEADROOM),
    )
    [got] = second.place([job])
    assert isinstance(got, Unplaceable)
    assert [(t.worker_class, t.verdict) for t in got.tried] == [(cls.name, "unfit")]


class _Meter:
    """A fake `CallMeter`: every call peaks 7 host bytes and 70 device bytes above its start."""

    def start(self, pids=None):
        return (0, 0)

    def stop(self, state):
        return (7, 70)


def test_samples_attribute_memory_to_their_residency(pool, tasks):
    """A worker serving residency A then B records each with the bytes gained across its own load
    (host and its own processes' device bytes, read again at each load) and the call's peaks; B's
    samples, taken with A still resident, are kept but never fitted."""
    from pydra.workers.pool.worker import Worker

    cls, hw = _gpu("l4", "NVIDIA L4", 24 * G, gpu_type="l4", gpu_mem=24 * G)
    pool.submitter.providers = (tasks.Grows(),)
    pool.classes(cls)
    a = [pool.job(tasks.UseGrows(model="a", x=i), f"a{i}") for i in range(2)]
    b = [pool.job(tasks.UseGrows(model="b", x=i), f"b{i}") for i in range(2)]
    pool.put(a, "l4")
    pool.put(b, "l4")
    worker = Worker(
        pool.path,
        cls,
        settings=_settings(),
        hardware=lambda: hw,
        memory=lambda: (
            tasks.MEM["rss"],
            10**12,
        ),  # a lifetime peak that must not be used
        device_memory=lambda: tasks.MEM["device"],
        meter=_Meter(),
    )
    assert worker.run() == 0
    cal = pool.calibration
    hardware = hardware_of(cls)
    ka, kb = fit_request(a[0]).key, fit_request(b[0]).key
    sa, sb = cal.samples(ka, hardware), cal.samples(kb, hardware)
    assert [(s.cold, s.others) for s in sa] == [(True, 0), (False, 0)]
    assert [(s.cold, s.others) for s in sb] == [(True, 1), (False, 1)]
    assert {(s.resident_bytes, s.peak_bytes) for s in sa} == {
        (1000 + 100, 1000 + 100 + 7)
    }
    assert {(s.resident_bytes, s.peak_bytes) for s in sb} == {
        (1000 + 300, 1000 + 300 + 7)
    }
    assert {
        (s.device_baseline_bytes, s.device_bytes, s.device_peak_bytes) for s in sa
    } == {(50, 1000, 1070)}
    assert {(s.device_bytes, s.device_peak_bytes) for s in sb} == {(3000, 3070)}
    assert DirectoryCalibration(cal.root, min_samples=1).memory(kb, hardware) is None
    fit = DirectoryCalibration(cal.root, min_samples=1).memory(ka, hardware)
    assert (fit.resident, fit.m0) == (50 + 1000, 70.0)


def test_device_bytes_count_the_workers_own_processes(monkeypatch, tmp_path):
    """By default a worker counts the device bytes of its own processes only, never a co-tenant's."""
    from pydra.workers.pool import worker as worker_module

    asked = []
    monkeypatch.setattr(
        worker_module, "device_memory", lambda pids=None: asked.append(pids) or 0
    )
    w = worker_module.Worker(tmp_path, _class())
    assert w.device_memory() == 0
    assert asked == [[os.getpid()]]


def test_recheck_not_run_is_recorded_and_never_audited(tmp_path, tasks, pools):
    """A re-check whose provider cannot load in the coordinator's process (here a load that runs
    out of memory there only) is recorded as not run, ``equal=None``, and never audited.
    """
    tasks.FAIL["load"] = True  # this process only: the workers import the module afresh
    _, parity, audits, _ = _recheck_run(
        tmp_path, pools, tasks, tasks.UseSizeds, 1.0, "sized"
    )
    assert len(parity) == 2 and all(r.equal is None for r in parity)
    assert all("MemoryError" in r.reason for r in parity)
    assert audits == []


def test_follower_continues_while_a_recheck_blocks(tmp_path, tasks, pools, monkeypatch):
    """A re-check runs on its own thread with a bounded queue. With A's re-check held, the follower
    still reports B and then C (each placed only after its upstream was reported) and the run
    completes; with B's re-check queued, C's is dropped and counted; `close` (called by the
    driver) returns within its bound while A's is still held, and B's is dropped once A's ends.
    """
    import threading

    from pydra.workers.pool.pool import WarmPool

    entered, release = threading.Event(), threading.Event()
    original = WarmPool._fresh_run

    def held(self, job, candidate):
        entered.set()
        assert release.wait(BOUND)
        return original(self, job, candidate)

    monkeypatch.setattr(WarmPool, "_fresh_run", held)
    monkeypatch.setattr(WarmPool, "recheck_queue", 1)
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Recorder(),)
    settings = _settings(idle_polls=40, recheck_fraction=1.0, drain_s=0.1)
    warm = _warm(tmp_path, pools, sub, settings=settings)
    runner = _drive(sub, tasks.Chain(x=1), warm)  # the driver closes the pool
    assert runner.done and entered.is_set() and not release.is_set()
    assert warm.rechecks_dropped == 1  # C's: the queue held B's
    assert warm._recheck_thread.is_alive()
    release.set()
    warm._recheck_thread.join(BOUND)
    assert not warm._recheck_thread.is_alive()
    assert warm.rechecks_dropped == 2  # and B's, left at close
    calibration = DirectoryCalibration(tmp_path / "pool" / "calibration")
    keys = [
        fit_request(Job(task=tasks.Same(model=m, x=1), submitter=sub, name=m)).key
        for m in "abc"
    ]
    assert [[r.equal for r in calibration.parity(k)] for k in keys] == [[True], [], []]


def test_workers_write_to_the_given_calibration_directory(tmp_path, tasks, pools):
    """A `DirectoryCalibration` passed to the pool is where its workers write; any other
    calibration, or a fit reading elsewhere, warns."""
    import warnings

    given = DirectoryCalibration(tmp_path / "given")
    sub = Submitter(cache_root=tmp_path / "store", worker="debug")
    sub.providers = (tasks.Recorder(),)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warm = _warm(
            tmp_path,
            pools,
            sub,
            calibration=given,
            fit=MeasuredFit(given, headroom=HEADROOM),
        )
    _drive(sub, tasks.Sames(xs=[1]), warm)
    warm.close()
    job = Job(task=tasks.Same(model="m", x=1), submitter=sub, name="k")
    hardware = hardware_of(_class())
    assert len(given.samples(fit_request(job).key, hardware)) == 1
    assert not (tmp_path / "pool" / "calibration").exists()

    class Elsewhere:
        def record(self, record):
            pass

    with pytest.warns(UserWarning, match="workers write"):
        _warm(tmp_path / "other", pools, sub, calibration=Elsewhere())
    with pytest.warns(UserWarning, match="no out-of-memory verdict"):
        _warm(tmp_path / "third", pools, sub, fit=MeasuredFit(given, headroom=HEADROOM))


# -- provider children on the host, unseen pids on the device, grants over names ---


def _fake_proc(monkeypatch, tasks, child):
    """Fake ``/proc``: this process holds 100 MB and peaks 0 above it; the child, once loaded,
    holds ``child`` bytes (None: unreadable) and peaks 1 GB above them."""
    from pydra.workers.pool import samples

    def status(field, pid="self"):
        if pid == tasks.CHILD:
            if child is None or "child" not in tasks.LOADS:
                return None
            return child + (GB if field == "VmHWM" else 0)
        return 100 * 10**6

    monkeypatch.setattr(samples, "_status_kb", status)
    monkeypatch.setattr(samples, "_reset_hwm", lambda pid: True, raising=False)


@pytest.mark.parametrize("child", [12 * GB, None])
def test_a_providers_child_process_counts_on_the_host(pool, tasks, monkeypatch, child):
    """A provider whose model lives in a child process: the host figures sum every process the
    worker owns, so a 12 GB child does not "fit" a 4 GB class; when a child's figures cannot be
    read there is no host peak, so no host fit, and placement falls back to the declared
    figures."""
    from pydra.workers.pool.worker import Worker

    _fake_proc(monkeypatch, tasks, child)
    pool.submitter.providers = (tasks.Child(),)
    small, large = _class("small", mem=4 * GB), _class("large", mem=16 * GB)
    pool.classes(small, large)
    jobs = [pool.job(tasks.UseChild(model="m", x=i), f"c{i}") for i in range(2)]
    pool.put(jobs, "small")
    worker = Worker(
        pool.path,
        small,
        settings=_settings(),
        memory=lambda: (100 * 10**6, None),
        device_memory=lambda: None,
    )
    assert worker.run() == 0
    key = fit_request(jobs[0]).key
    measured = DirectoryCalibration(pool.calibration.root, min_samples=1)
    [warm] = [s for s in measured.samples(key, hardware_of(small)) if not s.cold]
    got = place(jobs[0], [small, large], MeasuredFit(measured, headroom=HEADROOM), ok)
    if child is None:
        assert (warm.resident_bytes, warm.peak_bytes) == (None, None)
        assert measured.memory(key, hardware_of(small)) is None
        assert got.worker_class == "small" and got.tried[0].verdict == "unknown"
    else:
        assert warm.resident_bytes == 100 * 10**6 + child
        assert warm.peak_bytes == warm.resident_bytes + GB
        assert got.worker_class == "large"
        assert [t.verdict for t in got.tried] == ["too small", "fits"]


def test_a_device_read_that_sees_no_own_pid_is_unknown(tmp_path, monkeypatch):
    """``nvidia-smi`` listing pids of another namespace (a container): with CUDA initialised in
    this process, no row for this worker is unknown (None), never 0; torch's reserved bytes are the
    device peak a `CallMeter` reads."""
    if not POSIX:
        pytest.skip("a shell-script stand-in for nvidia-smi")
    from pydra.workers.pool import samples

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho '1, 4096'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    class Cuda:
        def reset_peak_memory_stats(self):
            self.peak = 0

        def memory_reserved(self):
            return 5 * G

        def max_memory_reserved(self):
            return 7 * G

        def memory_allocated(self):
            raise AssertionError("the allocated figure is not the device pressure")

        max_memory_allocated = memory_allocated

    monkeypatch.setattr(samples, "_torch_cuda", lambda: None)
    assert samples.device_memory([os.getpid()]) == 0  # no CUDA here: nothing held
    assert samples.device_memory() == 4096 << 20
    monkeypatch.setattr(samples, "_torch_cuda", lambda: Cuda())
    assert samples.device_memory([os.getpid()]) is None
    meter = samples.CallMeter()
    monkeypatch.setattr(samples, "_reset_hwm", lambda pid: False)
    assert meter.stop(meter.start([os.getpid()])) == (None, 2 * G)


def test_a_record_on_a_class_whose_grant_grew_does_not_bind(measured):
    """A record binds by grant when both are known: 16G recorded on ``cpu`` does not make ``cpu``
    unfit once it is raised to 64G (§3.3: a grant at least as large); by name only when a grant is
    unknown."""
    _, request, _ = measured
    record = UnfitRecord(request.key, "host", 16 * G, "cpu", request.code)

    class Fixed:
        def memory(self, key, hardware, task=None):
            return None

        def unfit(self, key):
            return [record]

        def times(self, key, hardware, task=None):
            return ()

    fit = MeasuredFit(Fixed(), headroom=HEADROOM)
    assert fit.judge(request, _class("cpu", mem=16 * G)).verdict == "unfit"
    assert fit.judge(request, _class("cpu", mem=64 * G)).verdict != "unfit"
    assert (
        fit.judge(request, _class("cpu")).verdict == "unfit"
    )  # grant unknown: by name


def test_an_unreachable_calibration_directory_is_not_made_locally(pool, tasks, caplog):
    """A calibration directory named in the pool whose parent is absent where the worker runs (a
    coordinator's path not mounted on the node): the worker warns once, writes nothing and makes
    no local tree."""
    import logging

    absent = pool.path.parent / "not-mounted" / "calibration"
    pool.pool.write_faults(pool.pool.take_epoch(), {"calibration": str(absent)})
    jobs = [pool.job(tasks.UseSized(model="m", x=i), f"s{i}") for i in range(2)]
    pool.put(jobs, "cpu")
    with caplog.at_level(logging.WARNING, logger="pydra.pool"):
        _, code = _run(pool, _class())
    assert code == 0
    assert not absent.parent.exists()
    assert not (pool.path / "calibration").exists()
    warned = [r for r in caplog.records if "not reachable here" in r.getMessage()]
    assert len(warned) == 1


def test_the_host_peak_sums_each_own_process_or_is_none(monkeypatch):
    """The call's host peak sums each own process's high-water mark above its resident set before
    the call; one process whose mark cannot be read gives no peak."""
    from pydra.workers.pool import samples

    figures = {
        (1, "VmRSS"): 100,
        (1, "VmHWM"): 130,
        (2, "VmRSS"): 1000,
        (2, "VmHWM"): 1500,
    }
    monkeypatch.setattr(
        samples, "_status_kb", lambda field, pid="self": figures.get((pid, field))
    )
    monkeypatch.setattr(samples, "_reset_hwm", lambda pid: True)
    monkeypatch.setattr(samples, "_torch_cuda", lambda: None)
    meter = samples.CallMeter()
    assert meter.stop(meter.start([1, 2])) == (30 + 500, None)
    state = meter.start([1, 2])
    del figures[(2, "VmHWM")]
    assert meter.stop(state) == (None, None)


def test_a_record_that_fails_to_encode_leaves_nothing_behind(tmp_path, monkeypatch):
    from pydra.workers.pool import calibration

    cal = DirectoryCalibration(tmp_path / "cal")
    unfit = UnfitRecord("k", "host", 8 * GB, "cpu", Code((("p", "1"),)))

    def fail(*args, **kwargs):
        raise ValueError("cannot encode")

    monkeypatch.setattr(calibration.json, "dump", fail)
    with pytest.raises(ValueError):
        cal.record(unfit)
    assert [p for p in (tmp_path / "cal").rglob("*") if p.is_file()] == []


def test_measured_fit_has_no_headroom_of_its_own():
    with pytest.raises(TypeError):
        MeasuredFit(DirectoryCalibration("unused"))
