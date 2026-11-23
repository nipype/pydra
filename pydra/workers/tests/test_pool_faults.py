"""Typed faults, one bisection engine and holds (RFC 0005 sections 4.4, 5.5, 6; W7).

The engine's contract (a job that crashes every call is charged after ``crashes_charged`` and its
neighbours complete, a crash another call survives is not charged, a load is never split, an
abandoned call is not retried), the classifier's (a fault's class survives every wrapper, a load is
never a job's) and out of memory (only the call that ran out is split, its halves
keep their results, a load out of memory is no batch's fault). The rest are rows of RFC 0005
section 11.

Nothing is timed: the engine runs fakes, workers in this process are driven to idle, and waits on
other processes poll for a state with a bound that only fails a hung test.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.submitter import Submitter

POSIX = sys.platform != "win32"
LEASE = LeaseSettings(
    stale_after=4.0, renew_fraction=0.4, attribute_delay=0.0, timeout=10.0
)
BOUND = 120.0  # seconds any wait on another process may take before the test fails

TASKS = """
import os, signal, sys
from pathlib import Path
from uuid import uuid4
from pydra.compose import python, workflow
from pydra.engine.procgroup import ProcessProvider
from pydra.engine.resident import Provided
from pydra.workers.pool.faults import FaultRecord, WorkerLost

HOOK = {}
LOADS = []


def _hook(name, *args):
    fn = HOOK.get(name)
    return None if fn is None else fn(*args)


class _InProcess:
    def close(self, obj):
        pass

    def footprint(self, spec):
        return None

    def measure(self, obj):
        return None

    def version(self):
        return "1"


class Recorder(_InProcess):
    # an in-process provider that records each load
    name = "recorder"

    def load(self, spec, device):
        LOADS.append("recorder")
        return spec


class OomLoad(_InProcess):
    # a provider whose load runs out of memory, re-raised by a wrapper as a loader does
    name = "oomload"

    def load(self, spec, device):
        LOADS.append("oomload")
        try:
            raise MemoryError("CUDA out of memory")
        except MemoryError as err:
            raise RuntimeError("loading the model failed") from err


class Broken(_InProcess):
    # a provider whose load always fails, as a missing model file does
    name = "broken"

    def load(self, spec, device):
        LOADS.append("broken")
        raise FileNotFoundError(2, "no such model", spec)


class FlakyLoad(_InProcess):
    # a provider whose first load succeeds and every later one runs out of memory
    name = "flakyload"

    def load(self, spec, device):
        LOADS.append("flakyload")
        if LOADS.count("flakyload") > 1:
            raise MemoryError("CUDA out of memory")
        return spec


class SleepServer(ProcessProvider):
    # a provider whose state is a child process that sleeps
    name = "sleepserver"

    def argv(self, spec, device):
        return [sys.executable, "-c", "import time; time.sleep(600)"]

    def load(self, spec, device):
        LOADS.append("sleepserver")
        return super().load(spec, device)


@python.define
def Echo(x: int) -> int:
    _hook("echo", x)
    return x


@python.define
def OomOnRecorder(model: Provided[str, "recorder"], x: int) -> int:
    if x == 1:
        raise MemoryError("CUDA out of memory")
    return x


@python.define
def OomOnServer(model: Provided[object, "sleepserver"], x: int) -> int:
    _hook("server", model, x)
    if x == 1:
        raise MemoryError("CUDA out of memory")
    return x


@python.define
def UseOomLoad(model: Provided[object, "oomload"], x: int) -> int:
    return x


@python.define
def UseBroken(model: Provided[object, "broken"], x: int) -> int:
    return x


@python.define
def OomOnFlaky(model: Provided[object, "flakyload"], x: int) -> int:
    if x == 1:
        raise MemoryError("CUDA out of memory")
    return x


@python.define
def CrashOnce(marker: str) -> str:
    if not os.path.exists(marker):
        Path(marker).touch()
        raise WorkerLost("the provider's process ended mid-request")
    return marker


@python.define
def Oom(x: int) -> int:
    raise MemoryError("CUDA out of memory")


@python.define
def Killed(x: int) -> int:
    _hook("killed", x)
    raise ChildProcessError("the command was killed by signal 9")


@python.define
def Crash(x: int) -> int:
    _hook("crash", x)
    raise WorkerLost("the provider's process ended mid-request")


@python.define
def Die(x: int, where: str) -> int:
    # one file per execution, then the worker process dies as a SIGKILL leaves it
    Path(where, f"{x}-{uuid4().hex}").touch()
    os.kill(os.getpid(), signal.SIGKILL)
    return x


@python.define
def Mark(x: int, where: str) -> int:
    Path(where, f"{x}-{uuid4().hex}").touch()
    return x


@python.define
def FailOnce(marker: str) -> str:
    if not os.path.exists(marker):
        Path(marker).touch()
        raise ValueError("the first attempt fails")
    return marker


@workflow.define
def Brokens(xs: list) -> list:
    return workflow.add(UseBroken(model="m").split(x=xs), name="B").out


@workflow.define(outputs=["b", "e"])
def Mixed(xs: list):
    from pydra.utils.resources import ComputeResources

    ignore = ComputeResources(residency="ignore")
    b = workflow.add(UseBroken(model="m").split(x=xs), name="B", resources=ignore)
    e = workflow.add(Echo().split(x=xs), name="E")
    return b.out, e.out


@workflow.define
def Flakys(markers: list) -> list:
    return workflow.add(FailOnce().split(marker=markers), name="F").out


@workflow.define
def Killeds(xs: list) -> list:
    return workflow.add(Killed().split(x=xs), name="K").out


@workflow.define
def Marks(xs: list, where: str) -> list:
    return workflow.add(Mark(where=where).split(x=xs), name="M").out


@workflow.define
def BigMark(x: int, where: str) -> int:
    from pydra.utils.resources import ComputeResources

    return workflow.add(
        Mark(x=x, where=where), name="M", resources=ComputeResources(mem="2G")
    ).out


@workflow.define
def Flaky(marker: str) -> str:
    return workflow.add(FailOnce(marker=marker), name="F").out


def infrastructure(exc, phase):
    # a stricter classifier: every exception is the environment's
    return FaultRecord("infrastructure", type(exc).__name__, str(exc), phase, gap=True)
"""


# -- fixtures and helpers ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def tasks(tmp_path_factory):
    """The task module, importable here and in workers started from here."""
    where = tmp_path_factory.mktemp("pool_fault_tasks")
    (where / "_pydra_pool_fault_tasks.py").write_text(TASKS)
    sys.path.insert(0, str(where))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(where), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    import _pydra_pool_fault_tasks

    return _pydra_pool_fault_tasks


@pytest.fixture(autouse=True)
def clean(monkeypatch, tasks):
    """Release every claim a test leaves held and clear the task hooks and loads, pass or fail."""
    from pydra.workers.pool.storage import Claim

    made = []
    init = Claim.__init__

    def recording(self, *args, **kwargs):
        init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(Claim, "__init__", recording)
    tasks.LOADS.clear()
    yield
    tasks.HOOK.clear()
    tasks.LOADS.clear()
    for claim in made:
        claim.lease.__exit__(None, None, None)


@pytest.fixture
def processes():
    """Processes a test starts, each in its own session; killed with their group at the end, and
    none left behind."""
    started: list[subprocess.Popen] = []
    yield started
    for proc in started:
        _kill_group(proc)
    if POSIX:
        for proc in started:
            found = subprocess.run(
                ["pgrep", "-g", str(proc.pid)], capture_output=True, text=True
            )
            assert found.stdout.strip() == "", f"processes left in group {proc.pid}"


@pytest.fixture
def pools():
    """Every `WarmPool` a test makes is closed, and every worker its substrate started is killed."""
    made = []
    yield made
    for pool in made:
        try:
            pool.close()
        finally:
            for handle in getattr(pool.substrate, "handles", []):
                if handle.process is not None:
                    _kill_group(handle.process)


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
    from pydra.workers.pool.classes import WorkerClass

    fields = dict(name=name, device="cpu", quantum_jobs=4, quantum_s=600.0)
    fields.update(kwargs)
    return WorkerClass(**fields)


class Pool:
    """A pool, a store and the jobs put on the pool, for one test."""

    def __init__(self, tmp_path: Path):
        from pydra.workers.pool.storage import PoolRoot

        self.pool = PoolRoot.open(tmp_path / "pool", LEASE, probe=False)
        self.submitter = Submitter(cache_root=tmp_path / "store", worker="debug")
        self.store = self.submitter.store
        self.tickets: list[tuple[str, str]] = []

    @property
    def path(self) -> Path:
        return self.pool.path

    def providers(self, tasks) -> None:
        """Jobs made from here on name the test providers."""
        self.submitter.providers = (
            tasks.FlakyLoad(),
            tasks.Recorder(),
            tasks.OomLoad(),
            tasks.Broken(),
            tasks.SleepServer(),
        )

    def job(self, task, name: str, *, resources=None) -> Job:
        return Job(task=task, submitter=self.submitter, name=name, resources=resources)

    def entry(self, job: Job):
        from pydra.workers.pool.codec import PrivatePickle, write_payload
        from pydra.workers.pool.storage import TicketJob

        fingerprint = self.store.identity.fingerprint(job.context)
        g = self.pool._next_generation((job.checksum, fingerprint))
        job.attempt = f"{self.pool.pool_id}.{g}"
        return TicketJob(
            checksum=job.checksum,
            fingerprint=fingerprint,
            payload=write_payload(PrivatePickle(), job, self.path),
            attempt=job.attempt,
            runner_attempt=0,
            units=1.0,
        )

    def put(self, jobs, queue="cpu", residency=None, **kwargs) -> str:
        from pydra.workers.pool.storage import Ticket

        if residency is None:
            residency = _residency(jobs[0])
        ticket = Ticket(
            queue=queue,
            node="wf/node",
            jobs=[self.entry(j) for j in jobs],
            residency=residency,
            **kwargs,
        )
        digest = self.pool.put(queue, ticket)
        self.tickets.append((queue, digest))
        return digest

    def key(self, job: Job) -> tuple[str, str]:
        return job.checksum, self.store.identity.fingerprint(job.context)

    def outcomes(self, queue="cpu", digest=None) -> list[dict]:
        found = []
        for q, d in self.tickets:
            if q == queue and (digest is None or d == digest):
                found.extend(self.pool.outcomes(q, d).values())
        return found

    def generations(self, job: Job) -> list[dict]:
        return [r for _, r in self.pool.generations(*self.key(job))]

    def newest(self, job: Job) -> tuple[str, dict | None]:
        """The queue of the job's newest generation's ticket and its outcome there."""
        _, record = self.pool.newest(*self.key(job))
        queue = self.pool._by_label(record["queue"]).name
        found = self.pool.read_outcome(
            queue, record["ticket"], self.key(job), record["attempt"]
        )
        return queue, found


def _residency(job: Job) -> tuple:
    """The residency a `WarmPool` writes in a job's ticket."""
    import attrs

    from pydra.engine import resident

    return tuple(
        attrs.asdict(resident.spec_key(job.task, f, p))
        for f, p in resident.provided_fields(job.task)
    )


@pytest.fixture
def pool(tmp_path):
    p = Pool(tmp_path)
    yield p
    p.pool.close()


def _worker(pool: Pool, cls=None, **kwargs):
    from pydra.workers.pool.worker import Worker

    kwargs.setdefault("settings", _settings())
    kwargs.setdefault("device_memory", lambda: None)
    return Worker(pool.path, cls or _class(), **kwargs)


def _run(pool: Pool, cls=None, **kwargs):
    worker = _worker(pool, cls, **kwargs)
    return worker, worker.run()


def _wait(predicate, what: str, bound: float = BOUND) -> None:
    deadline = time.monotonic() + bound
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _age(path: Path) -> None:
    """Stamp a lease directory and everything in it as a holder that stopped long ago leaves it."""
    old = 10**9
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in dirnames + filenames:
            os.utime(Path(dirpath) / name, (old, old))
    os.utime(path, (old, old))


def _classes(pool: Pool, *classes) -> None:
    from pydra.workers.pool.worker import class_record

    for cls in classes:
        pool.pool.put_class(cls.name, class_record(cls))


def _dead_call(
    pool: Pool, digest: str, jobs, *, expired=(), call=None, drained=()
) -> str:
    """Leave ticket ``digest`` as a worker killed mid-call leaves it: its jobs started under the dead
    worker's token, ``call`` (default every job) in its in-flight call record, ``expired`` given an
    ``expired`` outcome, ``drained`` marked cut short by a drain, and the claim aged. Return the
    dead token."""
    dead = "d" * 32
    claim = pool.pool.take("cpu", digest, beat="external")
    entries = {e.key: e for e in claim.ticket.jobs}
    listed = []
    for job in jobs:
        entry = entries[pool.key(job)]
        attempt = claim.start(entry, dead)
        if call is None or job in call:
            listed.append(
                {
                    "checksum": entry.checksum,
                    "fingerprint": entry.fingerprint,
                    "attempt": attempt,
                }
            )
        if job in expired:
            claim.outcome(entry, attempt, "expired")
        if job in drained:
            claim.drained(entry, dead)
    if listed:
        pool.pool.write_call(dead, 1, {"jobs": listed})
    _age(claim.lease.lockfile)
    return dead


# -- the classifier --------------------------------


def _wrapped_oom_by_cause():
    try:
        raise MemoryError("CUDA out of memory")
    except MemoryError as err:
        try:
            raise RuntimeError("loading the model failed") from err
        except RuntimeError as wrapper:
            return wrapper


def _wrapped_oom_by_context():
    try:
        try:
            raise MemoryError("CUDA out of memory")
        except MemoryError:
            raise KeyError("cleanup failed while handling it")
    except KeyError as wrapper:
        return wrapper


def _twice_wrapped_oom():
    try:
        raise _wrapped_oom_by_cause()
    except RuntimeError as err:
        try:
            raise OSError("the worker set gave up") from err
        except OSError as outer:
            return outer


@pytest.mark.parametrize(
    "make", [_wrapped_oom_by_cause, _wrapped_oom_by_context, _twice_wrapped_oom]
)
@pytest.mark.parametrize("phase", ["load", "call"])
def test_fault_class_survives_wrapper(make, phase):
    """An out-of-memory error wrapped by ``raise ... from``, raised during its handling, or wrapped
    twice stays a resource fault at a load and at a call; the record names the
    outermost exception and its cause."""
    from pydra.workers.pool.faults import classify, fault_of

    exc = make()
    for record in (classify(exc, phase), fault_of(exc, phase)):
        assert (record.kind, record.reason, record.phase) == ("resource", "oom", phase)
        assert record.type == type(exc).__name__
        assert "MemoryError" in record.message and not record.gap


@pytest.mark.parametrize("kind", ["content", "resource", "infrastructure"])
@pytest.mark.parametrize("phase", ["load", "call"])
def test_an_attached_record_survives_every_wrapper_and_classifier(kind, phase, tasks):
    """A record attached to an exception is the fault of anything that wraps it, whatever the
    classifier; a content record at a load is the residency's (infrastructure), since a load is
    not a job."""
    from pydra.workers.pool.faults import FaultRecord, attach, fault_of

    stated = FaultRecord(kind, "Stated", "stated by the provider", phase)
    try:
        try:
            raise attach(ValueError("stated"), stated)
        except ValueError as err:
            raise RuntimeError("wrapped") from err
    except RuntimeError as wrapper:
        record = fault_of(wrapper, phase, tasks.infrastructure)
    stands = "infrastructure" if (phase, kind) == ("load", "content") else kind
    assert (record.kind, record.type) == (stands, "Stated")


def test_default_table():
    """The default classifier: an exception from a call is content; from a load or the pool's
    machinery it is an infrastructure gap; `WorkerLost` anywhere in the chain is a crash.
    """
    from pydra.workers.pool.faults import WorkerLost, classify

    assert classify(ValueError("x"), "call").kind == "content"
    for phase in ("load", "pool"):
        record = classify(ValueError("x"), phase)
        assert (record.kind, record.gap) == ("infrastructure", True)
    try:
        try:
            raise WorkerLost("the process ended")
        except WorkerLost as err:
            raise BrokenPipeError("protocol") from err
    except BrokenPipeError as wrapper:
        record = classify(wrapper, "call")
    assert (record.kind, record.reason, record.crash) == (
        "infrastructure",
        "crash",
        True,
    )


def test_fault_budgets_defaults_and_json():
    """`FaultBudgets` carries pydra's defaults, ``start_failures`` among them (no longer a
    `PoolSettings` field), and travels as JSON."""
    import attrs

    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.settings import PoolSettings

    budgets = FaultBudgets()
    assert attrs.asdict(budgets) == {
        "content_attempts": 1,
        "resource_moves": 2,
        "infrastructure_replacements": 3,
        "crashes_charged": 2,
        "start_failures": 7,
        "listing_failures": 7,
    }
    assert "start_failures" not in {f.name for f in attrs.fields(PoolSettings)}
    changed = FaultBudgets(content_attempts=2, start_failures=3)
    assert FaultBudgets.from_json(json.loads(json.dumps(changed.to_json()))) == changed
    with pytest.raises(ValueError):
        FaultBudgets.from_json({"resource_replacements": 1})
    with pytest.raises(ValueError):
        FaultBudgets(start_failures=0)
    with pytest.raises(ValueError):
        FaultBudgets(listing_failures=0)


def test_no_alias_left_in_the_worker():
    """The budgets and the classifier live in `faults` only (renamed outright, pre-alpha)."""
    from pydra.workers.pool import faults, worker

    assert not hasattr(faults, "default_classifier")
    for name in ("FaultBudgets", "default_classifier", "classify", "FaultRecord"):
        assert name not in worker.__all__


# -- the bisection engine ----------------------

BATCH = 12


class Calls:
    """A fake batched call: ``poison`` maps an item to how many times it raises ``error()`` when it
    is in a call (None: always); ``oversize`` items run out of memory in any call of more than one
    item, and alone. It records the size of every call and every reload."""

    def __init__(self, poison=None, error=None, oversize=(), elements=()):
        from pydra.workers.pool.faults import WorkerLost

        self.poison = dict(poison or {})
        self.error = error or (lambda: WorkerLost("died"))
        self.oversize = set(oversize)
        self.elements = set(elements)
        self.sizes: list[int] = []
        self.reloads: list[str] = []

    def __call__(self, items):
        self.sizes.append(len(items))
        for item in items:
            left = self.poison.get(item, 0)
            if left is None or left > 0:
                if left is not None:
                    self.poison[item] = left - 1
                raise self.error()
        if self.oversize & set(items):
            raise MemoryError("CUDA out of memory")
        return [
            ValueError(f"item {i}") if i in self.elements else i * 10 for i in items
        ]

    def reload(self, fault):
        self.reloads.append(fault.kind if not fault.crash else "crash")


def _bisect(calls, items, **kwargs):
    from pydra.workers.pool.bisect import bisect
    from pydra.workers.pool.faults import classify

    kwargs.setdefault("crashes_charged", 2)
    return bisect(items, calls, classify=classify, reload=calls.reload, **kwargs)


def test_an_item_that_crashes_every_call_is_charged_and_the_rest_complete():
    """The poisoned item is ``JobCrashedWorker``, content; its eleven neighbours complete. Its calls
    are halved down to it, and it alone is lost ``crashes_charged`` times; each loss reloads.
    """
    calls = Calls({7: None})
    settled = _bisect(calls, list(range(BATCH)))
    [dead] = [s for s in settled if s.fault is not None]
    assert (dead.fault.kind, dead.fault.type) == ("content", "JobCrashedWorker")
    assert settled.index(dead) == 7
    assert [s.value for s in settled if s.fault is None] == [
        i * 10 for i in range(BATCH) if i != 7
    ]
    # 6 alone, 7 alone twice, 8 alone
    assert calls.sizes == [12, 6, 6, 3, 2, 1, 1, 1, 1, 3]
    assert calls.reloads == ["crash"] * 6


@pytest.mark.parametrize("charged", [1, 3])
def test_the_count_is_the_budget(charged):
    """A job alone is lost exactly ``crashes_charged`` times before it is charged."""
    calls = Calls({0: None})
    [dead] = _bisect(calls, [0], crashes_charged=charged)
    assert dead.fault.type == "JobCrashedWorker" and dead.charged
    assert calls.sizes == [1] * charged


@pytest.mark.parametrize("error", ["lost", "protocol"])
def test_a_crash_another_call_survives_is_not_charged(error):
    """The first call is lost once; its halves complete: no failure at all."""
    from pydra.workers.pool.faults import WorkerLost

    class ProtocolError(WorkerLost):
        pass

    make = {
        "lost": lambda: WorkerLost("died"),
        "protocol": lambda: ProtocolError("bad"),
    }
    calls = Calls({3: 1}, make[error])
    settled = _bisect(calls, list(range(BATCH)))
    assert all(s.fault is None for s in settled) and len(settled) == BATCH
    assert calls.sizes == [12, 6, 6] and calls.reloads == ["crash"]


def test_a_lone_job_lost_once_and_then_completing_is_not_charged():
    """One job lost is sent again after a reload, completes, and is never a failure."""
    calls = Calls({0: 1})
    [done] = _bisect(calls, [0])
    assert done.fault is None and done.value == 0 and calls.sizes == [1, 1]


def test_a_load_fault_inside_a_call_is_never_split():
    """A fault whose record says ``load`` (a reload inside the call) is raised out as it is: the
    call is not split or retried."""
    from pydra.workers.pool.faults import FaultRecord, attach

    record = FaultRecord("resource", "MemoryError", "reload", "load", reason="oom")
    calls = Calls({5: None}, lambda: attach(MemoryError("reload"), record))
    with pytest.raises(MemoryError):
        _bisect(calls, list(range(BATCH)))
    assert calls.sizes == [BATCH] and calls.reloads == []


def test_only_the_call_that_ran_out_is_split_and_its_halves_keep_their_results():
    """An out-of-memory call is halved until the oversize job is alone; every half that fits keeps
    its results, and the lone job is the resource fault (moved up its ladder by the caller).
    """
    calls = Calls(oversize={11})
    settled = _bisect(calls, list(range(BATCH)))
    assert [s.value for s in settled[:11]] == [i * 10 for i in range(11)]
    dead = settled[11]
    assert (dead.fault.kind, dead.fault.reason, dead.charged) == (
        "resource",
        "oom",
        False,
    )
    assert calls.sizes == [12, 6, 6, 3, 3, 2, 1]
    assert calls.reloads == ["resource"] * 4


def test_an_exception_element_is_that_jobs_content_fault_and_never_splits():
    """A batched return's exception element fails its own job (content) and splits nothing."""
    calls = Calls(elements={4})
    settled = _bisect(calls, list(range(BATCH)))
    assert calls.sizes == [BATCH] and calls.reloads == []
    assert settled[4].fault.kind == "content" and settled[4].fault.type == "ValueError"
    assert isinstance(settled[4].error, ValueError)
    assert sum(s.fault is None for s in settled) == BATCH - 1


def test_any_other_raise_fails_the_whole_call():
    """A whole call raising for a reason that is not a trigger (content here) fails each of its
    jobs once, unsplit."""
    calls = Calls({2: None}, lambda: ValueError("the batch is malformed"))
    settled = _bisect(calls, list(range(BATCH)))
    assert calls.sizes == [BATCH]
    assert {(s.fault.kind, s.fault.type) for s in settled} == {
        ("content", "ValueError")
    }


def test_a_resource_typed_crash_is_a_resource_fault_never_charged():
    """A lost process whose record says out of memory is a resource fault: split, then the lone job
    is resource, uncharged, never retried as a crash."""
    from pydra.workers.pool.faults import FaultRecord, WorkerLost, attach

    record = FaultRecord("resource", "MemoryError", "oom", "call", reason="oom")
    calls = Calls({5: None}, lambda: attach(WorkerLost("killed"), record))
    settled = _bisect(calls, list(range(BATCH)))
    [dead] = [s for s in settled if s.fault is not None]
    assert (dead.fault.kind, dead.charged) == ("resource", False)
    assert calls.sizes.count(1) == 1  # 5 alone, once


def test_an_abandoned_call_is_neither_split_nor_retried():
    """A call that ends without a verdict (its claim lost, the worker stopping) is re-raised."""
    from pydra.workers.pool.bisect import Abandon

    calls = Calls({7: None}, lambda: Abandon("the claim was lost"))
    with pytest.raises(Abandon):
        _bisect(calls, list(range(BATCH)))
    assert calls.sizes == [BATCH] and calls.reloads == []


def test_the_losses_are_counted_by_the_caller():
    """``lost`` records each loss of a lone job and returns its count, so a count kept elsewhere (a
    job's charged generations) is the one the budget reads."""
    calls = Calls({0: None})
    counts = iter([1, 5])
    seen = []

    def lost(item):
        seen.append(item)
        return next(counts)

    [dead] = _bisect(calls, [0], lost=lost, crashes_charged=3)
    assert (
        seen == [0, 0]
        and calls.sizes == [1, 1]
        and dead.fault.type == "JobCrashedWorker"
    )


def test_a_count_that_never_grows_is_capped_by_the_engine():
    """A ``lost`` hook whose count never grows cannot loop the engine: its own count of the job's
    losses in a row caps the retries at ``crashes_charged``."""
    calls = Calls({0: None})
    [dead] = _bisect(calls, [0], lost=lambda item: 0, crashes_charged=2)
    assert dead.fault.type == "JobCrashedWorker" and calls.sizes == [1, 1]


def test_survived_crashes_are_reported_and_a_charged_items_are_not():
    """Each loss whose jobs all completed afterwards is reported to ``survived`` (the environment's
    fault, charged to no job); the losses a job charged with ``JobCrashedWorker`` caused are not.
    """
    survived = []
    calls = Calls({3: 1})
    settled = _bisect(calls, list(range(BATCH)), survived=survived.append)
    assert all(s.fault is None for s in settled)
    assert [(f.crash, f.kind) for f in survived] == [(True, "infrastructure")]

    survived.clear()
    _bisect(Calls({7: None}), list(range(BATCH)), survived=survived.append)
    assert survived == []

    survived.clear()
    [done] = _bisect(Calls({0: 1}), [0], survived=survived.append)
    assert done.fault is None and len(survived) == 1


def test_a_task_raising_keyboard_interrupt_is_content_and_machinery_interrupts_propagate():
    """Only `Exception`s are typed: a `KeyboardInterrupt` from the call itself propagates out of the
    engine (the worker wraps a task's own as content before it reaches the engine)."""

    def call(items):
        raise KeyboardInterrupt()

    from pydra.workers.pool.bisect import bisect
    from pydra.workers.pool.faults import classify

    with pytest.raises(KeyboardInterrupt):
        bisect([0], call, classify=classify)


# -- the worker --------------------------------------------------------------------------------


def test_the_worker_runs_its_calls_through_the_one_engine(pool, tasks, monkeypatch):
    """Every call of a pool worker goes through `bisect.bisect`: one engine, not two."""
    from pydra.workers.pool import bisect as engine

    seen = []
    real = engine.bisect

    def counting(items, call, **kwargs):
        seen.append(len(items))
        return real(items, call, **kwargs)

    monkeypatch.setattr(engine, "bisect", counting)
    pool.put([pool.job(tasks.Echo(x=1), "a"), pool.job(tasks.Echo(x=2), "b")])
    _, code = _run(pool)
    assert code == 0 and seen == [1, 1]
    assert sorted(o["outcome"] for o in pool.outcomes()) == ["done", "done"]


def test_resource_fault_reloads_residency(pool, tasks):
    """An out-of-memory call closes the ticket's residency: the job moves up the ladder, and the next
    job of the ticket loads the provider again (``reload`` in its sample)."""
    pool.providers(tasks)
    oom, ok = pool.job(tasks.OomOnRecorder(model="m", x=1), "oom"), pool.job(
        tasks.OomOnRecorder(model="m", x=2), "ok"
    )
    pool.put([oom, ok], ladder=("big",))
    _classes(pool, _class("big"))
    _, code = _run(pool)
    assert code == 0
    assert tasks.LOADS == ["recorder", "recorder"]
    queue, moved = pool.newest(oom)
    assert queue == "big" and moved is None
    assert [(g["cause"], g["phase"]) for g in pool.generations(oom)] == [
        ("put", None),
        ("ladder", "call"),
    ]
    [done] = [o for o in pool.outcomes() if o["outcome"] == "done"]
    assert done["sample"]["reload"] is True


@pytest.mark.skipif(not POSIX, reason="process-owning providers are stopped by group")
def test_process_provider_restarts_after_resource_fault(pool, tasks):
    """A process-owning provider whose call ran out of memory is stopped with the residency, and the
    next job starts its process again."""
    pool.providers(tasks)
    pids = []
    tasks.HOOK["server"] = lambda model, x: pids.append(model.group.proc)
    pool.put(
        [
            pool.job(tasks.OomOnServer(model="m", x=1), "oom"),
            pool.job(tasks.OomOnServer(model="m", x=2), "ok"),
        ],
        ladder=("big",),
    )
    _classes(pool, _class("big"))
    _, code = _run(pool)
    assert code == 0
    assert tasks.LOADS == ["sleepserver", "sleepserver"]
    first, second = pids
    assert first.pid != second.pid and first.poll() is not None


@pytest.mark.parametrize("task", ["UseOomLoad", "OomDirect"])
def test_load_oom_moves_up_charges_nothing(pool, tasks, task, monkeypatch):
    """A load that runs out of memory, re-raised by a wrapper or not, is a resource fault: the
    ticket's jobs move up the ladder at the ``load`` phase, uncharged, and nothing is held.
    """
    pool.providers(tasks)
    if task == "OomDirect":

        def load(self, spec, device):
            tasks.LOADS.append("oomload")
            raise MemoryError("CUDA out of memory")

        monkeypatch.setattr(tasks.OomLoad, "load", load)
    jobs = [pool.job(tasks.UseOomLoad(model="m", x=i), f"j{i}") for i in range(2)]
    pool.put(jobs, ladder=("big",))
    _classes(pool, _class("big"))
    worker, code = _run(pool)
    assert code == 0 and worker.claims == [2] and tasks.LOADS == ["oomload"]
    for job in jobs:
        assert [
            (g["cause"], g["phase"], g["charged"]) for g in pool.generations(job)
        ] == [
            ("put", None, False),
            ("ladder", "load", False),
        ]
        assert pool.pool.charges(*pool.key(job)) == {
            "preemptions": 0,
            "crashes": 0,
            "infrastructure": 0,
            "resource_moves": 0,
        }
        assert pool.newest(job)[0] == "big"
    assert pool.pool.holds("cpu") == {}


def test_load_oom_on_the_top_class_is_unplaceable(pool, tasks):
    """With no larger class, a load out of memory fails the jobs ``Unplaceable``."""
    pool.providers(tasks)
    job = pool.job(tasks.UseOomLoad(model="m", x=1), "top")
    pool.put([job])
    _, code = _run(pool)
    assert code == 0
    [failed] = pool.outcomes()
    assert (failed["outcome"], failed["fault"]["type"]) == ("failed", "Unplaceable")
    assert failed["cause"]["kind"] == "resource"


@pytest.mark.skipif(not POSIX, reason="cgroup accounting is Linux's; faked here")
def test_sibling_oom_does_not_type_scancel(pool, tasks, tmp_path, monkeypatch):
    """An OOM kill a sibling of the same job recorded in the cgroup before this worker started does
    not turn the next killed call into a resource fault: the worker takes its baseline at start. One
    recorded during the call does."""
    from pydra.engine import procgroup

    events = tmp_path / "memory.events"
    events.write_text("low 0\nhigh 0\nmax 0\noom 1\noom_kill 1\n")
    monkeypatch.setattr(procgroup, "_BASELINE", procgroup.CgroupOomBaseline(events))
    killed = pool.job(tasks.Killed(x=1), "scancelled")
    pool.put([killed])
    _, code = _run(pool)
    assert code == 0
    [failed] = pool.outcomes()
    assert failed["outcome"] == "failed" and failed["fault"]["kind"] == "content"

    def oom_now(x):
        events.write_text("low 0\nhigh 0\nmax 0\noom 2\noom_kill 2\n")

    tasks.HOOK["killed"] = oom_now
    own = pool.job(tasks.Killed(x=2), "oom-killed")
    pool.put([own])
    _, code = _run(pool)
    assert code == 0
    [mine] = pool.outcomes(digest=pool.tickets[-1][1])
    assert (mine["fault"]["kind"], mine["fault"]["type"]) == (
        "resource",
        "ResourceExhausted",
    )


def test_lone_call_crashing_twice_is_job_crashed_worker(pool, tasks):
    """A job whose call crashes (`WorkerLost`) alone is retried on a reloaded residency under a fresh
    ``restart`` attempt, uncharged (a crash it survived would be the environment's); lost
    ``crashes_charged`` times in a row it is ``failed``, ``JobCrashedWorker``, a content fault.
    """
    ran = []
    tasks.HOOK["crash"] = ran.append
    job = pool.job(tasks.Crash(x=1), "crash")
    pool.put([job])
    _, code = _run(pool)
    assert code == 0 and ran == [1, 1]
    gens = pool.generations(job)
    assert [(g["cause"], g["charged"]) for g in gens] == [
        ("put", False),
        ("restart", False),
    ]
    [failed] = pool.outcomes()
    assert (failed["outcome"], failed["fault"]["kind"], failed["fault"]["type"]) == (
        "failed",
        "content",
        "JobCrashedWorker",
    )


@pytest.mark.skipif(not POSIX, reason="the task kills its worker by signal")
def test_lone_job_lost_twice_is_job_crashed_worker(pool, tasks, processes, tmp_path):
    """A job that kills every worker that runs it, alone in its call: each taker charges the loss
    (a ``takeover`` generation, ``charged``), and the third claimant fails it ``JobCrashedWorker``
    without running it."""
    marks = tmp_path / "marks"
    marks.mkdir()
    job = pool.job(tasks.Die(x=1, where=str(marks)), "dies")
    digest = pool.put([job])
    for n in (1, 2):
        proc = _hand_worker(pool, processes, tmp_path)
        assert proc.wait(BOUND) == -signal.SIGKILL
        assert len(list(marks.iterdir())) == n
        for d in (pool.path / "queues").glob(f"*/claims/{digest}.lease"):
            _age(d)
    _, code = _run(pool)
    assert code == 0 and len(list(marks.iterdir())) == 2
    gens = pool.generations(job)
    assert [(g["cause"], g["charged"]) for g in gens] == [
        ("put", False),
        ("takeover", True),
        ("takeover", True),
    ]
    [failed] = pool.outcomes()
    assert (failed["fault"]["kind"], failed["fault"]["type"]) == (
        "content",
        "JobCrashedWorker",
    )


def _hand_worker(pool: Pool, processes, tmp_path, cls=None):
    from pydra.workers.pool.worker import class_record

    cls = cls or _class()
    pool.pool.put_class(cls.name, class_record(cls))
    path = tmp_path / f"settings-{len(processes)}.json"
    path.write_text(json.dumps(_settings().to_json()))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pydra.workers.pool.worker",
            "--root",
            str(pool.path),
            "--class",
            cls.name,
            "--settings",
            str(path),
        ],
        start_new_session=POSIX,
    )
    processes.append(proc)
    return proc


def test_expired_prior_attempt_is_a_resource_fault_not_a_crash(pool, tasks):
    """A lost attempt with an ``expired`` outcome was a timeout: the taker moves the job up the
    ladder (``ladder`` at ``call``) uncharged, rather than charging a crash."""
    job = pool.job(tasks.Echo(x=1), "slow")
    digest = pool.put([job], ladder=("big",))
    _classes(pool, _class("big", time=7200), _class(time=60))
    _dead_call(pool, digest, [job], expired=[job])
    _, code = _run(pool, _class(time=60))
    assert code == 0
    gens = pool.generations(job)
    assert [(g["cause"], g["charged"]) for g in gens] == [
        ("put", False),
        ("takeover", False),
        ("ladder", False),
    ]
    assert gens[2]["phase"] == "call" and pool.newest(job)[0] == "big"
    assert pool.pool.charges(*pool.key(job))["crashes"] == 0


@pytest.mark.parametrize("budget, charged", [(1, False), (0, True)])
def test_a_drained_call_is_placed_again_uncharged_within_the_budget(
    pool, tasks, budget, charged
):
    """A call its worker's drain cut short (a scheduler preemption) is an infrastructure event: the
    taker places the job again under a fresh attempt, uncharged, while its preemptions stay within
    ``preemption_budget``; past it the cut is charged as a crash."""
    job = pool.job(tasks.Echo(x=1), "preempted")
    digest = pool.put([job])
    _dead_call(pool, digest, [job], drained=[job])
    _, code = _run(pool, settings=_settings(preemption_budget=budget))
    assert code == 0
    gens = pool.generations(job)
    assert [(g["cause"], g["charged"]) for g in gens[:2]] == [
        ("put", False),
        ("takeover", charged),
    ]
    charges = pool.pool.charges(*pool.key(job))
    assert charges["preemptions"] == 1 and charges["crashes"] == int(charged)


def test_timeout_kill_spares_gathered_jobs(pool, tasks):
    """A worker holding three gathered one-job tickets killed by one call's timeout: only that call's
    job is typed (a timeout, moved up the ladder); the two it had claimed but not called are
    re-placed uncharged and run."""
    jobs = [pool.job(tasks.Echo(x=i), f"g{i}") for i in range(3)]
    digests = [pool.put([j], ladder=("big",)) for j in jobs]
    _classes(pool, _class("big", time=7200), _class(time=60))
    for n, (digest, job) in enumerate(zip(digests, jobs)):
        _dead_call(
            pool,
            digest,
            [job] if n == 0 else [],
            expired=[job] if n == 0 else (),
            call=[job] if n == 0 else (),
        )
    _, code = _run(pool, _class(time=60))
    assert code == 0
    timed_out, *spared = jobs
    assert [g["cause"] for g in pool.generations(timed_out)] == [
        "put",
        "takeover",
        "ladder",
    ]
    for job in spared:
        gens = pool.generations(job)
        assert all(g["charged"] is False for g in gens)
        assert pool.newest(job)[1]["outcome"] == "done"


def test_expired_with_committed_result_settles_done(pool, tasks):
    """A call that published during its grace period, then expired: the taker's store lookup
    settles ``done`` (served), and nothing moves up the ladder."""
    job = pool.job(tasks.Echo(x=7), "late")
    digest = pool.put([job], ladder=("big",))
    _classes(pool, _class("big", time=7200))
    claim_attempt = pool.generations(job)[0]["attempt"]
    _dead_call(pool, digest, [job], expired=[job])
    job.attempt = claim_attempt
    job.run()  # what the expired call published before its worker was killed
    ran = []
    tasks.HOOK["echo"] = ran.append
    _, code = _run(pool)
    assert code == 0 and ran == []
    assert [g["cause"] for g in pool.generations(job)] == ["put", "takeover"]
    done = pool.newest(job)[1]
    assert (done["outcome"], done["served"]) == ("done", True)


def test_resource_moves_counted_from_generations(pool, tasks):
    """The ``resource_moves`` count is read from the job's ``ladder`` generations at ``call``, so a
    second taker (and a resumed coordinator) sees the moves an earlier worker made: with the budget
    spent the next out-of-memory call is ``ResourceExhausted``."""
    from pydra.workers.pool.faults import FaultBudgets

    job = pool.job(tasks.Oom(x=1), "oom")
    pool.put([job], ladder=("a", "b", "c"))
    for name in ("a", "b", "c"):
        _classes(pool, _class(name))
    budgets = FaultBudgets(resource_moves=2)
    for name in ("cpu", "a"):
        _, code = _run(pool, _class(name), faults=budgets)  # a fresh worker each time
        assert code == 0
    assert pool.pool.charges(*pool.key(job))["resource_moves"] == 2
    pool.tickets.append(("b", pool.pool.newest(*pool.key(job))[1]["ticket"]))
    _, code = _run(pool, _class("b"), faults=budgets)
    assert code == 0
    [failed] = pool.outcomes("b")
    assert (failed["outcome"], failed["fault"]["type"]) == (
        "failed",
        "ResourceExhausted",
    )


@pytest.mark.parametrize(
    "ladder,moved",
    [
        ((("same", 60, None),), None),
        ((("same", 60, None), ("longer", 600, None)), "longer"),
    ],
)
def test_timeout_moves_only_to_a_longer_or_other_gpu_type(pool, tasks, ladder, moved):
    """A timeout does not move to a class with the same ``time`` and ``gpu_type``: it skips to a
    longer one, or is ``ResourceExhausted`` when none is left."""
    job = pool.job(tasks.Echo(x=1), "slow")
    digest = pool.put([job], ladder=tuple(name for name, _, _ in ladder))
    _classes(pool, _class(time=60), *(_class(n, time=t) for n, t, _ in ladder))
    _dead_call(pool, digest, [job], expired=[job])
    _, code = _run(pool, _class(time=60))
    assert code == 0
    queue, found = pool.newest(job)
    if moved is None:
        assert (found["outcome"], found["fault"]["type"]) == (
            "failed",
            "ResourceExhausted",
        )
        assert found["cause"]["reason"] == "timeout"
    else:
        assert queue == moved and found is None


def test_gap_reported_once_per_type(pool, tasks, caplog):
    """An unclassified exception from the pool's machinery is an infrastructure gap, reported once
    per exception type however often it recurs."""
    pool.providers(tasks)
    pool.pool.take_epoch()
    jobs = [pool.job(tasks.UseBroken(model="m", x=i), f"b{i}") for i in range(2)]
    pool.put(jobs)
    with caplog.at_level(logging.WARNING, logger="pydra.pool"):
        _, code = _run(pool)
    assert code == 0 and tasks.LOADS == ["broken"] * 3
    gaps = [r for r in caplog.records if "unclassified" in r.getMessage()]
    assert len(gaps) == 1 and "FileNotFoundError" in gaps[0].getMessage()


# -- holds -------------------------------------------------------------------------------------


def test_hold_per_residency_and_epoch(pool, tasks):
    """After ``infrastructure_replacements`` load faults of one residency in a coordinator epoch the
    residency is held on its queue (``held-<epoch>-<label>``): the faulting ticket's jobs are
    ``held`` with a reason naming the residency and the latest cause, a second ticket of the
    residency is not claimed, and the count starts again in a newer epoch, which the hold does not
    bind."""
    from pydra.workers.pool.storage import Ticket

    pool.providers(tasks)
    first = pool.pool.take_epoch()
    a = pool.job(tasks.UseBroken(model="m", x=1), "a")
    b = pool.job(tasks.UseBroken(model="m", x=2), "b")
    da, db = pool.put([a]), pool.put([b])
    label = Ticket(
        queue="cpu", node="n", jobs=[pool.entry(a)], residency=_residency(a)
    ).residency_label
    worker, code = _run(pool)
    assert code == 0 and worker.claims == [1, 1, 1]
    assert tasks.LOADS == ["broken"] * 3
    assert pool.pool.holds("cpu", epoch=first) == {label: first}
    [held] = pool.outcomes(digest=da)
    assert held["outcome"] == "held"
    assert label in held["reason"] and "FileNotFoundError" in held["reason"]
    assert f"epoch {first}" in held["reason"]
    assert pool.outcomes(digest=db) == []
    assert [g["cause"] for g in pool.generations(a)] == ["put", "restart", "restart"]
    assert [g["cause"] for g in pool.generations(b)] == ["put"]

    second = pool.pool.take_epoch()
    assert pool.pool.holds("cpu", epoch=second) == {}
    tasks.LOADS.clear()
    worker, code = _run(pool)
    assert code == 0 and tasks.LOADS == ["broken"] * 3  # the count starts again
    assert pool.pool.holds("cpu", epoch=second) == {label: second}
    [held_b] = pool.outcomes(digest=db)
    assert held_b["outcome"] == "held" and f"epoch {second}" in held_b["reason"]


def test_call_infrastructure_faults_hold_the_residency(pool, tasks):
    """Under a classifier that types every exception infrastructure, a job that always fails is
    requeued under ``restart`` generations at the ``call`` phase and, at
    ``infrastructure_replacements``, its residency is held and the job ``held``."""
    pool.pool.take_epoch()
    job = pool.job(tasks.Killed(x=1), "infra")
    pool.put([job])
    _, code = _run(pool, classify=tasks.infrastructure)
    assert code == 0
    assert [(g["cause"], g["phase"]) for g in pool.generations(job)] == [
        ("put", None),
        ("restart", "call"),
        ("restart", "call"),
    ]
    [held] = pool.outcomes()
    assert held["outcome"] == "held" and "3 infrastructure faults" in held["reason"]
    assert held["fault"]["type"] == "ChildProcessError"


# -- the coordinator ---------------------------------------------------------------------------


def _submitter(tmp_path):
    return Submitter(cache_root=tmp_path / "store", worker="debug")


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


def _drive(sub, task, executor, rerun=False):
    from pydra.engine.drivers import drive
    from pydra.engine.runner import GraphRunner
    from pydra.engine.submitter import run_incomplete

    runner = GraphRunner(
        task, sub.store, on_error="raise_at_end", submitter=sub, rerun=rerun
    )
    try:
        asyncio.run(
            drive(
                runner,
                {"pool": executor},
                lambda job: "pool",
                on_error="raise_at_end",
            )
        )
    finally:
        executor.close()  # its owner closes it: drive never does
    if runner.errors:
        raise AssertionError("\n".join(runner.errors))
    if runner.not_run:
        raise run_incomplete(runner.job or task, runner.not_run)
    return runner


def test_held_residency_reports_its_jobs_not_run(tmp_path, tasks, pools):
    """A residency a worker held: the coordinator reports every job of it ``NotRun(held=True)``, the
    faulting ticket's and the unclaimed one's alike, naming the residency and the cause; the run ends
    `RunIncomplete`, and nothing is ``failed``."""
    from pydra.engine.submitter import RunIncomplete

    sub = _submitter(tmp_path)
    sub.providers = (tasks.Broken(),)
    pool = _warm(tmp_path, pools, sub, [_class(quantum_jobs=1)])
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Brokens(xs=[1, 2]), pool)
    not_run = info.value.not_run
    assert len(not_run) == 2 and all(o.held for _, o in not_run)
    assert all(
        "held" in o.reason and "FileNotFoundError" in o.reason for _, o in not_run
    )


def test_pool_passes_faults_and_classify_to_its_workers(tmp_path, tasks, pools):
    """`WarmPool` hands its fault budgets and classifier to the workers it starts, and warns of
    neither: ``content_attempts=2`` retries a task that fails once, and a classifier passed by
    reference reaches the worker (a call's exception, content by default, holds the residency under
    a classifier that types it infrastructure). A `DirectoryCalibration` is taken without a warning
    and its directory is handed to the workers; any other calibration warns that workers write
    to the pool's own directory, the default."""
    from pydra.workers.pool.faults import FaultBudgets

    sub = _submitter(tmp_path)
    marker = tmp_path / "failed-once"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pool = _warm(tmp_path, pools, sub, faults=FaultBudgets(content_attempts=2))
    runner = _drive(sub, tasks.Flaky(marker=str(marker)), pool)
    assert runner.done and not runner.failed

    sub2 = Submitter(cache_root=tmp_path / "store2", worker="debug")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        strict = _warm(
            tmp_path / "strict",
            pools,
            sub2,
            [_class(quantum_jobs=1)],
            classify=tasks.infrastructure,
            faults=FaultBudgets(infrastructure_replacements=1),
        )
    from pydra.engine.submitter import RunIncomplete

    with pytest.raises(RunIncomplete) as info:
        _drive(sub2, tasks.Killeds(xs=[1]), strict)
    [(_, held)] = info.value.not_run  # the default classifier would fail it as content
    assert held.held and "1 infrastructure faults" in held.reason

    from pydra.workers.pool.calibration import DirectoryCalibration

    given = DirectoryCalibration(tmp_path / "given")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibrated = _warm(tmp_path / "calibrated", pools, sub, calibration=given)
    assert calibrated.calibration is given

    def home(warm):
        record = warm.root / "coordinator" / str(warm.epoch) / "faults.json"
        return json.loads(record.read_text())["calibration"]

    assert home(calibrated) == str(given.root.absolute())
    assert home(pool) == str(tmp_path / "pool" / "calibration")
    with pytest.warns(UserWarning, match="workers write"):
        _warm(tmp_path / "other", pools, sub, calibration=object())


# -- resumed runs -------------------------------------------------------------------------------


def test_resumed_runner_attempt_zero_no_collision(tmp_path, tasks, pools):
    """A resubmitted run's fresh `GraphRunner` numbers its attempts from 0 again; the pool's attempts
    are ``<pool id>.<g>``, so the rerun job gets a new generation and a work directory distinct from
    the first run's."""
    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    first = _warm(tmp_path, pools, sub)
    _drive(sub, tasks.Marks(xs=[1], where=str(marks)), first)
    second = _warm(tmp_path, pools, sub)
    runner = _drive(sub, tasks.Marks(xs=[1], where=str(marks)), second, rerun=True)
    assert runner.done and len(list(marks.iterdir())) == 2
    from pydra.workers.pool.storage import PoolRoot

    with PoolRoot.open(tmp_path / "pool", LEASE, create=False, probe=False) as raw:
        follow = [p for p in (tmp_path / "pool" / "follow").iterdir() if p.is_dir()]
        [job_dir] = follow
        checksum, fingerprint = job_dir.name.rsplit("-", 1)
        gens = [r for _, r in raw.generations(checksum, fingerprint)]
    assert [g["cause"] for g in gens] == ["put", "put"]
    attempts = [g["attempt"] for g in gens]
    assert len(set(attempts)) == 2
    dirs = set()
    for attempt in attempts:
        started = json.loads(
            (job_dir / f"{attempt.rsplit('.', 1)[1]}.started").read_text()
        )
        dirs.add((attempt, started["worker"]))
    assert len(dirs) == 2


def test_refused_job_replaced_on_resume(tmp_path, tasks, pools):
    """A job a worker refused (``withdrawn``, reason ``refused``) whose coordinator died before
    re-placing it is re-placed one class up the refused ticket's ladder by the resumed coordinator,
    under a fresh attempt, and runs there."""
    from pydra.workers.pool.substrate import ExternalSubstrate

    marks = tmp_path / "marks"
    marks.mkdir()
    sub = _submitter(tmp_path)
    small, large = _class("small", mem="4G"), _class("large", mem="8G")
    dead = _warm(tmp_path, pools, sub, [small, large], substrate=ExternalSubstrate())
    from pydra.engine.runner import GraphRunner

    runner = GraphRunner(
        tasks.BigMark(x=1, where=str(marks)),
        sub.store,
        on_error="raise_at_end",
        submitter=sub,
    )
    dead.bind(lambda job, outcome: None)
    dead.submit(runner.ready().dispatch)
    from pydra.workers.pool.storage import queue_label

    tickets = tmp_path / "pool" / "queues" / queue_label("small") / "tickets"
    _wait(lambda: tickets.exists() and any(tickets.glob("*.json")), "the ticket")
    [path] = tickets.glob("*.json")
    digest = path.stem
    claim = dead.pool.take("small", digest)
    entry = claim.ticket.jobs[0]
    claim.withdraw(entry, "refused")
    claim.release()
    assert json.loads(path.read_text())["ladder"] == ["large"]
    dead.close()  # the coordinator dies before its follower re-places the job

    resumed = _warm(tmp_path, pools, sub, [small, large])
    done = _drive(sub, tasks.BigMark(x=1, where=str(marks)), resumed)
    assert done.done and not done.failed and len(list(marks.iterdir())) == 1
    from pydra.workers.pool.storage import PoolRoot

    with PoolRoot.open(tmp_path / "pool", LEASE, create=False, probe=False) as raw:
        gens = [r for _, r in raw.generations(entry.checksum, entry.fingerprint)]
    assert [g["queue"] for g in gens] == [queue_label("small"), queue_label("large")]
    assert gens[0]["attempt"] != gens[1]["attempt"]


# -- holds, storage errors and reloads -------------------------------------------------------------------------------


def test_gathered_tickets_do_not_run_after_hold(pool, tasks):
    """A warm worker that gathered more one-job tickets of a residency than the budget runs no call
    after it holds the residency: the rest are settled ``held`` unrun."""
    ran = []
    tasks.HOOK["killed"] = ran.append
    pool.pool.take_epoch()
    digests = [pool.put([pool.job(tasks.Killed(x=i), f"k{i}")]) for i in range(6)]
    _, code = _run(pool, _class(quantum_jobs=8), classify=tasks.infrastructure)
    assert code == 0
    assert len(ran) == 3, f"calls run: {ran} (budget 3)"
    kinds = [o["outcome"] for d in digests for o in pool.outcomes(digest=d)]
    assert kinds.count("held") >= 1 and "failed" not in kinds
    faults = pool.path / "queues"
    assert len([p for p in faults.glob("*/faults/*/*")]) == 3


def test_requeued_ticket_reclaimed_while_gathering_is_held(pool, tasks):
    """A requeued ticket is claimed again by the gathering worker that released it, until the hold."""
    pool.pool.take_epoch()
    job = pool.job(tasks.Killed(x=1), "infra")
    pool.put([job])
    _, code = _run(pool, _class(quantum_jobs=4), classify=tasks.infrastructure)
    assert code == 0
    assert [o["outcome"] for o in pool.outcomes()] == ["held"]


def test_storage_error_in_a_call_is_infrastructure_not_content(
    pool, tasks, monkeypatch
):
    """An `OSError` (``ESTALE``) from the pool's machinery around a call is an infrastructure fault
    of the residency (requeued), never the job's content fault, and the job runs again.
    """
    from pydra.workers.pool.storage import PoolRoot

    real = PoolRoot.write_call
    state = {"n": 0}

    def flaky(self, *args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise OSError(116, "Stale file handle")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(PoolRoot, "write_call", flaky)
    job = pool.job(tasks.Echo(x=1), "estale")
    pool.put([job])
    _, code = _run(pool)
    assert code == 0
    gens = pool.generations(job)
    assert [(g["cause"], g["phase"]) for g in gens][:2] == [
        ("put", None),
        ("restart", "call"),
    ]
    assert pool.newest(job)[1]["outcome"] == "done"


def test_keyboard_interrupt_from_the_machinery_propagates(pool, tasks, monkeypatch):
    """A `KeyboardInterrupt` raised by the pool's machinery around a call is not typed: it leaves the
    worker and writes no outcome."""
    from pydra.workers.pool.storage import PoolRoot

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(PoolRoot, "write_call", interrupted)
    pool.put([pool.job(tasks.Echo(x=1), "a")])
    with pytest.raises(KeyboardInterrupt):
        _run(pool)
    assert pool.outcomes() == []


def test_a_reload_that_runs_out_of_memory_is_a_load(pool, tasks):
    """After one job's call runs out of memory the residency is reloaded for the next job, and that reload
    running out of memory is a load fault: the remaining jobs move up at the ``load`` phase,
    uncharged, rather than as a call fault of one job."""
    pool.providers(tasks)
    jobs = [pool.job(tasks.OomOnFlaky(model="m", x=i), f"j{i}") for i in (1, 2, 3)]
    pool.put(jobs, ladder=("big",))
    _classes(pool, _class("big"))
    _, code = _run(pool)
    assert code == 0
    assert tasks.LOADS == ["flakyload", "flakyload"]
    first, *rest = jobs
    assert [(g["cause"], g["phase"]) for g in pool.generations(first)][-1] == (
        "ladder",
        "call",
    )
    for job in rest:
        assert [(g["cause"], g["phase"]) for g in pool.generations(job)] == [
            ("put", None),
            ("ladder", "load"),
        ]
        assert pool.pool.charges(*pool.key(job))["resource_moves"] == 0


def test_a_survived_crash_counts_against_the_residency(pool, tasks, tmp_path):
    """A crash the job survives on a reloaded residency is an infrastructure fault of the residency,
    charged to no job; at the budget the residency is held and its other tickets are not claimed.
    """
    from pydra.workers.pool.faults import FaultBudgets

    pool.pool.take_epoch()
    survivor = pool.job(tasks.CrashOnce(marker=str(tmp_path / "crashed")), "survivor")
    other = pool.job(tasks.Echo(x=5), "other")
    pool.put([survivor])
    second = pool.put([other])
    _, code = _run(
        pool, _class(quantum_jobs=1), faults=FaultBudgets(infrastructure_replacements=1)
    )
    assert code == 0
    assert all(g["charged"] is False for g in pool.generations(survivor))
    assert pool.newest(survivor)[1]["outcome"] == "done"
    assert len(pool.pool.holds("cpu")) == 1
    assert pool.outcomes(digest=second) == []  # not claimed under the hold
    [record] = [
        json.loads(p.read_text()) for p in pool.path.glob("queues/*/faults/*/*")
    ]
    assert record["fault"]["reason"] == "crash" and record["ticket"] != second


def test_mixed_queue_holds_only_the_faulting_residency(tmp_path, tasks, pools):
    """``residency="ignore"`` jobs share the class's queue with plain ones; holding the provider's
    residency there reports only its jobs ``NotRun(held=True)``: the plain jobs run."""
    from pydra.engine.submitter import RunIncomplete

    sub = _submitter(tmp_path)
    sub.providers = (tasks.Broken(),)
    pool = _warm(tmp_path, pools, sub)
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Mixed(xs=[1, 2]), pool)
    names = sorted(job.name for job, _ in info.value.not_run)
    assert all(o.held for _, o in info.value.not_run)
    assert len(names) == 2 and all("B" in n for n in names)


def test_budgets_and_classifier_reach_hand_started_workers(
    tmp_path, tasks, pools, processes
):
    """On `ExternalSubstrate` the pool's budgets and classifier reference are written into the pool
    root, so a worker started by hand with no options uses them."""
    from pydra.engine.submitter import RunIncomplete
    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.substrate import ExternalSubstrate

    sub = _submitter(tmp_path)
    pool = _warm(
        tmp_path,
        pools,
        sub,
        substrate=ExternalSubstrate(),
        faults=FaultBudgets(content_attempts=2, infrastructure_replacements=1),
        classify=tasks.infrastructure,
    )
    worker = _root_worker(pool.root, processes, tmp_path)
    with pytest.raises(RunIncomplete) as info:
        _drive(sub, tasks.Killeds(xs=[1]), pool)
    [(_, held)] = info.value.not_run  # the default classifier would fail it as content
    assert held.held and "1 infrastructure faults" in held.reason
    worker.wait(BOUND)

    sub2 = Submitter(cache_root=tmp_path / "store2", worker="debug")
    plain = _warm(
        tmp_path / "plain",
        pools,
        sub2,
        substrate=ExternalSubstrate(),
        faults=FaultBudgets(content_attempts=2),
    )
    _root_worker(plain.root, processes, tmp_path)
    marker = tmp_path / "failed-once"
    runner = _drive(sub2, tasks.Flakys(markers=[str(marker)]), plain)
    assert runner.done and not runner.failed


def _root_worker(root, processes, tmp_path):
    """``python -m pydra.workers.pool.worker`` on ``root`` with no fault options, by hand."""
    path = tmp_path / f"root-settings-{len(processes)}.json"
    path.write_text(json.dumps(_settings(idle_polls=40).to_json()))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pydra.workers.pool.worker",
            "--root",
            str(root),
            "--class",
            "cpu",
            "--settings",
            str(path),
        ],
        start_new_session=POSIX,
    )
    processes.append(proc)
    return proc


def test_a_held_residency_does_not_skip_its_queue(pool, tasks):
    """A worker that has read a ticket of a residency later held still claims another residency's
    ticket on the same queue (``residency="ignore"`` jobs share the class's queue)."""
    from pydra.workers.pool.storage import Ticket

    pool.providers(tasks)
    epoch = pool.pool.take_epoch()
    provided = pool.job(tasks.UseBroken(model="m", x=1), "provided")
    plain = pool.job(tasks.Echo(x=2), "plain")
    pool.put([provided])
    other = pool.put([plain])
    first = pool.pool.claim("cpu")  # reads the provided residency's ticket
    label = Ticket(
        queue="cpu",
        node="n",
        jobs=[pool.entry(provided)],
        residency=_residency(provided),
    ).residency_label
    first.release()
    pool.pool.hold("cpu", epoch, label)
    claim = pool.pool.claim("cpu")
    assert claim is not None and claim.digest == other


# -- holds across epochs -------------------------------------------------------------------------------


def _held_worker(pool, tasks):
    """A worker that holds the empty residency of ``cpu`` in the newest epoch (one job requeued until
    the budget) and keeps polling."""
    import threading

    pool.put([pool.job(tasks.Killed(x=0), "k0")])
    worker = _worker(
        pool,
        _class(quantum_jobs=1),
        classify=tasks.infrastructure,
        settings=_settings(idle_polls=4000),
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    _wait(lambda: pool.pool.holds("cpu"), "the hold")
    return worker, thread


def test_local_hold_does_not_outlive_its_epoch(pool, tasks):
    """A residency this worker held in one coordinator epoch does not bind it
    in a newer epoch; a later run's job of that residency runs."""
    pool.pool.take_epoch()
    worker, thread = _held_worker(pool, tasks)
    try:
        pool.pool.take_epoch()  # a later run: the hold no longer binds
        later = pool.put([pool.job(tasks.Echo(x=7), "later")])
        _wait(lambda: pool.outcomes(digest=later), "the later job's outcome")
    finally:
        worker.stop.set()
        thread.join(BOUND)
    [outcome] = pool.outcomes(digest=later)
    assert outcome["outcome"] == "done"


def test_released_hold_releases_the_worker(pool, tasks):
    """``pydracli pool release-hold`` lifts the storage hold, and the worker that wrote it honours
    the lift: a later job of the residency runs in the same epoch."""
    pool.pool.take_epoch()
    worker, thread = _held_worker(pool, tasks)
    try:
        assert pool.pool.release_holds("cpu") == 1
        later = pool.put([pool.job(tasks.Echo(x=8), "released")])
        _wait(lambda: pool.outcomes(digest=later), "the released job's outcome")
    finally:
        worker.stop.set()
        thread.join(BOUND)
    [outcome] = pool.outcomes(digest=later)
    assert outcome["outcome"] == "done"


def test_faults_reread_when_the_epoch_changes(pool, tasks, tmp_path):
    """A running worker reads the budgets again when a newer coordinator epoch writes them, even
    when it saw the epoch before the budgets were written: a job of the new run is retried under
    its ``content_attempts=2``."""
    import threading

    from pydra.workers.pool.faults import FaultBudgets

    first = pool.pool.take_epoch()
    pool.pool.write_faults(
        first, {"budgets": FaultBudgets().to_json(), "classify": None}
    )
    worker = _worker(pool, settings=_settings(idle_polls=4000))
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        _wait(lambda: worker.token is not None and worker.cls is not None, "the start")
        second = pool.pool.take_epoch()
        # the worker sees the new epoch before its coordinator writes the budgets (the window
        # between take_epoch and write_faults), and must still read them once written
        _wait(lambda: worker._epoch == second, "the worker to see the new epoch")
        pool.pool.write_faults(
            second,
            {"budgets": FaultBudgets(content_attempts=2).to_json(), "classify": None},
        )
        job = pool.job(tasks.FailOnce(marker=str(tmp_path / "once")), "flaky")
        digest = pool.put([job])
        _wait(
            lambda: any(
                o["outcome"] != "expired" for o in pool.outcomes(digest=digest)
            ),
            "the job's outcome",
        )
    finally:
        worker.stop.set()
        thread.join(BOUND)
    [outcome] = pool.outcomes(digest=digest)
    assert outcome["outcome"] == "done"


@pytest.mark.parametrize("reference", ["no_such_module_xyz:classify", "os:sep"])
def test_unusable_classifier_stops_the_worker_before_any_claim(pool, tasks, reference):
    """A classifier reference the worker cannot import, or that names something not callable, is an
    error: the worker exits non-zero before writing a ready mark or claiming anything, so its class
    breaks after ``start_failures`` and its jobs are held."""
    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.worker import EXIT_FAULTS

    epoch = pool.pool.take_epoch()
    pool.pool.write_faults(
        epoch, {"budgets": FaultBudgets().to_json(), "classify": reference}
    )
    digest = pool.put([pool.job(tasks.Echo(x=1), "unclaimed")])
    worker, code = _run(pool)
    assert code == EXIT_FAULTS
    assert (
        worker.token is None
        or pool.pool.worker_record(worker.token, "ready.json") is None
    )
    assert pool.outcomes(digest=digest) == []
    assert pool.pool.claim("cpu") is not None  # never claimed: the ticket is free


def _partial_classifier():
    import functools

    from pydra.workers.pool.faults import classify

    return functools.partial(classify)


class _Bound:
    def classify(self, exc, phase):
        from pydra.workers.pool.faults import classify

        return classify(exc, phase)


@pytest.mark.parametrize(
    "make",
    [_partial_classifier, lambda: _Bound().classify, lambda: (lambda exc, phase: None)],
    ids=["partial", "bound-method", "lambda"],
)
def test_pool_refuses_a_classifier_workers_cannot_import(tmp_path, tasks, pools, make):
    """`WarmPool` resolves the classifier's reference and refuses one that does not round-trip (a
    ``partial``, a bound method, a lambda) at construction, before anything is written.
    """
    sub = _submitter(tmp_path)
    with pytest.raises(ValueError, match="classif"):
        _warm(tmp_path, pools, sub, classify=make())
    assert not (tmp_path / "pool" / "coordinator").exists() or not list(
        (tmp_path / "pool" / "coordinator").glob("*/faults.json")
    )


def test_a_held_routed_queue_is_not_walked(pool, tasks, monkeypatch):
    """A routed queue (``<class>#<digest>``) carries one residency (RFC 0005 section 4.2), so a hold
    on it skips the queue without reading a slot."""
    from pydra.workers.pool.storage import PoolRoot

    epoch = pool.pool.take_epoch()
    queue = "cpu#" + "ab" * 8
    residency = ({"kind": "provider", "name": "m"},)
    for i in range(3):
        digest = pool.put(
            [pool.job(tasks.Echo(x=i), f"r{i}")], queue=queue, residency=residency
        )
    label = pool.pool.ticket(queue, digest).residency_label
    pool.pool.hold(queue, epoch, label)
    walked = []
    real = PoolRoot._target
    monkeypatch.setattr(
        PoolRoot, "_target", lambda self, *a: walked.append(a) or real(self, *a)
    )
    assert pool.pool.claim(queue) is None and walked == []


@pytest.mark.parametrize("taken", ["before", "after"])
def test_a_claim_made_before_the_new_epochs_budgets_waits_for_them(
    pool, tasks, tmp_path, taken
):
    """The worker refreshes, then the new coordinator writes its budgets and puts a ticket, then the
    worker claims it, all in one pass, with the new epoch taken before the refresh or after it:
    the job runs under the new epoch's budgets (``content_attempts=2``), never the older one's.
    """
    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.worker import Worker

    first = pool.pool.take_epoch()
    pool.pool.write_faults(
        first, {"budgets": FaultBudgets().to_json(), "classify": None}
    )
    epochs = []
    if taken == "before":
        epochs.append(pool.pool.take_epoch())  # its budgets not written yet
    job = pool.job(tasks.FailOnce(marker=str(tmp_path / "once")), "flaky")
    real_next = Worker._next
    done = []

    def interleaved(self, prefer, left):
        if not done:  # after this pass's refresh, before its claim
            done.append(True)
            if not epochs:
                epochs.append(pool.pool.take_epoch())
            pool.pool.write_faults(
                epochs[0],
                {
                    "budgets": FaultBudgets(content_attempts=2).to_json(),
                    "classify": None,
                },
            )
            pool.put([job])
        return real_next(self, prefer, left)

    import unittest.mock

    with unittest.mock.patch.object(Worker, "_next", interleaved):
        worker, code = _run(pool)
    assert code == 0 and done
    [outcome] = pool.outcomes()
    assert outcome["outcome"] == "done"
    assert worker.faults == FaultBudgets(content_attempts=2)


def test_a_claim_is_released_unrun_until_the_new_epochs_budgets_exist(
    pool, tasks, tmp_path
):
    """A ticket claimed while the newest epoch's coordinator has not yet written its budgets is
    released unrun (no attempt started); once they are written it runs under them."""
    import unittest.mock

    from pydra.workers.pool.faults import FaultBudgets
    from pydra.workers.pool.worker import Worker

    first = pool.pool.take_epoch()
    pool.pool.write_faults(
        first, {"budgets": FaultBudgets().to_json(), "classify": None}
    )
    second = pool.pool.take_epoch()
    job = pool.job(tasks.FailOnce(marker=str(tmp_path / "once")), "flaky")
    real_next = Worker._next
    calls = []

    def staged(self, prefer, left):
        calls.append(len(pool.generations(job)) if calls else 0)
        if len(calls) == 1:
            pool.put([job])  # claimed before the budgets exist
        elif len(calls) == 2:
            assert pool.outcomes() == [] and len(pool.generations(job)) == 1
            pool.pool.write_faults(
                second,
                {
                    "budgets": FaultBudgets(content_attempts=2).to_json(),
                    "classify": None,
                },
            )
        return real_next(self, prefer, left)

    with unittest.mock.patch.object(Worker, "_next", staged):
        _, code = _run(pool)
    assert code == 0 and len(calls) >= 2
    [outcome] = pool.outcomes()
    assert outcome["outcome"] == "done"


# -- which memory an out-of-memory load ran out of -----------------------------------------------


def _named(qualname: str, base: type) -> type:
    module, _, name = qualname.rpartition(".")
    kind = type(name, (base,), {})
    kind.__module__ = module
    return kind


@pytest.mark.parametrize(
    "qualname, base, gpu_class, resource",
    [
        # cupy's device OOM subclasses MemoryError: it is still the device's
        ("cupy.cuda.memory.OutOfMemoryError", MemoryError, True, "device"),
        ("torch.cuda.OutOfMemoryError", RuntimeError, True, "device"),
        ("torch.OutOfMemoryError", RuntimeError, True, "device"),
        ("builtins.MemoryError", MemoryError, True, "host"),
        ("cupy.cuda.memory.OutOfMemoryError", MemoryError, False, "host"),
    ],
)
def test_out_of_memory_resource(qualname, base, gpu_class, resource):
    """A device out-of-memory type, by qualified name anywhere in the chain, is the device's on a GPU
    class, before the generic `MemoryError` rule; a plain `MemoryError` is the host's.
    """
    from pydra.workers.pool.faults import classify, oom_resource

    kind = _named(qualname, base) if qualname != "builtins.MemoryError" else MemoryError
    exc = kind("out of memory")
    assert classify(exc, "load").reason == "oom"
    assert oom_resource(exc, gpu_class) == resource
    try:
        try:
            raise exc
        except BaseException as inner:
            raise RuntimeError("loading failed") from inner
    except RuntimeError as wrapped:
        assert oom_resource(wrapped, gpu_class) == resource


def test_out_of_memory_resource_follows_the_first_out_of_memory_link():
    """The first out-of-memory link in chain order decides: a CUDA OOM caught by a loader that falls
    back to the host and then runs out of host memory is the host's; the reverse is the
    device's."""
    from pydra.workers.pool.faults import oom_resource

    cuda = _named("torch.cuda.OutOfMemoryError", RuntimeError)
    try:
        try:
            raise cuda("CUDA out of memory")
        except RuntimeError:
            raise MemoryError("host memory exhausted while offloading")
    except MemoryError as fallback:
        assert oom_resource(fallback, True) == "host"
    try:
        try:
            raise MemoryError("host memory")
        except MemoryError as host:
            raise cuda("CUDA out of memory") from host
    except RuntimeError as device:
        assert oom_resource(device, True) == "device"


@pytest.mark.parametrize(
    "message",
    [
        "Out of memory while trying to allocate 2.0GiB on device cuda:0",
        "Out of memory on GPU_0",
        "Out of memory while trying to allocate 2GiB",
        "RESOURCE_EXHAUSTED: Out of memory allocating 4096 bytes",
    ],
)
def test_jax_resource_exhausted_is_judged_by_the_class_it_ran_on(message):
    """jax's ResourceExhaustedError is out of memory whatever its message says; on a GPU class it
    is the device's, on a CPU class the host's (the message is never read)."""
    from pydra.workers.pool.faults import classify, oom_resource

    exc = _named("jax.errors.ResourceExhaustedError", RuntimeError)(message)
    assert classify(exc, "load").reason == "oom"
    assert oom_resource(exc, True) == "device"
    assert oom_resource(exc, False) == "host"
