"""The external driver model of RFC 0002 (sections 3.1 and 3.5, R7): every job the runner
completes itself is reported in ``Round.settled``, every element blocked by a failure in
``Round.unrunnable``; same-checksum sharing is the runner's; ``NotRun`` outcomes, attempt
ids, ``upstream`` and the node-granular ``limit``"""

import asyncio
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

from pydra.compose import python, workflow
from pydra.engine.drivers import drive, drive_inline
from pydra.engine.executor import Handle
from pydra.engine.hooks import TaskHooks
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.runner import (
    Failed,
    GraphRunner,
    NotRun,
    Settled,
    Unrunnable,
)
from pydra.engine.store import SUCCESS_DIR, PickleStore, ResultStore
from pydra.engine.submitter import RunIncomplete, Submitter, WorkflowError
from pydra.engine.tests.utils import FunAddVar, ListSum
from pydra.workers import base

# -- an in-memory store and a driver that computes outputs itself ------------------------------------


class HeldLease:
    """A lease that is always held"""

    lost = None

    def __init__(self, checksum):
        self.checksum = checksum

    def try_take(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


class MemoryStore(ResultStore):
    """A `ResultStore` kept in a dict, as an external driver's adapter might be"""

    def __init__(self, root):
        self.root = Path(root)
        self.results = {}
        self.lookups = []

    def cache_dir(self, checksum, context):
        return self.root / checksum

    def lookup(self, checksum, context, task=None, stored=None):
        self.lookups.append(checksum)
        return self.results.get(checksum)

    def stored_contexts(self, checksum):
        return iter(())

    def lockfile(self, checksum, context):
        return self.root / f"{checksum}.lock"

    def acquire(self, checksum, context):
        return HeldLease(checksum)

    def commit(self, lease, result, job=None):
        if not result.errored:
            self.results[lease.checksum] = result


def compute(job) -> Result:
    """The outcome of a job, computed by the driver without the store"""
    task = job.task
    if isinstance(task, ListSum):
        out = sum(task.x)
    else:
        out = task.a + task.b
    return Result(cache_dir=job.cache_dir, outputs=task.Outputs(out=out), task=task)


def drive_by_hand(runner, outcome=compute, rounds=200):
    """The loop of RFC 0002 section 3.5, completing each job as it is handed out"""
    sent = []
    for _ in range(rounds):
        if runner.done:
            return sent
        rnd = runner.ready()
        for job in rnd.dispatch:
            sent.append(job)
            runner.complete(job, outcome(job))
        if not (rnd.dispatch or rnd.settled or rnd.unrunnable) and not runner.done:
            raise AssertionError(runner.stuck())
    raise AssertionError("the run did not finish")


def node_job(runner, name, index=None):
    return runner.exec_graph[name].job(index)


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        yield sub


@pytest.fixture
def memory(tmp_path):
    return MemoryStore(tmp_path / "mem")


def runner_for(submitter, task, **kwargs) -> GraphRunner:
    kwargs.setdefault("on_error", "raise_at_end")
    return GraphRunner(task, submitter.store, submitter=submitter, **kwargs)


def is_uuid(value) -> bool:
    try:
        uuid.UUID(hex=value)
    except (TypeError, ValueError):
        return False
    return True


# -- workflows ---------------------------------------------------------------------------------------


@workflow.define
def Chain(x):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=10), name="B")
    return b.out


@workflow.define
def SplitCombine(xs: list[int]) -> int:
    a = workflow.add(FunAddVar(b=1).split(a=xs).combine("a"), name="A")
    s = workflow.add(ListSum(x=a.out), name="S")
    return s.out


@workflow.define(outputs=["a", "c", "d"])
def Twins(x: int):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    d = workflow.add(FunAddVar(a=x, b=1), name="D")  # A's twin in the same round
    b = workflow.add(FunAddVar(a=x, b=0), name="B")
    c = workflow.add(FunAddVar(a=b.out, b=1), name="C")  # A's twin, released later
    return a.out, c.out, d.out


@workflow.define
def Inner(x):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=10), name="B")
    return b.out


@workflow.define
def Outer(x):
    i = workflow.add(Inner(x=x), name="I")
    return i.out


@workflow.define(outputs=["a", "c"])
def Independent(x: int):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    c = workflow.add(FunAddVar(a=x, b=2), name="C")
    return a.out, c.out


@workflow.define(outputs=["b", "d"])
def Siblings(xs: list[int]):
    a = workflow.add(FunAddVar(b=1).split(a=xs), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=1), name="B")
    c = workflow.add(FunAddVar(b=10).split(a=xs), name="C")
    d = workflow.add(FunAddVar(a=c.out, b=1), name="D")
    return b.out, d.out


@workflow.define
def Fanin(xs: list[int], y: int) -> list[int]:
    a = workflow.add(FunAddVar(b=1).split(a=xs), name="A")
    c = workflow.add(FunAddVar(a=y, b=2), name="C")
    b = workflow.add(FunAddVar(a=a.out, b=c.out), name="B")
    return b.out


@workflow.define(outputs=["b", "a"])
def TwoSplits(xs: list[int]):
    b = workflow.add(FunAddVar(b=1).split(a=xs), name="B")  # added first
    a = workflow.add(FunAddVar(b=2).split(a=xs), name="A")
    return b.out, a.out


@python.define
def Inc(x: int) -> int:
    return x + 1


@workflow.define(outputs=["first", "second"])
def IncPair(x: int):
    first = workflow.add(Inc(x=x), name="first")
    second = workflow.add(Inc(x=x + 100), name="second")
    return first.out, second.out


# -- the external driver ------------------------------------------------------------------------------


def test_external_driver_split_combine(memory):
    """The loop of RFC 0002 section 3.5 over an in-memory store, with no `Submitter`"""
    runner = GraphRunner(
        SplitCombine(xs=[1, 2, 3]),
        memory,
        on_error="raise_at_end",
        lookup_before_dispatch=True,
    )
    inbox: queue.Queue = queue.Queue()
    sent = []
    for _ in range(50):
        if runner.done:
            break
        rnd = runner.ready()
        for job in rnd.dispatch:
            sent.append((job.name, job.state_index, runner.upstream(job)))
            inbox.put((job, compute(job)))  # another thread or host, in a real driver
        if rnd.settled or rnd.unrunnable:
            continue
        while True:
            try:
                job, outcome = inbox.get(timeout=rnd.retry_after or 0.01)
            except queue.Empty:
                break
            runner.complete(job, outcome)
    assert runner.done and not runner.failed and runner.not_run == []
    assert [(n, i) for n, i, _ in sent] == [("A", 0), ("A", 1), ("A", 2), ("S", None)]
    a_jobs = [node_job(runner, "A", i) for i in range(3)]
    assert sent[3][2] == {"x": [(j, None) for j in a_jobs]}  # a combined read
    assert node_job(runner, "S").result().outputs.out == 2 + 3 + 4
    assert set(memory.lookups) >= {j.checksum for j in a_jobs}  # looked up first


def test_resume_settles_stored(submitter):
    drive_by_hand(runner_for(submitter, Chain(x=1)), lambda job: job.run())
    runner = runner_for(submitter, Chain(x=1))
    rnd = runner.ready()
    assert rnd.dispatch == []
    assert [(s.job.name, s.how) for s in rnd.settled] == [
        ("A", "stored"),
        ("B", "stored"),
    ]
    assert all(isinstance(s, Settled) for s in rnd.settled)
    assert rnd.settled[1].result.outputs.out == 12
    assert all(s.job.attempt is None for s in rnd.settled)  # never handed out
    assert runner.done


@pytest.mark.parametrize("order", ["first-in-flight", "first-finished"])
def test_shared_settled(memory, order):
    runner = GraphRunner(Twins(x=1), memory, on_error="raise_at_end")
    rnd = runner.ready()
    assert sorted(j.name for j in rnd.dispatch) == ["A", "B"]  # D is held
    assert rnd.settled == []
    jobs = {j.name: j for j in rnd.dispatch}
    a_result = compute(jobs["A"])
    if order == "first-in-flight":
        runner.complete(jobs["B"], compute(jobs["B"]))
        rnd = runner.ready()  # C is A's twin, and A is in flight: held
        assert rnd.dispatch == [] and rnd.settled == []
        runner.complete(jobs["A"], a_result)
        rnd = runner.ready()
        assert {(s.job.name, s.how) for s in rnd.settled} == {
            ("C", "shared"),
            ("D", "shared"),
        }
    else:
        runner.complete(jobs["A"], a_result)
        rnd = runner.ready()
        assert [(s.job.name, s.how) for s in rnd.settled] == [("D", "shared")]
        runner.complete(jobs["B"], compute(jobs["B"]))
        rnd = runner.ready()  # C is A's twin, and A has finished: settled at once
        assert [(s.job.name, s.how) for s in rnd.settled] == [("C", "shared")]
    assert rnd.dispatch == []
    assert all(s.result is a_result for s in rnd.settled)
    assert runner.ready().settled == []  # reported once
    assert runner.done and not runner.failed


def test_drive_holds_no_checksum_table():
    for fn in (drive, drive_inline):
        code = fn.__code__
        names = set(code.co_varnames) | set(code.co_cellvars) | set(code.co_names)
        assert not names & {"sharing", "futured", "finished", "share"}, fn.__name__


@pytest.mark.parametrize("path", ["success-record", "under-the-lease"])
def test_workflow_stored_settled(submitter, path):
    submitter(Outer(x=1))
    if path == "under-the-lease":
        shutil.rmtree(Path(submitter.cache_root) / SUCCESS_DIR)
    runner = runner_for(submitter, Outer(x=1))
    rnd = runner.ready()
    assert rnd.dispatch == []
    assert [(s.job.name, s.how) for s in rnd.settled] == [("I", "workflow-stored")]
    assert rnd.settled[0].result.outputs.out == 12
    assert rnd.settled[0].job.attempt is None  # not expanded
    assert runner.done


def test_unrunnable_reported_once_with_upstream(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    runner.complete(a, Failed(RuntimeError("no")))
    rnd = runner.ready()
    assert rnd.unrunnable == [Unrunnable(node="B", index=None, upstream=["A"])]
    assert runner.ready().unrunnable == []
    # a nested node is named by its workflow path
    runner = runner_for(submitter, Outer(x=1))
    (a,) = runner.ready().dispatch
    runner.complete(a, Failed(RuntimeError("no")))
    unrunnable = []
    for _ in range(5):
        unrunnable += runner.ready().unrunnable
    assert Unrunnable(node="I/B", index=None, upstream=["I/A"]) in unrunnable
    assert len(unrunnable) == len(set((u.node, u.index) for u in unrunnable))


# -- NotRun ---------------------------------------------------------------------------------------------


def test_notrun_held_blocks_readers(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    held = NotRun(reason="the queue is held", held=True)
    runner.complete(a, held)
    rnd = runner.ready()
    assert rnd.dispatch == [] and rnd.settled == [] and rnd.unrunnable == []
    assert not runner.failed and runner.entries == []
    assert runner.not_run == [(a, held)]
    assert runner.done  # nothing else can progress
    assert runner.stuck() is None
    assert runner.ready().dispatch == []  # B stays blocked, A is not handed out again


def test_retry_redispatches(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    first = a.attempt
    runner.complete(a, NotRun(reason="cancelled", held=False))
    assert runner.ready().dispatch == []
    runner.retry(a)
    assert runner.not_run == [] and not runner.done
    (again,) = runner.ready().dispatch
    assert again is a and is_uuid(a.attempt) and a.attempt != first
    runner.complete(a, a.run())
    (b,) = runner.ready().dispatch
    assert b.name == "B"
    with pytest.raises(ValueError, match="NotRun"):
        runner.retry(b)  # only a job reported NotRun is handed out again


def test_notrun_only_remaining_is_done(submitter):
    runner = runner_for(submitter, Independent(x=1))
    jobs = {j.name: j for j in runner.ready().dispatch}
    held = NotRun(reason="held by the queue", held=True)
    runner.complete(jobs["A"], held)
    runner.complete(jobs["C"], jobs["C"].run())
    runner.ready()
    assert runner.done and not runner.failed
    assert runner.not_run == [(jobs["A"], held)]


class CancelsA(base.Worker):
    """Raises ``CancelledError`` for the job ``first``, runs the rest"""

    _plugin_name = "cancels-first"

    async def run(self, job, rerun=False):
        if job.name == "first":
            raise asyncio.CancelledError
        return job.run(rerun=rerun)


class CancelsASync(base.Worker):
    _plugin_name = "cancels-first-sync"

    def run(self, job, rerun=False):
        if job.name == "first":
            raise asyncio.CancelledError
        return job.run(rerun=rerun)


class CancelsBatch(base.Worker):
    """A batching worker whose ``submit_many`` raises ``CancelledError`` itself"""

    _plugin_name = "cancels-batch"
    batches = True

    async def run(self, job, rerun=False):
        return job.run(rerun=rerun)

    async def submit_many(self, jobs):
        raise asyncio.CancelledError


class CancelsFirstInBatch(CancelsBatch):
    """A batching worker that reports ``first`` cancelled through ``done``, runs the rest"""

    _plugin_name = "cancels-first-in-batch"
    reports_done = True

    async def submit_many(self, jobs, done=None):
        outcomes = []
        for job in jobs:
            outcome = (
                asyncio.CancelledError() if job.name == "first" else job.run(job.rerun)
            )
            done(job, outcome)
            outcomes.append(outcome)
        return outcomes


@pytest.mark.timeout(60, method="thread")
@pytest.mark.parametrize(
    "worker, not_run",
    [
        (CancelsA, ["first"]),
        (CancelsASync, ["first"]),
        (CancelsBatch, ["first", "second"]),
        (CancelsFirstInBatch, ["first"]),
    ],
)
def test_submitter_raises_run_incomplete(tmp_path, worker, not_run):
    outcome = []

    def call():
        try:
            with Submitter(worker=worker(), cache_root=tmp_path) as sub:
                sub(IncPair(x=1), raise_errors=True)
            outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "the run hangs"
    (exc,) = outcome
    assert isinstance(exc, RunIncomplete) and isinstance(exc, WorkflowError)
    assert sorted((job.name, nr.held) for job, nr in exc.not_run) == [
        (name, False) for name in not_run
    ]
    assert exc.entries == []  # not a failure
    assert "first" in str(exc)


def test_close_does_not_report_notrun_as_stopped(submitter):
    task = Independent(x=1)
    job = Job(task, submitter=submitter, name="main")
    runner = GraphRunner(
        task, submitter.store, on_error="raise_at_end", submitter=submitter, job=job
    )
    jobs = {j.name: j for j in runner.ready().dispatch}
    runner.complete(jobs["A"], Failed(RuntimeError("A failed")))
    runner.complete(jobs["C"], NotRun(reason="held", held=True))
    runner.ready()
    runner.close()
    assert [(e.job_name, e.phase) for e in runner.entries] == [("A", "run")]


def test_failed_per_job_not_exception(submitter):
    runner = runner_for(submitter, Siblings(xs=[1, 2, 3]))

    def outcome(job):
        if job.name == "A":
            return Failed(RuntimeError(f"refused {job.state_index}"))
        return job.run()

    sent = drive_by_hand(runner, outcome)
    assert sorted((j.name, j.state_index) for j in sent if j.name != "A") == [
        ("C", 0),
        ("C", 1),
        ("C", 2),
        ("D", 0),
        ("D", 1),
        ("D", 2),
    ]
    assert [(e.job_name, e.state_index) for e in runner.entries] == [
        ("A", 0),
        ("A", 1),
        ("A", 2),
    ]
    assert runner.exec_graph["B"].unrunnable.keys() == {0, 1, 2}
    assert [node_job(runner, "D", i).result().outputs.out for i in range(3)] == [
        12,
        13,
        14,
    ]


# -- upstream, lookup, attempt and limit ---------------------------------------------------------------


def test_upstream_maps_fields(submitter):
    runner = runner_for(submitter, Fanin(xs=[1, 2, 3], y=5))
    drive_by_hand(runner, lambda job: job.run())
    a1, c = node_job(runner, "A", 1), node_job(runner, "C")
    assert runner.upstream(node_job(runner, "B", 1)) == {
        "a": [(a1, 1)],
        "b": [(c, None)],
    }
    assert runner.upstream(a1) == {}


def test_lookup_before_dispatch_hides_hits(submitter):
    drive_by_hand(runner_for(submitter, Chain(x=1)), lambda job: job.run())
    shutil.rmtree(Path(submitter.cache_root) / SUCCESS_DIR)  # stored, without records
    plain = runner_for(submitter, Chain(x=1)).ready()
    assert [j.name for j in plain.dispatch] == ["A"]
    runner = runner_for(submitter, Chain(x=1), lookup_before_dispatch=True)
    rnd = runner.ready()
    assert rnd.dispatch == []
    assert [(s.job.name, s.how) for s in rnd.settled] == [
        ("A", "stored"),
        ("B", "stored"),
    ]
    assert runner.done


def test_attempt_on_dispatch_only(memory):
    runner = GraphRunner(Chain(x=1), memory, on_error="raise_at_end")
    (a,) = runner.ready().dispatch
    assert is_uuid(a.attempt)
    b = runner.exec_graph["B"]
    assert b._tasks is None or all(j is None for j in b._tasks.values())
    runner.complete(a, compute(a))
    (b,) = runner.ready().dispatch
    assert is_uuid(b.attempt) and b.attempt != a.attempt


def test_attempt_resume_for_can_resume(tmp_path, monkeypatch):
    """A job that can resume runs in its stable ``work/resume-<hash>`` directory, and each
    dispatch still has an attempt of its own, which its outcome files carry (P4c)"""
    from pydra.engine.portable.store import PortableStore

    store = PortableStore(tmp_path / "root", trust="private")
    monkeypatch.setattr(Job, "_can_resume", True)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Chain(x=1), store, on_error="raise_at_end", submitter=sub)
        (a,) = runner.ready().dispatch
        first, wd = a.attempt, store.work_dir(a)
        assert is_uuid(first) and wd.name.startswith("resume-")
        runner.complete(a, NotRun(reason="cancelled", held=False))
        runner.ready()
        runner.retry(a)
        (a,) = runner.ready().dispatch
        assert is_uuid(a.attempt) and a.attempt != first
        assert store.work_dir(a) == wd


def test_attempt_for_expanded_workflow_and_outer_job(submitter):
    seen = []
    hooks = TaskHooks(
        pre_run=lambda job: seen.append(job.attempt),
        post_run=lambda job, result: seen.append(job.attempt),
    )
    submitter(Outer(x=1), hooks=hooks)
    # the outer job, from its creation, and kept when its expansion begins
    assert len(seen) == 2 and is_uuid(seen[0]) and seen[0] == seen[1]
    submitter.store = PickleStore(Path(submitter.cache_root) / "fresh")
    runner = runner_for(submitter, Outer(x=1))
    drive_by_hand(runner, lambda job: job.run())
    assert is_uuid(node_job(runner, "I").attempt)  # given when its expansion began
    runner = runner_for(submitter, Outer(x=1))
    (settled,) = runner.ready().settled
    assert settled.how == "workflow-stored" and settled.job.attempt is None


def test_limit_keeps_node_groups(submitter):
    runner = runner_for(submitter, TwoSplits(xs=[1, 2, 3]))
    rnd = runner.ready(limit=4)
    assert [(j.name, j.state_index) for j in rnd.dispatch] == [
        ("A", 0),
        ("A", 1),
        ("A", 2),
        ("B", 0),
    ]
    rest = runner.ready()
    assert [(j.name, j.state_index) for j in rest.dispatch] == [("B", 1), ("B", 2)]


class Batching:
    """A batching executor that runs each round it is given in the driver's loop"""

    batches = True
    checks_store = True
    inline = False

    def __init__(self):
        self.calls = []

    def bind(self, report):
        self.report = report

    def submit(self, jobs):
        self.calls.append([(j.name, j.state_index) for j in jobs])
        for job in jobs:
            self.report(job, job.run(rerun=job.rerun))
        return Handle(self, tuple(jobs), set())

    def cancel(self, handle):
        pass

    def close(self):
        pass


@workflow.define
def Split6(xs: list[int]) -> list[int]:
    a = workflow.add(FunAddVar(b=1).split(a=xs), name="A")
    return a.out


class SlowRelease:
    """Wraps a workflow job's lease; its release blocks until the test has seen the event
    loop run while the release waits, or gives up after ``PATIENCE`` seconds"""

    PATIENCE = 20.0
    blocked: list = []  # the releases waiting, each a threading.Event
    threads: list = []  # the thread each release ran on
    gave_up = 0
    released = 0

    def __init__(self, lease):
        self.lease = lease

    def __getattr__(self, name):
        return getattr(self.lease, name)

    def try_take(self):
        return self.lease.try_take()

    def __enter__(self):
        self.lease.__enter__()
        return self

    def __exit__(self, *exc):
        cls = type(self)
        cls.threads.append(threading.get_ident())
        go = threading.Event()
        cls.blocked.append(go)
        if not go.wait(self.PATIENCE):  # the loop never ran while this release waited
            cls.gave_up += 1
        self.lease.__exit__(*exc)
        cls.released += 1


class SlowReleaseStore(PickleStore):
    def acquire(self, checksum, context):
        lease = super().acquire(checksum, context)
        return SlowRelease(lease) if checksum.startswith("workflow-") else lease


def test_submitter_runs_a_workflow_on_a_portable_store(any_worker, tmp_path):
    """Every job handed out has an attempt, so it has a work directory; once the runner
    holds a job's outcome its ``_committed.json`` is forgotten"""
    from pydra.engine.portable.store import PortableStore

    store = PortableStore(tmp_path / "root", trust="private")
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc") as sub:
        result = sub(Outer(x=1))
    assert not result.errored and result.outputs.out == 12
    assert list((tmp_path / "root").rglob("_committed.json")) == []


def test_runner_leases_resumable_workflow_jobs(tmp_path, monkeypatch):
    from pydra.engine.portable.store import PortableStore

    store = PortableStore(tmp_path / "root", trust="private", dedupe=False)
    calls = []
    acquire = store.acquire

    def spy(checksum, context, **kwargs):
        calls.append((checksum.split("-")[0], kwargs.get("resumable", False)))
        return acquire(checksum, context, **kwargs)

    monkeypatch.setattr(store, "acquire", spy)
    monkeypatch.setattr(Job, "_can_resume", True)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Outer(x=1), store, on_error="raise_at_end", submitter=sub)
        drive_by_hand(runner, lambda job: job.run())
    assert ("workflow", True) in calls  # the nested workflow job, as Job.run would
    assert all(resumable for _, resumable in calls)


def test_read_back_reads_a_settled_result():
    from pydra.engine.executor import WorkerExecutor
    from pydra.workers.debug import DebugWorker

    calls = []
    found = object()

    class Reported:
        def result(self, settled=False):
            calls.append(settled)
            return found

    ex = WorkerExecutor(DebugWorker(), visibility_timeout=0.1)
    assert asyncio.run(ex._read_back(Reported())) is found
    assert calls == [True]


class ArrayWorker(base.Worker):
    """A batching worker like slurm's: one ``submit_many`` per array, each job reported
    through ``done`` as it ends; records how many jobs it holds at once"""

    _plugin_name = "array-worker"
    batches = True
    reports_done = True

    def __init__(self):
        super().__init__()
        self.calls = []
        self.holding = 0
        self.peak = 0

    async def run(self, job, rerun=False):
        return job.run(rerun=rerun)

    async def submit_many(self, jobs, done=None):
        self.calls.append(len(jobs))
        self.holding += len(jobs)
        self.peak = max(self.peak, self.holding)

        async def one(job):
            await asyncio.sleep(0.05)
            job.run(rerun=job.rerun)
            self.holding -= 1
            if done is not None:
                done(job, True)
            return True

        return list(await asyncio.gather(*(one(j) for j in jobs)))


class HoldsJobs(base.Worker):
    """A batching worker that keeps every job it is given until the test lets go"""

    _plugin_name = "holds-jobs"
    batches = True
    reports_done = True

    def __init__(self):
        super().__init__()
        self.given = []

    async def run(self, job, rerun=False):
        raise AssertionError("jobs are given through submit_many")

    async def submit_many(self, jobs, done=None):
        self.given.extend(j.state_index for j in jobs)
        await asyncio.Event().wait()


class CountingReads(PickleStore):
    """A store that counts its success-record reads"""

    reads = 0

    def read_succeeded(self, checksum, context, task=None):
        type(self).reads += 1
        return super().read_succeeded(checksum, context, task)


@workflow.define
def SplitInc(xs: list[int]) -> list[int]:
    return workflow.add(Inc().split(x=xs).combine("x"), name="inc").out


@pytest.mark.parametrize("n", [20, 80])
def test_a_job_held_back_by_the_limit_is_not_read_again(tmp_path, n):
    """``ready(limit)`` reads each released job's success record once, not once per call
    while the limit holds it back"""
    store = CountingReads(tmp_path / "cache")
    CountingReads.reads = 0
    with Submitter(worker="debug", store=store) as sub:
        runner = GraphRunner(
            SplitInc(xs=list(range(n))), store, on_error="raise_at_end", submitter=sub
        )
        while not runner.done:
            for job in runner.ready(1).dispatch:
                runner.complete(job, job.run())
    assert not runner.failed
    assert CountingReads.reads <= 2 * n  # held back and read every call: n(n+1)/2
