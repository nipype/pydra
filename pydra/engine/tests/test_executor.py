"""Executors and the asynchronous driver"""

import asyncio
import threading
import time
import pytest
from pydra.compose import python, workflow
from pydra.engine.drivers import drive
from pydra.engine.executor import Executor, WorkerExecutor, no_result_diagnosis
from pydra.engine.runner import Failed, GraphRunner
from pydra.engine.store import LEASE_SUFFIX
from pydra.engine.submitter import Submitter
from pydra.engine.tests.utils import FunAddVar
from pydra.utils.resources import ComputeResources
from pydra.workers import base, debug

GPU = ComputeResources(gpus=1, tags=("gpu",))


@python.define
def AddOne(x: int) -> int:
    return x + 1


@workflow.define
def Split(xs: list[int]) -> list[int]:
    n = workflow.add(AddOne().split(x=xs), name="N")
    return n.out


@workflow.define(outputs=["out"])
def Pipeline(x: int):
    prep = workflow.add(AddOne(x=x), name="prep")
    train = workflow.add(AddOne(x=prep.out), name="train", resources=GPU)
    post = workflow.add(AddOne(x=train.out), name="post")
    return post.out


def make_runner(sub, task, on_error="raise_at_end"):
    return GraphRunner(task, sub.store, on_error=on_error, submitter=sub)


def run_drive(sub, runner, executors, router=lambda job: "default", **kwargs):
    kwargs.setdefault("on_error", "raise_at_end")
    sub.loop.run_until_complete(drive(runner, executors, router, **kwargs))


class Counting(WorkerExecutor):
    """Counts the reports of each job and the jobs of each submit call"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reports = {}
        self.calls = []
        self.cancelled = []

    def bind(self, report):
        def counted(job, outcome):
            self.reports[job] = self.reports.get(job, 0) + 1
            report(job, outcome)

        super().bind(counted)

    def submit(self, jobs):
        self.calls.append([j.name for j in jobs])
        return super().submit(jobs)

    def cancel(self, handle):
        self.cancelled.append(handle)
        super().cancel(handle)


def test_every_submitted_job_reported_once(any_worker, tmp_path):
    with Submitter(worker=any_worker, cache_root=tmp_path) as sub:
        runner = make_runner(sub, Split(xs=list(range(6))))
        ex = Counting(sub.worker)
        assert isinstance(ex, Executor)
        run_drive(sub, runner, {"default": ex})
        assert runner.done and not runner.failed
        assert len(ex.reports) == 6 and set(ex.reports.values()) == {1}
        assert sum(map(len, ex.calls)) == 6


class WritesLate(base.Worker):
    """Returns True at once, as a scheduler worker does, and the result appears later"""

    _plugin_name = "debug"

    def __init__(self, delay, write=True):
        super().__init__()
        self.delay, self.write = delay, write

    async def run(self, job, rerun=False):
        if self.write:
            self.loop.call_later(self.delay, lambda: job.run(rerun=rerun))
        return True


def one_job(sub, task=None):
    runner = make_runner(sub, task or Pipeline(x=1))
    return runner.ready().dispatch[0]


async def submit_and_wait(ex, job):
    reported = []
    ex.bind(lambda j, o: reported.append(o))
    ex.submit([job])
    while not reported:
        await asyncio.sleep(0.01)
    return reported


def test_true_returning_worker_result_read_late(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        job = one_job(sub)
        worker = WritesLate(0.3)
        worker.loop = sub.loop
        ex = WorkerExecutor(worker, visibility_timeout=2.0)
        (outcome,) = sub.loop.run_until_complete(submit_and_wait(ex, job))
        assert not isinstance(outcome, Failed)
        assert outcome.outputs.out == 2


def test_true_returning_worker_never_visible(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        job = one_job(sub)
        worker = WritesLate(0.0, write=False)
        worker.loop = sub.loop
        ex = WorkerExecutor(worker, visibility_timeout=0.5)
        (outcome,) = sub.loop.run_until_complete(submit_and_wait(ex, job))
        assert isinstance(outcome, Failed)
        assert str(outcome.error) == no_result_diagnosis(job)


def test_read_back_waits_the_lease_visibility_by_default():
    from pydra.engine.lease import LeaseSettings

    ex = WorkerExecutor(debug.Worker())
    assert ex.visibility_timeout == LeaseSettings().visibility_timeout


def test_router_dispatches_to_executors(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = make_runner(sub, Pipeline(x=1))
        default, gpu = Counting(debug.Worker()), Counting(debug.Worker())
        default.worker.loop = gpu.worker.loop = sub.loop
        run_drive(
            sub,
            runner,
            {"default": default, "gpu": gpu},
            router=lambda job: "gpu" if job.resources is not None else "default",
        )
        assert runner.done and not runner.failed
        assert default.calls == [["prep"], ["post"]]
        assert gpu.calls == [["train"]]


class CallSoonBatching(base.Worker):
    """Collects the jobs that reach it in one loop iteration, as a batching plugin does"""

    _plugin_name = "debug"

    def __init__(self):
        super().__init__()
        self.pending = []
        self.batches = []

    async def run(self, job, rerun=False):
        fut = self.loop.create_future()
        if not self.pending:
            self.loop.call_soon(self.flush)
        self.pending.append((job, fut, rerun))
        return await fut

    def flush(self):
        batch, self.pending = self.pending, []
        self.batches.append(len(batch))
        for job, fut, rerun in batch:
            fut.set_result(job.run(rerun=rerun))


def test_call_soon_batching_worker_sees_round(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = make_runner(sub, Split(xs=[0, 1, 2, 3]))
        worker = CallSoonBatching()
        worker.loop = sub.loop
        run_drive(sub, runner, {"default": WorkerExecutor(worker)})
        assert runner.done and not runner.failed
        assert worker.batches == [4]


class LeaseHang(base.Worker):
    """Takes the job's lease, as ``Job.run_async`` does, then waits forever"""

    _plugin_name = "debug"
    cancelled = 0

    async def run(self, job, rerun=False):
        async with job.store.acquire(job.checksum, job.context):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                type(self).cancelled += 1
                raise


def test_interrupt_cancels_open_handles(tmp_path, monkeypatch):
    monkeypatch.setattr(LeaseHang, "cancelled", 0)
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = make_runner(sub, Split(xs=[0, 1, 2, 3]))
        worker = LeaseHang()
        worker.loop = sub.loop
        ex = Counting(worker)

        async def interrupted():
            task = asyncio.ensure_future(
                drive(
                    runner,
                    {"default": ex},
                    lambda job: "default",
                    on_error="raise_at_end",
                )
            )
            deadline = time.monotonic() + 60  # bound only
            while len(ex.calls) < 4 or len(leases(tmp_path)) < 4:  # all four hang
                assert time.monotonic() < deadline, (ex.calls, leases(tmp_path))
                await asyncio.sleep(0.01)
            assert len(ex.calls) == 4
            assert len(leases(tmp_path)) == 4
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        sub.loop.run_until_complete(interrupted())
        assert len(ex.cancelled) == 4
        assert LeaseHang.cancelled == 4  # every run coroutine was cancelled
        assert not leases(tmp_path)  # and released its lease
        assert not ex._tasks  # nothing is left scheduled


@python.define
def Inc(x: int) -> int:
    return x + 1


@workflow.define
def IncTwice(x: int) -> int:
    first = workflow.add(Inc(x=x), name="first")
    second = workflow.add(Inc(x=first.out), name="second")
    return second.out


class CancelsSecond(base.Worker):
    _plugin_name = "cancels-second"

    async def run(self, job, rerun=False):
        if job.name == "second":
            raise asyncio.CancelledError
        return job.run(rerun=rerun)


@pytest.mark.timeout(60, method="thread")
def test_a_cancellation_raised_by_a_worker_ends_the_run(tmp_path):
    # the run happens on a daemon thread so a regression fails the join instead of hanging the suite
    outcome = []

    def call():
        try:
            with Submitter(worker=CancelsSecond(), cache_root=tmp_path) as sub:
                sub(IncTwice(x=1), raise_errors=True)
            outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the run hangs: the cancelled job is never reported"
    assert outcome and outcome[0] is not None, "the run must not succeed"


def leases(root):
    return [p for p in root.rglob("*") if p.name.endswith(LEASE_SUFFIX)]


@python.define
def Boom(x: int) -> int:
    raise ValueError("BOOM")


@python.define
def Touch(x: int, mark: str) -> int:
    import pathlib

    pathlib.Path(mark, f"ran-{x}").touch()
    return x


@python.define
def Remote(x: int) -> int:
    return x


@workflow.define(outputs=["a", "t1", "t2", "t3"])
def FailThenTouch(x: int, mark: str):
    a = workflow.add(Boom(x=x), name="a")
    t1 = workflow.add(Touch(x=1, mark=mark), name="t1")
    t2 = workflow.add(Touch(x=2, mark=mark), name="t2")
    t3 = workflow.add(Touch(x=3, mark=mark), name="t3")
    return a.out, t1.out, t2.out, t3.out


@workflow.define(outputs=["r", "a", "t1", "t2", "t3"])
def MixedFailThenTouch(x: int, mark: str):
    r = workflow.add(Remote(x=x), name="r", resources=GPU)
    a = workflow.add(Boom(x=x), name="a")
    t1 = workflow.add(Touch(x=1, mark=mark), name="t1")
    t2 = workflow.add(Touch(x=2, mark=mark), name="t2")
    t3 = workflow.add(Touch(x=3, mark=mark), name="t3")
    return r.out, a.out, t1.out, t2.out, t3.out


@workflow.define(outputs=["r", "inner"])
def NestedMixed(x: int, mark: str):
    r = workflow.add(Remote(x=x), name="r", resources=GPU)
    inner = workflow.add(FailThenTouch(x=x, mark=mark), name="inner")
    return r.out, inner.a


def mixed_submitter(tmp_path):
    return Submitter(
        worker="debug",
        workers={"pool": "cf"},
        router=lambda job: "pool" if job.resources is not None else None,
        cache_root=tmp_path / "cache",
    )


@pytest.mark.parametrize("task", [MixedFailThenTouch, NestedMixed])
def test_mixed_run_runs_no_job_after_the_first_failure(tmp_path, task):
    """A debug main worker routed beside cf keeps the fail_fast contract, also in a
    nested workflow, whose default is the submitter's"""
    mark = tmp_path / "mark"
    mark.mkdir()
    with mixed_submitter(tmp_path) as sub:
        assert sub.on_error == "fail_fast"
        with pytest.raises(ValueError, match="BOOM"):
            sub(task(x=1, mark=str(mark)))
    assert sorted(p.name for p in mark.iterdir()) == []


def test_all_inline_workers_use_drive_inline(tmp_path, monkeypatch):
    import pydra.engine.submitter as submitter_mod

    calls = []
    for name in ("drive", "drive_inline"):
        orig = getattr(submitter_mod, name)
        monkeypatch.setattr(
            submitter_mod,
            name,
            lambda *a, _o=orig, _n=name, **k: (calls.append(_n), _o(*a, **k))[1],
        )
    with Submitter(
        worker="debug", workers={"local": "debug"}, cache_root=tmp_path / "a"
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert set(calls) == {"drive_inline"}
    calls.clear()
    with Submitter(
        worker="debug",
        workers={"pool": "cf"},
        router=lambda job: "pool" if job.resources is not None else None,
        cache_root=tmp_path / "b",
    ) as sub:
        assert sub(Pipeline(x=1)).outputs.out == 4
    assert calls[0] == "drive"


def test_blocking_worker_runs_on_the_running_loop(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        job = one_job(sub, Chain1())
        other = asyncio.new_event_loop()
        try:
            sub.loop = other  # a stale loop must not be used
            result = asyncio.run(sub._run_blocking(debug.Worker(), job, False))
        finally:
            other.close()
        assert result.outputs.out == 2


@workflow.define
def Chain1():
    return workflow.add(FunAddVar(a=1, b=1), name="A").out
