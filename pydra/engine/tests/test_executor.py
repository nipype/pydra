"""Executors and the asynchronous driver"""

import asyncio
import pytest
from pydra.compose import python, workflow
from pydra.engine.drivers import drive
from pydra.engine.executor import Executor, WorkerExecutor
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
    kwargs.setdefault("max_concurrent", float("inf"))
    kwargs.setdefault("on_error", "raise_at_end")
    kwargs.setdefault("audit", sub.audit)
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
        assert "no result" in str(outcome.error)


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


class Hangs(base.Worker):
    _plugin_name = "debug"

    async def run(self, job, rerun=False):
        await asyncio.Event().wait()


def test_interrupt_cancels_open_handles(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = make_runner(sub, Split(xs=[0, 1, 2, 3]))
        worker = Hangs()
        worker.loop = sub.loop
        ex = Counting(worker)

        async def interrupted():
            task = asyncio.ensure_future(
                drive(
                    runner,
                    {"default": ex},
                    lambda job: "default",
                    max_concurrent=float("inf"),
                    on_error="raise_at_end",
                    audit=sub.audit,
                )
            )
            await asyncio.sleep(0.3)
            assert len(ex.calls) == 4
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        sub.loop.run_until_complete(interrupted())
        assert len(ex.cancelled) == 4
        assert not [p for p in tmp_path.rglob("*") if p.name.endswith(LEASE_SUFFIX)]
        assert not ex._tasks  # nothing is left scheduled


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
    assert "drive" in calls


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
