"""Completion is reported to the runner, not polled from the store"""

import asyncio
import time
from types import SimpleNamespace
import pytest
from pydra.engine.drivers import drive
from pydra.engine.executor import InlineExecutor
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner, Round
from pydra.engine.submitter import Submitter
from pydra.engine.tests.test_runner import Chain, runner_for  # noqa: F401
from pydra.workers.debug import DebugWorker


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        yield sub


def test_completed_result_read_without_store(submitter, monkeypatch):
    """A job completed with a Result is read back from the job, not the store"""
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    result = a.run()
    a._cached_result = None  # as if run in another process
    lookups = []
    real = submitter.store.lookup
    monkeypatch.setattr(
        submitter.store,
        "lookup",
        lambda *args, **kw: lookups.append(args) or real(*args, **kw),
    )
    runner.complete(a, result)
    assert a.result() is result
    (b,) = runner.ready().dispatch
    assert b.task.a == 2
    assert [args for args in lookups if args[0] == a.checksum] == []


def test_dispatched_jobs_not_polled(submitter, monkeypatch):
    """Rounds taken while a job is dispatched do not poll it"""
    polled = []
    real_done = Job.done
    monkeypatch.setattr(
        Job,
        "done",
        property(lambda self: polled.append(self.name) or real_done.fget(self)),
    )
    real_start = Job.run_start_time
    monkeypatch.setattr(
        Job,
        "run_start_time",
        property(lambda self: polled.append(self.name) or real_start.fget(self)),
    )
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    for _ in range(3):
        assert runner.ready().dispatch == []
        assert not runner.done
    assert polled == []
    runner.complete(a, a.run())
    assert [j.name for j in runner.ready().dispatch] == ["B"]


def test_stuck_is_immediate(submitter):
    """With nothing in flight and nothing released, the driver reports the diagnosis
    at once rather than waiting for results to become visible"""

    class Wedged(GraphRunner):
        def ready(self, limit=None):
            return Round()

    runner = runner_for(submitter, Chain(x=1))
    runner.__class__ = Wedged
    executor = InlineExecutor(DebugWorker())

    async def run():
        await drive(
            runner,
            {"debug": executor},
            lambda job: "debug",
            max_concurrent=float("inf"),
            on_error="raise_at_end",
            audit=submitter.audit,
        )

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="not done"):
        asyncio.run(run())
    assert time.monotonic() - start < 1


def test_complete_keeps_committed_attribute(submitter):
    """Whatever the Result carries, ``committed`` included, reaches readers unchanged"""
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    committed = object()
    stub = SimpleNamespace(errored=False, committed=committed, outputs=None)
    runner.complete(a, stub)
    assert a.result() is stub
    assert a.result().committed is committed


def test_errored_result_sets_failed(submitter):
    """An errored Result reported to complete() is a failure of the run, and its
    readers are unrunnable"""
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    assert not runner.failed
    runner.complete(
        a, Result(cache_dir=a.cache_dir, outputs=None, errored=True, task=a.task)
    )
    assert runner.failed
    assert a.errored
    assert runner.ready().dispatch == []
    assert [e.job_name for e in runner.entries] == ["A"]


def test_cf_worker_runs_on_the_running_loop(tmp_path):
    """The cf worker submits to the loop it is awaited on, not to a stale ``loop``"""
    from pydra.workers.cf import ConcurrentFuturesWorker

    stale = asyncio.new_event_loop()
    stale.close()
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = runner_for(sub, Chain(x=1))
        (a,) = runner.ready().dispatch
    worker = ConcurrentFuturesWorker(n_procs=1)
    worker.loop = stale
    try:
        result = asyncio.run(worker.run(a))
    finally:
        worker.close()
    assert result.outputs.out == 2


def test_unrunnable_spreads_in_one_round(submitter):
    """One ``ready()`` after a failure marks every transitive reader, so the driver
    needs no waiting round to see that the graph is done"""
    from pydra.compose import workflow
    from pydra.engine.runner import Failed
    from pydra.engine.tests.utils import FunAddVar

    @workflow.define
    def Three(x):
        a = workflow.add(FunAddVar(a=x, b=1), name="A")
        b = workflow.add(FunAddVar(a=a.out, b=1), name="B")
        c = workflow.add(FunAddVar(a=b.out, b=1), name="C")
        return c.out

    runner = runner_for(submitter, Three(x=1))
    (a,) = runner.ready().dispatch
    runner.complete(a, Failed(RuntimeError("no")))
    assert runner.ready().dispatch == []
    assert runner.done
    assert runner.stuck() is None
