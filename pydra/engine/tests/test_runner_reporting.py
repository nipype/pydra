"""Completion is reported to the runner, not polled from the store"""

import asyncio
import sys
import traceback
from types import SimpleNamespace
import pytest
from pydra.compose import python, workflow
from pydra.engine.drivers import drive
from pydra.engine.executor import InlineExecutor
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner, RecordedError, Round, error_entry
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


def test_stuck_is_immediate(submitter, monkeypatch):
    """With nothing in flight and nothing released, the driver reports the diagnosis
    at once rather than waiting for results to become visible"""
    rounds, waits = [], []

    class Wedged(GraphRunner):
        def ready(self, limit=None):
            rounds.append(limit)
            return Round()

    real_wait = asyncio.Event.wait

    async def wait(self):
        waits.append(self)
        return await real_wait(self)

    monkeypatch.setattr(asyncio.Event, "wait", wait)

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
        )

    with pytest.raises(RuntimeError, match="not done"):
        asyncio.run(run())
    # the first round reported it: no second round, no wait for an outcome
    assert rounds == [None]
    assert waits == []


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
    from pydra.engine.runner import Failed
    from pydra.engine.tests.utils import FunAddVar

    @workflow.define
    def Deep(x):
        prev = workflow.add(FunAddVar(a=x, b=1), name="n0")
        for i in range(1, 7):
            prev = workflow.add(FunAddVar(a=prev.out, b=1), name=f"n{i}")
        return prev.out

    runner = runner_for(submitter, Deep(x=1))
    (a,) = runner.ready().dispatch
    runner.complete(a, Failed(RuntimeError("no")))
    assert runner.ready().dispatch == []
    assert runner.done
    assert runner.stuck() is None


@python.define
def Inc(x: int) -> int:
    return x + 1


@python.define
def Total(xs: list[int]) -> int:
    return sum(xs)


@workflow.define
def Long(x: int, tag: str = "") -> int:
    prev = workflow.add(Inc(x=x), name="n0")
    for i in range(1, 5):
        prev = workflow.add(Inc(x=prev.out), name=f"n{i}")
    return prev.out


@workflow.define
def EmptySplit(xs: list[int]) -> int:
    a = workflow.add(Inc().split(x=xs).combine("x"), name="a")
    b = workflow.add(Total(xs=a.out), name="b")
    return b.out


def test_cached_chain_rerun_finishes(tmp_path, any_worker):
    """Every inner job of a five-node chain is cached on the second run, which settles
    through ``ready()`` alone; the driver must not take that for stuck"""
    for tag in ("1", "2"):
        with Submitter(worker=any_worker, cache_root=tmp_path) as sub:
            result = sub(Long(x=1, tag=tag), raise_errors=True)
        assert result.outputs.out == 6


def test_empty_split_reader_finishes(tmp_path, any_worker):
    """A node with no elements starts without a job; its reader still runs"""
    with Submitter(worker=any_worker, cache_root=tmp_path) as sub:
        result = sub(EmptySplit(xs=[]), raise_errors=True)
    assert result.outputs.out == 0


def test_cached_chain_drive_inline(tmp_path):
    """The synchronous driver settles a fully cached chain too"""
    from pydra.engine.drivers import drive_inline

    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        sub(Long(x=1, tag="1"), raise_errors=True)
        runner = GraphRunner(
            Long(x=1, tag="2"), sub.store, on_error="raise_at_end", submitter=sub
        )
        drive_inline(
            runner,
            InlineExecutor(DebugWorker()),
            on_error="raise_at_end",
        )
        assert runner.done and not runner.failed


def test_errored_result_entry_carries_the_recorded_error(submitter, tmp_path):
    """The entry of an errored Result is built from the error file the worker left"""
    from pydra.engine.result import record_error

    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    cache_dir = tmp_path / "slurm-like"
    cache_dir.mkdir()
    try:
        raise ValueError("remote boom")
    except ValueError as raised:
        record_error(
            cache_dir,
            error=traceback.format_exception(*sys.exc_info()),
            exception=raised,
        )
    runner.complete(
        a, Result(cache_dir=cache_dir, outputs=None, errored=True, task=a.task)
    )
    (entry,) = runner.entries
    assert entry.exception_type == "builtins.ValueError"
    assert entry.message == "remote boom"
    assert "Traceback" in entry.traceback and "remote boom" in entry.traceback
    assert "remote boom" in runner.errors[0]


class Unpicklable(Exception):
    def __reduce__(self):
        raise TypeError("not picklable")


def _recorded(tmp_path, exc):
    from pydra.engine.result import record_error
    from pydra.engine.runner import result_error

    try:
        raise exc
    except BaseException as raised:
        lines = traceback.format_exception(*sys.exc_info())
        record_error(tmp_path, error=lines, exception=raised)
    job = SimpleNamespace(name="job", cache_dir=tmp_path)
    return result_error(job, Result(cache_dir=tmp_path, outputs=None, errored=True))


@pytest.mark.parametrize(
    "exc",
    [KeyError("x"), OSError(2, "gone", "f"), ValueError("a: b"), Unpicklable("u")],
)
def test_a_recorded_error_keeps_its_type(tmp_path, exc):
    """The error read back names the type raised; a picklable one is the exception itself"""
    got = _recorded(tmp_path, exc)
    name = f"{type(exc).__module__}.{type(exc).__qualname__}"
    assert error_entry("job", None, got).exception_type == name
    assert "Traceback" in got.recorded_traceback
    if isinstance(exc, Unpicklable):
        assert isinstance(got, RecordedError)
    else:
        assert type(got) is type(exc) and got.args == exc.args


def test_an_exception_whose_class_the_reader_cannot_import_is_a_recorded_error(
    tmp_path, monkeypatch
):
    """The record stays readable where the exception's module is missing"""
    module = tmp_path / "mod" / "only_where_the_job_ran.py"
    module.parent.mkdir()
    module.write_text("class Boom(Exception):\n    pass\n")
    monkeypatch.syspath_prepend(str(module.parent))
    from only_where_the_job_ran import Boom

    record = tmp_path / "record"
    record.mkdir()
    try:
        raise Boom("remote")
    except Boom as raised:
        from pydra.engine.result import record_error

        record_error(record, error=traceback.format_exc(), exception=raised)
    monkeypatch.delitem(sys.modules, "only_where_the_job_ran")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(module.parent)])
    from pydra.engine.runner import result_error

    job = SimpleNamespace(name="job", cache_dir=record)
    got = result_error(job, Result(cache_dir=record, outputs=None, errored=True))
    assert isinstance(got, RecordedError)
    assert got.exception_type == "only_where_the_job_ran.Boom"
    assert "remote" in got.recorded_traceback


def test_errored_result_without_error_file(submitter, tmp_path):
    """With no error file the entry says so rather than naming an invented cause"""
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    runner.complete(
        a, Result(cache_dir=tmp_path / "none", outputs=None, errored=True, task=a.task)
    )
    (entry,) = runner.entries
    assert entry.job_name == "A"
    assert "no error record" in entry.message


def chain_of(n: int):
    @workflow.define
    def ChainN(x: int, tag: str = "") -> int:
        prev = workflow.add(Inc(x=x), name="n0")
        for i in range(1, n):
            prev = workflow.add(Inc(x=prev.out), name=f"n{i}")
        return prev.out

    return ChainN


@pytest.mark.parametrize("n", [16, 64])
def test_a_stored_chain_is_released_with_work_linear_in_its_length(
    tmp_path, monkeypatch, n
):
    """Settling a chain from the store takes one pass of ``ready`` per node; each pass
    looks at a bounded number of nodes, not at the whole graph"""
    from pydra.engine.drivers import drive_inline
    from pydra.engine.runner import NodeExecution

    looks = []
    done = NodeExecution.done

    def counted(node):
        looks.append(node.name)
        return done.fget(node)

    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        sub(chain_of(n)(x=1, tag="1"), raise_errors=True)
        runner = GraphRunner(
            chain_of(n)(x=1, tag="2"), sub.store, on_error="raise_at_end", submitter=sub
        )
        monkeypatch.setattr(NodeExecution, "done", property(counted))
        drive_inline(
            runner,
            InlineExecutor(DebugWorker()),
            on_error="raise_at_end",
        )
        assert runner.done and not runner.failed
    assert len(looks) <= 16 * n  # a quadratic walk looks 2462 times at n=64


def test_progress_is_a_count_read_without_walking_the_graph(tmp_path, monkeypatch):
    """``ready`` compares a count of changes between passes; reading it walks no node"""
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            chain_of(4)(x=1, tag="p"), sub.store, on_error="raise_at_end", submitter=sub
        )
        runner.ready()

        def walked(*args, **kwargs):
            raise AssertionError("a node table was read")

        monkeypatch.setattr(GraphRunner, "_window", staticmethod(walked))
        before = runner._progress()
        assert isinstance(before, int)
        node = runner.exec_graph.sorted_nodes[1]
        node.blocked = {0: ["changed"]}
        assert runner._progress() > before
