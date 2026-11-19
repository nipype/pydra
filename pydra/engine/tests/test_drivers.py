"""The synchronous driver and InlineExecutor: a debug run happens in the caller's
thread and context, inside or outside an event loop"""

import asyncio
import contextvars
import sys
import threading
import pytest
from pydra.compose import python, workflow
from pydra.engine.audit import Audit
from pydra.engine.drivers import drive_inline
from pydra.engine.executor import Handle, InlineExecutor
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.engine.tests.utils import FunAddVar
from pydra.workers.debug import DebugWorker

VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pydra_test_var", default="unset"
)


@python.define
def ThreadId(x: int) -> int:
    return threading.get_ident()


@python.define
def ReadVar(x: int) -> str:
    import contextvars

    return {v.name: val for v, val in contextvars.copy_context().items()}.get(
        "pydra_test_var", "unset"
    )


@python.define
def RunsAsyncio(x: int) -> int:
    async def inner():
        await asyncio.sleep(0)
        return x + 1

    return asyncio.run(inner())


class Boom(Exception):
    pass


BOOM = Boom("task failed")


@python.define
def Raises(x: int) -> int:
    raise BOOM


@python.define
def Touch(path: str) -> int:
    with open(path, "w") as f:
        f.write("ran")
    return 1


@workflow.define
def Chain(x):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=10), name="B")
    return b.out


@workflow.define(outputs=["r", "t"])
def FailThenIndependent(path: str):
    r = workflow.add(Raises(x=1), name="R")
    t = workflow.add(Touch(path=path), name="T")
    return r.out, t.out


def test_debug_runs_in_caller_thread(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        result = sub(ThreadId(x=1))
    assert result.outputs.out == threading.get_ident()


def test_debug_contextvar_visible(tmp_path):
    VAR.set("caller")
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        result = sub(ReadVar(x=1))
    assert result.outputs.out == "caller"


def test_debug_task_may_call_asyncio_run(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        assert sub(RunsAsyncio(x=1)).outputs.out == 2


def test_debug_inside_running_loop(tmp_path):
    async def outer():
        with Submitter(worker="debug", cache_root=tmp_path) as sub:
            return sub(Chain(x=1)).outputs.out

    assert asyncio.run(outer()) == 12
    assert "nest_asyncio" not in sys.modules


def test_debug_fail_fast_raises_task_exception_object(tmp_path):
    side_effect = tmp_path / "side.txt"
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        with pytest.raises(Boom) as info:
            sub(FailThenIndependent(path=str(side_effect)))
    assert info.value is BOOM
    assert not side_effect.exists()


def test_inline_executor_reports_outcomes(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            Chain(x=1), sub.store, on_error="raise_at_end", submitter=sub
        )
        executor = InlineExecutor(DebugWorker())
        assert (executor.batches, executor.checks_store, executor.inline) == (
            False,
            True,
            True,
        )
        reported = []
        executor.bind(lambda job, outcome: reported.append((job.name, outcome)))
        jobs = runner.ready().dispatch
        handle = executor.submit(jobs)
        assert isinstance(handle, Handle) and handle.executor is executor
        assert handle.jobs == tuple(jobs) and handle.open == set()
        assert [name for name, _ in reported] == ["A"]
        executor.cancel(handle)
        executor.close()


def test_drive_inline_completes_runner(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(Chain(x=1), sub.store, on_error="fail_fast", submitter=sub)
        drive_inline(
            runner, InlineExecutor(DebugWorker()), on_error="fail_fast", audit=sub.audit
        )
        assert runner.done and not runner.failed
        assert isinstance(sub.audit, Audit)


def test_deprecated_wrappers_warn(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        task = Chain(x=1)
        wf_job = _workflow_job(sub, task)
        with pytest.warns(DeprecationWarning, match="expand_workflow"):
            sub.expand_workflow(wf_job, rerun=False)
        runner = GraphRunner(task, sub.store, on_error="raise_at_end", submitter=sub)
        with pytest.warns(DeprecationWarning, match="get_runnable_tasks"):
            sub.get_runnable_tasks(runner.exec_graph)

    async def go():
        with Submitter(worker="debug", cache_root=tmp_path / "b") as sub2:
            job = _workflow_job(sub2, Chain(x=2))
            with pytest.warns(DeprecationWarning, match="expand_workflow_async"):
                await sub2.expand_workflow_async(job, rerun=False)

    asyncio.run(go())


def _workflow_job(sub, task):
    from pydra.engine.job import Job

    return Job(task, submitter=sub, name="main", environment=sub.environment)
