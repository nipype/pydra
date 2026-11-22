"""The runner's growth of a loop (RFC 0002 section 2.3(d), R11): each round of a
``workflow.loop`` is added to the loop job's graph only once the round before it is done
and its ``until`` output is false"""

import pytest

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.utils.general import get_fields

# -- bodies ------------------------------------------------------------------------------------------


@python.define(outputs=["n", "done"])
def Count(n: int, stop: int) -> tuple[int, bool]:
    return n + 1, n + 1 >= stop


@python.define(outputs=["x", "done"])
def Halve(x: int, target: int) -> tuple[int, bool]:
    y = x // 2
    return y, y <= target


CountLoop = workflow.loop(Count, until="done")
CountTo1 = workflow.loop(Count, until="done", max_rounds=1)


@workflow.define(outputs=["n", "rounds", "after"])
def Outer(stop: int) -> tuple[int, int, int]:
    loop = workflow.add(CountLoop(n=0, stop=stop), name="count")
    after = workflow.add(Count(n=loop.n, stop=0), name="after")
    return loop.n, loop.rounds, after.n


# -- helpers -----------------------------------------------------------------------------------------


def run_inline(job) -> Result:
    """The outcome of a python job, computed by the test without the store"""
    task = job.task
    inputs = {
        f.name: getattr(task, f.name)
        for f in type(task).__attrs_attrs__
        if f.name != "function" and not f.name.startswith("_")
    }
    out = task.function(**inputs)
    names = [f.name for f in get_fields(task.Outputs)]
    if len(names) == 1:
        out = (out,)
    return Result(
        cache_dir=job.cache_dir,
        outputs=task.Outputs(**dict(zip(names, out))),
        task=task,
    )


def drive(runner, complete=run_inline, limit=2000):
    """Drive a runner by hand to the end, returning every round"""
    rounds = []
    for _ in range(limit):
        if runner.done:
            return rounds
        rnd = runner.ready()
        rounds.append(rnd)
        for job in rnd.dispatch:
            runner.complete(job, complete(job))
        if not (rnd.dispatch or rnd.settled or rnd.unrunnable or rnd.absent_nodes):
            if runner.done:  # as `drive_inline` decides
                return rounds
            raise AssertionError(runner.stuck())
    raise AssertionError("the run did not finish")


def runner_for(sub, task, **kwargs):
    return GraphRunner(
        task,
        sub.store,
        submitter=sub,
        on_error="raise_at_end",
        lookup_before_dispatch=True,
        **kwargs,
    )


def dispatched(rounds):
    return [j.name for r in rounds for j in r.dispatch]


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        yield sub


# -- growth ------------------------------------------------------------------------------------------


def test_rounds_added_one_at_a_time(submitter):
    """The loop's graph holds only the rounds instantiated so far: never one ahead"""
    runner = runner_for(submitter, CountLoop(n=0, stop=5))
    names = []
    while not runner.done:
        rnd = runner.ready()
        assert len(rnd.dispatch) <= 1
        names += [j.name for j in rnd.dispatch]
        assert [n.name for n in runner.exec_graph.nodes] == names
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
    assert names == ["r0", "r1", "r2", "r3", "r4"]


def test_not_done_between_rounds_unowned_root(submitter):
    """The runner is not done after a round that does not end the loop completes, before
    the next ``ready`` adds the next round"""
    runner = runner_for(submitter, CountLoop(n=0, stop=3))
    (r0,) = runner.ready().dispatch
    runner.complete(r0, run_inline(r0))
    assert not runner.done
    (r1,) = runner.ready().dispatch
    assert r1.name == "r1"


def test_not_done_between_rounds_owned_root(submitter):
    task = CountLoop(n=0, stop=3)
    job = Job(task, submitter=submitter, name="main")
    runner = runner_for(submitter, task, job=job)
    (r0,) = runner.ready().dispatch
    runner.complete(r0, run_inline(r0))
    assert not runner.done
    rounds = drive(runner)
    assert dispatched(rounds) == ["r1", "r2"]
    outputs = job.result().outputs
    assert (outputs.n, outputs.done, outputs.rounds) == (3, True, 3)


def test_nested_loop_feeds_reader(submitter):
    rounds = drive(runner_for(submitter, Outer(stop=4)))
    assert dispatched(rounds) == ["r0", "r1", "r2", "r3", "after"]
    loop_jobs = [j for r in rounds for j in r.dispatch if j.name.startswith("r")]
    assert {j.workflow_path for j in loop_jobs} == {("count",)}
    outputs = submitter(Outer(stop=4)).outputs
    assert (outputs.n, outputs.rounds, outputs.after) == (4, 4, 5)


def test_loop_through_submitter(submitter):
    outputs = submitter(workflow.loop(Halve, until="done")(x=16, target=1)).outputs
    assert (outputs.x, outputs.done, outputs.rounds) == (1, True, 4)
    assert (outputs.converged, outputs.stopped) == (True, "until")


# -- max_rounds --------------------------------------------------------------------------------------


def test_max_rounds_one(submitter):
    rounds = drive(runner_for(submitter, CountTo1(n=0, stop=5)))
    assert dispatched(rounds) == ["r0"]
    outputs = submitter(CountTo1(n=0, stop=5)).outputs
    assert (outputs.n, outputs.rounds, outputs.converged) == (1, 1, False)
    assert outputs.stopped == "max-rounds"


def test_unbounded_loop_runs_fifty_rounds(submitter):
    rounds = drive(runner_for(submitter, CountLoop(n=0, stop=50)))
    assert dispatched(rounds) == [f"r{k}" for k in range(50)]
    outputs = submitter(CountLoop(n=0, stop=50)).outputs
    assert (outputs.n, outputs.rounds, outputs.stopped) == (50, 50, "until")


# -- until is read as a gate -------------------------------------------------------------------------


class Reference:
    """What an external driver might complete a job with in place of a value"""


def test_until_of_wrong_type_fails_the_loop(submitter):
    def complete(job):
        result = run_inline(job)
        if job.name == "r1":
            object.__setattr__(result.outputs, "done", Reference())
        return result

    runner = runner_for(submitter, Outer(stop=5))
    drive(runner, complete)
    assert runner.failed
    (entry,) = runner.entries
    assert entry.job_name == "count"
    assert "until" in entry.message and "'done'" in entry.message
    assert "Reference" in entry.message
    assert entry.exception_type.endswith("GateTypeError")


def test_until_of_wrong_type_unowned_root(submitter):
    def complete(job):
        result = run_inline(job)
        object.__setattr__(result.outputs, "done", 1)
        return result

    runner = runner_for(submitter, CountLoop(n=0, stop=5))
    drive(runner, complete)
    assert runner.failed
    assert "until" in runner.errors[0]
    (entry,) = runner.entries
    assert entry.exception_type.endswith("GateTypeError")
