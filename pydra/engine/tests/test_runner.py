"""The graph runner: which jobs it releases, and what it does with their outcomes,
without a worker"""

import os
import time
import pytest
from pydra.compose import python, workflow
from pydra.engine.runner import Failed, GraphRunner, Round
from pydra.engine.submitter import Submitter
from pydra.engine.tests.utils import FunAddVar


@workflow.define
def Chain(x):
    a = workflow.add(FunAddVar(a=x, b=1), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=10), name="B")
    return b.out


@workflow.define
def Fan(x):
    a = workflow.add(FunAddVar(b=1).split("a", a=x), name="A")
    return a.out


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        yield sub


def runner_for(submitter, task, **kwargs) -> GraphRunner:
    kwargs.setdefault("on_error", "raise_at_end")
    return GraphRunner(task, submitter.store, submitter=submitter, **kwargs)


def test_ready_releases_roots_only(submitter):
    runner = runner_for(submitter, Chain(x=1))
    round = runner.ready()
    assert isinstance(round, Round)
    assert [j.name for j in round.dispatch] == ["A"]
    assert [j.name for j in runner.ready().dispatch] == []
    assert not runner.done


def test_complete_releases_reader(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    result = a.run()
    runner.complete(a, result)
    (b,) = runner.ready().dispatch
    assert b.name == "B"
    assert b.task.a == 2
    runner.complete(b, b.run())
    assert runner.ready().dispatch == []
    assert runner.done
    assert not runner.failed
    assert runner.stuck() is None


def test_failed_marks_readers_unrunnable(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    error = RuntimeError("could not run A")
    runner.complete(a, Failed(error))
    assert runner.ready().dispatch == []  # B is never dispatched
    assert runner.done
    assert runner.failed
    assert runner.first_error is error
    (entry,) = runner.entries
    assert entry.phase == "run"
    assert entry.job_name == "A"
    assert "could not run A" in entry.message
    (b,) = [n for n in runner.exec_graph.nodes if n.name == "B"]
    assert b.unrunnable == {None: [runner.exec_graph["A"]]}


def test_each_job_handed_out_once(submitter):
    runner = runner_for(submitter, Fan(x=[1, 2, 3, 4]))
    handed = []
    for _ in range(3):
        handed.extend(runner.ready().dispatch)
    assert len(handed) == 4
    assert len({id(j) for j in handed}) == 4
    assert sorted(j.state_index for j in handed) == [0, 1, 2, 3]


def test_ready_limit_caps_hand_out(submitter):
    runner = runner_for(submitter, Fan(x=[1, 2, 3, 4]))
    first = runner.ready(limit=3).dispatch
    second = runner.ready(limit=3).dispatch
    assert (len(first), len(second)) == (3, 1)


@pytest.mark.parametrize("propagate, inner_expected", [(False, False), (True, True)])
def test_rerun_propagation(submitter, propagate, inner_expected):
    runner = runner_for(submitter, Chain(x=1), rerun=True, propagate_rerun=propagate)
    (a,) = runner.ready().dispatch
    assert a.rerun is inner_expected


def test_no_rerun_by_default(submitter):
    (a,) = runner_for(submitter, Chain(x=1)).ready().dispatch
    assert a.rerun is False


def test_stuck_returns_diagnosis(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch  # handed out, never completed or run
    diagnosis = runner.stuck()
    assert diagnosis is not None
    assert (
        "Something has gone wrong when retrieving the predecessor results" in diagnosis
    )
    assert "B: started=False" in diagnosis
    assert "A: started=True" in diagnosis
    assert "'B' node blocked due to" in diagnosis


def test_runner_without_a_submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(Chain(x=1), sub.store, on_error="raise_at_end")
        assert [j.name for j in runner.ready().dispatch] == ["A"]


def test_deprecated_submitter_wrappers(submitter):
    runner = runner_for(submitter, Chain(x=1))
    with pytest.warns(DeprecationWarning, match="get_runnable_tasks"):
        jobs = submitter.get_runnable_tasks(runner.exec_graph)
    assert [j.name for j in jobs] == ["A"]
    with pytest.warns(DeprecationWarning, match="read_succeeded"):
        assert submitter.read_succeeded(jobs) == jobs
    with pytest.warns(DeprecationWarning, match="pop_resolution_errors"):
        assert submitter.pop_resolution_errors(runner.exec_graph) == []
