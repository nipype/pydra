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


@python.define
def Hold(x: int, logdir: str, partners: int = 0, total: int = 0) -> int:
    """Records when it ran; given ``partners``, first waits (a bound only) until that
    many of the ``total`` jobs run at once, itself included, or all have started"""
    t0 = time.time()
    sync = logdir + "-sync"
    if partners:
        os.makedirs(sync, exist_ok=True)
        open(os.path.join(sync, f"start-{x}"), "w").close()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            names = os.listdir(sync)
            started = sum(n.startswith("start-") for n in names)
            running = started - sum(n.startswith("end-") for n in names)
            if running >= partners or started == total:
                break
            time.sleep(0.02)
    if partners:  # before its end is read, so a job that saw it running overlaps it
        open(os.path.join(sync, f"end-{x}"), "w").close()
    with open(os.path.join(logdir, f"{x}.txt"), "w") as f:
        f.write(f"{t0} {time.time()}")
    return x


@workflow.define
def HoldMany(
    xs: list[int], logdir: str, partners: int = 0, total: int = 0
) -> list[int]:
    h = workflow.add(Hold(logdir=logdir, partners=partners, total=total).split(x=xs))
    return h.out


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


def count_in_flight(monkeypatch) -> dict[str, int]:
    """The jobs the runner has handed out and had no outcome reported for, counted in
    this process: ``now``, and the most at once, ``most``"""
    counts = {"now": 0, "most": 0}
    real_ready, real_complete = GraphRunner.ready, GraphRunner.complete

    def ready(self, *args, **kwargs):
        rnd = real_ready(self, *args, **kwargs)
        counts["now"] += len(rnd.dispatch)
        counts["most"] = max(counts["most"], counts["now"])
        return rnd

    def complete(self, job, outcome):
        counts["now"] -= 1
        return real_complete(self, job, outcome)

    monkeypatch.setattr(GraphRunner, "ready", ready)
    monkeypatch.setattr(GraphRunner, "complete", complete)
    return counts


def test_max_concurrent_caps_running_jobs(any_worker, tmp_path, monkeypatch):
    """No more than ``max_concurrent`` jobs of a workflow run at once"""
    in_flight = count_in_flight(monkeypatch)
    logdir = tmp_path / "log"
    logdir.mkdir()
    # on cf each job waits for a second one to run beside it, so the cap is reached
    # whatever the machine's speed; a serial worker has no second one to wait for
    # (the cf pool is larger than the cap whatever the host)
    partners, pool = (2, {"n_procs": 4}) if any_worker == "cf" else (0, {})
    with Submitter(
        worker=any_worker, cache_root=tmp_path / "cache", max_concurrent=2, **pool
    ) as sub:
        result = sub(
            HoldMany(xs=list(range(8)), logdir=str(logdir), partners=partners, total=8)
        )
    assert not result.errored
    spans = [tuple(map(float, f.read_text().split())) for f in logdir.iterdir()]
    assert len(spans) == 8
    # the cap: never more than two handed out without an outcome
    assert in_flight["most"] <= 2 and in_flight["now"] == 0
    if any_worker == "cf":  # and the cap is reached: two ran at once
        peak = max(sum(a2 <= a < b2 for a2, b2 in spans) for a, _ in spans)
        assert peak >= 2


def test_deprecated_submitter_wrappers(submitter):
    runner = runner_for(submitter, Chain(x=1))
    with pytest.warns(DeprecationWarning, match="get_runnable_tasks"):
        jobs = submitter.get_runnable_tasks(runner.exec_graph)
    assert [j.name for j in jobs] == ["A"]
    with pytest.warns(DeprecationWarning, match="read_succeeded"):
        assert submitter.read_succeeded(jobs) == jobs
    with pytest.warns(DeprecationWarning, match="pop_resolution_errors"):
        assert submitter.pop_resolution_errors(runner.exec_graph) == []
