"""The runner's growth of a loop (RFC 0002 section 2.3(d), R11): each round of a
``workflow.loop`` is added to the loop job's graph only once the round before it is done
and its ``until`` output is false"""

import typing as ty

import pytest

from pydra.compose import python, workflow
from pydra.engine.absent import Absent
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.runner import Failed, GraphRunner, NotRun
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


# -- stopping and the loop's outputs (L3) ------------------------------------------------------------


@python.define(outputs=["x", "done"])
def UnsureAt2(x: int) -> tuple[int, bool]:
    """A content ``Absent`` until at ``x == 2``"""
    if x == 2:
        return x + 1, Absent(cause="unknown", source="UnsureAt2")
    return x + 1, False


@python.define(outputs=["x", "done"])
def GoneAt1(x: int) -> tuple[int, bool]:
    """A content ``Absent`` carried value at ``x == 1``, with until false"""
    if x == 1:
        return Absent(cause="gone", source="GoneAt1"), False
    return x + 1, False


@python.define(outputs=["x", "done"])
def Clamp(x: int) -> tuple[int, bool]:
    """Reaches the fixed point 0 and never reports done"""
    return max(x - 1, 0), False


@python.define(outputs=["x", "done"])
def Flip(x: int) -> tuple[int, bool]:
    """A cycle of length 2"""
    return 1 - x, False


@python.define(outputs=["x", "done"])
def Poll(x: int, attempt: int = 0) -> tuple[int, bool]:
    """Carries nothing new forward: its output equals its input"""
    return x, False


@python.define(outputs=["x", "done"])
def BoomAt2(x: int) -> tuple[int, bool]:
    if x == 2:
        raise RuntimeError("boom at 2")
    return x + 1, False


@python.define(outputs=["x", "payload", "maybe", "items", "done"])
def Typed(
    x: int,
) -> tuple[int, ty.Any, ty.Optional[int], list[int], bool]:
    return x + 1, {"k": x}, x, list(range(x + 1)), x + 1 >= 2


@python.define(outputs=["x", "payload", "done"])
def Fetch(x: int) -> tuple[int, ty.Any, bool]:
    """Converges at ``x == 2`` with a content ``Absent`` payload"""
    if x >= 2:
        return x, Absent(cause="missing", source="Fetch"), True
    return x + 1, x, False


@python.define(outputs=["n", "done"])
def CountFrom(stop: int, n: int = 10) -> tuple[int, bool]:
    return n + 1, n + 1 >= stop


def settled(rounds, name=None):
    return [s for r in rounds for s in r.settled if name is None or s.job.name == name]


def loop_outputs(submitter, task):
    result = submitter(task)
    assert not result.errored
    return result.outputs


def test_content_absent_until_stops_absent(submitter):
    """I1 probe W1: the round whose until is an Absent ran, and ends the loop"""
    task = workflow.loop(UnsureAt2, until="done")(x=0)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0", "r1", "r2"]
    out = loop_outputs(submitter, task)
    assert out.stopped == "absent"
    assert out.converged == Absent(cause="unknown", source="UnsureAt2")
    assert (out.rounds, out.x) == (3, 3)


def test_carried_absent_propagates_and_stops(submitter):
    """I1 probe W2: round 1 returns x=Absent("gone") with until false; round 2 reads it
    under propagate, completes absent without running, and ends the loop"""
    task = workflow.loop(GoneAt1, until="done")(x=0)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0", "r1"]
    (r2,) = settled(rounds, "r2")
    assert r2.how == "absent"
    out = loop_outputs(submitter, task)
    assert out.stopped == "absent"
    assert out.rounds == 2
    assert out.converged == Absent(cause="gone", source="GoneAt1")
    assert out.x == Absent(cause="gone", source="GoneAt1")


def test_fixed_point_stops_on_cycle(submitter):
    """2 -> 1 -> 0 -> 0: the repeated round is served, never dispatched, and ends it"""
    task = workflow.loop(Clamp, until="done", max_rounds=20)(x=2)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0", "r1", "r2"]
    (folded,) = settled(rounds, "r3")
    assert folded.how in {"shared", "stored"}
    jobs = {s.job.name: s.job for s in settled(rounds)}
    jobs.update({j.name: j for r in rounds for j in r.dispatch})
    assert jobs["r3"].checksum == jobs["r2"].checksum
    out = loop_outputs(submitter, task)
    assert (out.x, out.rounds, out.converged, out.stopped) == (0, 4, False, "cycle")


def test_cycle_of_length_two(submitter):
    task = workflow.loop(Flip, until="done", max_rounds=20)(x=0)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0", "r1"]
    assert [s.how for s in settled(rounds, "r2")] in (["shared"], ["stored"])
    out = loop_outputs(submitter, task)
    assert (out.x, out.rounds, out.stopped) == (1, 3, "cycle")


def test_index_gives_distinct_checksums(submitter):
    task = workflow.loop(Poll, until="done", carry={}, index="attempt", max_rounds=3)(
        x=7
    )
    rounds = drive(runner_for(submitter, task))
    jobs = [j for r in rounds for j in r.dispatch]
    assert [j.name for j in jobs] == ["r0", "r1", "r2"]
    assert [j.task.attempt for j in jobs] == [0, 1, 2]
    assert len({j.checksum for j in jobs}) == 3
    out = loop_outputs(submitter, task)
    assert (out.x, out.rounds, out.stopped) == (7, 3, "max-rounds")


def test_poll_without_index_stops_after_two_rounds(submitter):
    """A poll that forgot its index folds round 1 onto round 0 and stops, visibly"""
    task = workflow.loop(Poll, until="done", carry={"x": "x"})(x=7)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0"]
    assert [s.how for s in settled(rounds, "r1")] in (["shared"], ["stored"])
    out = loop_outputs(submitter, task)
    assert (out.rounds, out.converged, out.stopped) == (2, False, "cycle")


def test_failed_round_fails_the_loop(submitter):
    @workflow.define(outputs=["x"])
    def Wf(x: int) -> int:
        return workflow.add(workflow.loop(BoomAt2, until="done")(x=x), name="boom").x

    def complete(job):
        try:
            return run_inline(job)
        except RuntimeError as exc:
            return Failed(exc)

    runner = runner_for(submitter, Wf(x=0))
    rounds = drive(runner, complete)
    assert dispatched(rounds) == ["r0", "r1", "r2"]
    assert runner.failed
    (entry,) = runner.entries
    assert entry.job_name == "boom"
    assert "boom at 2" in entry.message


def test_not_run_round_waits_for_retry(submitter):
    runner = runner_for(submitter, CountLoop(n=0, stop=3))
    (r0,) = runner.ready().dispatch
    runner.complete(r0, run_inline(r0))
    (r1,) = runner.ready().dispatch
    runner.complete(r1, NotRun(reason="cancelled", held=False))
    assert runner.ready().dispatch == []
    assert runner.done and [j.name for j, _ in runner.not_run] == ["r1"]
    runner.retry(r1)
    rounds = drive(runner)
    assert dispatched(rounds) == ["r1", "r2"]


def test_untyped_and_optional_outputs_returned(submitter):
    """B2: outputs typed Any, Optional[int] and list[int] are returned as computed"""
    out = loop_outputs(submitter, workflow.loop(Typed, until="done")(x=0))
    assert (out.x, out.payload, out.maybe, out.items) == (2, {"k": 1}, 1, [0, 1])
    assert (out.rounds, out.stopped) == (2, "until")


def test_last_round_content_absent_returned(submitter):
    out = loop_outputs(submitter, workflow.loop(Fetch, until="done")(x=0))
    assert out.payload == Absent(cause="missing", source="Fetch")
    assert (out.rounds, out.converged, out.stopped) == (3, True, "until")


def test_carried_input_default(submitter):
    """N5: a carried input with a default the loop is not given starts from it"""
    out = loop_outputs(submitter, workflow.loop(CountFrom, until="done")(stop=13))
    assert (out.n, out.rounds) == (13, 3)
