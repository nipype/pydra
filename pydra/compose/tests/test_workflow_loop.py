"""Bounded loops unrolled at construction (RFC 0002 section 2.3(d), R11): ``workflow.loop``"""

import typing as ty

import pytest

from pydra.compose import python, workflow
from pydra.engine.absent import Absent
from pydra.engine.result import Result
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter

# -- bodies ------------------------------------------------------------------------------------------


@python.define(outputs=["x", "done"])
def Halve(x: int, target: int) -> tuple[int, bool]:
    y = x // 2
    return y, y <= target


@python.define(outputs=["x", "done"])
def Poll(x: int, attempt: int = 0) -> tuple[int, bool]:
    """Carries nothing forward: its output equals its input"""
    return x, False


@python.define(outputs=["x", "done"])
def Clamp(x: int) -> tuple[int, bool]:
    """Reaches the fixed point 0 and never reports done"""
    return max(x - 1, 0), False


@python.define(outputs=["x", "payload", "done"])
def Fetch(x: int) -> tuple[int, ty.Any, bool]:
    """Converges at ``x == 2`` with a content ``Absent`` payload"""
    if x >= 2:
        return x, Absent(cause="missing", source="Fetch"), True
    return x + 1, x, False


# -- workflows ---------------------------------------------------------------------------------------


@workflow.define(outputs=["x", "rounds", "converged"])
def HalveLoop(x: int, target: int, max_rounds: int) -> tuple[int, int, bool]:
    out = workflow.loop(
        Halve, until="done", max_rounds=max_rounds, name="halve", x=x, target=target
    )
    return out.x, out.rounds, out.converged


@workflow.define(outputs=["x", "rounds"])
def PollLoop(x: int, indexed: bool) -> tuple[int, int]:
    out = workflow.loop(
        Poll,
        until="done",
        max_rounds=3,
        index="attempt" if indexed else None,
        name="poll",
        x=x,
    )
    return out.x, out.rounds


@workflow.define(outputs=["x", "rounds", "converged"])
def ClampLoop(x: int) -> tuple[int, int, bool]:
    out = workflow.loop(Clamp, until="done", max_rounds=4, name="clamp", x=x)
    return out.x, out.rounds, out.converged


@workflow.define(outputs=["payload", "rounds", "converged"])
def FetchLoop(x: int) -> tuple[ty.Any, int, bool]:
    out = workflow.loop(Fetch, until="done", max_rounds=5, name="fetch", x=x)
    return out.payload, out.rounds, out.converged


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
    names = [
        f.name for f in type(task).Outputs.__attrs_attrs__ if not f.name.startswith("_")
    ]
    if len(names) == 1:
        out = (out,)
    return Result(
        cache_dir=job.cache_dir,
        outputs=task.Outputs(**dict(zip(names, out))),
        task=task,
    )


def drive(runner):
    """Drive a runner by hand to the end, returning every round"""
    rounds = []
    for _ in range(200):
        if runner.done:
            return rounds
        rnd = runner.ready()
        rounds.append(rnd)
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
        if not (rnd.dispatch or rnd.settled or rnd.unrunnable or rnd.absent_nodes):
            raise AssertionError(runner.stuck())
    raise AssertionError("the run did not finish")


def runner_for(sub, task):
    return GraphRunner(
        task,
        sub.store,
        submitter=sub,
        on_error="raise_at_end",
        lookup_before_dispatch=True,
    )


def settled(rounds, name):
    return [s for r in rounds for s in r.settled if s.job.name == name]


def is_round(name, prefix):
    """Whether ``name`` is a round ``<prefix><k>`` rather than a merge node of the loop"""
    return name.startswith(prefix) and name[len(prefix) :].isdigit()


def dispatched(rounds, prefix):
    return [j.name for r in rounds for j in r.dispatch if is_round(j.name, prefix)]


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        yield sub


# -- tests -------------------------------------------------------------------------------------------


def test_loop_converges_at_round_k(submitter):
    """16 -> 8 -> 4 -> 2: done at round 2, so rounds 3..5 are unused and never dispatched"""
    task = HalveLoop(x=16, target=2, max_rounds=6)
    names = sorted(n for n in task.construct().node_names if is_round(n, "halve."))
    assert names == [f"halve.{k}" for k in range(6)]

    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds, "halve.") == ["halve.0", "halve.1", "halve.2"]
    (first_unused,) = settled(rounds, "halve.3")
    assert first_unused.how == "gated"
    assert first_unused.result.outputs.x == Absent(cause="not-routed", source="halve.3")
    for k in (4, 5):
        # a round whose predecessor was gated off reads an Absent gate and propagates
        (later,) = settled(rounds, f"halve.{k}")
        assert later.how == "absent"
        assert later.result.outputs.x.cause == "not-routed"

    outputs = submitter(task).outputs
    assert (outputs.x, outputs.rounds, outputs.converged) == (2, 3, True)


def test_loop_exhausts_max_rounds(submitter):
    """16 -> 8 -> 4 -> 2 never reaches 0: three rounds run and the loop did not converge"""
    outputs = submitter(HalveLoop(x=16, target=0, max_rounds=3)).outputs
    assert (outputs.x, outputs.rounds, outputs.converged) == (2, 3, False)
    # converging on the very last round is still convergence
    outputs = submitter(HalveLoop(x=16, target=2, max_rounds=3)).outputs
    assert (outputs.x, outputs.rounds, outputs.converged) == (2, 3, True)


def test_index_gives_distinct_checksums(submitter):
    """A body that carries nothing forward: index= keys each round, index=None folds them"""
    rounds = drive(runner_for(submitter, PollLoop(x=7, indexed=True)))
    jobs = [j for r in rounds for j in r.dispatch if is_round(j.name, "poll.")]
    assert sorted(j.name for j in jobs) == ["poll.0", "poll.1", "poll.2"]
    assert sorted(j.task.attempt for j in jobs) == [0, 1, 2]
    assert len({j.checksum for j in jobs}) == 3

    rounds = drive(runner_for(submitter, PollLoop(x=7, indexed=False)))
    jobs = [s.job for r in rounds for s in r.settled if is_round(s.job.name, "poll.")]
    jobs += [j for r in rounds for j in r.dispatch if is_round(j.name, "poll.")]
    assert len({j.checksum for j in jobs}) == 1

    outputs = submitter(PollLoop(x=7, indexed=True)).outputs
    assert (outputs.x, outputs.rounds) == (7, 3)


def test_fixed_point_folds_equal_rounds(submitter):
    """2 -> 1 -> 0 -> 0 -> 0: rounds past the fixed point share one checksum and run once"""
    rounds = drive(runner_for(submitter, ClampLoop(x=2)))
    assert dispatched(rounds, "clamp.") == ["clamp.0", "clamp.1", "clamp.2"]
    (folded,) = settled(rounds, "clamp.3")
    assert folded.how == "shared"
    jobs = {s.job.name: s.job for r in rounds for s in r.settled}
    jobs.update({j.name: j for r in rounds for j in r.dispatch})
    assert jobs["clamp.3"].checksum == jobs["clamp.2"].checksum
    assert jobs["clamp.2"].checksum != jobs["clamp.1"].checksum

    outputs = submitter(ClampLoop(x=2)).outputs
    assert (outputs.x, outputs.rounds, outputs.converged) == (0, 4, False)


def test_last_round_content_absent_returned(submitter):
    """The last round's content ``Absent`` is returned with its cause, chosen on the gates"""
    outputs = submitter(FetchLoop(x=0)).outputs
    assert outputs.payload == Absent(cause="missing", source="Fetch")
    assert (outputs.rounds, outputs.converged) == (3, True)


def test_loop_refuses_bad_wiring():
    """``until`` must name a bool output, ``index`` an input, and ``max_rounds`` be >= 1"""

    def build(**kwargs):
        @workflow.define(outputs=["x"])
        def Wf(x: int) -> int:
            return workflow.loop(Halve, name="h", x=x, target=0, **kwargs).x

        Wf(x=1).construct()

    with pytest.raises(ValueError, match="until"):
        build(until="x", max_rounds=2)
    with pytest.raises(ValueError, match="until"):
        build(until="missing", max_rounds=2)
    with pytest.raises(ValueError, match="index"):
        build(until="done", max_rounds=2, index="missing")
    with pytest.raises(ValueError, match="max_rounds"):
        build(until="done", max_rounds=0)
