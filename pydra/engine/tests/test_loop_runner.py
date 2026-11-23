"""The runner's growth of a loop (RFC 0002 section 2.3(d), R11): each round of a
``workflow.loop`` is added to the loop job's graph only once the round before it is done
and its ``until`` output is false"""

import gc
import shutil
import typing as ty
import weakref

import pytest

from pydra.compose import python, workflow
from pydra.engine.absent import Absent
from pydra.engine.job import Job
from pydra.engine.loop import MAX_ADVANCES
from pydra.engine.result import Result
from pydra.engine.runner import Failed, GraphRunner, NotRun
from pydra.engine.store import SUCCESS_DIR, PickleStore
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
        progress = rnd.dispatch or rnd.settled or rnd.unrunnable or rnd.absent_nodes
        if not (progress or rnd.loops):
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
        assert [n.name for n in runner.exec_graph.nodes] == names[-2:]
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
    """The round whose until is an Absent ran, and ends the loop"""
    task = workflow.loop(UnsureAt2, until="done")(x=0)
    rounds = drive(runner_for(submitter, task))
    assert dispatched(rounds) == ["r0", "r1", "r2"]
    out = loop_outputs(submitter, task)
    assert out.stopped == "absent"
    assert out.converged == Absent(cause="unknown", source="UnsureAt2")
    assert (out.rounds, out.x) == (3, 3)


def test_carried_absent_propagates_and_stops(submitter):
    """Round 1 returns x=Absent("gone") with until false; round 2 reads it
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
    # r0 is released at the end of the call that adds r2, so r2 is still served r0's
    # outcome there
    assert dispatched(rounds) == ["r0", "r1"]
    assert [s.how for s in settled(rounds, "r2")] == ["shared"]
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
    """Outputs typed Any, Optional[int] and list[int] are returned as computed"""
    out = loop_outputs(submitter, workflow.loop(Typed, until="done")(x=0))
    assert (out.x, out.payload, out.maybe, out.items) == (2, {"k": 1}, 1, [0, 1])
    assert (out.rounds, out.stopped) == (2, "until")


def test_last_round_content_absent_returned(submitter):
    out = loop_outputs(submitter, workflow.loop(Fetch, until="done")(x=0))
    assert out.payload == Absent(cause="missing", source="Fetch")
    assert (out.rounds, out.converged, out.stopped) == (3, True, "until")


def test_carried_input_default(submitter):
    """A carried input with a default the loop is not given starts from it"""
    out = loop_outputs(submitter, workflow.loop(CountFrom, until="done")(stop=13))
    assert (out.n, out.rounds) == (13, 3)


# -- split, gate, absence and the store (L4) ---------------------------------------------------------


@python.define
def Ident(v: int) -> int:
    return v


@python.define
def Upto(n: int) -> list[int]:
    return [4 * 2**k for k in range(n)]


@python.define
def Positive(x: int) -> bool:
    return x > 0


@python.define
def MaybeAbsent(x: int) -> int:
    if x < 0:
        return Absent(cause="negative", source="MaybeAbsent")
    return x


@python.define(
    inputs={"x": python.arg(type=int, absent="accept")}, outputs=["x", "done"]
)
def Accepting(x) -> tuple[int, bool]:
    if isinstance(x, Absent):
        return 0, True
    return x, True


HalveLoop = workflow.loop(Halve, until="done")


@workflow.define(outputs=["x", "rounds"])
def OverSplit(xs: list[int]) -> tuple[list[int], list[int]]:
    s = workflow.add(Ident().split(v=xs), name="s")
    loop = workflow.add(HalveLoop(x=s.out, target=1), name="halve")
    return loop.x, loop.rounds


@workflow.define(outputs=["x", "rounds"])
def OverPartial(n: int) -> tuple[list[int], list[int]]:
    up = workflow.add(Upto(n=n), name="up")
    loop = workflow.add(HalveLoop(target=1).split(x=up.out), name="halve")
    return loop.x, loop.rounds


@workflow.define(outputs=["x", "rounds", "stopped"])
def GatedLoop(x: int) -> tuple[int, int, str]:
    pos = workflow.add(Positive(x=x), name="pos")
    loop = workflow.add(HalveLoop(x=x, target=1), name="halve", when=pos.out)
    return loop.x, loop.rounds, loop.stopped


@workflow.define(outputs=["x", "rounds"])
def AbsentIntoLoop(x: int, accept: bool) -> tuple[int, int]:
    m = workflow.add(MaybeAbsent(x=x), name="m")
    if accept:
        loop = workflow.add(workflow.loop(Accepting, until="done")(x=m.out), name="lp")
    else:
        loop = workflow.add(HalveLoop(x=m.out, target=1), name="lp")
    return loop.x, loop.rounds


@workflow.define(outputs=["a", "b"])
def TwoLoops(stop: int) -> tuple[int, int]:
    a = workflow.add(workflow.loop(Count, max_rounds=10)(n=0, stop=stop), name="a")
    b = workflow.add(workflow.loop(Count, max_rounds=11)(n=0, stop=stop), name="b")
    return a.n, b.n


def test_loop_downstream_of_split(submitter):
    """One loop per element, each with its own round count"""
    rounds = drive(runner_for(submitter, OverSplit(xs=[16, 4])))
    paths = sorted(
        {j.workflow_path for r in rounds for j in r.dispatch if j.name[0] == "r"}
    )
    assert paths == [("halve[0]",), ("halve[1]",)]
    out = loop_outputs(submitter, OverSplit(xs=[16, 4]))
    assert (out.x, out.rounds) == ([1, 1], [4, 2])


def test_loop_under_partial_split(submitter):
    """A loop split over an upstream output (E2) loops per element"""
    out = loop_outputs(submitter, OverPartial(n=3))  # 4, 8, 16
    assert (out.x, out.rounds) == ([1, 1, 1], [2, 3, 4])


def test_gated_off_loop_never_entered(submitter):
    rounds = drive(runner_for(submitter, GatedLoop(x=-8)))
    assert dispatched(rounds) == ["pos"]
    (loop,) = settled(rounds, "halve")
    assert loop.how == "gated"
    out = loop_outputs(submitter, GatedLoop(x=-8))
    for value in (out.x, out.rounds, out.stopped):
        assert value == Absent(cause="not-routed", source="halve")
    assert loop_outputs(submitter, GatedLoop(x=8)).rounds == 3


def test_absent_input_propagates_or_is_accepted(submitter):
    rounds = drive(runner_for(submitter, AbsentIntoLoop(x=-1, accept=False)))
    assert dispatched(rounds) == ["m"]
    assert [s.how for s in settled(rounds, "lp")] == ["absent"]
    out = loop_outputs(submitter, AbsentIntoLoop(x=-1, accept=False))
    assert out.x == out.rounds == Absent(cause="negative", source="MaybeAbsent")

    rounds = drive(runner_for(submitter, AbsentIntoLoop(x=-1, accept=True)))
    assert [n for n in dispatched(rounds) if n != "m"] == ["r0"]
    out = loop_outputs(submitter, AbsentIntoLoop(x=-1, accept=True))
    assert (out.x, out.rounds) == (0, 1)


def test_rerun_of_finished_loop_is_workflow_stored(submitter):
    loop_outputs(submitter, Outer(stop=4))
    rounds = drive(runner_for(submitter, Outer(stop=4)))
    assert [s.how for s in settled(rounds, "count")] == ["workflow-stored"]
    assert not [s for s in settled(rounds) if s.job.name[0] == "r"]
    assert [j for j in dispatched(rounds) if j[0] == "r"] == []


def test_rerun_without_loop_slot_replays_rounds(tmp_path):
    """The store is the loop's state: without the loop job's slot each round is read
    from its own and the next is instantiated from it"""
    cache_root = tmp_path / "pickled"
    with Submitter(worker="debug", store=PickleStore(cache_root)) as submitter:
        _replay_without_loop_slot(submitter, cache_root)


def _replay_without_loop_slot(submitter, cache_root):
    loop_outputs(submitter, Outer(stop=4))
    checksum = Job(CountLoop(n=0, stop=4), submitter=submitter, name="count").checksum
    shutil.rmtree(cache_root / checksum)
    for record in (cache_root / SUCCESS_DIR).glob(f"{checksum}-*"):
        record.unlink()
    rounds = drive(runner_for(submitter, Outer(stop=4)))
    assert [j for j in dispatched(rounds) if j[0] == "r"] == []
    assert [(s.job.name, s.how) for s in settled(rounds) if s.job.name[0] == "r"] == [
        (f"r{k}", "stored") for k in range(4)
    ]
    out = loop_outputs(submitter, Outer(stop=4))
    assert (out.n, out.rounds) == (4, 4)


def test_two_loops_with_equal_inputs_share_rounds(submitter):
    rounds = drive(runner_for(submitter, TwoLoops(stop=3)))
    ran = [(j.workflow_path, j.name) for r in rounds for j in r.dispatch]
    shared = [(s.job.workflow_path, s.job.name, s.how) for s in settled(rounds)]
    assert len(ran) == 3
    assert sorted(path for path, _ in ran) in ([("a",)] * 3, [("b",)] * 3)
    assert sorted(name for _, name, _ in shared) == ["r0", "r1", "r2"]
    assert {how for *_, how in shared} <= {"shared", "stored"}
    out = loop_outputs(submitter, TwoLoops(stop=3))
    assert (out.a, out.b) == (3, 3)


# -- release: a two-round window (L5) ----------------------------------------------------------------


@python.define
def NoList(x: int) -> list[int]:
    return Absent(cause="no-list", source="NoList")


@workflow.define(outputs=["n", "done"])
def CountWf(n: int, stop: int) -> tuple[int, bool]:
    """A round that is a workflow, with a node absent as a whole in each round"""
    gone = workflow.add(NoList(x=n), name="gone")
    workflow.add(Ident().split(v=gone.out), name="each")
    step = workflow.add(Count(n=n, stop=stop), name="step")
    return step.n, step.done


CountWfLoop = workflow.loop(CountWf, until="done")


def test_released_rounds_are_freed(submitter):
    """Only the two newest rounds are held: round 0's node and job are freed once
    round 3 is done, and the runner's tables do not grow with the rounds"""
    runner = runner_for(submitter, CountLoop(n=0, stop=200))
    rnd = runner.ready()
    (job,) = rnd.dispatch
    node_ref = weakref.ref(runner.exec_graph.node("r0").node)
    job_ref = weakref.ref(job)
    sizes = []
    for k in range(200):
        assert job.name == f"r{k}"
        runner.complete(job, run_inline(job))
        del job, rnd
        if k == 3:
            gc.collect()
            assert node_ref() is None and job_ref() is None
        rnd = runner.ready()
        sizes.append(
            (len(runner._handed), len(runner._frame_of), len(runner._finished))
        )
        if not rnd.dispatch:
            break
        (job,) = rnd.dispatch
        assert len(runner.exec_graph.nodes) <= 2
    assert runner.done
    assert all(max(size) <= 2 for size in sizes), sizes
    out = loop_outputs(submitter, CountLoop(n=0, stop=200))
    assert (out.n, out.rounds) == (200, 200)


def test_workflow_rounds_release_child_frames(submitter):
    """A round that is a workflow has a frame of its own, dropped with the round; each
    round's absent-node record is reported, even where a released node's id is reused"""
    runner = runner_for(submitter, CountWfLoop(n=0, stop=60))
    frames = set()
    records = []
    for _ in range(2000):
        if runner.done:
            break
        rnd = runner.ready()
        records += [a.node for a in rnd.absent_nodes]
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
        frames.add(len(runner._frames))
    assert runner.done
    assert max(frames) <= 4, frames
    assert records == [f"r{k}/each" for k in range(60)]


def test_record_of_a_round_released_in_the_same_call_is_reported(submitter):
    """A workflow round whose absent node is found in the call that also releases it
    (the later rounds are served from the store) still has its record reported once"""
    submitter(CountWfLoop(n=1, stop=4))  # stores the rounds from n=1
    rounds = drive(runner_for(submitter, CountWfLoop(n=0, stop=4)))
    hows = [(s.job.name, s.how) for s in settled(rounds) if not s.job.workflow_path]
    assert hows == [(f"r{k}", "workflow-stored") for k in (1, 2, 3)]
    assert [a.node for r in rounds for a in r.absent_nodes] == ["r0/each"]


def test_upstream_of_a_released_round_is_refused(submitter):
    runner = runner_for(submitter, CountLoop(n=0, stop=5))
    jobs = []
    for _ in range(3):
        (job,) = runner.ready().dispatch
        runner.complete(job, run_inline(job))
        jobs.append(job)
    assert [j.name for j in jobs] == ["r0", "r1", "r2"]
    (r3,) = runner.ready().dispatch  # r1 is released
    assert [j.name for j, _ in runner.upstream(r3)["n"]] == ["r2"]
    with pytest.raises(LookupError, match="r1"):
        runner.upstream(jobs[1])


# -- reporting loops (L6) ----------------------------------------------------------------------------


def test_driver_sees_one_loop_closed_per_loop_instance(submitter):
    """Each loop job that is entered is reported once; an element whose loop job is a
    twin of another's is served its result and is not a loop instance of its own"""
    runner = runner_for(submitter, OverSplit(xs=[16, 4, 8]))
    closed = []
    for _ in range(200):
        if runner.done:
            break
        rnd = runner.ready()
        closed += rnd.loops
        for job in rnd.dispatch:
            runner.complete(job, run_inline(job))
    assert runner.done
    assert sorted((c.loop, c.rounds, c.stopped) for c in closed) == [
        ("halve[0]", 4, "until"),
        ("halve[1]", 2, "until"),
        ("halve[2]", 3, "until"),
    ]


def test_loop_closed_reports_error_and_cycle(submitter):
    @workflow.define(outputs=["x"])
    def Wf(x: int) -> int:
        return workflow.add(workflow.loop(BoomAt2, until="done")(x=x), name="boom").x

    def complete(job):
        try:
            return run_inline(job)
        except RuntimeError as exc:
            return Failed(exc)

    rounds = drive(runner_for(submitter, Wf(x=0)), complete)
    assert [(c.loop, c.rounds, c.stopped) for r in rounds for c in r.loops] == [
        ("boom", 3, "error")
    ]
    task = workflow.loop(Clamp, until="done")(x=1)
    rounds = drive(runner_for(submitter, task))
    assert [(c.loop, c.rounds, c.stopped) for r in rounds for c in r.loops] == [
        ("ClampLoop", 3, "cycle")
    ]


def test_loop_closing_alone_is_reported_before_done(submitter):
    """The ``ready`` that closes an unowned root loop reports it, and ``done`` waits for
    that report"""
    runner = runner_for(submitter, CountLoop(n=0, stop=1))
    (r0,) = runner.ready().dispatch
    runner.complete(r0, run_inline(r0))
    assert not runner.done
    rnd = runner.ready()
    assert (rnd.dispatch, rnd.settled) == ([], [])
    assert [(c.loop, c.rounds, c.stopped) for c in rnd.loops] == [
        ("CountLoop", 1, "until")
    ]
    assert runner.done


@pytest.mark.parametrize("cap", [None, 8])
def test_replayed_loop_yields_to_its_driver(tmp_path, cap):
    """A loop replayed from the store adds at most ``max_advances`` rounds per ``ready``"""
    cap_kw = {} if cap is None else {"max_advances": cap}
    cap = MAX_ADVANCES if cap is None else cap
    cache_root = tmp_path / "pickled"
    with Submitter(worker="debug", store=PickleStore(cache_root)) as submitter:
        loop_outputs(submitter, Outer(stop=150))
        checksum = Job(
            CountLoop(n=0, stop=150), submitter=submitter, name="count"
        ).checksum
        shutil.rmtree(cache_root / checksum)
        for record in (cache_root / SUCCESS_DIR).glob(f"{checksum}-*"):
            record.unlink()
        rounds = drive(runner_for(submitter, Outer(stop=150), **cap_kw))
    per_call = [len([s for s in r.settled if s.job.name[0] == "r"]) for r in rounds]
    assert sum(per_call) == 150
    assert max(per_call) <= cap + 1
    assert len([n for n in per_call if n]) >= 3
