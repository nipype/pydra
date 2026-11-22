"""``GraphRunner.plan()`` (RFC 0002 section 3.1, R8): what would run, be reused or wait,
read from the runner's graph and the store without running, leasing or writing anything
"""

import os
import time
from pathlib import Path

import pytest

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.result import Result, save
from pydra.engine.plan import Plan, PlanNode
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.engine.tests.test_runner_driver_model import (
    Chain,
    Outer,
    MemoryStore,
    Siblings,
    compute,
    drive_by_hand,
    runner_for,
)
from pydra.engine.tests.utils import FunAddVar
from pydra.utils.resources import ComputeResources

GPU = ComputeResources(gpus=1, tags=("gpu",))


@pytest.fixture
def submitter(tmp_path):
    with Submitter(
        worker="debug", store="pickle", cache_root=tmp_path / "cache"
    ) as sub:
        yield sub


@pytest.fixture
def memory(tmp_path):
    return MemoryStore(tmp_path / "mem")


def snapshot(root: Path):
    return (
        sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        if root.exists()
        else []
    )


def by_path(plan: Plan) -> dict[str, PlanNode]:
    return {n.path: n for n in plan.nodes}


@workflow.define(outputs=["b", "d"])
def Fan(xs: list[int]):
    a = workflow.add(FunAddVar(b=1).split(a=xs), name="A", resources=GPU)
    b = workflow.add(FunAddVar(a=a.out, b=1), name="B")
    c = workflow.add(FunAddVar(a=7, b=10), name="C")
    return b.out, c.out


@workflow.define
def OverOutput(xs: list[int]):
    a = workflow.add(FunAddVar(b=1).split(a=xs).combine("a"), name="A")
    b = workflow.add(FunAddVar(b=2).split(a=a.out), name="B")
    return b.out


def test_plan_runs_nothing(submitter, tmp_path):
    runner = runner_for(submitter, Chain(x=1))
    before = snapshot(tmp_path)
    progress = runner._progress()
    plan = runner.plan()
    assert isinstance(plan, Plan)
    assert [n.path for n in plan.nodes] == ["A", "B"]
    assert snapshot(tmp_path) == before
    # the runner is as it was: nothing started, handed out or settled
    assert runner._progress() == progress and not runner._handed
    assert [j.name for j in runner.ready().dispatch] == ["A"]
    assert [n.path for n in runner.plan().nodes] == ["A", "B"]  # A is not done
    assert runner.ready().dispatch == []  # B was not released by planning
    runner = runner_for(submitter, Chain(x=2))
    runner.plan()
    drive_by_hand(runner)  # a run that was planned still completes
    assert runner.done and runner.plan().nodes == ()


def test_plan_split_sizes_and_workers(tmp_path):
    def router(job):
        return "gpu" if job.resources is not None and job.resources.gpus else None

    with Submitter(
        worker="debug",
        workers={"gpu": "debug"},
        router=router,
        cache_root=tmp_path / "c",
    ) as sub:
        runner = runner_for(sub, Fan(xs=[1, 2, 3]))
        nodes = by_path(runner.plan())
        runner.ready()
        started = by_path(runner.plan())
    assert started["B"].elements == 3 and started["B"].state == "pending"
    assert started["B"].worker == "default"
    # B maps over A's elements: its count is read once A has started
    assert {k: v.elements for k, v in nodes.items()} == {"A": 3, "B": None, "C": 1}
    assert nodes["B"].state == "deferred"
    assert nodes["A"].worker == "gpu" and nodes["A"].resources == GPU
    assert nodes["B"].worker == "default" and nodes["B"].resources is None
    assert all(n.state == "pending" and n.gated == 0 for n in started.values())
    assert all(n.placement is None for n in started.values())


def test_plan_expected_hits_from_records(submitter):
    submitter(Chain(x=1))  # stores A and B, and the workflow
    runner = runner_for(submitter, Chain(x=1))
    nodes = by_path(runner.plan())
    assert nodes["A"].expected_hits == 1
    # B reads A: its job cannot be formed before A's result is read, and is not counted
    assert nodes["B"].expected_hits == 0
    other = runner_for(submitter, Chain(x=5))
    assert [n.expected_hits for n in other.plan().nodes] == [0, 0]


def test_plan_counts_lookups_when_enabled(submitter, memory):
    job = Job(FunAddVar(a=1, b=1), submitter=submitter, name="A")
    memory.results[job.checksum] = compute(job)
    plain = GraphRunner(
        Chain(x=1), memory, on_error="raise_at_end", submitter=submitter
    ).plan()
    assert [n.expected_hits for n in plain.nodes] == [0, 0]  # an adapter has no records
    assert memory.lookups == []
    looked = GraphRunner(
        Chain(x=1),
        memory,
        on_error="raise_at_end",
        submitter=submitter,
        lookup_before_dispatch=True,
    ).plan()
    assert by_path(looked)["A"].expected_hits == 1
    assert memory.lookups == [job.checksum]


def test_plan_expected_hits_store_hook(submitter, memory):
    calls = []

    def expected_hits(jobs):
        calls.append([j.name for j in jobs])
        return 7

    memory.expected_hits = expected_hits
    job = Job(FunAddVar(a=1, b=1), submitter=submitter, name="A")
    memory.results[job.checksum] = compute(job)  # ignored: the hook answers
    runner = GraphRunner(
        Chain(x=1),
        memory,
        on_error="raise_at_end",
        submitter=submitter,
        lookup_before_dispatch=True,
    )
    nodes = by_path(runner.plan())
    assert nodes["A"].expected_hits == 7
    assert calls[0] == ["A"]
    assert memory.lookups == []


def test_plan_elements_uncapped(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "c", max_concurrent=2) as sub:
        runner = runner_for(sub, Fan(xs=list(range(9))))
        assert by_path(runner.plan())["A"].elements == 9
        rnd = runner.ready(2)  # the cap slices the group
        assert len(rnd.dispatch) == 2
        assert by_path(runner.plan())["A"].elements == 9


def test_plan_elements_unknown_until_upstream_output_is_read(submitter):
    nodes = by_path(runner_for(submitter, OverOutput(xs=[1, 2])).plan())
    assert nodes["A"].elements == 2 and nodes["A"].state == "pending"
    assert nodes["B"].elements is None and nodes["B"].state == "deferred"


def test_plan_constructor_workflow_deferred(submitter):
    runner = runner_for(submitter, Outer(x=1))
    nodes = by_path(runner.plan())
    assert list(nodes) == ["I"]
    assert nodes["I"].state == "deferred" and nodes["I"].elements == 1
    # once expanded the nested nodes are planned, by their path
    for _ in range(3):
        runner.ready()
    assert {"I/A", "I/B"} <= set(by_path(runner.plan()))
    assert "I" not in by_path(runner.plan())


class WaitingLease:
    lost = None

    def try_take(self):
        return 60.0

    def __exit__(self, *exc):
        pass


class WaitingStore(PickleStore):
    def acquire(self, checksum, context, **kwargs):
        return WaitingLease()


def test_plan_waiting_lease(tmp_path):
    store = WaitingStore(tmp_path / "cache")
    sub = Submitter(worker="debug", store=store)
    task = Chain(x=1)
    job = Job(task, submitter=sub, name="main")
    runner = GraphRunner(task, store, on_error="raise_at_end", submitter=sub, job=job)
    # before the first attempt nothing is known of the workflow's contents
    (first,) = runner.plan().nodes
    assert first.state == "deferred" and first.path == "main"
    assert runner.ready().retry_after is not None
    (node,) = runner.plan().nodes
    assert node.state == "waiting" and node.elements == 1
    assert runner.stuck() is None
    t = time.monotonic()
    runner.plan()
    assert time.monotonic() - t < 1  # plan did not wait for or take the lease


def test_plan_carries_executor_placement(tmp_path):
    placed = []

    class Placer:
        batches = False

        def place(self, jobs):
            placed.append([j.name for j in jobs])
            return [("slot", j.name) for j in jobs]

    class Plain:
        batches = False

    with Submitter(
        worker="debug",
        workers={"gpu": "debug"},
        router=lambda job: "gpu" if job.resources and job.resources.gpus else None,
        cache_root=tmp_path / "c",
    ) as sub:
        runner = runner_for(sub, Fan(xs=[1, 2]))
        nodes = by_path(runner.plan(executors={"gpu": Placer(), "default": Plain()}))
        assert nodes["A"].placement == [("slot", "A"), ("slot", "A")]
        assert nodes["B"].placement is None and nodes["C"].placement is None
        assert placed == [["A", "A"]]


# -- plan() leaves the run as it was ---------------------------------------------------------------------


def outcome(job, value=None):
    t = job.task
    out = t.a + t.b if value is None else value
    return Result(cache_dir=job.cache_dir, outputs=t.Outputs(out=out), task=t)


def stale_run(tmp_path, do_plan, *, errored_slot, rerun):
    """Siblings(xs=[1, 2]) stored; A[1] is handed out and held while the run goes on; plan()
    is (or is not) called; A[1] then completes. Returns what B read and how the run ended.
    """
    with Submitter(
        worker="debug", store="pickle", cache_root=tmp_path / f"c{do_plan}"
    ) as sub:
        sub(Siblings(xs=[1, 2]))
        runner = runner_for(sub, Siblings(xs=[1, 2]), rerun=rerun)
        if errored_slot:  # an earlier attempt at A[1] failed
            for j in runner_for(sub, Siblings(xs=[1, 2]), rerun=True).ready().dispatch:
                if j.name == "A" and j.state_index == 1:
                    d = sub.store.cache_dir(j.checksum, j.context)
                    save(
                        d, Result(cache_dir=d, outputs=None, task=j.task, errored=True)
                    )
                    os.unlink(sub.store.success_record(j.checksum, j.context))
        held, seen = [], []
        for j in runner.ready().dispatch:
            if j.name == "A" and j.state_index == 1:
                held.append(j)
            else:
                runner.complete(j, outcome(j))
        for j in runner.ready().dispatch:
            seen.append((j.name, j.state_index, j.task.a))
            runner.complete(j, outcome(j))
        if do_plan:
            runner.plan()
        (a1,) = held
        errored = a1.errored
        runner.complete(a1, outcome(a1, 50 if rerun else None))
        for _ in range(20):
            if runner.done:
                break
            for j in runner.ready().dispatch:
                seen.append((j.name, j.state_index, j.task.a))
                runner.complete(j, outcome(j))
        return (
            errored,
            sorted(x for x in seen if x[0] == "B"),
            runner.failed,
            runner.entries,
        )


def test_plan_does_not_mark_an_in_flight_upstream_errored(tmp_path):
    kwargs = dict(errored_slot=True, rerun=False)
    plain = stale_run(tmp_path, False, **kwargs)
    planned = stale_run(tmp_path, True, **kwargs)
    assert planned[0] is False and planned[2] is False and planned[3] == []
    assert planned == plain


def test_plan_does_not_cache_a_stale_input_in_a_live_node(tmp_path):
    kwargs = dict(errored_slot=False, rerun=True)
    plain = stale_run(tmp_path, False, **kwargs)
    planned = stale_run(tmp_path, True, **kwargs)
    assert dict(((n, i), a) for n, i, a in plain[1])[("B", 1)] == 50
    assert planned == plain


def test_plan_leaves_node_tables_and_state_alone(submitter):
    runner = runner_for(submitter, Siblings(xs=[1, 2, 3]))
    runner.ready()  # A and C released; B and D not started
    graph = runner._frames[0].graph
    tables = lambda: {  # noqa: E731
        n.name: (
            dict(n._tasks or {}),
            dict(n.queued),
            dict(n.blocked or {}),
            bool(n.started),
        )
        for n in graph.nodes
    }
    state = vars(graph.node("B").node.state)
    ids = {k: id(v) for k, v in state.items()}
    before = tables()
    plan = runner.plan()
    assert sorted(n.path for n in plan.nodes) == ["A", "B", "C", "D"]
    assert tables() == before
    assert {k: id(v) for k, v in vars(graph.node("B").node.state).items()} == ids


class CountingLease:
    lost = None
    takes = 0

    def try_take(self):
        type(self).takes += 1
        return 60.0

    def __exit__(self, *exc):
        pass


class CountingStore(PickleStore):
    def acquire(self, checksum, context, **kwargs):
        return CountingLease()


def test_plan_takes_no_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(CountingLease, "takes", 0)
    store = CountingStore(tmp_path / "cache")
    sub = Submitter(worker="debug", store=store)
    task = Chain(x=1)
    job = Job(task, submitter=sub, name="main")
    runner = GraphRunner(task, store, on_error="raise_at_end", submitter=sub, job=job)
    runner.plan()
    runner.ready()
    assert CountingLease.takes == 1
    runner.plan()
    runner.plan()
    assert CountingLease.takes == 1


def test_plan_settles_no_frame(submitter):
    runner = runner_for(submitter, Outer(x=1))
    states = lambda: [f.state for f in runner._frames]  # noqa: E731
    seen = set()
    for _ in range(10):
        for j in runner.ready().dispatch:
            runner.complete(j, outcome(j))
        before = states()  # the nested frame's nodes are done, not yet committed
        runner.plan()
        assert states() == before
        seen.add(tuple(before))
    assert runner.done
    assert ("expanding", "expanding") in seen


def test_plan_forms_no_job_in_a_live_node(submitter):
    runner = runner_for(submitter, Chain(x=1))
    (a,) = runner.ready().dispatch
    runner.ready()  # B starts, blocked on A
    runner.complete(a, outcome(a))  # A accepted, B not yet released
    b = runner._frames[0].graph.node("B")
    assert b._tasks == {None: None} and set(b.blocked) == {None}
    plan = runner.plan()
    assert by_path(plan)["B"].expected_hits == 0
    assert b._tasks == {None: None} and set(b.blocked) == {None} and not b.queued


def test_plan_does_not_count_handed_out_jobs(submitter):
    submitter(Fan(xs=[4, 5, 6]))  # a record for each element of A
    fresh = runner_for(submitter, Fan(xs=[4, 5, 6]))
    assert by_path(fresh.plan())["A"].expected_hits == 3
    # A[0]'s slot is errored under its record: read_succeeded refuses it, so it is
    # dispatched, while A[1] and A[2] settle from the store
    probe = runner_for(submitter, Fan(xs=[4, 5, 6]), rerun=True)
    job = next(
        j for j in probe.ready().dispatch if j.name == "A" and j.state_index == 0
    )
    d = submitter.store.cache_dir(job.checksum, job.context)
    save(d, Result(cache_dir=d, outputs=None, task=job.task, errored=True))
    rnd = fresh.ready()
    assert [j.state_index for j in rnd.dispatch if j.name == "A"] == [0]
    assert len([s for s in rnd.settled if s.job.name == "A"]) == 2
    node = by_path(fresh.plan())["A"]
    assert node.elements == 3 and node.expected_hits == 0  # all three are handed out


def test_plan_expected_hits_honours_rerun(submitter):
    submitter(Chain(x=1))
    assert by_path(runner_for(submitter, Chain(x=1)).plan())["A"].expected_hits == 1
    rerun = runner_for(submitter, Chain(x=1), rerun=True)
    assert [n.expected_hits for n in rerun.plan().nodes] == [0, 0]
    no_propagate = runner_for(submitter, Chain(x=1), rerun=True, propagate_rerun=False)
    assert by_path(no_propagate.plan())["A"].expected_hits == 1


def test_pickle_store_expected_hits_is_one_listing(submitter, monkeypatch):
    submitter(Chain(x=1))
    store = submitter.store
    jobs = [
        Job(FunAddVar(a=1, b=1), submitter=submitter, name="A"),
        Job(FunAddVar(a=9, b=9), submitter=submitter, name="A"),
    ]
    scans = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda *a, **k: scans.append(a) or real(*a, **k))
    assert store.expected_hits(jobs) == 1
    assert len(scans) == 1


# -- routing and placement -------------------------------------------------------------------------------


def test_plan_propagates_a_router_error_for_a_real_job(tmp_path):
    with Submitter(
        worker="debug", router=lambda j: "gpu", cache_root=tmp_path / "c"
    ) as sub:
        runner = runner_for(sub, Chain(x=1))
        with pytest.raises(ValueError, match="gpu"):
            runner.plan()


def test_plan_router_error_for_a_stand_in_is_provisional_none(tmp_path):
    def router(job):
        if job.task.a != 1 and not isinstance(job.task.a, int):
            raise RuntimeError("reads the input")
        return None

    with Submitter(worker="debug", router=router, cache_root=tmp_path / "c") as sub:
        nodes = by_path(runner_for(sub, Chain(x=1)).plan())
    assert nodes["A"].worker == "default" and not nodes["A"].provisional
    assert nodes["B"].worker is None and nodes["B"].provisional


def test_plan_placement_for_unformable_nodes_is_provisional(tmp_path):
    seen = []

    class Placer:
        batches = False

        def place(self, jobs):
            seen.append([(type(j).__name__, j.name) for j in jobs])
            return "placed"

    with Submitter(worker="debug", cache_root=tmp_path / "c") as sub:
        runner = runner_for(sub, Chain(x=1))
        nodes = by_path(runner.plan(executors={"default": Placer()}))
    assert nodes["A"].placement == "placed" and not nodes["A"].provisional
    assert nodes["B"].placement == "placed" and nodes["B"].provisional
    assert seen[0][0][0] == "Job" and seen[1][0][0] != "Job"


def test_plan_defaults_executors_to_the_submitters_workers(tmp_path):
    from pydra.workers import debug

    class PlacingDebug(debug.Worker):
        _plugin_name = "debug"

        def place(self, jobs):
            return len(jobs)

    with Submitter(worker=PlacingDebug(), cache_root=tmp_path / "c") as sub:
        nodes = by_path(runner_for(sub, Fan(xs=[1, 2])).plan())
    assert nodes["A"].placement == 2 and nodes["C"].placement == 1


# -- expected_hits against what ready() would settle ----------------------------------------------------


def test_plan_counts_lookups_on_a_pickle_store_with_read_only_caches(tmp_path):
    with Submitter(worker="debug", store="pickle", cache_root=tmp_path / "old") as sub:
        sub(Chain(x=1))
    store = PickleStore(tmp_path / "new", readonly_caches=[tmp_path / "old"])
    with Submitter(worker="debug", store=store) as sub:
        for lbd, expected in ((False, 0), (True, 1)):
            runner = GraphRunner(
                Chain(x=1),
                store,
                on_error="raise_at_end",
                submitter=sub,
                lookup_before_dispatch=lbd,
            )
            assert by_path(runner.plan())["A"].expected_hits == expected
            rnd = runner.ready()
            assert len([s for s in rnd.settled if s.how == "stored"]) == expected * 2
            assert len(rnd.dispatch) == (0 if lbd else 1)


def test_plan_lists_the_record_directory_once(submitter, monkeypatch):
    submitter(Siblings(xs=[1, 2]))
    runner = runner_for(submitter, Siblings(xs=[1, 2]))
    scans = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda *a, **k: scans.append(a) or real(*a, **k))
    nodes = by_path(runner.plan())
    assert nodes["A"].expected_hits == 2 and nodes["C"].expected_hits == 2
    assert len(scans) == 1


def test_plan_counts_records_per_job_for_a_store_without_the_hook(submitter, tmp_path):
    submitter(Chain(x=1))

    class Recording(MemoryStore):
        success_record = submitter.store.success_record

    store = Recording(tmp_path / "mem")
    store.cache_root = submitter.store.cache_root
    runner = GraphRunner(
        Chain(x=1), store, on_error="raise_at_end", submitter=submitter
    )
    assert not hasattr(store, "expected_hits")
    assert by_path(runner.plan())["A"].expected_hits == 1
    other = GraphRunner(Chain(x=3), store, on_error="raise_at_end", submitter=submitter)
    assert by_path(other.plan())["A"].expected_hits == 0


# -- loops (RFC 0002 section 2.3(d), R11) ------------------------------------------------------------


@python.define(outputs=["n", "done"])
def Step(n: int, stop: int) -> tuple[int, bool]:
    return n + 1, n + 1 >= stop


@workflow.define(outputs=["n"])
def LoopThen(stop: int) -> int:
    loop = workflow.add(workflow.loop(Step, max_rounds=7)(n=0, stop=stop), name="count")
    after = workflow.add(FunAddVar(a=loop.n, b=1), name="after")
    return after.out


def test_plan_loop_deferred_then_looping(submitter):
    runner = runner_for(submitter, LoopThen(stop=3))
    assert by_path(runner.plan())["count"].state == "deferred"

    (r0,) = runner.ready().dispatch
    plan = runner.plan()
    assert [n.path for n in plan.nodes] == ["after", "count", "count/r0"]
    loop = by_path(plan)["count"]
    assert (loop.state, loop.rounds, loop.max_rounds) == ("looping", 0, 7)
    assert (loop.elements, loop.known_elements) == (None, None)

    runner.complete(r0, r0.run())
    (frame,) = [f for f in runner._frames if f.loop is not None]
    plan = runner.plan()  # never adds a round, and never predicts a total
    assert [n.path for n in plan.nodes] == ["after", "count"]
    assert by_path(plan)["count"].rounds == 1
    assert (frame.loop.rounds, frame.loop.stopped) == (1, None)
    assert [n.name for n in frame.graph.nodes] == ["r0"]
    (r1,) = runner.ready().dispatch
    assert r1.name == "r1"
