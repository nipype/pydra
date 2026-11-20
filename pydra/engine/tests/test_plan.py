"""``GraphRunner.plan()`` (RFC 0002 section 3.1, R8): what would run, be reused or wait,
read from the runner's graph and the store without running, leasing or writing anything
"""

import time
from pathlib import Path

import pytest

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.plan import Plan, PlanNode
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.engine.tests.test_runner_driver_model import (
    Chain,
    Outer,
    MemoryStore,
    compute,
    drive_by_hand,
    runner_for,
)
from pydra.engine.tests.utils import FunAddVar
from pydra.utils.resources import ComputeResources

GPU = ComputeResources(gpus=1, tags=("gpu",))


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
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
