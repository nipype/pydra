"""Per-element readiness: an element of a downstream node is released as soon as the
upstream elements its inputs read have completed, rather than once the whole upstream
node is done"""

import pytest
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter
from pydra.engine.workflow import Workflow
from pydra.engine.tests.utils import (
    FunAddVar,
    FunAddVar3,
    Identity,
    ListOutput,
    Multiply,
)


@workflow.define
def Outer(x, y):
    a = workflow.add(FunAddVar().split(["a", "b"], a=x, b=y), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=1000), name="B")
    return b.out


@workflow.define
def OuterCombined(x, y):
    a = workflow.add(FunAddVar().split(["a", "b"], a=x, b=y).combine("a"), name="A")
    b = workflow.add(Identity(x=a.out), name="B")
    return b.out


@workflow.define
def Scalar(x, y):
    a = workflow.add(FunAddVar().split(("a", "b"), a=x, b=y), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=1000), name="B")
    return b.out


@workflow.define
def ScalarCombined(x, y):
    a = workflow.add(FunAddVar().split(("a", "b"), a=x, b=y).combine("a"), name="A")
    b = workflow.add(Identity(x=a.out), name="B")
    return b.out


@workflow.define
def Nested(x, y, z):
    a = workflow.add(FunAddVar3().split([("a", "b"), "c"], a=x, b=y, c=z), name="A")
    b = workflow.add(FunAddVar(a=a.out, b=1000), name="B")
    return b.out


@workflow.define
def NestedCombined(x, y, z):
    a = workflow.add(
        FunAddVar3().split([("a", "b"), "c"], a=x, b=y, c=z).combine("a"), name="A"
    )
    b = workflow.add(Identity(x=a.out), name="B")
    return b.out


@workflow.define
def DownstreamSplit(x, y):
    a = workflow.add(FunAddVar(b=10).split("a", a=x), name="A")
    b = workflow.add(FunAddVar(a=a.out).split("b", b=y), name="B")
    return b.out


@workflow.define
def DownstreamSplitCombined(x, y):
    a = workflow.add(FunAddVar(b=10).split("a", a=x), name="A")
    b = workflow.add(FunAddVar(a=a.out).split("b", b=y).combine("b"), name="B")
    c = workflow.add(Identity(x=b.out), name="C")
    return c.out


@workflow.define
def TwoUpstreamOuter(x, y):
    a = workflow.add(FunAddVar(b=0).split("a", a=x), name="A")
    b = workflow.add(FunAddVar(b=0).split("a", a=y), name="B")
    c = workflow.add(FunAddVar(a=a.out, b=b.out), name="C")
    return c.out


@workflow.define
def TwoUpstreamScalar(x, y):
    a = workflow.add(FunAddVar(b=0).split("a", a=x), name="A")
    b = workflow.add(FunAddVar(b=0).split("a", a=y), name="B")
    c = workflow.add(FunAddVar(a=a.out, b=b.out).split(("_A", "_B")), name="C")
    return c.out


@workflow.define
def Resplit(x, y):
    a = workflow.add(ListOutput().split("x", x=x), name="A")
    b = workflow.add(Multiply().split(["y", "x"], x=a.out, y=y), name="B")
    return b.out


def on(node, *indices):
    return {(node, i) for i in indices}


# Each case: the workflow, the nodes run first, the upstream nodes whose elements are
# run one at a time, the downstream node, the upstream elements each downstream element
# needs (worked out by hand from the splitters), and the workflow output.
CASES = {
    "outer": (
        Outer(x=[1, 2, 3], y=[10, 20]),
        [],
        ["A"],
        "B",
        {i: on("A", i) for i in range(6)},
        [1011, 1021, 1012, 1022, 1013, 1023],
    ),
    "outer-combined": (
        OuterCombined(x=[1, 2, 3], y=[10, 20]),
        [],
        ["A"],
        "B",
        {0: on("A", 0, 2, 4), 1: on("A", 1, 3, 5)},
        [[11, 12, 13], [21, 22, 23]],
    ),
    "scalar": (
        Scalar(x=[1, 2, 3], y=[10, 20, 30]),
        [],
        ["A"],
        "B",
        {i: on("A", i) for i in range(3)},
        [1011, 1022, 1033],
    ),
    "scalar-combined": (
        ScalarCombined(x=[1, 2, 3], y=[10, 20, 30]),
        [],
        ["A"],
        "B",
        {None: on("A", 0, 1, 2)},
        [11, 22, 33],
    ),
    "nested": (
        Nested(x=[1, 2], y=[10, 20], z=[100, 200, 300]),
        [],
        ["A"],
        "B",
        {i: on("A", i) for i in range(6)},
        [1111, 1211, 1311, 1122, 1222, 1322],
    ),
    "nested-combined": (
        NestedCombined(x=[1, 2], y=[10, 20], z=[100, 200, 300]),
        [],
        ["A"],
        "B",
        {i: on("A", i, i + 3) for i in range(3)},
        [[111, 122], [211, 222], [311, 322]],
    ),
    "downstream-split": (
        DownstreamSplit(x=[1, 2, 3], y=[100, 200]),
        [],
        ["A"],
        "B",
        {i * 2 + j: on("A", i) for i in range(3) for j in range(2)},
        [111, 211, 112, 212, 113, 213],
    ),
    "downstream-split-combined": (
        DownstreamSplitCombined(x=[1, 2, 3], y=[100, 200]),
        ["A"],
        ["B"],
        "C",
        {i: on("B", 2 * i, 2 * i + 1) for i in range(3)},
        [[111, 211], [112, 212], [113, 213]],
    ),
    "two-upstream-outer": (
        TwoUpstreamOuter(x=[1, 2, 3], y=[10, 20]),
        [],
        ["A", "B"],
        "C",
        {i * 2 + j: on("A", i) | on("B", j) for i in range(3) for j in range(2)},
        [11, 21, 12, 22, 13, 23],
    ),
    "two-upstream-scalar": (
        TwoUpstreamScalar(x=[1, 2], y=[10, 20]),
        [],
        ["A", "B"],
        "C",
        {i: on("A", i) | on("B", i) for i in range(2)},
        [11, 22],
    ),
    # splitting over an upstream output still waits for the whole upstream node
    "resplit": (
        Resplit(x=[1, 2], y=[10, 100]),
        [],
        ["A"],
        "B",
        {i: on("A", 0, 1) for i in range(12)},
        [10, 20, 30, 20, 40, 60, 100, 200, 300, 200, 400, 600],
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_readiness_per_element(case, tmp_path):
    """Run the upstream elements one at a time and check that, after each, exactly the
    downstream elements whose upstream elements have all completed are released"""
    task, first, upstream, downstream, needs, expected = CASES[case]
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        wf = Workflow.construct(task)
        graph = wf.execution_graph(submitter=sub)
        for name in first:
            for job in graph[name].get_runnable_tasks(graph):
                job.run()
            graph[name].update_status()
        jobs = [
            job for name in upstream for job in graph[name].get_runnable_tasks(graph)
        ]
        done = set()
        for job in [None] + jobs:
            if job is not None:
                job.run()
                graph[job.name].update_status()
                done.add((job.name, job.state_index))
            released = graph[downstream].get_runnable_tasks(graph)
            assert {j.state_index for j in released} == {
                i for i, n in needs.items() if n <= done
            }
        for job in released:
            job.run()
        graph[downstream].update_status()
        assert graph[downstream].done
        assert wf.outputs.out._get_value(workflow=wf, graph=graph) == expected


@pytest.mark.parametrize("case", CASES)
def test_readiness_debug_parity(case, worker, tmp_path):
    """Whole workflows give the same outputs under each worker as under debug"""
    task, *_, expected = CASES[case]
    reference = task(worker="debug", cache_root=tmp_path / "debug")
    outputs = task(worker=worker, cache_root=tmp_path / "worker")
    assert reference.out == expected
    assert outputs.out == reference.out


def test_readiness_async(tmp_path):
    """Under the cf worker, a downstream element runs while an upstream sibling is still
    running: the second upstream element here only finishes once the downstream element
    of the first has written a marker, so waiting for the whole upstream node would
    time out"""

    @python.define
    def Wait(x: int, marker: str) -> int:
        import time
        from pathlib import Path

        if x:
            deadline = time.monotonic() + 60
            while not Path(marker).exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{marker} was not written")
                time.sleep(0.1)
        return x

    @python.define
    def Mark(x: int, marker: str) -> int:
        from pathlib import Path

        if not x:
            Path(marker).touch()
        return x

    @workflow.define
    def Handshake(xs, marker):
        a = workflow.add(Wait(marker=marker).split(x=xs), name="A")
        b = workflow.add(Mark(x=a.out, marker=marker), name="B")
        return b.out

    task = Handshake(xs=[0, 1], marker=str(tmp_path / "marker"))
    outputs = task(worker="cf", n_procs=2, cache_root=tmp_path / "cache")
    assert outputs.out == [0, 1]
