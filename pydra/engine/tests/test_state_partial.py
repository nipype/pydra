"""Partial ``State`` (proposal 0001 §3.1 step 2, RFC 0002 E2): a node that splits over
an upstream output is expanded one block per upstream element, so each element's
sub-state is released as soon as that upstream element has succeeded"""

import typing as ty
import pytest
from pydra.compose import python, workflow
from pydra.engine.runner import Failed, GraphRunner
from pydra.engine.submitter import Submitter
from pydra.engine.workflow import Workflow
from pydra.engine.state import PydraStateError
from pydra.engine.tests.test_readiness import BadInput


@python.define
def Seg(n: int) -> list[int]:
    return [n * 10 + i for i in range(n)]


@python.define
def L(x: int) -> list[int]:
    return [x, 2 * x, 3 * x]


@python.define(outputs=["a", "b"])
def Two(x: int) -> tuple[list[int], list[int]]:
    return [x, 2 * x], [10 * x, 20 * x]


@python.define(outputs=["a", "b"])
def Uneven(x: int) -> tuple[list[int], list[int]]:
    return [x, 2 * x], ([10 * x] if x == 2 else [10 * x, 20 * x])


@python.define
def FailOn2(x: int) -> list[int]:
    if x == 2:
        raise ValueError("record 2 failed")
    return [x, 2 * x, 3 * x]


@python.define
def AnyOut(x: int) -> ty.Any:
    return [x, 2 * x] if x != 2 else x


@python.define
def Route(go: bool, x: int) -> list[int]:
    return [x] if go else []


@python.define
def ListP(n: int) -> list[int]:
    return list(range(n))


@python.define
def Mul(x: int, y: int) -> int:
    return x * y


@python.define
def Add(x: int, d: int) -> int:
    return x + d


@python.define
def Ident(x: int) -> int:
    return x


@python.define
def Inc(x: int) -> int:
    return x + 1


@python.define
def Double(x: int) -> int:
    return 2 * x


@python.define
def IdentL(x: list[int]) -> list[int]:
    return x


@python.define
def SumL(x: list[int]) -> int:
    return sum(x)


# ----------------------------------------------------------------------------------
# Workflows


@workflow.define
def YOuter(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(["y", "x"], x=a.out, y=y), name="B")
    return b.out


@workflow.define
def XOuter(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(["x", "y"], x=a.out, y=y), name="B")
    return b.out


@workflow.define
def YOuterCombX(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(["y", "x"], x=a.out, y=y).combine("x"), name="B")
    return b.out


@workflow.define
def YOuterCombY(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(["y", "x"], x=a.out, y=y).combine("y"), name="B")
    return b.out


@workflow.define
def XOuterCombY(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(["x", "y"], x=a.out, y=y).combine("y"), name="B")
    return b.out


@workflow.define
def MixedD(x: list[int], d: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    dd = workflow.add(Ident().split("x", x=d), name="D")
    b = workflow.add(Add(d=dd.out).split("x", x=a.out), name="B")
    return b.out


@workflow.define
def CombX(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out).combine("x"), name="B")
    return b.out


@workflow.define
def Downstream(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Mul(x=b.out, y=10), name="C")
    return c.out


@workflow.define
def DownCombBx(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine("B.x"), name="C")
    return c.out


@workflow.define
def DownCombAx(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine("A.x"), name="C")
    return c.out


@workflow.define
def RaggedDownCombAx(n: list[int]):
    a = workflow.add(Seg().split("n", n=n), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine("A.n"), name="C")
    return c.out


@workflow.define
def Sibling(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b1 = workflow.add(Mul(y=1).split("x", x=a.out), name="B1")
    b2 = workflow.add(Mul(y=1000).split("x", x=a.out), name="B2")
    c = workflow.add(Add(x=b1.out, d=b2.out), name="C")
    return c.out


@workflow.define
def UpCombAll(x: list[int]):
    a = workflow.add(Ident().split("x", x=x).combine("x"), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    return b.out


@workflow.define
def Unsplit(n: int):
    p = workflow.add(ListP(n=n), name="P")
    b = workflow.add(Mul(y=10).split("x", x=p.out), name="B")
    return b.out


@workflow.define
def ScalarSame(x: list[int]):
    a = workflow.add(Two().split("x", x=x), name="A")
    b = workflow.add(Add().split(("x", "d"), x=a.a, d=a.b), name="B")
    return b.out


@workflow.define
def UnevenZip(x: list[int]):
    a = workflow.add(Uneven().split("x", x=x), name="A")
    b = workflow.add(Add().split(("x", "d"), x=a.a, d=a.b), name="B")
    return b.out


@workflow.define
def StaticD(x: list[int], d: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Add().split(["x", "d"], x=a.out, d=d), name="B")
    c = workflow.add(Ident(x=b.out).combine("B.d"), name="C")
    return c.out


@workflow.define
def CombBoth(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out).combine(["x", "A.x"]), name="B")
    return b.out


@workflow.define
def CombAOnNode(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out).combine("A.x"), name="B")
    return b.out


@workflow.define
def UpComb(x: list[int], y: list[int]):
    a = workflow.add(Mul().split(["x", "y"], x=x, y=y).combine("y"), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    return b.out


@workflow.define
def Ragged(n: list[int]):
    a = workflow.add(Seg().split("n", n=n), name="A")
    b = workflow.add(Inc().split("x", x=a.out), name="B")
    return b.out


@workflow.define
def RaggedComb(n: list[int]):
    a = workflow.add(Seg().split("n", n=n), name="A")
    b = workflow.add(Inc().split("x", x=a.out).combine("x"), name="B")
    return b.out


@workflow.define
def EmptySplit(go: list[bool], x: int):
    p = workflow.add(Route(x=x).split("go", go=go), name="P")
    b = workflow.add(Double().split("x", x=p.out).combine("x"), name="B")
    return b.out


@workflow.define
def EmptySplitReader(go: list[bool], x: int):
    p = workflow.add(Route(x=x).split("go", go=go), name="P")
    b = workflow.add(Double().split("x", x=p.out).combine("x"), name="B")
    c = workflow.add(IdentL(x=b.out), name="C")
    return c.out


@workflow.define
def MixedAny(x: list[int]):
    a = workflow.add(AnyOut().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    return b.out


@workflow.define
def Records(n: list[int]):
    """Segments within records: split each record into its segments, process each
    segment, and gather the segments of each record back into one value per record"""
    a = workflow.add(Seg().split("n", n=n), name="A")
    b = workflow.add(Inc().split("x", x=a.out), name="B")
    c = workflow.add(Double(x=b.out).combine("B.x"), name="C")
    d = workflow.add(SumL(x=c.out), name="D")
    return d.out


@workflow.define
def PerRecord(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out).combine("x"), name="B")
    c = workflow.add(SumL(x=b.out), name="C")
    return c.out


@workflow.define
def FailRecord(x: list[int]):
    a = workflow.add(FailOn2().split("x", x=x), name="A")
    b = workflow.add(Mul(y=1).split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine(["B.x", "A.x"]), name="C")
    e = workflow.add(SumL(x=c.out), name="E")
    return e.out


@workflow.define
def FailRecordGather(x: list[int]):
    a = workflow.add(FailOn2().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out).combine("x"), name="B")
    c = workflow.add(SumL(x=b.out), name="C")
    return c.out


@workflow.define
def FailRecordReader(x: list[int]):
    a = workflow.add(FailOn2().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine("A.x"), name="C")
    return c.out


@workflow.define
def ZipStatic(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Mul().split(("x", "y"), x=a.out, y=y), name="B")
    return b.out


@workflow.define
def ZipPartials(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b1 = workflow.add(Mul(y=1).split("x", x=a.out), name="B1")
    b2 = workflow.add(Mul(y=1000).split("x", x=a.out), name="B2")
    c = workflow.add(Add(x=b1.out, d=b2.out).split(("_B1", "_B2")), name="C")
    return c.out


@workflow.define
def SameSrc(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    dd = workflow.add(SumL(x=a.out), name="D")
    b = workflow.add(Add(d=dd.out).split("x", x=a.out), name="B")
    return b.out


# ----------------------------------------------------------------------------------
# Helpers: a pure runner, driven by hand so that the test controls completion order


@pytest.fixture
def submitter(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        yield sub


def runner_for(submitter, task) -> GraphRunner:
    return GraphRunner(
        task, submitter.store, submitter=submitter, on_error="raise_at_end"
    )


def finish(runner, job):
    """Run a handed-out job and report its outcome"""
    try:
        result = job.run()
    except Exception as exc:
        runner.complete(job, Failed(exc))
    else:
        runner.complete(job, result)


def by_name(jobs, name):
    return [j for j in jobs if j.name == name]


def outputs(runner):
    wf = runner.workflow
    return wf.outputs.out._get_value(workflow=wf, graph=runner.exec_graph)


def drain(runner, keep=None):
    """Run every handed-out job, round after round, except those ``keep`` holds back,
    which are returned"""
    held = []
    while True:
        jobs = runner.ready().dispatch
        if not jobs:
            return held
        for job in jobs:
            if keep is not None and keep(job):
                held.append(job)
            else:
                finish(runner, job)


def run_wf(task, worker, tmp_path):
    return task(worker=worker, cache_root=tmp_path).out


# ----------------------------------------------------------------------------------
# The brief's tests (pure runner)


def test_outer_split_over_upstream_output_per_element(submitter):
    """Upstream element 0 completed, element 1 not: element 0's sub-state is
    dispatched. Elements carry their key before the node is sealed and get today's flat
    index once it is"""
    runner = runner_for(submitter, YOuter(x=[1, 2], y=[10, 100]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    released = runner.ready().dispatch
    assert {j.name for j in released} == {"B"}
    # key: (y index, A element, position in A's output), in today's ``["y","x"]`` order
    assert {j.state_key for j in released} == {
        (k, 0, j) for k in range(2) for j in range(3)
    }
    assert all(j.state_index is None for j in released)
    assert sorted(j.task.x * 1000 + j.task.y for j in released) == sorted(
        x * 1000 + y for x in (1, 2, 3) for y in (10, 100)
    )
    for job in released:
        finish(runner, job)
    finish(runner, a1)
    drain(runner)
    assert runner.done and not runner.failed
    node = runner.exec_graph["B"]
    first = {j.state_index for j in released}
    assert first == {0, 1, 2, 6, 7, 8}
    assert [j.state_index for j in node.tasks] == list(range(12))
    assert outputs(runner) == [10, 20, 30, 20, 40, 60, 100, 200, 300, 200, 400, 600]


def test_scalar_splitter_per_element(submitter):
    """A scalar splitter over two outputs of the same upstream is expanded per
    upstream element"""
    runner = runner_for(submitter, ScalarSame(x=[1, 2]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    released = runner.ready().dispatch
    assert sorted((j.task.x, j.task.d) for j in released) == [(1, 10), (2, 20)]
    for job in released:
        finish(runner, job)
    finish(runner, a1)
    drain(runner)
    assert outputs(runner) == [11, 22, 22, 44]


def test_nested_segments_within_records(submitter):
    """Two levels (records, then segments; a third inner level is refused at
    construction, ``lazy.py:83``). Record 1's segments, their per-segment readers and
    record 1's gather all run while record 0 is still running"""
    runner = runner_for(submitter, Records(n=[2, 3]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a1)
    for name in ("B", "C", "D"):
        released = runner.ready().dispatch
        assert {j.name for j in released} == {name}, name
        for job in released:
            finish(runner, job)
    assert not runner.done
    finish(runner, a0)
    drain(runner)
    assert runner.done and not runner.failed
    # record 0: segments 20, 21 -> +1 -> x2 -> 42 + 44; record 1: 30, 31, 32
    assert outputs(runner) == [42 + 44, 62 + 64 + 66]


def test_combine_after_partial_failure(submitter):
    """A failed record makes only its own gather unrunnable; the other record's gather
    runs, and a whole read is failed"""
    runner = runner_for(submitter, FailRecordGather(x=[1, 2, 3]))
    drain(runner)
    assert runner.done and runner.failed
    c = runner.exec_graph["C"]
    assert sorted(j.task.x for j in c.successful.values()) == [[1, 2, 3], [3, 6, 9]]
    b = runner.exec_graph["B"]
    assert [type(k).__name__ for k in b.unrunnable] == ["Block"]
    assert len(c.unrunnable) == 1


def test_empty_split_routes_per_element(any_worker, tmp_path):
    """The per-element empty split gives one list per element"""
    assert run_wf(EmptySplit(go=[True, False], x=3), any_worker, tmp_path) == [[6], []]
    assert run_wf(EmptySplit(go=[False, True], x=3), any_worker, tmp_path / "2") == [
        [],
        [6],
    ]


# ----------------------------------------------------------------------------------
# Release algebra


def test_zero_element_block_wakes_reader(submitter):
    """A block that expands to no element closes its group, which wakes the reader of
    the group although no element count changed"""
    runner = runner_for(submitter, EmptySplitReader(go=[True, False], x=3))
    p0, p1 = runner.ready().dispatch
    finish(runner, p0)
    (b,) = runner.ready().dispatch
    finish(runner, b)
    (c0,) = runner.ready().dispatch
    assert c0.name == "C" and c0.task.x == [6]
    finish(runner, c0)
    finish(runner, p1)
    (c1,) = runner.ready().dispatch
    assert c1.name == "C" and c1.task.x == []
    finish(runner, c1)
    assert runner.ready().dispatch == []
    assert runner.done
    assert outputs(runner) == [[6], []]


def test_reader_without_a_block_registers_its_first_group(submitter):
    """A reader of a combined partial node starts with no block and registers each
    group as it closes"""
    runner = runner_for(submitter, PerRecord(x=[1, 2]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    for job in runner.ready().dispatch:
        finish(runner, job)
    (c0,) = runner.ready().dispatch
    assert c0.name == "C" and c0.task.x == [1, 2, 3]
    finish(runner, c0)
    finish(runner, a1)
    drain(runner)
    assert outputs(runner) == [6, 12]


def test_downstream_not_done_between_blocks(submitter):
    """A reader of a partial node is not done between the node's blocks"""
    runner = runner_for(submitter, Downstream(x=[1, 2]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    drain(runner, keep=lambda j: j is a1)
    c = runner.exec_graph["C"]
    assert len(c.successful) == 3
    assert not c.done and not runner.done
    finish(runner, a1)
    drain(runner)
    assert runner.done
    assert outputs(runner) == [10, 20, 30, 20, 40, 60]


def test_last_block_runs_when_source_finishes_before_reader_is_visited(tmp_path):
    """Every element of B completes before C is first visited: C still expands B's
    last block before it is done"""
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        wf = Workflow.construct(Downstream(x=[1, 2]))
        graph = wf.execution_graph(submitter=sub)
        a, b, c = graph["A"], graph["B"], graph["C"]
        for job in a.get_runnable_tasks(graph):
            job.run()
        a.update_status()
        for job in b.get_runnable_tasks(graph):
            job.run()
        b.update_status()
        assert b.done
        released = c.get_runnable_tasks(graph)
        assert len(released) == 6
        for job in released:
            job.run()
        c.update_status()
        assert c.done


def test_failed_source_block_reaches_readers(submitter):
    """A failed block of a source registers a failed block in each reader, so a
    reader of every element is unrunnable and the readers are never sealed"""
    runner = runner_for(submitter, FailRecord(x=[1, 2]))
    drain(runner)
    assert runner.done and runner.failed
    b, c, e = (runner.exec_graph[n] for n in "BCE")
    assert len(b.successful) == 3
    assert [type(k).__name__ for k in b.unrunnable] == ["Block"]
    assert c.unrunnable and len(c.successful) == 3
    assert e.unrunnable == {None: [c]}
    assert not b.sealed and not c.sealed
    assert all(j.state_index is None for j in b.tasks)
    with pytest.raises(RuntimeError, match="not sealed"):
        b.read_value(lambda job: None, None)


def test_barrier_reader_of_never_sealing_upstream(submitter):
    """A barrier reader of a partial node with a failed block is unrunnable as a whole,
    naming the partial node"""
    runner = runner_for(submitter, FailRecordReader(x=[1, 2]))
    drain(runner)
    assert runner.done and runner.failed
    b, c = runner.exec_graph["B"], runner.exec_graph["C"]
    assert c.mode == "barrier"
    assert c.unrunnable == {None: [b]}


def test_expansion_error_is_reported(any_worker, tmp_path):
    """Unequal lengths under a scalar splitter error that block only; the workflow
    raises with the expansion error"""
    with pytest.raises(RuntimeError, match=r"B\[block") as exc_info:
        run_wf(UnevenZip(x=[1, 2, 3]), any_worker, tmp_path)
    assert "same length" in str(exc_info.value)


def test_expansion_error_entry(submitter):
    runner = runner_for(submitter, UnevenZip(x=[1, 2, 3]))
    drain(runner)
    assert runner.done and runner.failed
    (entry,) = runner.entries
    assert entry.phase == "expansion"
    assert entry.job_name == "B"
    assert entry.state_index is None and entry.state_key is None
    assert entry.block == (1,)
    b = runner.exec_graph["B"]
    assert len(b.successful) == 4


def test_out_of_order_blocks_keep_keys(submitter):
    """Completing upstream element 1 before element 0 assigns element 1's keys first;
    no key changes, and the flat order is today's"""
    runner = runner_for(submitter, XOuter(x=[1, 2], y=[10, 100]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a1)
    first = runner.ready().dispatch
    keys = {id(j): j.state_key for j in first}
    assert set(keys.values()) == {(1, j, k) for j in range(3) for k in range(2)}
    for job in first:
        finish(runner, job)
    finish(runner, a0)
    drain(runner)
    assert {id(j): j.state_key for j in first} == keys
    assert sorted(j.state_index for j in first) == list(range(6, 12))
    assert outputs(runner) == [10, 100, 20, 200, 30, 300, 20, 200, 40, 400, 60, 600]


def test_reads_of_an_unsealed_upstream(submitter):
    """An element read is released per element, a group read when its group closes,
    and a whole read only once the upstream is sealed"""
    runner = runner_for(submitter, PerRecord(x=[1, 2]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    b_jobs = runner.ready().dispatch  # element reads of A[0]
    assert len(b_jobs) == 3
    b = runner.exec_graph["B"]
    with pytest.raises(RuntimeError, match="not sealed"):
        b.read_value(lambda job: None, None)  # whole read
    for job in b_jobs:
        finish(runner, job)
    (c0,) = runner.ready().dispatch  # group read
    assert c0.task.x == [1, 2, 3]
    finish(runner, c0)
    finish(runner, a1)
    drain(runner)
    assert b.sealed
    assert b.read_value(lambda job: job.task.x, None) == [[1, 2, 3], [2, 4, 6]]


def test_stuck_names_pending_block(submitter):
    runner = runner_for(submitter, YOuter(x=[1, 2], y=[10]))
    runner.ready()  # A's jobs handed out, never completed
    runner.ready()  # B starts, with a pending block per A element
    diagnosis = runner.stuck()
    assert "pending blocks" in diagnosis
    assert "B" in diagnosis


# ----------------------------------------------------------------------------------
# Order and identity: today's results are kept wherever they are correct


# workflow, inputs, output
UNCHANGED = {
    "y-outer": (
        YOuter,
        dict(x=[1, 2], y=[10, 100]),
        [10, 20, 30, 20, 40, 60, 100, 200, 300, 200, 400, 600],
    ),
    "x-outer": (
        XOuter,
        dict(x=[1, 2], y=[10, 100]),
        [10, 100, 20, 200, 30, 300, 20, 200, 40, 400, 60, 600],
    ),
    "y-outer-combine-x": (
        YOuterCombX,
        dict(x=[1, 2], y=[10, 100]),
        [[10, 20, 30], [100, 200, 300], [20, 40, 60], [200, 400, 600]],
    ),
    "y-outer-combine-y": (
        YOuterCombY,
        dict(x=[1, 2], y=[10, 100]),
        [[10, 100], [20, 200], [30, 300], [20, 200], [40, 400], [60, 600]],
    ),
    "x-outer-combine-y": (
        XOuterCombY,
        dict(x=[1, 2], y=[10, 100]),
        [[10, 100], [20, 200], [30, 300], [20, 200], [40, 400], [60, 600]],
    ),
    "d-major": (
        MixedD,
        dict(x=[1, 2], d=[1000, 2000]),
        [1001, 1002, 1003, 1002, 1004, 1006, 2001, 2002, 2003, 2002, 2004, 2006],
    ),
    "combine-x": (CombX, dict(x=[1, 2]), [[1, 2, 3], [2, 4, 6]]),
    "downstream": (Downstream, dict(x=[1, 2]), [10, 20, 30, 20, 40, 60]),
    "downstream-combine-inner": (DownCombBx, dict(x=[1, 2]), [[1, 2, 3], [2, 4, 6]]),
    "downstream-combine-source": (
        DownCombAx,
        dict(x=[1, 2]),
        [[1], [2], [3], [2], [4], [6]],
    ),
    "sibling": (
        Sibling,
        dict(x=[1, 2]),
        [
            b1 + b2
            for b1 in (1, 2, 3, 2, 4, 6)
            for b2 in (1000, 2000, 3000, 2000, 4000, 6000)
        ],
    ),
    "upstream-combined-all": (UpCombAll, dict(x=[1, 2, 3]), [1, 2, 3]),
    "unsplit-source": (Unsplit, dict(n=3), [0, 10, 20]),
    "scalar-same-source": (ScalarSame, dict(x=[1, 2]), [11, 22, 22, 44]),
    "zip-partials": (ZipPartials, dict(x=[1, 2]), [1001, 2002, 3003, 2002, 4004, 6006]),
}


@pytest.mark.parametrize("case", UNCHANGED)
def test_partial_order_matches_full_state(case, tmp_path):
    """Elements, groups and values are today's for every case that is correct today"""
    wf, kwargs, expected = UNCHANGED[case]
    assert run_wf(wf(**kwargs), "debug", tmp_path) == expected


CHECKSUM_INPUTS = {
    # node, the fields compared, today's inputs of each element in today's order
    "y-outer": (
        "B",
        ("x", "y"),
        [(x, y) for y in (10, 100) for a in (1, 2) for x in (a, 2 * a, 3 * a)],
    ),
    "d-major": (
        "B",
        ("x", "d"),
        [(x, d) for d in (1000, 2000) for a in (1, 2) for x in (a, 2 * a, 3 * a)],
    ),
    "sibling": (
        "C",
        ("x", "d"),
        [
            (b1, b2)
            for b1 in (1, 2, 3, 2, 4, 6)
            for b2 in (1000, 2000, 3000, 2000, 4000, 6000)
        ],
    ),
    "scalar-same-source": ("B", ("x", "d"), [(1, 10), (2, 20), (2, 20), (4, 40)]),
}


@pytest.mark.parametrize("case", CHECKSUM_INPUTS)
def test_partial_checksums_match_full_state(case, tmp_path):
    """Each element resolves to the inputs it resolves to today, in today's order, so it
    has the checksum a full-state run gives it, so the two share their cache"""
    wf, kwargs, _ = UNCHANGED[case]
    name, fields, expected = CHECKSUM_INPUTS[case]
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = runner_for(sub, wf(**kwargs))
        drain(runner)
        node = runner.exec_graph[name]
        assert [
            tuple(getattr(j.task, f) for f in fields) for j in node.tasks
        ] == expected
        assert [j.state_index for j in node.tasks] == list(range(len(expected)))


@pytest.mark.parametrize("case", UNCHANGED)
def test_partial_cases_any_worker(case, any_worker, tmp_path):
    wf, kwargs, expected = UNCHANGED[case]
    assert run_wf(wf(**kwargs), any_worker, tmp_path) == expected


# ----------------------------------------------------------------------------------
# Wrong today: corrected (#945 lists them; checksums change only here)


CORRECTED = {
    "ragged": (Ragged, dict(n=[1, 3]), [11, 31, 32, 33]),
    "ragged-combined": (RaggedComb, dict(n=[1, 3]), [[11], [31, 32, 33]]),
    "combined-source": (UpComb, dict(x=[1, 2], y=[10, 100]), [10, 100, 20, 200]),
    "mixed-scalar-and-list": (MixedAny, dict(x=[1, 2, 3]), [1, 2, 2, 3, 6]),
    "reader-combines-static-field": (
        StaticD,
        dict(x=[1, 2], d=[100, 200]),
        [[101, 201], [102, 202], [103, 203], [102, 202], [104, 204], [106, 206]],
    ),
    "empty-split": (EmptySplit, dict(go=[True, False], x=3), [[6], []]),
    "combine-inner-and-source": (CombBoth, dict(x=[1, 2]), [1, 2, 3, 2, 4, 6]),
}


@pytest.mark.parametrize("case", CORRECTED)
def test_wrong_today_corrected(case, any_worker, tmp_path):
    wf, kwargs, expected = CORRECTED[case]
    assert run_wf(wf(**kwargs), any_worker, tmp_path) == expected


def test_ragged_inner_split(submitter):
    runner = runner_for(submitter, Ragged(n=[1, 3]))
    drain(runner)
    b = runner.exec_graph["B"]
    assert [j.state_key for j in b.tasks] == [(0, 0), (1, 0), (1, 1), (1, 2)]
    assert [j.state_index for j in b.tasks] == [0, 1, 2, 3]
    assert outputs(runner) == [11, 31, 32, 33]


def test_scalar_upstream_is_one_element_block(submitter):
    """A non-sequence upstream value is a block of one element, as today"""
    runner = runner_for(submitter, BadInput(xs=[0, 2]))
    drain(runner)
    b = runner.exec_graph["B"]
    assert [j.state_key for j in b.tasks] == [(0, 0), (1, 0)]
    assert outputs(runner) == [1, 3]


# ----------------------------------------------------------------------------------
# Combiners and barrier shapes


def test_combine_source_without_inner_on_node_raises(any_worker, tmp_path):
    """Combining the block axis without its inner input on the node itself is a
    ``PydraStateError``, reported as a resolution error"""
    with pytest.raises(RuntimeError, match="PydraStateError|combining"):
        run_wf(CombAOnNode(x=[1, 2]), any_worker, tmp_path)


@python.define
def LL(x: int) -> list[list[int]]:
    return [[x, x + 1], [x + 2, x + 3]]


@python.define
def AnyId(x: ty.Any) -> ty.Any:
    return x


@workflow.define
def UserNdim(x: list[int]):
    a = workflow.add(LL().split("x", x=x), name="A")
    b = workflow.add(AnyId().split("x", x=a.out, container_ndim={"x": 2}), name="B")
    return b.out


@workflow.define
def AmbigComb(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b1 = workflow.add(Mul(y=1).split("x", x=a.out), name="B1")
    b2 = workflow.add(Mul(y=1000).split("x", x=a.out), name="B2")
    c = workflow.add(Add(x=b1.out, d=b2.out).combine("A.x"), name="C")
    return c.out


# the outputs for shapes 2-4
BARRIER_VALUES = {
    2: [1001, 2002, 3003, 2002, 4004, 6006],
    3: [1, 2, 3, 4],
    4: [[1], [2], [3], [2], [4], [6]],
}


BARRIER_SHAPES = {
    # shape: (workflow, the barrier node, its output, or the error its entry names)
    1: (ZipStatic(x=[1, 2], y=[1, 2, 3, 4, 5, 6]), "B", "do not have same shape"),
    2: (ZipPartials(x=[1, 2]), "C", BARRIER_VALUES[2]),
    3: (UserNdim(x=[1, 2]), "B", BARRIER_VALUES[3]),
    4: (DownCombAx(x=[1, 2]), "C", BARRIER_VALUES[4]),
    5: (SameSrc(x=[1, 2]), "B", "unsupported operand"),
    6: (AmbigComb(x=[1, 2]), "C", "KeyError"),
}


@pytest.mark.parametrize("shape", sorted(BARRIER_SHAPES))
def test_barrier_shape_waits(shape, submitter):
    """Each of the six barrier shapes waits for every upstream node: holding A's last
    element back, the node has no job; then it gives today's output or error, recorded
    as an entry rather than raised out of ``ready()``"""
    task, name, expected = BARRIER_SHAPES[shape]
    runner = runner_for(submitter, task)
    jobs = runner.ready().dispatch
    held = [j for j in jobs if j.name == "A"][-1]
    while jobs:
        for job in jobs:
            if job is not held:
                finish(runner, job)
        jobs = runner.ready().dispatch
    node = runner.exec_graph[name]
    assert node.mode == "barrier"
    assert not node.started and not node.queued and not node.successful
    finish(runner, held)
    drain(runner)
    assert runner.done
    if isinstance(expected, str):
        assert runner.failed
        assert any(
            e.job_name == name and expected in e.message + e.traceback
            for e in runner.entries
        ), runner.entries
    else:
        assert not runner.failed and outputs(runner) == expected


def test_barrier_shape_errors_unchanged(tmp_path):
    with pytest.raises(Exception, match="do not have same shape"):
        run_wf(ZipStatic(x=[1, 2], y=[1, 2, 3, 4, 5, 6]), "debug", tmp_path / "1")
    with pytest.raises(Exception, match="unsupported operand"):
        run_wf(SameSrc(x=[1, 2]), "debug", tmp_path / "2")


def test_barrier_reader_of_ragged_partial_node_raises(submitter):
    """A barrier reader of a ragged partial node is a ``PydraStateError``, recorded
    as a resolution entry for the whole node rather than raised out of ``ready()``"""
    runner = runner_for(submitter, RaggedDownCombAx(n=[1, 3]))
    drain(runner)
    assert runner.done and runner.failed
    c = runner.exec_graph["C"]
    assert isinstance(c.errored[None], PydraStateError)
    (entry,) = runner.entries
    assert entry.phase == "resolution" and entry.job_name == "C"
    assert entry.state_index is None and entry.state_key is None


# ----------------------------------------------------------------------------------
# Labels and workers


def test_error_names_use_the_key(any_worker, tmp_path):
    """On a partial node an error names the element by its key, whatever the timing"""
    with pytest.raises(
        RuntimeError, match=r"Job 'B\[1,0\]', inputs could not be resolved"
    ):
        run_wf(BadInput(xs=[0, 1, 2, 3]), any_worker, tmp_path)


def test_entry_of_partial_node_carries_key(submitter):
    runner = runner_for(submitter, BadInput(xs=[0, 1, 2]))
    drain(runner)
    (entry,) = runner.entries
    assert (entry.job_name, entry.state_index, entry.state_key, entry.phase) == (
        "B",
        None,
        (1, 0),
        "resolution",
    )


def test_slurm_array_eligible_partial():
    from pydra.workers.slurm import Worker as SlurmWorker

    worker = SlurmWorker.__new__(SlurmWorker)
    worker.max_array_size = 10
    worker.sbatch_args = ""

    class FakeJob:
        def __init__(self, state_index, state_key):
            self.state_index = state_index
            self.state_key = state_key

    assert worker._array_eligible(FakeJob(None, (0, 1)))
    assert worker._array_eligible(FakeJob(3, None))
    assert not worker._array_eligible(FakeJob(None, None))


# ----------------------------------------------------------------------------------
# plan(), upstream(), entries and the ready() fixpoint


def _plan(runner):
    return {n.path: n for n in runner.plan().nodes}


def _live_tables(node):
    return (dict(node._tasks or {}), dict(node.blocked or {}), dict(node.queued))


def test_plan_counts_the_elements_of_expanded_blocks(submitter):
    """A partial node's plan counts the elements of the blocks whose upstream elements
    the runner has accepted, including blocks ``ready`` has not expanded yet, and
    leaves the live node's tables alone"""
    runner = runner_for(submitter, Records(n=[2, 3]))
    a0, a1 = runner.ready().dispatch
    b = _plan(runner)["B"]
    assert (b.elements, b.known_elements, b.state) == (None, 0, "deferred")
    finish(runner, a1)  # no ready() since: the live node has not expanded the block
    live = runner.exec_graph.node("B")
    before = _live_tables(live)
    b = _plan(runner)["B"]
    assert (b.elements, b.known_elements, b.state) == (None, 3, "pending")
    assert _live_tables(live) == before
    released = runner.ready().dispatch
    assert sorted(j.task.x for j in released) == [30, 31, 32]
    finish(runner, a0)
    b = _plan(runner)["B"]
    assert (b.elements, b.known_elements) == (5, 5)  # every block is expanded
    for job in released:
        finish(runner, job)
    drain(runner)
    assert runner.done and not runner.failed and outputs(runner) == [86, 192]


def test_planned_partial_run_equals_unplanned(tmp_path):
    def run(plan_each_round):
        with Submitter(
            worker="debug", cache_root=tmp_path / str(plan_each_round)
        ) as sub:
            runner = runner_for(sub, Records(n=[2, 3]))
            seen = []
            while True:
                if plan_each_round:
                    runner.plan()
                jobs = runner.ready().dispatch
                if not jobs:
                    break
                for job in reversed(jobs):  # complete later elements first
                    seen.append((job.name, job.state_key or job.state_index))
                    finish(runner, job)
                    if plan_each_round:
                        runner.plan()
            return seen, outputs(runner), runner.failed

    planned = run(True)
    assert planned == run(False)
    assert len(planned[0]) == 2 + 5 + 5 + 2


def test_upstream_names_the_source_element_of_a_block(submitter):
    runner = runner_for(submitter, Records(n=[2, 3]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a1)
    released = runner.ready().dispatch
    assert {j.name for j in released} == {"B"}
    for job in released:
        assert runner.upstream(job) == {"x": [(a1, 1)]}


def test_partial_bookkeeping_is_progress_for_the_ready_fixpoint(submitter):
    """A partial node registers blocks, closes groups and seals without changing its
    tables (each bumps ``version``), so ``_progress`` must see the version and the
    seal"""
    runner = runner_for(submitter, Records(n=[2, 3]))
    runner.ready()
    assert runner.ready().dispatch == []  # B starts
    live = runner.exec_graph.node("B")
    assert live.is_partial
    before, tables = runner._progress(), _live_tables(live)
    live._partial.version += 1  # what registering a block or closing a group does
    assert _live_tables(live) == tables
    assert runner._progress() != before
    before = runner._progress()
    live._partial.is_sealed = True
    assert runner._progress() != before


def test_partial_entries_name_the_key(submitter):
    runner = runner_for(submitter, Records(n=[2, 3]))
    a0, a1 = runner.ready().dispatch
    finish(runner, a0)
    finish(runner, a1)
    b = runner.ready().dispatch
    victim = next(j for j in b if j.state_key is not None)
    runner.complete(victim, Failed(ValueError("segment failed")))
    for job in b:
        if job is not victim:
            finish(runner, job)
    drain(runner)
    (entry,) = [e for e in runner.entries if e.job_name == "B"]
    assert entry.state_key == victim.state_key and entry.phase == "run"
    assert entry.state_index is None


# ----------------------------------------------------------------------------------
# Early-release guards: groups that span blocks, whole reads, failed blocks


@workflow.define
def AllComb(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Ident().split("x", x=a.out), name="B")
    c = workflow.add(Ident(x=b.out).combine(["B.x", "A.x"]), name="C")
    d = workflow.add(SumL(x=c.out), name="D")
    return d.out


@workflow.define
def GroupByY(x: list[int], y: list[int]):
    a = workflow.add(FailOn2().split("x", x=x), name="A")
    b = workflow.add(
        Mul().split(["x", "y"], x=a.out, y=y).combine(["x", "A.x"]), name="B"
    )
    c = workflow.add(SumL(x=b.out), name="C")
    return c.out


@workflow.define
def GroupByYOk(x: list[int], y: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(
        Mul().split(["x", "y"], x=a.out, y=y).combine(["x", "A.x"]), name="B"
    )
    c = workflow.add(SumL(x=b.out), name="C")
    return c.out


def _drain_holding(runner, keep):
    """Drain, holding back the jobs ``keep`` selects; return the names dispatched"""
    seen, held = [], []
    jobs = runner.ready().dispatch
    while jobs:
        for job in jobs:
            seen.append(job.name)
            if keep(job):
                held.append(job)
            else:
                finish(runner, job)
        jobs = runner.ready().dispatch
    return seen, held


def test_group_spanning_blocks_waits_for_every_block(submitter):
    """B groups by y across every block of A, so C's read of a group waits for all of
    A's elements, not only those whose blocks are expanded"""
    runner = runner_for(submitter, GroupByYOk(x=[1, 2, 3], y=[10, 100]))
    seen, held = _drain_holding(runner, lambda j: j.name == "A" and j.state_index == 1)
    assert "B" in seen and "C" not in seen
    (a1,) = held
    finish(runner, a1)
    drain(runner)
    assert runner.done and not runner.failed
    assert outputs(runner) == [360, 3600]


def test_whole_read_waits_for_the_seal(submitter):
    """D reads all of C, which combines every element: it waits for the seal"""
    runner = runner_for(submitter, AllComb(x=[1, 2, 3]))
    seen, held = _drain_holding(runner, lambda j: j.name == "A" and j.state_index != 0)
    assert "B" in seen and "D" not in seen and not runner.failed
    for job in held:
        finish(runner, job)
    drain(runner)
    assert runner.done and not runner.failed
    assert outputs(runner) == 6 * (1 + 2 + 3)


def test_group_with_a_failed_block_is_unrunnable(submitter):
    """A[1] fails, so every group of B (by y) misses its members: C never runs, and
    each of C's two elements is reported unrunnable exactly once"""
    runner = runner_for(submitter, GroupByY(x=[1, 2, 3], y=[10, 100]))
    reported = []
    jobs = runner.ready().dispatch
    dispatched = []
    while True:
        dispatched += [j.name for j in jobs]
        for job in jobs:
            finish(runner, job)
        rnd = runner.ready()
        reported += [(u.node, u.index) for u in rnd.unrunnable]
        jobs = rnd.dispatch
        if not jobs and not rnd.unrunnable and not rnd.settled:
            break
    assert runner.done and runner.failed
    assert "C" not in dispatched
    of_c = [i for n, i in reported if n == "C"]
    assert len(of_c) == 2 and set(of_c) == {(0,), (1,)}, of_c


@workflow.define
def PerElement(x: list[int]):
    a = workflow.add(L().split("x", x=x), name="A")
    b = workflow.add(Inc().split("x", x=a.out), name="B")
    c = workflow.add(Inc().split("x", x=a.out).combine("x"), name="C")
    d = workflow.add(SumL(x=c.out), name="D")
    return d.out


def test_expand_ready_visits_only_changed_blocks(submitter, monkeypatch):
    """With one upstream element completing per round (an async pool), each pending
    block is visited when it is registered and when its source element finishes, not
    on every round: the visits grow linearly, not quadratically"""
    from pydra.engine import partial

    visits = []
    candidates = partial.PartialExpansion._candidates

    def counted(self):
        found = candidates(self)
        visits.append(len(found))
        return found

    monkeypatch.setattr(partial.PartialExpansion, "_candidates", counted)
    n = 60
    runner = runner_for(submitter, PerElement(x=list(range(n))))
    held = []
    while True:
        jobs = runner.ready().dispatch
        held += [j for j in jobs if j.name == "A"]
        others = [j for j in jobs if j.name != "A"]
        batch, held = held[:1], held[1:]
        if not (batch or others):
            break
        for job in batch + others:
            finish(runner, job)
    assert runner.done and not runner.failed
    # B and C, n blocks each: a visit at registration (first pass), one when the
    # block's A element finishes, and a few more for their D/group closures; visiting
    # every pending block on every round would be about n * n
    assert sum(visits) <= 6 * n, sum(visits)
